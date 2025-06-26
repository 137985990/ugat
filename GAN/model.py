# src/model.py

"""
model.py

Module defining the GAN architecture with GAT-based generator and dual discriminators.
Replaces the original U-Net structure with GAN for time series completion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GATConv
except ImportError:
    print("Warning: torch_geometric not found. Please install: pip install torch_geometric")
    # Fallback implementation
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, heads=1):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels * heads)
            self.heads = heads
            self.out_channels = out_channels
            
        def forward(self, x, edge_index, return_attention_weights=False):
            out = self.linear(x)
            if return_attention_weights:
                return out, None
            return out

# ===================== GAN MODELS =====================

# ====== GAT-based GAN Generator for Time Series ======
class GANGenerator(nn.Module):
    """
    基于GAT的生成器：使用Graph Attention Network结构
    输入mask后的[batch, channels, time]，输出补全[batch, channels, time]
    """
    def __init__(self, in_channels, out_channels, seq_len, hidden_dim=128, time_k=1):
        super().__init__()
        self.time_k = time_k
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # GAT-based encoder-decoder architecture
        self.encoder = GraphEncoder(in_channels, hidden_dim, num_layers=3, heads=4)
        self.bottleneck = TransformerBottleneck(hidden_dim, nhead=4, num_layers=2)
        self.decoder = GraphDecoder(hidden_dim, out_channels, num_layers=3, heads=4)

    def forward(self, x):
        # x: [batch, channels, time] (mask后)
        batch_size = x.size(0)
        outputs = []
        
        for i in range(batch_size):
            # 处理单个样本: [channels, time] -> [time, channels]
            window = x[i].t()
            
            # 简化的图结构构建，直接使用时间序列特征
            # 创建简单的序列图：每个时间步连接到下一个时间步
            T, C = window.shape
            
            # 构建边索引（时间序列图）
            edge_index = []
            for t in range(T-1):
                edge_index.append([t, t+1])
                edge_index.append([t+1, t])  # 双向边
            
            if len(edge_index) > 0:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_index = edge_index.to(window.device)
            else:
                # 如果序列太短，创建自环
                edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(window.device)
            
            # 编码
            h = self.encoder(window, edge_index)
            
            # Transformer bottleneck
            h_trans = h.unsqueeze(0)  # [1, T, hidden]
            h_trans = self.bottleneck(h_trans)
            h = h_trans.squeeze(0)  # [T, hidden]
            
            # 解码
            out = self.decoder(h, edge_index)  # [T, out_channels]
            outputs.append(out.t())  # [out_channels, T]
        
        outputs = torch.stack(outputs, dim=0)  # [batch, channels, time]
        return torch.tanh(outputs)

# ====== Primary GAN Discriminator ======
class GANDiscriminator(nn.Module):
    """
    主要的GAN判别器：判断输入的时间序列是真实的还是生成的
    """
    def __init__(self, in_channels, seq_len, hidden_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim//4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
        )
        
        # 计算卷积后的序列长度
        conv_out_len = seq_len // 8  # 经过3次stride=2的卷积
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * conv_out_len, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [batch, channels, time]
        features = self.conv_layers(x)
        return self.classifier(features)

# ====== Secondary Classifier Discriminator ======
class ClassifierDiscriminator(nn.Module):
    """
    分类器判别器：原U-Net中的分类模型，作为辅助判别器
    不仅判断真假，还进行多类别分类任务
    """
    def __init__(self, in_channels, seq_len, num_classes=3, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim//4, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 真假判别头
        self.discriminator_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # 分类头
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x, return_features=False):
        # x: [batch, channels, time]
        features = self.feature_extractor(x)  # [batch, hidden_dim, 1]
        features = features.squeeze(-1)  # [batch, hidden_dim]
        
        # 真假判别
        disc_output = self.discriminator_head(features)
        
        # 分类输出
        class_output = self.classifier_head(features)
        
        if return_features:
            return disc_output, class_output, features
        return disc_output, class_output
# =============== END GAN MODELS ===============


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer: in_channels -> hidden_channels via multi-head
        self.layers.append(
            GATConv(in_channels, hidden_channels // heads, heads=heads)
        )
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads)
            )
        self.act = nn.ReLU()

    def forward(self, x, edge_index, return_attention=False):
        # x: [N_nodes, in_channels]
        attentions = []
        for i in range(len(self.layers)):
            gat = self.layers[i]
            if return_attention:
                x, attn = gat(x, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                x = gat(x, edge_index)
            x = self.act(x)
        if return_attention:
            return x, attentions  # [N_nodes, hidden_channels], list of attention weights
        return x  # [N_nodes, hidden_channels]


class TransformerBottleneck(nn.Module):
    def __init__(self, hidden_channels, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch_size, N_nodes, hidden_channels]
        return self.transformer(x)


class GraphDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # First decoder GAT
        self.layers.append(
            GATConv(hidden_channels, hidden_channels // heads, heads=heads)
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads)
            )
        # Last layer maps to out_channels
        self.layers.append(
            GATConv(hidden_channels, out_channels, heads=1)
        )
        self.act = nn.ReLU()

    def forward(self, x, edge_index, return_attention=False):
        attentions = []
        # 使用索引迭代
        for i in range(len(self.layers) - 1):
            gat = self.layers[i]
            if return_attention:
                x, attn = gat(x, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                x = gat(x, edge_index)
            x = self.act(x)
        
        # Last layer
        if return_attention:
            x, attn = self.layers[-1](x, edge_index, return_attention_weights=True)
            attentions.append(attn)
            return x, attentions  # [N_nodes, out_channels], list of attention weights
        x = self.layers[-1](x, edge_index)
        return x  # [N_nodes, out_channels]


class TGATUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 encoder_layers=3, decoder_layers=3,
                 heads=4, time_k=1,
                 trans_nhead=4, trans_layers=2, trans_dim_feedforward=512):
        super().__init__()
        self.time_k = time_k
        # Modules
        self.encoder = GraphEncoder(in_channels, hidden_channels, num_layers=encoder_layers, heads=heads)
        self.bottleneck = TransformerBottleneck(hidden_channels,
                                                nhead=trans_nhead,
                                                num_layers=trans_layers,
                                                dim_feedforward=trans_dim_feedforward)
        self.decoder = GraphDecoder(hidden_channels, out_channels,
                                    num_layers=decoder_layers, heads=heads)

    def forward(self, window, return_attention=False):
        # window: [T, C] tensor for a single sample
        device = window.device
        T, C = window.shape
        
        # 构建简单的时间序列图
        edge_index = []
        for t in range(T-1):
            edge_index.append([t, t+1])
            edge_index.append([t+1, t])  # 双向边
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        else:
            # 如果序列太短，创建自环
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        
        x = window  # [T, C]
        
        # Encode
        if return_attention:
            h, encoder_attn = self.encoder(x, edge_index, return_attention=True)  # [T, hidden], list
        else:
            h = self.encoder(x, edge_index)
        
        # Prepare for transformer: add batch dim
        h_trans = h.unsqueeze(0)  # [1, T, hidden]
        h_trans = self.bottleneck(h_trans)  # [1, T, hidden]
        h = h_trans.squeeze(0)  # [T, hidden]
        
        # Decode
        if return_attention:
            out, decoder_attn = self.decoder(h, edge_index, return_attention=True)  # [T, out_channels], list
            # Return output and attention maps
            return out.t(), encoder_attn, decoder_attn
        out = self.decoder(h, edge_index)  # [T, out_channels]
        
        # Return as (out_channels, T)
        return out.t()
