# src/model.py

"""
model.py

Module defining the Temporal Graph Attention U-Net (T-GAT-UNet) architecture,
with Graph-based encoder (GAT), Transformer bottleneck, and Graph-based decoder (GAT).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from graph import build_graph


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
        for gat in self.layers:
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
        for gat in self.layers[:-1]:
            if return_attention:
                x, attn = gat(x, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                x = gat(x, edge_index)
            x = self.act(x)
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
                 trans_nhead=4, trans_layers=2, trans_dim_feedforward=512,
                 num_classes=2,
                 use_discriminator=False):
        super().__init__()
        print(f"[DEBUG] TGATUNet in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}")
        self.time_k = time_k
        # Modules
        self.encoder = GraphEncoder(in_channels, hidden_channels, num_layers=encoder_layers, heads=heads)
        self.bottleneck = TransformerBottleneck(hidden_channels,
                                                nhead=trans_nhead,
                                                num_layers=trans_layers,
                                                dim_feedforward=trans_dim_feedforward)
        self.decoder = GraphDecoder(hidden_channels, out_channels,
                                    num_layers=decoder_layers, heads=heads)
        # 分类头：全局池化后MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        # 判别器分支（可选，后续可扩展为独立判别器）
        self.use_discriminator = use_discriminator
        if use_discriminator:
            # 简单判别器：输入为解码后特征，输出为一标量（可扩展为更复杂结构）
            self.discriminator = nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Linear(out_channels // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.discriminator = None

    def forward(self, window, return_attention=False, phase="encode"):
        """
        phase: "encode" or "decode"
        - encode: 返回 (out, logits)
        - decode: 返回 (out, disc_pred)
        """
        device = window.device
        data = build_graph(window, time_k=self.time_k)
        x = data.x.to(device) if data.x is not None else None
        edge_index = data.edge_index.to(device) if data.edge_index is not None else None
        # Encode
        if return_attention:
            h, encoder_attn = self.encoder(x, edge_index, return_attention=True)  # [T, hidden], list
        else:
            h = self.encoder(x, edge_index)
        # Prepare for transformer: add batch dim
        h_trans = h.unsqueeze(0)  # [1, T, hidden]
        h_trans = self.bottleneck(h_trans)  # [1, T, hidden]
        h = h_trans.squeeze(0)  # [T, hidden]
        # 分类分支：全局平均池化
        h_cls = h.mean(dim=0)  # [hidden]
        logits = self.classifier(h_cls)  # [num_classes]
        # Decode
        if return_attention:
            out, decoder_attn = self.decoder(h, edge_index, return_attention=True)  # [T, out_channels], list
            # Return output, attention maps, and logits
            return out.t(), encoder_attn, decoder_attn, logits
        out = self.decoder(h, edge_index)  # [T, out_channels]
        out_t = out.t()  # [out_channels, T]
        if phase == "encode":
            # encode 阶段：返回 (out, logits)
            return out_t, logits
        elif phase == "decode":
            # decode 阶段：返回 (out, disc_pred)
            if self.use_discriminator and self.discriminator is not None:
                # 判别器输入：对每个时间点的补全结果做池化（可自定义）
                # 这里简单取均值池化
                pooled = out_t.mean(dim=1)  # [out_channels]
                disc_pred = self.discriminator(pooled)  # [1]
            else:
                disc_pred = None
            return out_t, disc_pred
        else:
            # 默认兼容旧接口
            return out_t, logits

    def forward_batch(self, windows_batch):
        """
        批量前向传播 - 充分利用16GB显存
        Args:
            windows_batch: [batch_size, T, C] 批量窗口数据
        Returns:
            batch_out: [batch_size, C, T] 批量重建输出            batch_logits: [batch_size, num_classes] 批量分类输出
        """
        batch_size, T, C = windows_batch.size()
        device = windows_batch.device
        
        # 批量处理所有样本的图构建
        batch_outputs = []
        batch_logits = []
          # 可以进一步优化：尝试向量化图构建
        for i in range(batch_size):
            window = windows_batch[i]  # [T, C]
            result = self.forward(window)  # 可能返回2个或4个值
            if len(result) == 2:
                out, logits = result
            elif len(result) == 4:
                out, _, _, logits = result
            else:
                out, logits = result[0], result[-1]  # 取第一个和最后一个
            batch_outputs.append(out)
            batch_logits.append(logits)
        
        # 堆叠结果
        batch_out = torch.stack(batch_outputs, dim=0)  # [batch_size, C, T]
        batch_logits = torch.stack(batch_logits, dim=0)  # [batch_size, num_classes]
        
        return batch_out, batch_logits
    
    def forward_batch_parallel(self, windows_batch):
        """
        并行批量前向传播 - 最大化显存利用
        使用torch.jit.script或其他并行化技术
        """
        batch_size, T, C = windows_batch.size()
        
        # 尝试使用编译优化的批量处理
        try:
            # 使用torch.compile进行批量优化
            @torch.compile
            def compiled_batch_forward(windows):
                return self.forward_batch(windows)
            
            return compiled_batch_forward(windows_batch)
        except:
            # 回退到标准批量处理
            return self.forward_batch(windows_batch)
