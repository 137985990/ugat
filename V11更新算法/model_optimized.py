# model_optimized.py - 优化版模型架构

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from graph_cache import build_graph_cached
import math

class LightweightGATConv(nn.Module):
    """轻量化GAT卷积层"""
    
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        
        # 使用分组卷积减少参数量
        self.lin_src = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.lin_dst = nn.Linear(in_channels, out_channels * heads, bias=False)
        
        # 简化的注意力机制
        self.att_src = nn.Parameter(torch.randn(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.randn(1, heads, out_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.lin_src.weight, gain=gain)
        nn.init.xavier_uniform_(self.lin_dst.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_src, gain=gain)
        nn.init.xavier_uniform_(self.att_dst, gain=gain)
    
    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E]
        
        H, C = self.heads, self.out_channels
        N = x.size(0)
        
        # 线性变换
        x_src = self.lin_src(x).view(N, H, C)  # [N, H, C]
        x_dst = self.lin_dst(x).view(N, H, C)  # [N, H, C]
        
        # 计算注意力分数
        alpha_src = (x_src * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)  # [N, H]
        
        # 边注意力
        row, col = edge_index
        alpha = alpha_src[row] + alpha_dst[col]  # [E, H]
        alpha = torch.softmax(alpha, dim=0)
        alpha = self.dropout(alpha)
        
        # 消息传递
        out = torch.zeros_like(x_dst)  # [N, H, C]
        for h in range(H):
            x_h = x_src[:, h, :]  # [N, C]
            alpha_h = alpha[:, h]  # [E]
            
            # 稀疏矩阵乘法
            messages = x_h[row] * alpha_h.unsqueeze(-1)  # [E, C]
            out[:, h, :] = torch.zeros(N, C, device=x.device).scatter_add_(0, col.unsqueeze(-1).expand(-1, C), messages)
        
        return out.mean(dim=1)  # [N, C] 平均多头结果

class EfficientTransformerBlock(nn.Module):
    """高效的Transformer块"""
    
    def __init__(self, d_model, nhead=4, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        dim_feedforward = dim_feedforward or d_model * 2  # 减少FFN维度
        
        # 使用分组注意力减少计算量
        assert d_model % nhead == 0
        self.head_dim = d_model // nhead
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
        # 简化的FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, T, d_model]
        B, T, C = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, nhead, T, head_dim]
        
        # 计算注意力（简化版）
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, nhead, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)  # [B, T, C]
        out = self.proj(out)
        
        x = residual + self.dropout(out)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x

class OptimizedTGATUNet(nn.Module):
    """优化版T-GAT-UNet"""
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 encoder_layers=2, decoder_layers=2,  # 减少层数
                 heads=2, time_k=1,  # 减少头数
                 trans_layers=1,  # 减少Transformer层数
                 num_classes=2,
                 use_checkpoint=True):  # 梯度检查点
        super().__init__()
        
        self.time_k = time_k
        self.use_checkpoint = use_checkpoint
        
        # 轻量化编码器
        self.encoder_layers = nn.ModuleList()
        current_dim = in_channels
        for i in range(encoder_layers):
            self.encoder_layers.append(
                LightweightGATConv(current_dim, hidden_channels, heads=heads)
            )
            current_dim = hidden_channels
        
        # 简化的Transformer
        self.transformer_layers = nn.ModuleList()
        for _ in range(trans_layers):
            self.transformer_layers.append(
                EfficientTransformerBlock(hidden_channels, nhead=heads)
            )
        
        # 轻量化解码器
        self.decoder_layers = nn.ModuleList()
        current_dim = hidden_channels
        for i in range(decoder_layers - 1):
            self.decoder_layers.append(
                LightweightGATConv(current_dim, hidden_channels, heads=heads)
            )
        # 最后一层输出
        self.decoder_layers.append(
            LightweightGATConv(hidden_channels, out_channels, heads=1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes)
        )
        
        self.act = nn.ReLU(inplace=True)  # 使用inplace操作节省内存
    
    def _checkpoint_forward(self, func, *args):
        """梯度检查点包装"""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args)
        else:
            return func(*args)
    
    def forward(self, window, return_attention=False, phase="encode"):
        device = window.device
        
        # 使用缓存的图构建
        data = build_graph_cached(window, time_k=self.time_k)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # 编码阶段
        for layer in self.encoder_layers:
            x = self._checkpoint_forward(layer, x, edge_index)
            x = self.act(x)
        
        # Transformer处理
        h_trans = x.unsqueeze(0)  # [1, T, hidden]
        for transformer in self.transformer_layers:
            h_trans = self._checkpoint_forward(transformer, h_trans)
        h = h_trans.squeeze(0)  # [T, hidden]
        
        # 分类分支
        h_cls = h.mean(dim=0)  # 全局平均池化
        logits = self.classifier(h_cls.unsqueeze(0).unsqueeze(-1)).squeeze()
        
        # 解码阶段
        for layer in self.decoder_layers:
            h = self._checkpoint_forward(layer, h, edge_index)
            if layer != self.decoder_layers[-1]:  # 最后一层不加激活
                h = self.act(h)
        
        out = h.t()  # [out_channels, T]
        
        return out, logits

def create_optimized_model(config):
    """创建优化版模型"""
    return OptimizedTGATUNet(
        in_channels=config.get('in_channels', 32),
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=config.get('out_channels', 32),
        encoder_layers=config.get('encoder_layers', 2),
        decoder_layers=config.get('decoder_layers', 2),
        heads=config.get('attention_heads', 2),
        time_k=config.get('time_k', 1),
        trans_layers=config.get('transformer_layers', 1),
        num_classes=config.get('num_classes', 2),
        use_checkpoint=config.get('use_gradient_checkpoint', True)
    )
