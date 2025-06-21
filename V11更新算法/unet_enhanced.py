# unet_enhanced.py - 增强版U-Net架构

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from graph_cache import build_graph_cached
import math
from typing import List, Tuple, Dict

class SkipConnection(nn.Module):
    """跳跃连接模块"""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, reduction_ratio: int = 4):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # 特征对齐层：将编码器特征映射到解码器维度
        self.feature_align = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力门控：决定使用多少跳跃信息
        self.attention_gate = nn.Sequential(
            nn.Linear(encoder_dim + decoder_dim, decoder_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_dim // reduction_ratio, decoder_dim),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_feat: [N, encoder_dim] 编码器特征
            decoder_feat: [N, decoder_dim] 解码器特征
        Returns:
            fused_feat: [N, decoder_dim] 融合特征
        """
        # 特征对齐
        aligned_encoder = self.feature_align(encoder_feat)  # [N, decoder_dim]
        
        # 注意力门控
        concat_feat = torch.cat([encoder_feat, decoder_feat], dim=-1)  # [N, encoder_dim + decoder_dim]
        attention_weights = self.attention_gate(concat_feat)  # [N, decoder_dim]
        
        # 加权融合
        gated_encoder = aligned_encoder * attention_weights  # [N, decoder_dim]
        fused_input = torch.cat([gated_encoder, decoder_feat], dim=-1)  # [N, 2*decoder_dim]
        
        return self.fusion(fused_input)  # [N, decoder_dim]

class MultiScaleGATEncoder(nn.Module):
    """多尺度GAT编码器，输出多层特征用于跳跃连接"""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 4, heads: int = 4):
        super().__init__()
        self.num_layers = num_layers
        
        # 构建多层编码器
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        current_dim = in_channels
        self.layer_dims = [current_dim]
        
        for i in range(num_layers):
            # 逐层增加特征维度
            out_dim = hidden_channels * (2 ** min(i, 2))  # 最多增加到4倍
            
            self.layers.append(
                GATConv(current_dim, out_dim // heads, heads=heads, dropout=0.1)
            )
            self.norms.append(nn.LayerNorm(out_dim))
            
            current_dim = out_dim
            self.layer_dims.append(current_dim)
        
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            features: List[Tensor] 每层的特征，用于跳跃连接
        """
        features = [x]  # 包含输入特征
        
        for i, (gat_layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = gat_layer(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = self.dropout(x)
            features.append(x)
        
        return features

class MultiScaleGATDecoder(nn.Module):
    """多尺度GAT解码器，使用跳跃连接"""
    
    def __init__(self, encoder_dims: List[int], hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.num_layers = len(encoder_dims) - 1
        
        # 解码器层（逆序）
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() 
        self.skip_connections = nn.ModuleList()
        
        # 从最深层开始解码
        current_dim = encoder_dims[-1]  # 最深层维度
        
        for i in range(self.num_layers):
            # 对应的编码器层维度（逆序）
            encoder_layer_dim = encoder_dims[-(i+2)]  # 跳过当前层，取上一层
            
            # 目标维度：逐层减少
            if i == self.num_layers - 1:
                target_dim = out_channels
            else:
                target_dim = hidden_channels // (2 ** min(i, 2))
            
            # GAT解码层
            self.layers.append(
                GATConv(current_dim, target_dim // heads, heads=heads, dropout=0.1)
            )
            self.norms.append(nn.LayerNorm(target_dim))
            
            # 跳跃连接
            self.skip_connections.append(
                SkipConnection(encoder_layer_dim, target_dim)
            )
            
            current_dim = target_dim
        
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, encoder_features: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: List[Tensor] 编码器各层特征
            edge_index: 图连接信息
        """
        # 从最深层特征开始
        x = encoder_features[-1]
        
        for i, (gat_layer, norm, skip_conn) in enumerate(zip(self.layers, self.norms, self.skip_connections)):
            # GAT解码
            x = gat_layer(x, edge_index)
            x = norm(x)
            x = self.act(x)
            
            # 跳跃连接：融合对应编码器层的特征
            encoder_feat = encoder_features[-(i+2)]  # 对应编码器层
            x = skip_conn(encoder_feat, x)
            
            x = self.dropout(x)
        
        return x

class UNetTGAT(nn.Module):
    """真正的U-Net架构的T-GAT模型"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 encoder_layers: int = 4, heads: int = 4, time_k: int = 1,
                 trans_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.time_k = time_k
        
        # 多尺度编码器
        self.encoder = MultiScaleGATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels, 
            num_layers=encoder_layers,
            heads=heads
        )
        
        # Transformer瓶颈（在最深层）
        bottleneck_dim = hidden_channels * (2 ** min(encoder_layers-1, 2))
        self.transformer_layers = nn.ModuleList()
        for _ in range(trans_layers):
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=bottleneck_dim,
                    nhead=heads,
                    dim_feedforward=bottleneck_dim * 2,
                    dropout=0.1,
                    batch_first=True
                )
            )
        
        # 多尺度解码器
        encoder_dims = [in_channels] + [hidden_channels * (2 ** min(i, 2)) for i in range(encoder_layers)]
        self.decoder = MultiScaleGATDecoder(
            encoder_dims=encoder_dims,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, window: torch.Tensor, return_skip_info: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        device = window.device
        
        # 构建图
        data = build_graph_cached(window, time_k=self.time_k)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        
        # 多尺度编码
        encoder_features = self.encoder(x, edge_index)
        
        # Transformer瓶颈处理
        bottleneck_feat = encoder_features[-1].unsqueeze(0)  # [1, N, D]
        for transformer in self.transformer_layers:
            bottleneck_feat = transformer(bottleneck_feat)
        encoder_features[-1] = bottleneck_feat.squeeze(0)  # [N, D]
        
        # 分类分支
        global_feat = encoder_features[-1].mean(dim=0)  # 全局池化
        logits = self.classifier(global_feat.unsqueeze(0).unsqueeze(-1)).squeeze()
        
        # 多尺度解码 + 跳跃连接
        decoded_feat = self.decoder(encoder_features, edge_index)
        
        out = decoded_feat.t()  # [out_channels, T]
        
        if return_skip_info:
            return out, logits, encoder_features
        return out, logits
