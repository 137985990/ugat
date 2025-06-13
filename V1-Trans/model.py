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
        data = build_graph(window, time_k=self.time_k)
        # 兼容data.x/data.edge_index为None的情况
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
        # Decode
        if return_attention:
            out, decoder_attn = self.decoder(h, edge_index, return_attention=True)  # [T, out_channels], list
            # Return output and attention maps
            return out.t(), encoder_attn, decoder_attn
        out = self.decoder(h, edge_index)  # [T, out_channels]
        # Return as (out_channels, T)
        return out.t()
