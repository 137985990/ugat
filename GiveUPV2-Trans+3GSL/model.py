# src/model.py

"""
Updated model.py for multi-layer GSL

Temporal Graph Attention UNet (T-GAT-UNet) with:
- GAT-based encoder/decoder
- Transformer bottleneck
- One GSL module per layer (MultiLayerGraphLearner)
- Skip connections
- Optional feedback from GAT attention to GSL
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from graph import MultiLayerGraphLearner


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        self.layers.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.act = nn.ReLU()

    def forward(self, x, edge_indices, return_attention=False):
        attentions = []
        for i, gat in enumerate(self.layers):
            if return_attention:
                x, attn = gat(x, edge_indices[i], return_attention_weights=True)
                attentions.append(attn)
            else:
                x = gat(x, edge_indices[i])
            x = self.act(x)
        if return_attention:
            return x, attentions
        return x


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
        return self.transformer(x)


class GraphDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.layers.append(GATConv(hidden_channels, out_channels, heads=1))
        self.act = nn.ReLU()

    def forward(self, x, edge_indices):
        for i, gat in enumerate(self.layers[:-1]):
            x = self.act(gat(x, edge_indices[i]))
        x = self.layers[-1](x, edge_indices[-1])
        return x


class TGATUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 encoder_layers=3, decoder_layers=3,
                 heads=4, top_k=3,
                 trans_nhead=4, trans_layers=2, trans_dim_feedforward=512,
                 use_skip=True, use_attention_feedback=False):
        super().__init__()
        self.use_skip = use_skip
        self.use_attention_feedback = use_attention_feedback

        self.graph_learner = MultiLayerGraphLearner(in_channels, hidden_dim=hidden_channels, top_k=top_k,
                                                    num_layers=encoder_layers + decoder_layers)

        self.encoder = GraphEncoder(in_channels, hidden_channels, num_layers=encoder_layers, heads=heads)
        self.bottleneck = TransformerBottleneck(hidden_channels, nhead=trans_nhead,
                                                num_layers=trans_layers, dim_feedforward=trans_dim_feedforward)
        self.decoder = GraphDecoder(hidden_channels, out_channels, num_layers=decoder_layers, heads=heads)

        if self.use_skip:
            self.skip_proj = nn.Linear(in_channels, hidden_channels)

    def forward(self, window, return_attention=False):
        device = window.device
        x = window  # [T, C]

        graph_structs = self.graph_learner(x)
        adjs, edge_indices = zip(*graph_structs)
        edge_indices = [ei.to(device) for ei in edge_indices]

        encoder_edge_indices = edge_indices[:len(self.encoder.layers)]
        decoder_edge_indices = edge_indices[len(self.encoder.layers):]

        if return_attention or self.use_attention_feedback:
            h_enc, encoder_attn = self.encoder(x, encoder_edge_indices, return_attention=True)
        else:
            h_enc = self.encoder(x, encoder_edge_indices)

        skip_x = x if self.use_skip else None
        h_trans = self.bottleneck(h_enc.unsqueeze(0)).squeeze(0)

        if self.use_skip:
            skip_x_proj = self.skip_proj(skip_x)
            h_trans = h_trans + skip_x_proj

        out = self.decoder(h_trans, decoder_edge_indices)

        if return_attention:
            return out.t(), encoder_attn, adjs  # [C, T], list of attention, list of adjs
        return out.t()

    def get_graph_structure(self, x):
        return self.graph_learner(x)
