# src/model.py

"""
Updated model.py

Temporal Graph Attention UNet (T-GAT-UNet) with:
- GAT-based encoder/decoder
- Transformer bottleneck
- Learnable Graph Structure (GSL)
- Skip connections
- Optional feedback from GAT attention to GSL
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from graph import GraphLearner


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.heads = heads

        self.layers.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.act = nn.ReLU()

    def forward(self, x, edge_index, return_attention=False):
        attentions = []
        for gat in self.layers:
            if return_attention:
                x, attn = gat(x, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                x = gat(x, edge_index)
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
        self.num_layers = num_layers

        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.layers.append(GATConv(hidden_channels, out_channels, heads=1))
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        for gat in self.layers[:-1]:
            x = self.act(gat(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


class TGATUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 encoder_layers=3, decoder_layers=3,
                 heads=4, time_k=1, top_k=3,
                 trans_nhead=4, trans_layers=2, trans_dim_feedforward=512,
                 shared_graph=True, use_skip=True, use_attention_feedback=False):
        super().__init__()
        self.time_k = time_k
        self.use_skip = use_skip
        self.use_attention_feedback = use_attention_feedback
        self.shared_graph = shared_graph

        self.graph_learner = GraphLearner(in_channels, hidden_dim=hidden_channels, top_k=top_k)

        self.encoder = GraphEncoder(in_channels, hidden_channels, num_layers=encoder_layers, heads=heads)
        self.bottleneck = TransformerBottleneck(hidden_channels, nhead=trans_nhead,
                                                num_layers=trans_layers, dim_feedforward=trans_dim_feedforward)
        self.decoder = GraphDecoder(hidden_channels, out_channels, num_layers=decoder_layers, heads=heads)

        if self.use_skip:
            self.skip_proj = nn.Linear(in_channels, hidden_channels)


    def forward(self, window, return_attention=False):
        device = window.device
        x = window  # shape: [T, C]

        adj, edge_index = self.graph_learner(x)
        edge_index = edge_index.to(device)

        # Encoder
        if return_attention or self.use_attention_feedback:
            h_enc, encoder_attn = self.encoder(x, edge_index, return_attention=True)
        else:
            h_enc = self.encoder(x, edge_index)

        # Skip connection storage (assumes 3-layer symmetry)
        skip_x = x if self.use_skip else None

        # Transformer
        h_trans = self.bottleneck(h_enc.unsqueeze(0)).squeeze(0)  # [1, T, H] -> [T, H]

        # Optionally combine skip (project skip_x to hidden_channels)
        if self.use_skip:
            skip_x_proj = self.skip_proj(skip_x)
            h_trans = h_trans + skip_x_proj

        # Decoder
        out = self.decoder(h_trans, edge_index)  # [T, C]

        if return_attention:
            return out.t(), encoder_attn, adj  # [C, T], list of attn, soft adj
        return out.t()  # [C, T]

    def get_graph_structure(self, x):
        """Expose graph structure externally for inspection/visualization."""
        adj, edge_index = self.graph_learner(x)
        return adj, edge_index
