# src/graph.py

"""
Graph module with:
- Static graph builder (build_graph)
- Learnable multi-layer Graph Structure Learning (GSL)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data


def build_graph(window: torch.Tensor, time_k: int = 1) -> Data:
    x = window
    T = x.size(0)
    edges = []
    for i in range(T):
        for j in range(max(0, i - time_k), min(T, i + time_k + 1)):
            edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def build_topk_edge_index(adj: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Builds edge_index from an adjacency matrix by selecting top-k neighbors for each node.
    This version is vectorized for efficiency.
    """
    T = adj.size(0)
    k = min(top_k, T)

    # Get top-k scores and their indices for each node
    _, topk_indices = torch.topk(adj, k=k, dim=-1)

    # Create source node indices (e.g., [0,0,0, 1,1,1, ...])
    row_indices = torch.arange(T, device=adj.device).view(-1, 1).repeat(1, k)

    # Stack to create edge_index: [2, T * k]
    edge_index = torch.stack([row_indices.flatten(), topk_indices.flatten()], dim=0)
    
    return edge_index.contiguous()


class GraphLearner(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)  # [T, hidden]
        sim = torch.matmul(h, h.t())  # [T, T]
        adj = torch.softmax(sim, dim=-1)
        edge_index = build_topk_edge_index(adj, self.top_k)
        return adj, edge_index


class MultiLayerGraphLearner(nn.Module):
    """
    Graph learner with separate learnable graphs for each layer.
    """
    def __init__(self, in_dim, hidden_dim=32, top_k=3, num_layers=3):
        super().__init__()
        self.learners = nn.ModuleList([
            GraphLearner(in_dim, hidden_dim, top_k)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [learner(x) for learner in self.learners]