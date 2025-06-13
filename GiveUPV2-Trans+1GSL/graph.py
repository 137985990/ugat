# src/graph.py

"""
graph.py

Module for:
- Building graphs from time series
- Learning dynamic graph structures (GSL)
- Supporting both static and learned graph creation
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data


def build_graph(window: torch.Tensor, time_k: int = 1) -> Data:
    """
    Static graph builder based on time adjacency.
    Each time node connects to +/- time_k neighbors.
    """
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
    Build sparse edge_index from soft adjacency by keeping top-k edges per node.
    """
    T = adj.size(0)
    edge_index = []
    for i in range(T):
        scores = adj[i]
        topk = torch.topk(scores, k=min(top_k, T), largest=True)
        neighbors = topk.indices
        for j in neighbors:
            edge_index.append([i, j.item()])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


class GraphLearner(nn.Module):
    """
    Learnable Graph Learner: generates soft adjacency matrix and sparse edge_index.
    """
    def __init__(self, in_dim, hidden_dim=32, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [T, C]

        Returns:
            adj: Soft adjacency matrix [T, T]
            edge_index: Sparse edge_index built from top-k neighbors
        """
        h = self.proj(x)  # [T, hidden]
        sim = torch.matmul(h, h.t())  # [T, T] similarity matrix
        adj = torch.softmax(sim, dim=-1)  # normalize per row
        edge_index = build_topk_edge_index(adj, self.top_k)
        return adj, edge_index
