# src/graph.py

"""
graph.py

Module for building time-level graph data structures from windowed time series tensors,  # 从窗口化时间序列张量构建时间级图数据结构的模块
suitable for Graph Attention Networks (GAT).  # 适用于图注意力网络 (GAT)
"""

import torch
from torch_geometric.data import Data

def build_graph(
    window: torch.Tensor,
    time_k: int = 1
) -> Data:
    """
    Build a time-level graph Data object from a window tensor for GAT.  # 为 GAT 从窗口张量构建时间级图数据对象

    Parameters  # 参数
    ----------
    window : torch.Tensor  # 窗口张量
        Tensor of shape (T, C) containing node features (time nodes).  # 形状为 (T, C) 的张量，包含节点特征（时间节点）
    time_k : int  # 时间步数
        Each time node connects to neighbors within +/- time_k steps.  # 每个时间节点连接到其 +/- time_k 步内的邻居

    Returns  # 返回值
    -------
    data : torch_geometric.data.Data  # 图数据对象
        Graph with attributes:  # 图的属性：
          x : node feature matrix of shape [T, C]  # 节点特征矩阵，形状为 [T, C]
          edge_index : LongTensor of shape [2, num_edges]  # 边索引，形状为 [2, num_edges]
    """
    # window: [T, C]  # 窗口张量，形状为 [T, C]
    x = window
    T = x.size(0)  # 时间节点数量
    edges = []
    for i in range(T):
        for j in range(max(0, i - time_k), min(T, i + time_k + 1)):  # 遍历时间步范围内的邻居
            edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # 构建边索引
    return Data(x=x, edge_index=edge_index)  # 返回图数据对象
