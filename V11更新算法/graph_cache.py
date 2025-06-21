# graph_cache.py - 图构建缓存优化模块

import torch
from torch_geometric.data import Data
import pickle
import os
from functools import lru_cache
import hashlib

class GraphCache:
    """图构建缓存管理器，避免重复构建相同结构的图"""
    
    def __init__(self, cache_dir="graph_cache", max_cache_size=1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.memory_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, T, time_k):
        """生成缓存键"""
        return f"T{T}_k{time_k}"
    
    def _build_edge_index(self, T, time_k):
        """构建边索引 - 只依赖于图结构参数"""
        edges = []
        for i in range(T):
            for j in range(max(0, i - time_k), min(T, i + time_k + 1)):
                edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def get_edge_index(self, T, time_k):
        """获取缓存的边索引"""
        cache_key = self._get_cache_key(T, time_k)
        
        # 1. 检查内存缓存
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # 2. 检查磁盘缓存
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    edge_index = pickle.load(f)
                self.memory_cache[cache_key] = edge_index
                return edge_index
            except:
                pass
        
        # 3. 构建新的边索引
        edge_index = self._build_edge_index(T, time_k)
        
        # 4. 保存到缓存
        self.memory_cache[cache_key] = edge_index
        if len(self.memory_cache) <= self.max_cache_size:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(edge_index, f)
            except:
                pass
        
        return edge_index

# 全局缓存实例
_graph_cache = GraphCache()

def build_graph_cached(window: torch.Tensor, time_k: int = 1) -> Data:
    """
    优化版本的图构建函数，使用缓存避免重复计算边索引
    
    Args:
        window: [T, C] 时间窗口数据
        time_k: 时间邻接范围
    
    Returns:
        Data: 图数据对象
    """
    T, C = window.shape
    x = window
    
    # 从缓存获取边索引
    edge_index = _graph_cache.get_edge_index(T, time_k)
    
    return Data(x=x, edge_index=edge_index)

@lru_cache(maxsize=128)
def get_static_edge_index(T: int, time_k: int):
    """静态边索引缓存装饰器版本"""
    edges = []
    for i in range(T):
        for j in range(max(0, i - time_k), min(T, i + time_k + 1)):
            edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
