# optimized_data_loader.py
"""
优化的数据加载器，减少内存占用，提升显存利用率
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple, Optional
import gc


class MemoryEfficientDataset(Dataset):
    """内存高效的数据集，按需加载数据"""
    
    def __init__(self, 
                 data_files: List[str],
                 feature_cols: List[str],
                 block_col: str,
                 label_col: str,
                 window_size: int,
                 step_size: int,
                 normalize: str = 'zscore',
                 need_indices: Optional[List[int]] = None,
                 phase: str = "encode"):
        
        self.data_files = data_files
        self.feature_cols = feature_cols
        self.block_col = block_col
        self.label_col = label_col
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        self.need_indices = need_indices if need_indices is not None else []
        self.phase = phase
        
        # 只存储索引信息，不加载实际数据
        self._build_index()
        
    def _build_index(self):
        """构建数据索引，不加载实际数据到内存"""
        self.file_indices = []  # (file_idx, block_idx, start_idx, seg_label)
        
        for file_idx, file_path in enumerate(self.data_files):
            # 只读取必要的列来构建索引
            df_info = pd.read_csv(file_path, usecols=[self.block_col, self.label_col])
            
            for block_id, block_info in df_info.groupby(self.block_col):
                labels = block_info[self.label_col].values
                # 找到标签变化点
                change_points = np.where(np.diff(labels) != 0)[0] + 1
                seg_starts = np.concatenate(([0], change_points))
                seg_ends = np.concatenate((change_points, [len(labels)]))
                
                for seg_start, seg_end in zip(seg_starts, seg_ends):
                    seg_label = labels[seg_start]
                    seg_len = seg_end - seg_start
                    if seg_len < self.window_size:
                        continue
                    
                    for start in range(seg_start, seg_end - self.window_size + 1, self.step_size):
                        self.file_indices.append((file_idx, block_id, start, seg_label))
            
            # 释放临时数据
            del df_info
            gc.collect()
            
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        """按需加载单个样本"""
        file_idx, block_id, start_idx, seg_label = self.file_indices[idx]
        
        # 只加载需要的窗口数据
        df = pd.read_csv(self.data_files[file_idx])
        block_data = df[df[self.block_col] == block_id]
        
        # 提取窗口
        window_data = block_data.iloc[start_idx:start_idx + self.window_size]
        data_array = window_data[self.feature_cols].values.astype(np.float32)
        
        # 数据清理
        del df, block_data, window_data
        
        # 归一化
        if self.normalize == 'zscore':
            mean = data_array.mean(axis=0, keepdims=True)
            std = data_array.std(axis=0, keepdims=True)
            data_array = (data_array - mean) / (std + 1e-6)
        elif self.normalize == 'minmax':
            min_v = data_array.min(axis=0, keepdims=True)
            max_v = data_array.max(axis=0, keepdims=True)
            data_array = (data_array - min_v) / (max_v - min_v + 1e-6)
        
        # 转换为tensor [C, T]
        tensor = torch.from_numpy(data_array.T).float()
        label = torch.tensor(int(seg_label), dtype=torch.long)
        
        # 通道可信mask
        is_real_mask = torch.ones(len(self.feature_cols), dtype=torch.bool)
        for idx in self.need_indices:
            if idx < len(is_real_mask):
                is_real_mask[idx] = False
        
        # 根据阶段处理
        if self.phase == "encode" and len(self.need_indices) > 0:
            mask_idx = np.random.choice(self.need_indices)
            tensor_masked = tensor.clone()
            tensor_masked[mask_idx, :] = 0
            return tensor_masked, label, mask_idx, is_real_mask
        
        return tensor, label, -1, is_real_mask


class OptimizedDataLoader:
    """优化的数据加载器，支持内存高效和显存最大利用"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def create_datasets(self, data_files: List[str], feature_cols: List[str]) -> Tuple[Dataset, Dataset, Dataset]:
        """创建训练、验证、测试数据集"""
        
        # 使用内存高效的数据集
        full_dataset = MemoryEfficientDataset(
            data_files=data_files,
            feature_cols=feature_cols,
            block_col=self.config['block_col'],
            label_col=self.config['label_col'],
            window_size=self.config['window_size'],
            step_size=self.config['step_size'],
            normalize=self.config.get('norm_method', 'zscore'),
            need_indices=self.config.get('need_indices', []),
            phase="encode"
        )
        
        # 数据集分割
        train_split = self.config.get('train_split', 0.6)
        val_split = 0.2
        
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # 使用torch的random_split来分割
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, val_dataset, test_dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        
        # 优化的DataLoader参数
        loader_kwargs = {
            'batch_size': self.config.get('batch_size', 64),
            'num_workers': self.config.get('num_workers', 0),
            'pin_memory': self.config.get('pin_memory', True),
            'drop_last': self.config.get('dataloader_drop_last', True),
            'persistent_workers': self.config.get('persistent_workers', False),
        }
        
        # 只在num_workers > 0时设置prefetch_factor
        if loader_kwargs['num_workers'] > 0:
            loader_kwargs['prefetch_factor'] = self.config.get('prefetch_factor', 4)
        
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs
        )
        
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_kwargs
        )
        
        return train_loader, val_loader, test_loader


def get_memory_usage():
    """获取当前内存和显存使用情况"""
    import psutil
    
    # CPU内存
    memory = psutil.virtual_memory()
    cpu_usage = {
        'total': memory.total / (1024**3),  # GB
        'available': memory.available / (1024**3),
        'percent': memory.percent,
        'used': (memory.total - memory.available) / (1024**3)
    }
    
    # GPU显存
    gpu_usage = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            
            gpu_usage[f'gpu_{i}'] = {
                'total': total,
                'allocated': allocated,
                'cached': cached,
                'free': total - cached,
                'utilization': (allocated / total) * 100 if total > 0 else 0
            }
    
    return cpu_usage, gpu_usage


def print_memory_usage(prefix=""):
    """打印内存使用情况"""
    cpu_usage, gpu_usage = get_memory_usage()
    
    print(f"\n{prefix} Memory Usage:")
    print(f"CPU: {cpu_usage['used']:.2f}GB / {cpu_usage['total']:.2f}GB ({cpu_usage['percent']:.1f}%)")
    
    for gpu_id, usage in gpu_usage.items():
        print(f"GPU {gpu_id}: {usage['allocated']:.2f}GB / {usage['total']:.2f}GB ({usage['utilization']:.1f}%)")
