# memory_optimizer.py - 内存优化模块

import torch
import numpy as np
from torch.utils.data import DataLoader
import gc
from typing import Iterator, Tuple
import psutil
import warnings

class MemoryEfficientDataLoader:
    """内存高效的数据加载器"""
    
    def __init__(self, dataset, batch_size=32, shuffle=True, 
                 memory_threshold=0.8, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.memory_threshold = memory_threshold
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
    def _check_memory_usage(self):
        """检查内存使用率"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        return memory_percent
    
    def _adaptive_batch_size(self):
        """根据内存使用情况自适应调整批大小"""
        memory_usage = self._check_memory_usage()
        if memory_usage > self.memory_threshold:
            new_batch_size = max(1, self.batch_size // 2)
            warnings.warn(f"内存使用率 {memory_usage:.1%} 过高，批大小调整为 {new_batch_size}")
            return new_batch_size
        return self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        """内存优化的迭代器"""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        current_batch_size = self._adaptive_batch_size()
        
        for i in range(0, len(indices), current_batch_size):
            # 检查内存并可能调整批大小
            if i % (current_batch_size * 10) == 0:
                current_batch_size = self._adaptive_batch_size()
            
            batch_indices = indices[i:i + current_batch_size]
            batch_data = []
            
            for idx in batch_indices:
                item = self.dataset[idx]
                batch_data.append(item)
            
            # 转换为张量批次
            if batch_data:
                batch = self._collate_fn(batch_data)
                yield batch
                
                # 强制垃圾回收
                if len(batch_indices) > 16:
                    del batch_data, batch
                    gc.collect()
    
    def _collate_fn(self, batch):
        """自定义批次整理函数"""
        if len(batch) == 0:
            return None
        
        # 假设每个样本返回 (x, label, mask_idx, is_real_mask)
        xs, labels, mask_idxs, is_real_masks = zip(*batch)
        
        # 堆叠张量
        x_batch = torch.stack(xs, dim=0)
        label_batch = torch.stack(labels, dim=0) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels)
        
        # 处理mask_idx（可能是int或tensor）
        if isinstance(mask_idxs[0], torch.Tensor):
            mask_idx_batch = torch.stack(mask_idxs, dim=0)
        else:
            mask_idx_batch = torch.tensor(mask_idxs)
        
        # 处理is_real_mask
        if isinstance(is_real_masks[0], torch.Tensor):
            is_real_mask_batch = torch.stack(is_real_masks, dim=0)
        else:
            is_real_mask_batch = torch.tensor(is_real_masks)
        
        return x_batch, label_batch, mask_idx_batch, is_real_mask_batch

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, warning_threshold=0.8, critical_threshold=0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
    def check_and_warn(self, context=""):
        """检查内存并发出警告"""
        memory_info = psutil.virtual_memory()
        usage_percent = memory_info.percent / 100.0
        
        if usage_percent > self.critical_threshold:
            print(f"🚨 严重警告 {context}: 内存使用率 {usage_percent:.1%} 过高！")
            self._emergency_cleanup()
            return "critical"
        elif usage_percent > self.warning_threshold:
            print(f"⚠️ 警告 {context}: 内存使用率 {usage_percent:.1%}")
            return "warning"
        return "normal"
    
    def _emergency_cleanup(self):
        """紧急内存清理"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def optimize_tensor_memory(tensor, dtype=torch.float16):
    """张量内存优化"""
    if tensor.dtype == torch.float32 and dtype == torch.float16:
        return tensor.half()
    return tensor

def create_memory_efficient_loader(dataset, config):
    """创建内存高效的数据加载器"""
    batch_size = config.get('batch_size', 32)
    
    # 根据可用内存调整批大小
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if available_memory_gb < 4:
        batch_size = min(batch_size, 8)
        print(f"可用内存较少({available_memory_gb:.1f}GB)，批大小调整为 {batch_size}")
    elif available_memory_gb < 8:
        batch_size = min(batch_size, 16)
    
    return MemoryEfficientDataLoader(
        dataset, 
        batch_size=batch_size,
        memory_threshold=0.8,
        pin_memory=torch.cuda.is_available()
    )
