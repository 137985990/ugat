# data_optimizations.py - 数据处理的微调优化

import torch
import numpy as np

def add_data_validation(tensor, name="tensor"):
    """
    为您的数据处理流程添加验证检查
    可以在关键位置调用这个函数来确保数据质量
    """
    if torch.isnan(tensor).any():
        print(f"⚠️ 检测到NaN在 {name}")
        return False
    
    if torch.isinf(tensor).any():
        print(f"⚠️ 检测到Inf在 {name}")
        return False
    
    if tensor.std() < 1e-6:
        print(f"⚠️ {name} 的标准差过小，可能存在问题")
        return False
    
    return True

def stabilize_masking(batch, source_datasets, dataset, add_noise=True, noise_std=0.01):
    """
    稳定化的遮掩函数 - 在您现有mask_channel基础上的改进
    
    参数:
        add_noise: 是否在遮掩位置添加小量噪声而不是完全置零
        noise_std: 噪声标准差
    """
    masked = batch.clone()
    batch_size = batch.size(0)
    
    if not source_datasets or len(source_datasets) != batch_size:
        source_datasets = ['UNKNOWN'] * batch_size
    
    for i in range(batch_size):
        src = source_datasets[i]
        have_indices = dataset.get_have_indices_for_dataset(src) if dataset is not None else []
        
        for idx in have_indices:
            if idx < masked.size(1):
                if add_noise:
                    # 添加小量噪声而不是完全置零，可能有助于梯度流动
                    noise = torch.randn_like(masked[i, idx, :]) * noise_std
                    masked[i, idx, :] = noise
                else:
                    masked[i, idx, :] = 0
    
    return masked

def validate_batch_data(batch_data, step="unknown"):
    """
    批次数据验证 - 可以在dataloader的关键位置调用
    """
    issues = []
    
    if len(batch_data) >= 2:
        batch, labels = batch_data[:2]
        
        # 检查batch
        if not add_data_validation(batch, f"batch at {step}"):
            issues.append("batch数据异常")
        
        # 检查标签分布
        label_dist = torch.bincount(labels)
        if len(label_dist) < 2:
            issues.append("标签分布异常：只有一个类别")
        elif label_dist.min() / label_dist.max() < 0.1:
            issues.append(f"标签分布不平衡: {label_dist.tolist()}")
    
    if issues:
        print(f"🔍 数据验证 ({step}): {issues}")
    
    return len(issues) == 0

def enhance_batch_normalization(batch, method='zscore', eps=1e-8):
    """
    增强的批次标准化 - 可能有助于稳定训练
    """
    if method == 'zscore':
        # 按特征维度标准化
        mean = batch.mean(dim=(0, 2), keepdim=True)  # [1, C, 1]
        std = batch.std(dim=(0, 2), keepdim=True) + eps
        normalized = (batch - mean) / std
    elif method == 'minmax':
        # 最小最大标准化
        min_val = batch.min(dim=2, keepdim=True)[0].min(dim=0, keepdim=True)[0]
        max_val = batch.max(dim=2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        normalized = (batch - min_val) / (max_val - min_val + eps)
    else:
        normalized = batch
    
    return normalized

# 建议的集成方式
"""
在您的train.py中的关键位置添加这些验证：

1. 在dataloader循环开始时：
   valid = validate_batch_data(batch_data, f"epoch_{epoch}_batch_{batch_idx}")
   if not valid:
       continue  # 跳过有问题的batch

2. 在mask_channel调用时：
   from data_optimizations import stabilize_masking
   masked = stabilize_masking(batch, source_datasets, dataset, add_noise=True)

3. 在前向传播前：
   batch = enhance_batch_normalization(batch, method='zscore')

4. 在关键计算后：
   add_data_validation(recon_loss, "recon_loss")
   add_data_validation(total_classification_loss, "total_classification_loss")
"""