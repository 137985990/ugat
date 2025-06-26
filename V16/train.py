# train.py - V13多模态时间序列模型训练脚本（完全重构版本）

"""
V13 多模态时间序列模型训练脚本

核心特性：
- 支持多数据集（FM/OD/MEFAR）混合训练
- 动态遮掩have通道，按source_dataset动态处理损失
- 使用重构的数据处理流程，完全移除旧逻辑
"""

import os
import sys
import yaml
import argparse
import logging
import collections
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

# 添加混合精度训练支持
try:
    from torch.cuda.amp.grad_scaler import GradScaler
    from torch.cuda.amp.autocast_mode import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Warning: Automatic Mixed Precision not available.")

# 导入V13重构的数据处理模块
from data import create_multimodal_dataset_from_config, load_config, check_label_distribution

# 导入自定义模块（可选）
try:
    from simple_multimodal_integration import create_simple_multimodal_criterion
    MULTIMODAL_CRITERION_AVAILABLE = True
except ImportError:
    MULTIMODAL_CRITERION_AVAILABLE = False
    print("Warning: Simple multimodal criterion not available, using standard MSE.")

try:
    from enhanced_validation_integration import EnhancedValidationManager
    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError:
    ENHANCED_VALIDATION_AVAILABLE = False
    print("Warning: Enhanced validation not available.")


def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 配置UTF-8编码的日志文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 创建logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def custom_collate_fn(batch):
    """
    自定义collate函数来处理不同长度的列表
    batch中每个元素格式: (tensor, label, indices_list, is_real_mask, source_dataset)
    """
    print("🚨 DEBUG: custom_collate_fn 函数被调用了!")
    tensors = []
    labels = []
    indices_lists = []
    is_real_masks = []
    source_datasets = []
    
    for i, item in enumerate(batch):
        if not isinstance(item, (list, tuple)) or len(item) != 5:
            raise ValueError(f"Batch item {i} has wrong format, expected 5 elements, got {len(item) if hasattr(item, '__len__') else 'unknown'}")
            
        tensor, label, indices_list, is_real_mask, source_dataset = item
        
        tensors.append(tensor)
        labels.append(label)
        indices_lists.append(indices_list)
        is_real_masks.append(is_real_mask)
        source_datasets.append(source_dataset)
    
    # 将tensors和labels堆叠
    try:
        batched_tensors = torch.stack(tensors)
        batched_labels = torch.stack(labels)
        batched_is_real_masks = torch.stack(is_real_masks)
        
        # indices_lists和source_datasets保持为列表
        return batched_tensors, batched_labels, indices_lists, batched_is_real_masks, source_datasets
    except Exception as e:
        print(f"[ERROR] Failed to stack tensors: {e}")
        print(f"[DEBUG] Tensor shapes: {[t.shape for t in tensors]}")
        print(f"[DEBUG] Label types: {[type(l) for l in labels]}")
        raise

def check_label_distribution(dataset):
    """检查并输出数据集标签分布和所有标签种类"""
    label_counter = collections.Counter()
    all_labels = set()
    for i in range(len(dataset)):
        item = dataset[i]
        label = item[1]
        if hasattr(label, 'item'):
            label = label.item()
        label_counter[label] += 1
        all_labels.add(label)
    print("标签分布:", dict(label_counter))
    print("所有标签:", sorted(list(all_labels)))
    return label_counter, all_labels

def complete_need_with_model(model, dataset, device, need_indices):
    """
    用模型对整个数据集的need通道进行补全，并写回dataset（循环补全逻辑）
    这是关键的循环学习机制：用当前模型补全need通道，为下一轮训练提供更好的数据
    """
    model.eval()
    from torch.utils.data import DataLoader
    import torch
    
    # 获取原始数据集（如果是Subset需要取出原始dataset）
    original_dataset = dataset
    if hasattr(dataset, 'dataset'):
        original_dataset = dataset.dataset
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_need_predictions = []
    
    print(f"开始循环补全need通道 (共{len(need_indices)}个need通道)")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="补全need通道")):
            if len(batch) == 4:
                batch_x, _, _, _ = batch
            else:
                batch_x = batch[0]
            
            batch_x = batch_x.to(device)
            batch_size, C, T = batch_x.size()
            
            # 对batch中每个样本进行need通道补全
            for i in range(batch_size):
                window = batch_x[i].t()  # [T, C]
                out, _ = model(window)   # 模型重建输出 [C, T]
                out = out.t()           # 转回 [T, C]
                
                # 只保存need通道的预测结果
                need_pred = {}
                for need_idx in need_indices:
                    if need_idx < out.size(1):
                        need_pred[need_idx] = out[:, need_idx].cpu()
                
                all_need_predictions.append(need_pred)
    
    # 将预测的need通道写回原始数据集
    if hasattr(original_dataset, 'update_need_channels'):
        # 如果数据集支持批量更新need通道
        original_dataset.update_need_channels(all_need_predictions, need_indices)
    else:
        # 逐个更新（fallback方案）
        for idx, need_pred in enumerate(all_need_predictions):
            if hasattr(original_dataset, 'update_need'):
                original_dataset.update_need(idx, need_pred)
    
    print(f"循环补全完成，已更新{len(all_need_predictions)}个样本的need通道")

def mask_channel(batch, source_datasets, dataset):
    """对batch中的每个样本按其source_dataset动态遮掩have通道"""
    masked = batch.clone()
    batch_size = batch.size(0)
    # 修正：如果source_datasets为空或长度不对，自动填充
    if not source_datasets or len(source_datasets) != batch_size:
        source_datasets = ['UNKNOWN'] * batch_size
    for i in range(batch_size):
        src = source_datasets[i]
        have_indices = dataset.get_have_indices_for_dataset(src) if dataset is not None else []
        for idx in have_indices:
            if idx < masked.size(1):
                masked[i, idx, :] = 0
    return masked

def train_phased_with_grad_accumulation(model, dataloader, optimizer, criterion, device, mask_indices, 
                                      accumulate_grad_batches=2, use_mixed_precision=True, scaler=None, need_indices=None,
                                      training_strategy="mask_have", common_indices=None, have_indices=None, 
                                      recon_weight=1.0, cls_improvement_weight=1.0, dataset=None, current_epoch=1):
    """训练函数 - 实现双路径分类训练，优化重建数据的分类性能"""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_improvement_loss = 0.0  # 分类改进损失
    total_original_correct = 0  # 原始数据分类正确数
    total_reconstructed_correct = 0  # 重建数据分类正确数
    total_samples = 0
    
    # 获取common模态索引    common_indices = getattr(criterion, 'common_indices', [])
    need_indices = need_indices if need_indices is not None else []
    
    # 梯度累积相关
    accumulated_loss = 0.0
    step_count = 0
    
    # 统计各source_dataset的采样分布
    source_dataset_counter = {}
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training")):
        if len(batch_data) == 5:
            batch, labels, _, is_real_mask, source_datasets = batch_data
        elif len(batch_data) == 4:
            batch, labels, _, is_real_mask = batch_data
            source_datasets = ['UNKNOWN'] * batch.size(0)  # 默认值
        else:
            batch, labels, _, is_real_mask, source_datasets = batch_data
        
        # 统计采样分布
        for src in source_datasets:
            source_dataset_counter[src] = source_dataset_counter.get(src, 0) + 1
        
        # 立即转移到GPU
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        is_real_mask = is_real_mask.to(device, non_blocking=True)
        
        # 动态遮掩
        masked = mask_channel(batch, source_datasets, dataset)
        batch_size, C, T = batch.size()          # 梯度累积：只在累积周期开始时清零梯度
        if step_count % accumulate_grad_batches == 0:
            optimizer.zero_grad()
        
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            with autocast():
                # 阶段1: 重建训练 - 输入时间序列 → GAT编码器 → Transformer瓶颈 → GAT解码器 → 重建输出
                batch_reconstructed, _ = forward_batch_parallel(model, masked, device)
                
                # 计算重建损失
                recon_loss = compute_batch_recon_loss(batch, batch_reconstructed, is_real_mask, 
                                                    common_indices, criterion, C, batch_size, need_indices,
                                                    training_strategy, have_indices, source_datasets, dataset, current_epoch)
                
                # 阶段2: 双路径分类训练
                # 路径1: 原始数据分类 (N初始0+C真+H真) → GAT特征提取 → Transformer瓶颈 → 分类器 → 损失1
                original_data = batch.clone()  # 使用原始数据（包含初始Need=0）
                _, original_logits = forward_batch_parallel(model, original_data, device)
                
                # 路径2: 重建数据分类 (N重建+C真+H真) → GAT特征提取 → Transformer瓶颈 → 分类器 → 损失2
                # 合成完整数据：重建的Need + 真实的C和H
                enhanced_data = batch.clone()
                if need_indices:
                    for need_idx in need_indices:
                        if need_idx < batch_reconstructed.size(1):
                            enhanced_data[:, need_idx, :] = batch_reconstructed[:, need_idx, :]
                
                _, enhanced_logits = forward_batch_parallel(model, enhanced_data, device)
                
                # 计算分类准确率
                original_preds = torch.argmax(original_logits, dim=1)
                enhanced_preds = torch.argmax(enhanced_logits, dim=1)
                
                original_accuracy = (original_preds == labels).float().mean()
                enhanced_accuracy = (enhanced_preds == labels).float().mean()                # 分类改进损失：鼓励重建数据提升分类性能
                # 总损失 = α×重建损失 + β×(分类准确率1 - 分类准确率2)
                # 当重建数据分类更准确时，损失更小，鼓励这种行为
                cls_improvement_loss = original_accuracy - enhanced_accuracy
                
                # 总损失 = 加权重建损失 + 加权分类改进损失
                loss = (recon_weight * recon_loss + cls_improvement_weight * cls_improvement_loss) / accumulate_grad_batches
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            
            # 在累积周期结束时更新参数
            if (step_count + 1) % accumulate_grad_batches == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            # 标准精度训练 - 相同逻辑
            # 阶段1: 重建训练
            batch_reconstructed, _ = forward_batch_parallel(model, masked, device)            # 计算重建损失
            recon_loss = compute_batch_recon_loss(batch, batch_reconstructed, is_real_mask, 
                                                common_indices, criterion, C, batch_size, need_indices,
                                                training_strategy, have_indices, source_datasets, dataset, current_epoch)
            
            # 阶段2: 双路径分类训练
            # 路径1: 原始数据分类
            original_data = batch.clone()
            _, original_logits = forward_batch_parallel(model, original_data, device)
            
            # 路径2: 重建数据分类
            enhanced_data = batch.clone()
            if need_indices:
                for need_idx in need_indices:
                    if need_idx < batch_reconstructed.size(1):
                        enhanced_data[:, need_idx, :] = batch_reconstructed[:, need_idx, :]
            
            _, enhanced_logits = forward_batch_parallel(model, enhanced_data, device)
            
            # 计算分类准确率
            original_preds = torch.argmax(original_logits, dim=1)
            enhanced_preds = torch.argmax(enhanced_logits, dim=1)
            
            original_accuracy = (original_preds == labels).float().mean()
            enhanced_accuracy = (enhanced_preds == labels).float().mean()            # 分类改进损失：总损失 = α×重建损失 + β×(分类准确率1 - 分类准确率2)
            cls_improvement_loss = original_accuracy - enhanced_accuracy
            
            # 总损失 = 加权重建损失 + 加权分类改进损失
            loss = (recon_weight * recon_loss + cls_improvement_weight * cls_improvement_loss) / accumulate_grad_batches
            loss.backward()
            
            # 在累积周期结束时更新参数
            if (step_count + 1) % accumulate_grad_batches == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 统计信息（用原始损失值）
        original_loss = loss * accumulate_grad_batches
        original_recon = recon_loss
        original_cls_improvement = cls_improvement_loss
        
        # 统计分类正确数
        total_original_correct += (original_preds == labels).sum().item()
        total_reconstructed_correct += (enhanced_preds == labels).sum().item()
        total_samples += batch_size        # 安全地获取损失值
        if isinstance(original_loss, torch.Tensor):
            loss_val = original_loss.item()
        else:
            loss_val = float(original_loss)
            
        if isinstance(original_recon, torch.Tensor):
            recon_val = original_recon.item()
        else:
            recon_val = float(original_recon)
            
        if isinstance(original_cls_improvement, torch.Tensor):
            cls_improvement_val = original_cls_improvement.item()
        else:
            cls_improvement_val = float(original_cls_improvement)
        
        total_loss += loss_val * batch_size
        total_recon_loss += recon_val * batch_size
        total_cls_improvement_loss += cls_improvement_val * batch_size
        step_count += 1
    
    # 处理最后的不完整累积批次
    if step_count % accumulate_grad_batches != 0:
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    n = len(dataloader.dataset)
    original_acc = total_original_correct / total_samples if total_samples > 0 else 0.0
    reconstructed_acc = total_reconstructed_correct / total_samples if total_samples > 0 else 0.0
    print(f"[采样分布] 本轮训练各source_dataset采样数: {source_dataset_counter}")
    return total_loss / n, total_recon_loss / n, total_cls_improvement_loss / n, original_acc, reconstructed_acc


def forward_batch_parallel(model, input_batch, device):
    batch_size, C, T = input_batch.size()
    
    # 批量转置: [batch_size, C, T] -> [batch_size, T, C] 
    windows = input_batch.transpose(1, 2)  # [batch_size, T, C]
      # 尝试真正的批量处理 - 直接处理整个batch
    try:
        batch_out, batch_logits = model.forward_batch(windows)
        return batch_out, batch_logits  # [batch_size, C, T], [batch_size, num_classes]
    except AttributeError:
        try:
            def single_forward(window):
                out, logits = model(window)
                return out.t(), logits  # [C, T], [num_classes]
            
            # 使用vmap进行真正的并行化
            vmapped_forward = torch.vmap(single_forward, in_dims=0, out_dims=0)
            batch_out, batch_logits = vmapped_forward(windows)
            return batch_out, batch_logits  # [batch_size, C, T], [batch_size, num_classes]
        except:
            return forward_large_batch_optimized(model, windows, device)


def forward_large_batch_optimized(model, windows, device):

    batch_size, T, C = windows.size()
    
    # 使用固定的chunk大小
    chunk_size = min(batch_size, 16)
    
    batch_outputs = []
    batch_logits = []
    
    # 使用更大的chunk进行并行处理
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_windows = windows[i:end_idx]  # [chunk_size, T, C]
        chunk_size_actual = chunk_windows.size(0)
        
        # 批量处理chunk
        chunk_out_list = []
        chunk_logits_list = []        
        # 并行处理chunk中的所有样本
        for j in range(chunk_size_actual):
            try:
                out, logits = model(chunk_windows[j])
                chunk_out_list.append(out.t())  # [C, T]
                chunk_logits_list.append(logits)
            except Exception as e:
                print(f"Error processing chunk sample {j}: {e}")
                # 创建默认输出
                C = chunk_windows.size(-1)
                T = chunk_windows.size(-2)
                default_out = torch.zeros((C, T), device=device)
                default_logits = torch.zeros((2,), device=device)  # 假设2分类
                chunk_out_list.append(default_out)
                chunk_logits_list.append(default_logits)
        
        # 批量堆叠chunk结果
        if chunk_out_list:
            chunk_out = torch.stack(chunk_out_list, dim=0)  # [chunk_size, C, T]
            chunk_logits = torch.stack(chunk_logits_list, dim=0)  # [chunk_size, num_classes]
        else:
            # 创建空的tensor
            chunk_out = torch.zeros((chunk_size_actual, C, T), device=device)
            chunk_logits = torch.zeros((chunk_size_actual, 2), device=device)
        
        batch_outputs.append(chunk_out)
        batch_logits.append(chunk_logits)
    
    # 最终堆叠所有chunk
    final_batch_out = torch.cat(batch_outputs, dim=0)  # [batch_size, C, T]
    final_batch_logits = torch.cat(batch_logits, dim=0)  # [batch_size, num_classes]
    
    return final_batch_out, final_batch_logits


def compute_batch_recon_loss(targets, predictions, is_real_mask, common_indices, criterion, C, batch_size, need_indices, training_strategy="mask_have", have_indices=None, source_datasets=None, dataset=None, current_epoch=1):
    """
    高效的批量重建损失计算 - 支持混合数据集中每个样本可能有不同通道定义
    
    Args:
        targets: [batch_size, C, T] 目标数据
        predictions: [batch_size, C, T] 预测数据
        is_real_mask: 通道可信度mask (如果是混合batch，则为默认值)
        common_indices: Common通道索引 (如果是混合batch，则为默认值)
        criterion: 损失函数
        C: 通道数
        batch_size: 批量大小
        need_indices: Need通道索引 (如果是混合batch，则为默认值)
        training_strategy: 训练策略 ("mask_have" 或 "no_mask")
        have_indices: Have通道索引 (如果是混合batch，则为默认值)
        source_datasets: list of str, 每个样本的数据集来源 (用于混合batch)
        dataset: SlidingWindowDataset对象 (用于查询每个样本的通道定义)
        current_epoch: 当前epoch数，用于决定N通道是否参与损失计算
    """
    total_recon_loss = 0.0
    
    # UNet范式：第一个epoch N不参与损失，第二个epoch开始N为生成值参与训练
    include_need_in_loss = (current_epoch > 1)
    
    # 检查输入数据是否正常
    if torch.isnan(targets).any() or torch.isnan(predictions).any():
        return torch.tensor(0.001, device=targets.device, requires_grad=True)
    
    if torch.isinf(targets).any() or torch.isinf(predictions).any():
        return torch.tensor(0.001, device=targets.device, requires_grad=True)
    
    # 检查是否为混合数据集batch
    is_mixed_batch = (source_datasets is not None and 
                     dataset is not None and 
                     hasattr(dataset, 'get_have_indices_for_dataset') and
                     len(source_datasets) > 0)
    
    if is_mixed_batch:
        # 混合batch：每个样本单独计算损失
        loss_count = 0
        for b in range(batch_size):
            if source_datasets and b < len(source_datasets):
                source_ds = source_datasets[b]
            else:
                continue
            
            # 获取该样本的通道定义
            sample_common_indices = []  # Common通道对所有数据集都是一样的，可以从dataset获取
            sample_have_indices = dataset.get_have_indices_for_dataset(source_ds) if dataset else []
            sample_need_indices = dataset.get_need_indices_for_dataset(source_ds) if dataset else []
            
            if training_strategy == "mask_have":
                # 对被mask的have通道计算损失
                for h_idx in sample_have_indices:
                    if h_idx >= C:
                        continue
                    target_seq = targets[b, h_idx, :]  # [T]
                    pred_seq = predictions[b, h_idx, :]  # [T]
                    try:
                        # 简化损失函数调用，兼容标准MSE
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            loss_item = criterion(pred_seq, target_seq, channel_idx=h_idx, is_common=False)
                        else:
                            loss_item = criterion(pred_seq, target_seq)
                        if not (torch.isnan(loss_item) or torch.isinf(loss_item)):
                            total_recon_loss += loss_item
                            loss_count += 1
                    except Exception:
                        continue
                
                # UNet范式：第二个epoch开始，Need通道也参与损失计算（因为此时N为生成值）
                if include_need_in_loss:
                    for n_idx in sample_need_indices:
                        if n_idx >= C:
                            continue
                        target_seq = targets[b, n_idx, :]  # [T]
                        pred_seq = predictions[b, n_idx, :]  # [T]
                        try:
                            if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                                loss_item = criterion(pred_seq, target_seq, channel_idx=n_idx, is_common=False)
                            else:
                                loss_item = criterion(pred_seq, target_seq)
                            if not (torch.isnan(loss_item) or torch.isinf(loss_item)):
                                total_recon_loss += loss_item
                                loss_count += 1
                        except Exception:
                            continue
            elif training_strategy == "no_mask":
                # 对common通道计算损失（have通道也可以考虑）
                for c_idx in sample_common_indices:
                    if c_idx >= C:
                        continue
                    target_seq = targets[b, c_idx, :]  # [T]
                    pred_seq = predictions[b, c_idx, :]  # [T]
                    try:
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            loss_item = criterion(pred_seq, target_seq, channel_idx=c_idx, is_common=True)
                        else:
                            loss_item = criterion(pred_seq, target_seq)
                        if not (torch.isnan(loss_item) or torch.isinf(loss_item)):
                            total_recon_loss += loss_item
                            loss_count += 1
                    except Exception:
                        continue
        
        if loss_count > 0:
            total_recon_loss = total_recon_loss / loss_count
    else:
        # 统一batch：使用原有逻辑
        need_set = set(need_indices)
        
        if training_strategy == "mask_have":
            # 策略1: 遮掩Have通道，只对被遮掩的Have通道计算损失
            if have_indices is None:
                # 自动计算have_indices
                all_indices = set(range(C))
                common_set = set(common_indices)
                have_indices = list(all_indices - need_set - common_set)
            
            # 只对have通道计算损失（这些通道被遮掩了，需要重建）
            loss_count = 0
            for h_idx in have_indices:
                if h_idx >= C:
                    continue
                
                target_batch = targets[:, h_idx, :]  # [batch_size, T]
                pred_batch = predictions[:, h_idx, :]
                
                loss_sum = 0.0
                for b in range(batch_size):
                    try:
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            loss_item = criterion(pred_batch[b], target_batch[b], channel_idx=h_idx, is_common=False)
                        else:
                            loss_item = criterion(pred_batch[b], target_batch[b])
                        if torch.isnan(loss_item) or torch.isinf(loss_item):
                            continue
                        loss_sum += loss_item
                        loss_count += 1
                    except Exception:
                        continue
                
                if loss_count > 0:
                    total_recon_loss += loss_sum / batch_size
                
        elif training_strategy == "no_mask":
            # 策略2: 不遮掩任何通道，对Common+Have计算损失（跳过Need）
            # 向量化处理相同类型的通道
            common_mask = torch.zeros(C, dtype=torch.bool, device=targets.device)
            if common_indices:
                common_mask[common_indices] = True
            
            loss_count = 0
            
            # 处理common通道（批量）- Common始终参与损失计算
            if common_mask.any():
                common_targets = targets[:, common_mask, :]  # [batch_size, n_common, T]
                common_preds = predictions[:, common_mask, :]
                
                # 批量计算所有common通道的损失
                for c_idx, global_c in enumerate(torch.where(common_mask)[0]):
                    target_batch = common_targets[:, c_idx, :]  # [batch_size, T]
                    pred_batch = common_preds[:, c_idx, :]
                    
                    # 批量调用criterion
                    loss_sum = 0.0
                    for b in range(batch_size):
                        try:
                            if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                                loss_item = criterion(pred_batch[b], target_batch[b], channel_idx=global_c.item(), is_common=True)
                            else:
                                loss_item = criterion(pred_batch[b], target_batch[b])
                            if torch.isnan(loss_item) or torch.isinf(loss_item):
                                continue
                            loss_sum += loss_item
                            loss_count += 1
                        except Exception:
                            continue
                    
                    if loss_count > 0:
                        total_recon_loss += loss_sum / batch_size
            
            # 处理have通道（批量）- Have也参与损失计算，但跳过Need通道
            for c in range(C):
                if c in common_indices or c in need_set:
                    continue  # 跳过common（已处理）和need通道
                
                # 这是have通道，参与损失计算
                target_batch = targets[:, c, :]  # [batch_size, T]
                pred_batch = predictions[:, c, :]
                
                loss_sum = 0.0
                for b in range(batch_size):
                    try:
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            loss_item = criterion(pred_batch[b], target_batch[b], channel_idx=c, is_common=False)
                        else:
                            loss_item = criterion(pred_batch[b], target_batch[b])
                        if torch.isnan(loss_item) or torch.isinf(loss_item):
                            continue
                        loss_sum += loss_item
                        loss_count += 1
                    except Exception:
                        continue
                        
                if loss_count > 0:
                    total_recon_loss += loss_sum / batch_size
    
    # 最终检查
    if isinstance(total_recon_loss, torch.Tensor):
        if torch.isnan(total_recon_loss) or torch.isinf(total_recon_loss):
            return torch.tensor(0.001, device=targets.device, requires_grad=True)  # 返回一个小的非零值
    else:
        # 如果是float，转换为tensor
        if total_recon_loss == 0.0 or total_recon_loss != total_recon_loss or total_recon_loss == float('inf'):
            return torch.tensor(0.001, device=targets.device, requires_grad=True)
        total_recon_loss = torch.tensor(total_recon_loss, device=targets.device, requires_grad=True)
    
    if isinstance(total_recon_loss, torch.Tensor) and total_recon_loss.item() == 0.0:
        return torch.tensor(0.001, device=targets.device, requires_grad=True)
    elif not isinstance(total_recon_loss, torch.Tensor) and total_recon_loss == 0.0:
        return torch.tensor(0.001, device=targets.device, requires_grad=True)
    
    return total_recon_loss

def eval_loop(model, dataloader, criterion, device, mask_indices):

    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    
    # 获取common模态索引
    common_indices = getattr(criterion, 'common_indices', [])
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Eval"):
            if len(batch_data) == 4:
                batch, _, _, is_real_mask = batch_data
            else:
                batch, _, _, is_real_mask, _ = batch_data
            
            batch = batch.to(device)
            is_real_mask = is_real_mask.to(device)
            
            masked = mask_channel(batch, [], None) # V13中使用动态遮掩，这里传入空参数
            batch_size, C, T = batch.size()
            loss = 0.0
            recon_loss = 0.0
            
            for i in range(batch_size):
                window = masked[i].t()
                out, _ = model(window)
                
                # 获取真实通道信息
                if is_real_mask.dim() == 2:
                    real_channels = is_real_mask[i]
                else:
                    real_channels = is_real_mask
                
                recon_loss_i = 0.0
                real_count = 0
                
                for c in range(C):
                    target = batch[i, c, :]
                    pred = out[c, :]
                    
                    # 判断是否为common模态
                    is_common_channel = c in common_indices
                    
                    if is_common_channel:
                        # Common模态：始终计算损失
                        recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=True)
                        real_count += 1
                    elif real_channels[c]:                        # Have模态：只对真实通道计算损失
                        recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=False)
                        real_count += 1
                
                if real_count > 0:
                    recon_loss_i = recon_loss_i / real_count
                
                loss += recon_loss_i
                recon_loss += recon_loss_i
            
            loss = loss / batch_size
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss
    
    n = len(dataloader.dataset)
    return total_loss / n, total_recon_loss / n, 0.0, 0.0

def parse_args():
    parser = argparse.ArgumentParser(description='V13 多模态时间序列模型训练 - 支持FM/OD/MEFAR混合训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    return parser.parse_args()

# =========================
# 主训练入口
# =========================
def main():
    parser = argparse.ArgumentParser(description='V13 Multimodal Time Series Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    set_seed(config.get('seed', 42))
    logger = setup_logging(config.get('log_dir', 'logs'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = config.get('checkpoint_dir', 'Checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 数据集    
    dataset = create_multimodal_dataset_from_config(config, phase='encode')
    logger.info(f"数据集创建完成，滑动窗口数量: {len(dataset)}")
    
    dataset_size = len(dataset)
    train_size = int(config.get('train_ratio', 0.7) * dataset_size)
    val_size = int(config.get('val_ratio', 0.15) * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    batch_size = config.get('batch_size', 32)
    
    # 创建自定义collate函数和DataLoader
    def working_collate_fn(batch):
        tensors, labels, indices_lists, is_real_masks, source_datasets = [], [], [], [], []
        
        for item in batch:
            tensor, label, indices_list, is_real_mask, source_dataset = item
            tensors.append(tensor)
            labels.append(label)
            indices_lists.append(indices_list)
            is_real_masks.append(is_real_mask)
            source_datasets.append(source_dataset)
        
        batched_tensors = torch.stack(tensors)
        batched_labels = torch.stack(labels)
        batched_is_real_masks = torch.stack(is_real_masks)
        
        return batched_tensors, batched_labels, indices_lists, batched_is_real_masks, source_datasets
      # 创建DataLoader
    print(f"DEBUG: working_collate_fn = {working_collate_fn}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=(device.type == 'cuda'), 
                             collate_fn=working_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=(device.type == 'cuda'), 
                           collate_fn=working_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=(device.type == 'cuda'), 
                            collate_fn=working_collate_fn)
    # 采样一个batch并打印结构，验证collate_fn是否生效
    try:
        sample_batch = next(iter(train_loader))
        logger.info(f"采样batch类型: {type(sample_batch)}，长度: {len(sample_batch)}")
        for i, item in enumerate(sample_batch):
            logger.info(f"batch[{i}] 类型: {type(item)}, 形状/长度: {getattr(item, 'shape', len(item) if hasattr(item, '__len__') else 'NA')}")
        input_channels = sample_batch[0].size(1)
    except Exception as e:
        logger.error(f"DataLoader采样失败: {e}")
        logger.info("检查数据集第一个元素结构...")
        try:
            first_item = train_dataset[0]
            logger.info(f"数据集第一个元素: type={type(first_item)}")
            if isinstance(first_item, (list, tuple)):
                logger.info(f"元素数量: {len(first_item)}")
                for i, sub_item in enumerate(first_item):
                    logger.info(f"  element[{i}]: type={type(sub_item)}, shape={getattr(sub_item, 'shape', 'NA')}")
            
            # 如果采样失败，从配置文件获取输入通道数
            input_channels = config.get('input_channels', 32)
            logger.info(f"使用配置文件中的输入通道数: {input_channels}")
        except Exception as e2:
            logger.error(f"无法检查数据集元素: {e2}")
            input_channels = config.get('input_channels', 32)
            logger.info(f"使用默认输入通道数: {input_channels}")
    from model import TGATUNet
    model = TGATUNet(
        in_channels=input_channels,
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=input_channels,
        num_classes=config.get('num_classes', 2)
    ).to(device)
    logger.info(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=config.get('learning_rate', 1e-4), weight_decay=config.get('weight_decay', 1e-5))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.get('lr_factor', 0.5), patience=config.get('lr_patience', 10), verbose=True)
    scaler = GradScaler() if AMP_AVAILABLE and config.get('use_amp', False) else None
    writer = SummaryWriter(config.get('tensorboard_dir', 'runs'))
    epochs = config.get('epochs', 100)
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 20)
    patience_counter = 0

    # 损失函数
    if config.get('loss_config', {}).get('type') == 'multimodal' and MULTIMODAL_CRITERION_AVAILABLE:
        criterion = create_simple_multimodal_criterion(config)
        logger.info("使用多模态损失函数")
    else:
        criterion = nn.MSELoss()
        logger.info("使用标准MSE损失函数")    # 训练主循环
    logger.info(f"开始训练，总轮数: {epochs}")
    for epoch in range(1, epochs + 1):
        train_loss, train_recon, train_cls_improvement, train_original_acc, train_reconstructed_acc = train_phased_with_grad_accumulation(
            model, train_loader, optimizer, criterion, device, [],
            accumulate_grad_batches=config.get('accumulate_grad_batches', 2),
            use_mixed_precision=AMP_AVAILABLE and config.get('use_amp', False), scaler=scaler,
            need_indices=[], training_strategy=config.get('training_strategy', 'mask_have'),
            common_indices=[], have_indices=None, recon_weight=1.0, cls_improvement_weight=1.0,
            dataset=dataset, current_epoch=epoch
        )
        
        # 验证阶段
        val_loss, val_recon, val_cls, val_acc = eval_loop(
            model, val_loader, criterion, device, []
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志记录
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, train_recon={train_recon:.6f}, "
                   f"train_original_acc={train_original_acc:.4f}, train_reconstructed_acc={train_reconstructed_acc:.4f}, "
                   f"val_loss={val_loss:.6f}, val_recon={val_recon:.6f}, lr={current_lr:.6e}")
        
        # TensorBoard记录
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Recon_Loss', train_recon, epoch)
        writer.add_scalar('Train/Cls_Improvement_Loss', train_cls_improvement, epoch)
        writer.add_scalar('Train/Original_Acc', train_original_acc, epoch)
        writer.add_scalar('Train/Reconstructed_Acc', train_reconstructed_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Recon_Loss', val_recon, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
            logger.info(f"保存最佳模型，验证损失: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"早停触发，在第 {epoch} 轮停止训练")
            break
    
    # 测试评估
    logger.info("开始测试集评估...")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best_model.pth')))
    test_loss, test_recon, test_cls, test_acc = eval_loop(
        model, test_loader, criterion, device, []
    )
    logger.info(f"测试结果: loss={test_loss:.6f}, recon_loss={test_recon:.6f}, accuracy={test_acc:.4f}")
      # 记录测试结果到TensorBoard
    writer.add_scalar('Test/Loss', test_loss)
    writer.add_scalar('Test/Recon_Loss', test_recon)
    writer.add_scalar('Test/Accuracy', test_acc)
    
    writer.close()
    logger.info("训练完成！")

if __name__ == "__main__":
    main()
