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

def complete_need_with_model(model, dataset, device, need_indices=None):
    """
    用模型对整个数据集的need通道进行补全，并写回dataset（循环补全逻辑）
    这是关键的循环学习机制：用当前模型补全need通道，为下一轮训练提供更好的数据
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        device: 设备
        need_indices: 废弃参数，保持兼容性。实际Need通道根据source_dataset动态确定
    """
    model.eval()
    from torch.utils.data import DataLoader
    import torch
    
    # 获取原始数据集（如果是Subset需要取出原始dataset）
    original_dataset = dataset
    if hasattr(dataset, 'dataset'):
        original_dataset = dataset.dataset
    
    # 使用自定义collate_fn来获取source_dataset信息
    def collate_with_source(batch):
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
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_with_source)
    all_need_predictions = []
    
    print("开始基于source_dataset的循环补全need通道")
    
    with torch.no_grad():
        global_idx = 0  # 全局样本索引
        for batch_idx, batch_data in enumerate(tqdm(loader, desc="补全need通道")):
            if len(batch_data) == 5:
                batch_x, _, _, _, source_datasets = batch_data
            else:
                # fallback to first element if format is unexpected
                batch_x = batch_data[0]
                source_datasets = ['UNKNOWN'] * batch_x.size(0)
            
            batch_x = batch_x.to(device)
            batch_size, C, T = batch_x.size()
            
            # 对batch中每个样本进行need通道补全
            for i in range(batch_size):
                src = source_datasets[i] if i < len(source_datasets) else 'UNKNOWN'
                
                # 根据source_dataset动态获取该样本的Need通道
                sample_need_indices = original_dataset.get_need_indices_for_dataset(src) if hasattr(original_dataset, 'get_need_indices_for_dataset') else []
                
                # 调试信息
                if batch_idx == 0 and i < 3:  # 只打印前几个样本的调试信息
                    print(f"  样本 {global_idx}: source={src}, need_indices={sample_need_indices}")
                
                if sample_need_indices:  # 只有当该数据集有Need通道时才处理
                    window = batch_x[i].t()  # [T, C]
                    out, _ = model(window)   # 模型重建输出 [C, T]
                    out = out.t()           # 转回 [T, C]
                    
                    # 只保存该样本的need通道预测结果
                    need_pred = {}
                    for need_idx in sample_need_indices:
                        if need_idx < out.size(1):
                            need_pred[need_idx] = out[:, need_idx].cpu()
                    
                    all_need_predictions.append((global_idx, need_pred, src))
                else:
                    # 该数据集没有Need通道，跳过
                    all_need_predictions.append((global_idx, {}, src))
                
                global_idx += 1
    
    # 将预测的need通道写回原始数据集
    if hasattr(original_dataset, 'update_need_channels_by_source'):
        # 如果数据集支持按source批量更新need通道
        original_dataset.update_need_channels_by_source(all_need_predictions)
    elif hasattr(original_dataset, 'update_need_channels'):
        # 兼容旧接口 - 但需要正确处理need_indices
        print("开始批量更新need通道...")
        print(f"   - 预测结果数量: {len(all_need_predictions)}")
        
        # 收集所有实际使用的need通道索引
        all_used_need_indices = set()
        valid_predictions = []
        
        for global_idx, need_pred, src in all_need_predictions:
            if need_pred:  # 只处理非空的预测
                all_used_need_indices.update(need_pred.keys())
                valid_predictions.append((global_idx, need_pred, src))
        
        all_used_need_indices = sorted(list(all_used_need_indices))
        print(f"   - 实际使用的need通道索引: {all_used_need_indices}")
        print(f"   - 有效预测数量: {len(valid_predictions)}")
        
        if all_used_need_indices and valid_predictions:
            # 转换为旧格式：需要为所有样本（包括没有need通道的）构建预测结果
            need_preds_for_old_api = []
            valid_pred_map = {idx: pred for idx, pred, _ in valid_predictions}
            
            for global_idx, _, _ in all_need_predictions:
                if global_idx in valid_pred_map:
                    # 有预测结果的样本
                    need_pred = valid_pred_map[global_idx]
                    old_format_pred = {}
                    for need_idx in all_used_need_indices:
                        if need_idx in need_pred:
                            old_format_pred[need_idx] = need_pred[need_idx]
                    need_preds_for_old_api.append(old_format_pred)
                else:
                    # 没有预测结果的样本，添加空字典
                    need_preds_for_old_api.append({})
            
            # 调用旧接口，传入正确的need_indices
            original_dataset.update_need_channels(need_preds_for_old_api, all_used_need_indices)
        else:
            print("   - 没有有效的need通道预测，跳过更新")
    else:
        # 逐个更新（fallback方案）
        updated_count = 0
        for global_idx, need_pred, src in all_need_predictions:
            if need_pred and hasattr(original_dataset, 'update_need'):
                original_dataset.update_need(global_idx, need_pred)
                updated_count += 1
        print(f"   - 通过fallback方案更新了{updated_count}个样本")
    
    total_updated = sum(1 for _, pred, _ in all_need_predictions if pred)
    print(f"循环补全完成，已更新{total_updated}个样本的need通道（根据source_dataset动态确定）")

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
                                      recon_weight=1.0, cls_improvement_weight=1.0, dataset=None, current_epoch=1,
                                      loss_config=None):
    """训练函数 - 实现双路径分类训练，优化重建数据的分类性能（V13方式）"""
    model.train()
    
    # 从loss_config获取参数
    if loss_config is None:
        loss_config = {}
    
    accuracy_reward_scale = loss_config.get('accuracy_reward_scale', 2.0)
    accuracy_threshold = loss_config.get('accuracy_threshold', 0.05)
    min_improvement_margin = loss_config.get('min_improvement_margin', 0.05)
    dynamic_weighting = loss_config.get('dynamic_weighting', True)
    
    # 简化的双路径分类监督策略（新修正）
    enable_classification_supervision = loss_config.get('classification_supervision', True)
    
    # 损失平滑配置（新增）
    loss_smoothing = loss_config.get('loss_smoothing', False)
    smoothing_factor = loss_config.get('smoothing_factor', 0.9)
    
    # 损失历史追踪（用于平滑）
    if not hasattr(train_phased_with_grad_accumulation, '_loss_history'):
        train_phased_with_grad_accumulation._loss_history = {
            'recon_loss': 0.0,
            'cls_improvement_loss': 0.0,
            'accuracy_improvement': 0.0
        }
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_improvement_loss = 0.0  # 分类改进损失
    total_input_correct = 0   # 输入数据分类正确数（Common+Have+Need初始值）
    total_reconstructed_correct = 0  # 重建数据分类正确数（Common+Have+Need生成值）
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
                batch_reconstructed, _ = forward_batch_parallel_compat(model, masked, device)
                
                # 计算重建损失
                recon_loss = compute_batch_recon_loss(batch, batch_reconstructed, is_real_mask, 
                                                    common_indices, criterion, C, batch_size, need_indices,
                                                    training_strategy, have_indices, source_datasets, dataset, current_epoch)
                
                # 阶段2: 双路径分类训练 - 使用UNet内置分类器（V13方式）
                # 路径1: 输入数据分类 (Common真+Have真+Need初始/上轮生成) → UNet分类器 → 基准性能
                # 注意：输入batch本身就包含了当前的Need值（0或上轮生成的结果）
                _, input_logits = forward_batch_parallel_compat(model, batch, device)
                
                # 路径2: 重建数据分类 (Common真+Have真+Need当前生成) → UNet分类器 → 目标性能
                # 构建增强数据：保留Common+Have，用当前模型生成的Need替换
                # 关键：根据每个样本的source_dataset动态确定Need通道
                enhanced_data = batch.clone()
                
                # 逐样本处理，根据source_dataset动态确定Need通道
                for i in range(batch_size):
                    src = source_datasets[i] if i < len(source_datasets) else 'UNKNOWN'
                    sample_need_indices = dataset.get_need_indices_for_dataset(src) if dataset is not None else []
                    
                    # 用重建结果替换该样本的Need通道
                    for need_idx in sample_need_indices:
                        if need_idx < batch_reconstructed.size(1):
                            enhanced_data[i, need_idx, :] = batch_reconstructed[i, need_idx, :]
                
                _, enhanced_logits = forward_batch_parallel_compat(model, enhanced_data, device)
                
                # 计算分类损失（用于梯度优化）
                input_cls_loss = nn.CrossEntropyLoss()(input_logits, labels)
                enhanced_cls_loss = nn.CrossEntropyLoss()(enhanced_logits, labels)
                
                # 计算准确率用于监控
                input_preds = torch.argmax(input_logits, dim=1)
                enhanced_preds = torch.argmax(enhanced_logits, dim=1)
                input_accuracy = (input_preds == labels).float().mean()
                enhanced_accuracy = (enhanced_preds == labels).float().mean()
                
                # 分类改进损失：循环渐进策略，让每轮生成的数据都比输入数据分类效果更好
                # 优化策略：基于输入数据做监督 + 鼓励生成数据分类性能提升
                
                # 1. 基础分类监督损失：使用输入数据做监督信号
                # 输入数据包含真实的Common+Have + 当前的Need值（初始0或上轮生成结果）
                base_classification_loss = input_cls_loss
                
                # 2. 循环改进损失：鼓励生成数据的分类性能优于输入数据
                # 这样每轮训练都会让Need生成值变得更好
                classification_improvement_loss = enhanced_cls_loss - input_cls_loss
                
                # 3. 准确率差值（用于监控和奖励）
                accuracy_improvement = enhanced_accuracy - input_accuracy
                
                # 4. 稳定的准确率奖励：使用平滑函数避免梯度震荡
                # 使用tanh函数将准确率差值映射到[-1, 1]范围，然后施加奖励
                accuracy_improvement_clamped = torch.clamp(accuracy_improvement, -0.5, 0.5)  # 限制范围避免极值
                accuracy_reward = -torch.tanh(accuracy_improvement_clamped * 10) * accuracy_reward_scale
                
                # 5. 渐进式margin机制：当模型学习稳定时逐渐增加要求
                adaptive_margin = min_improvement_margin * (1.0 + 0.1 * (current_epoch / 100))
                
                # 6. 组合分类损失 = 基础分类监督 + 循环改进损失 + 准确率奖励 + 自适应margin
                cls_improvement_loss = classification_improvement_loss + accuracy_reward + adaptive_margin
                total_classification_loss = base_classification_loss + cls_improvement_loss
            
            # 自适应权重调整：更稳定的动态调整策略
            dynamic_recon_weight = recon_weight
            dynamic_cls_weight = cls_improvement_weight
            
            if dynamic_weighting:
                # 使用指数衰减来平滑权重调整，避免激进变化
                improvement_factor = torch.sigmoid(accuracy_improvement * 10)  # 平滑映射到[0,1]
                
                if accuracy_improvement > accuracy_threshold:
                    # 准确率显著提升时，适度增加分类权重
                    dynamic_cls_weight *= (1.0 + 0.5 * improvement_factor)
                elif accuracy_improvement < -accuracy_threshold:
                    # 准确率显著下降时，增加惩罚但避免过度
                    dynamic_cls_weight *= (1.0 + 1.0 * (1 - improvement_factor))
            
            loss = (dynamic_recon_weight * recon_loss + dynamic_cls_weight * total_classification_loss) / accumulate_grad_batches
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            
            # 在累积周期结束时更新参数
            if (step_count + 1) % accumulate_grad_batches == 0:
                scaler.unscale_(optimizer)
                # 对UNet模型参数进行梯度裁剪（V13方式）
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            # 标准精度训练 - 相同逻辑
            # 阶段1: 重建训练
            batch_reconstructed, _ = forward_batch_parallel_compat(model, masked, device)            # 计算重建损失
            recon_loss = compute_batch_recon_loss(batch, batch_reconstructed, is_real_mask, 
                                                common_indices, criterion, C, batch_size, need_indices,
                                                training_strategy, have_indices, source_datasets, dataset, current_epoch)
            
            # 阶段2: 双路径分类训练 - 使用UNet内置分类器（V13方式）
            # 路径1: 输入数据分类 (Common真+Have真+Need初始/上轮生成) → UNet分类器 → 基准性能
            _, input_logits = forward_batch_parallel_compat(model, batch, device)
            
            # 路径2: 重建数据分类 (Common真+Have真+Need当前生成) → UNet分类器 → 目标性能
            # 构建增强数据：保留Common+Have，用当前模型生成的Need替换  
            # 关键：根据每个样本的source_dataset动态确定Need通道
            enhanced_data = batch.clone()
            
            # 逐样本处理，根据source_dataset动态确定Need通道
            for i in range(batch_size):
                src = source_datasets[i] if i < len(source_datasets) else 'UNKNOWN'
                sample_need_indices = dataset.get_need_indices_for_dataset(src) if dataset is not None else []
                
                # 用重建结果替换该样本的Need通道
                for need_idx in sample_need_indices:
                    if need_idx < batch_reconstructed.size(1):
                        enhanced_data[i, need_idx, :] = batch_reconstructed[i, need_idx, :]
            
            _, enhanced_logits = forward_batch_parallel_compat(model, enhanced_data, device)
            
            # 计算各路径的分类损失
            input_cls_loss = nn.CrossEntropyLoss()(input_logits, labels)
            enhanced_cls_loss = nn.CrossEntropyLoss()(enhanced_logits, labels)
            
            # 计算准确率用于监控
            input_preds = torch.argmax(input_logits, dim=1)
            enhanced_preds = torch.argmax(enhanced_logits, dim=1)
            input_accuracy = (input_preds == labels).float().mean()
            enhanced_accuracy = (enhanced_preds == labels).float().mean()
            
            # 分类改进损失：循环渐进策略，让每轮生成的数据都比输入数据分类效果更好
            # 优化策略：基于输入数据做监督 + 鼓励生成数据分类性能提升
            
            # 1. 基础分类监督损失：使用输入数据做监督信号
            # 输入数据包含真实的Common+Have + 当前的Need值（初始0或上轮生成结果）
            base_classification_loss = input_cls_loss
            
            # 2. 循环改进损失：鼓励生成数据的分类性能优于输入数据
            # 这样每轮训练都会让Need生成值变得更好
            classification_improvement_loss = enhanced_cls_loss - input_cls_loss
            
            # 3. 准确率差值（用于监控和奖励）
            accuracy_improvement = enhanced_accuracy - input_accuracy
            
            # 4. 稳定的准确率奖励：使用平滑函数避免梯度震荡
            # 使用tanh函数将准确率差值映射到[-1, 1]范围，然后施加奖励
            accuracy_improvement_clamped = torch.clamp(accuracy_improvement, -0.5, 0.5)  # 限制范围避免极值
            accuracy_reward = -torch.tanh(accuracy_improvement_clamped * 10) * accuracy_reward_scale
            
            # 5. 渐进式margin机制：当模型学习稳定时逐渐增加要求
            adaptive_margin = min_improvement_margin * (1.0 + 0.1 * (current_epoch / 100))
            
            # 6. 组合分类损失 = 基础分类监督 + 循环改进损失 + 准确率奖励 + 自适应margin
            cls_improvement_loss = classification_improvement_loss + accuracy_reward + adaptive_margin
            total_classification_loss = base_classification_loss + cls_improvement_loss
            
            # 自适应权重调整：更稳定的动态调整策略
            dynamic_recon_weight = recon_weight
            dynamic_cls_weight = cls_improvement_weight
            
            if dynamic_weighting:
                # 使用指数衰减来平滑权重调整，避免激进变化
                improvement_factor = torch.sigmoid(accuracy_improvement * 10)  # 平滑映射到[0,1]
                
                if accuracy_improvement > accuracy_threshold:
                    # 准确率显著提升时，适度增加分类权重
                    dynamic_cls_weight *= (1.0 + 0.5 * improvement_factor)
                elif accuracy_improvement < -accuracy_threshold:
                    # 准确率显著下降时，增加惩罚但避免过度
                    dynamic_cls_weight *= (1.0 + 1.0 * (1 - improvement_factor))
            
            loss = (dynamic_recon_weight * recon_loss + dynamic_cls_weight * total_classification_loss) / accumulate_grad_batches
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
        total_input_correct += (input_preds == labels).sum().item()
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
        
        # 应用损失平滑（新增）
        if loss_smoothing and step_count > 0:
            # 指数移动平均平滑
            train_phased_with_grad_accumulation._loss_history['recon_loss'] = \
                smoothing_factor * train_phased_with_grad_accumulation._loss_history['recon_loss'] + \
                (1 - smoothing_factor) * recon_val
            
            train_phased_with_grad_accumulation._loss_history['cls_improvement_loss'] = \
                smoothing_factor * train_phased_with_grad_accumulation._loss_history['cls_improvement_loss'] + \
                (1 - smoothing_factor) * cls_improvement_val
            
            # 获取准确率改进值用于平滑
            current_accuracy_improvement = (enhanced_preds == labels).float().mean().item() - \
                                         (input_preds == labels).float().mean().item()
            train_phased_with_grad_accumulation._loss_history['accuracy_improvement'] = \
                smoothing_factor * train_phased_with_grad_accumulation._loss_history['accuracy_improvement'] + \
                (1 - smoothing_factor) * current_accuracy_improvement
        else:
            # 初始化损失历史
            train_phased_with_grad_accumulation._loss_history['recon_loss'] = recon_val
            train_phased_with_grad_accumulation._loss_history['cls_improvement_loss'] = cls_improvement_val
            current_accuracy_improvement = (enhanced_preds == labels).float().mean().item() - \
                                         (input_preds == labels).float().mean().item()
            train_phased_with_grad_accumulation._loss_history['accuracy_improvement'] = current_accuracy_improvement
        
        total_loss += loss_val * batch_size
        total_recon_loss += recon_val * batch_size
        total_cls_improvement_loss += cls_improvement_val * batch_size
        step_count += 1
    
    # 处理最后的不完整累积批次
    if step_count % accumulate_grad_batches != 0:
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            scaler.unscale_(optimizer)
            # 对UNet模型参数进行梯度裁剪（V13方式）
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 对UNet模型参数进行梯度裁剪（V13方式）
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    n = len(dataloader.dataset)
    input_acc = total_input_correct / total_samples if total_samples > 0 else 0.0
    reconstructed_acc = total_reconstructed_correct / total_samples if total_samples > 0 else 0.0
    print(f"[采样分布] 本轮训练各source_dataset采样数: {source_dataset_counter}")
    print(f"[分类性能] 输入数据准确率: {input_acc:.4f}, 重建数据准确率: {reconstructed_acc:.4f}")
    return total_loss / n, total_recon_loss / n, total_cls_improvement_loss / n, input_acc, reconstructed_acc, 0.0


def forward_batch_parallel(model, input_batch, device, use_generation_validator=False):
    """
    批量前向传播的包装器函数，兼容现有代码
    Args:
        model: 模型
        input_batch: 输入批量数据
        device: 设备
        use_generation_validator: 是否使用验证生成分类器
    Returns:
        如果use_generation_validator=False: (batch_out, batch_logits)
        如果use_generation_validator=True: (batch_out, batch_logits, batch_gen_logits)
    """
    batch_size, C, T = input_batch.size()
    
    # 批量转置: [batch_size, C, T] -> [batch_size, T, C] 
    windows = input_batch.transpose(1, 2)  # [batch_size, T, C]
      # 尝试真正的批量处理 - 直接处理整个batch
    try:
        if use_generation_validator:
            batch_out, batch_logits, batch_gen_logits = model.forward_batch(windows, use_generation_validator=True)
            return batch_out, batch_logits, batch_gen_logits  # [batch_size, C, T], [batch_size, num_classes], [batch_size, num_classes]
        else:
            batch_out, batch_logits = model.forward_batch(windows, use_generation_validator=False)
            return batch_out, batch_logits  # [batch_size, C, T], [batch_size, num_classes]
    except AttributeError:
        try:
            def single_forward(window):
                if use_generation_validator:
                    result = model(window, use_generation_validator=True)
                    if len(result) == 3:
                        out, logits, gen_logits = result
                    else:
                        out, logits, gen_logits = result[0], result[-2], result[-1]
                    return out.t(), logits, gen_logits  # [C, T], [num_classes], [num_classes]
                else:
                    out, logits = model(window)
                    return out.t(), logits  # [C, T], [num_classes]
            
            # 使用vmap进行真正的并行化
            vmapped_forward = torch.vmap(single_forward, in_dims=0, out_dims=0)
            if use_generation_validator:
                batch_out, batch_logits, batch_gen_logits = vmapped_forward(windows)
                return batch_out, batch_logits, batch_gen_logits
            else:
                batch_out, batch_logits = vmapped_forward(windows)
                return batch_out, batch_logits
        except Exception as e:
            print(f"[DEBUG] vmap failed: {e}, falling back to sequential processing")
            
            # 最终回退到循环处理
            batch_outputs = []
            batch_logits_list = []
            batch_gen_logits_list = []
            
            for i in range(batch_size):
                window = windows[i]  # [T, C]
                if use_generation_validator:
                    result = model(window, use_generation_validator=True)
                    if len(result) == 3:
                        out, logits, gen_logits = result
                    else:
                        out, logits, gen_logits = result[0], result[-2], result[-1]
                    batch_outputs.append(out.t())  # [C, T]
                    batch_logits_list.append(logits)
                    batch_gen_logits_list.append(gen_logits)
                else:
                    out, logits = model(window)
                    batch_outputs.append(out.t())  # [C, T]
                    batch_logits_list.append(logits)
            
            batch_out = torch.stack(batch_outputs, dim=0)  # [batch_size, C, T]
            batch_logits = torch.stack(batch_logits_list, dim=0)  # [batch_size, num_classes]
            
            if use_generation_validator:
                batch_gen_logits = torch.stack(batch_gen_logits_list, dim=0)  # [batch_size, num_classes]
                return batch_out, batch_logits, batch_gen_logits
            else:
                return batch_out, batch_logits


def forward_batch_parallel_compat(model, input_batch, device):
    """
    兼容旧代码的前向传播函数，始终返回2个值
    """
    result = forward_batch_parallel(model, input_batch, device, use_generation_validator=False)
    return result[0], result[1]  # 只返回 batch_out, batch_logits


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

def eval_loop(model, dataloader, criterion, device, mask_indices, need_indices=None, dataset=None):
    """
    评估函数 - 计算重建损失和双路径分类准确率（V13方式）
    
    Returns:
        tuple: (total_loss, total_recon_loss, input_acc, reconstructed_acc)
               其中 input_acc 是输入数据准确率 (Common真+Have真+Need当前值)
               reconstructed_acc 是重建数据准确率 (Common真+Have真+Need生成值)
               
    注意：Need通道使用当前数据集中的值（循环更新后的值），确保与训练时一致
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_input_correct = 0   # 输入数据分类正确数 (Common+Have+Need=0)
    total_reconstructed_correct = 0  # 重建数据分类正确数 (Common+Have+Need生成)
    total_samples = 0
    
    # 获取common模态索引
    common_indices = getattr(criterion, 'common_indices', [])
    need_indices = need_indices if need_indices is not None else []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Eval"):
            if len(batch_data) == 4:
                batch, labels, _, is_real_mask = batch_data
                source_datasets = ['UNKNOWN'] * batch.size(0)
            else:
                batch, labels, _, is_real_mask, source_datasets = batch_data
            
            batch = batch.to(device)
            labels = labels.to(device)
            is_real_mask = is_real_mask.to(device)
            
            # 动态遮掩
            masked = mask_channel(batch, source_datasets, dataset)
            batch_size, C, T = batch.size()
            loss = 0.0
            recon_loss = 0.0
            
            # 批量处理重建和分类
            batch_reconstructed, _ = forward_batch_parallel_compat(model, masked, device)
            
            # 构建输入数据用于分类 (Common真+Have真+Need当前值)
            # 注意：Need通道使用当前数据集中的值（可能是0或循环更新后的值）
            # 这确保了验证/测试与训练时的一致性
            input_data = batch.clone()  # 直接使用当前batch，包含循环更新后的Need值
            
            # 构建重建数据用于分类 (Common真+Have真+Need生成)
            # 根据每个样本的source_dataset动态确定Need通道并用生成值替换
            enhanced_data = batch.clone()
            for i in range(batch_size):
                src = source_datasets[i] if i < len(source_datasets) else 'UNKNOWN'
                sample_need_indices = dataset.get_need_indices_for_dataset(src) if dataset is not None else []
                
                # 用重建结果替换该样本的Need通道
                for need_idx in sample_need_indices:
                    if need_idx < batch_reconstructed.size(1):
                        enhanced_data[i, need_idx, :] = batch_reconstructed[i, need_idx, :]
            
            # 分类评估 - 使用UNet内置分类器（V13方式）
            _, input_logits = forward_batch_parallel_compat(model, input_data, device)
            _, enhanced_logits = forward_batch_parallel_compat(model, enhanced_data, device)
            
            # 计算准确率
            input_preds = torch.argmax(input_logits, dim=1)
            enhanced_preds = torch.argmax(enhanced_logits, dim=1)
            
            total_input_correct += (input_preds == labels).sum().item()
            total_reconstructed_correct += (enhanced_preds == labels).sum().item()
            total_samples += batch_size
            
            # 计算重建损失 (保持原有逻辑)
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
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=True)
                        else:
                            recon_loss_i = recon_loss_i + criterion(pred, target)
                        real_count += 1
                    elif real_channels[c]:
                        # Have模态：只对真实通道计算损失
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=False)
                        else:
                            recon_loss_i = recon_loss_i + criterion(pred, target)
                        real_count += 1
                
                if real_count > 0:
                    recon_loss_i = recon_loss_i / real_count
                
                loss += recon_loss_i
                recon_loss += recon_loss_i
            
            loss = loss / batch_size
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss
    
    n = len(dataloader.dataset)
    input_acc = total_input_correct / total_samples if total_samples > 0 else 0.0
    reconstructed_acc = total_reconstructed_correct / total_samples if total_samples > 0 else 0.0
    
    return total_loss / n, total_recon_loss / n, input_acc, reconstructed_acc

def eval_with_generation_validator(model, dataloader, criterion, device, mask_indices, need_indices=None, dataset=None):
    """
    增强评估函数 - 计算重建损失和三路径分类准确率（含验证生成分类器）
    
    Returns:
        tuple: (total_loss, total_recon_loss, input_acc, reconstructed_acc, generation_validator_acc)
               其中 input_acc 是输入数据准确率 (Common真+Have真+Need当前值)
               reconstructed_acc 是重建数据准确率 (Common真+Have真+Need生成值)
               generation_validator_acc 是验证生成分类器准确率 (基于生成数据的专门分类器)
               
    注意：验证生成分类器直接基于解码器输出进行分类，可以监控生成质量
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_input_correct = 0   # 输入数据分类正确数
    total_reconstructed_correct = 0  # 重建数据分类正确数
    total_generation_validator_correct = 0  # 验证生成分类正确数
    total_samples = 0
    
    # 获取common模态索引
    common_indices = getattr(criterion, 'common_indices', [])
    need_indices = need_indices if need_indices is not None else []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Eval with Generation Validator"):
            if len(batch_data) == 4:
                batch, labels, _, is_real_mask = batch_data
                source_datasets = ['UNKNOWN'] * batch.size(0)
            else:
                batch, labels, _, is_real_mask, source_datasets = batch_data
            
            batch = batch.to(device)
            labels = labels.to(device)
            is_real_mask = is_real_mask.to(device)
            
            # 动态遮掩
            masked = mask_channel(batch, source_datasets, dataset)
            batch_size, C, T = batch.size()
            loss = 0.0
            recon_loss = 0.0
            
            # 批量处理重建和分类（带验证生成分类器）
            result = forward_batch_parallel(model, masked, device, use_generation_validator=True)
            if len(result) == 3:
                batch_reconstructed, _, batch_gen_logits = result
            else:
                # 回退处理：如果返回2个值，说明模型不支持验证生成分类器
                batch_reconstructed, _ = result
                # 创建一个基础的验证生成分类结果
                batch_gen_logits = torch.zeros((batch_size, 2), device=device)
            
            # 构建输入数据用于分类 (Common真+Have真+Need当前值)
            input_data = batch.clone()  # 直接使用当前batch，包含循环更新后的Need值
            
            # 构建重建数据用于分类 (Common真+Have真+Need生成)
            # 根据每个样本的source_dataset动态确定Need通道并用生成值替换
            enhanced_data = batch.clone()
            for i in range(batch_size):
                src = source_datasets[i] if i < len(source_datasets) else 'UNKNOWN'
                sample_need_indices = dataset.get_need_indices_for_dataset(src) if dataset is not None else []
                
                # 用重建结果替换该样本的Need通道
                for need_idx in sample_need_indices:
                    if need_idx < batch_reconstructed.size(1):
                        enhanced_data[i, need_idx, :] = batch_reconstructed[i, need_idx, :]
            
            # 三路径分类评估
            _, input_logits = forward_batch_parallel_compat(model, input_data, device)
            _, enhanced_logits = forward_batch_parallel_compat(model, enhanced_data, device)
            
            # 计算准确率
            input_preds = torch.argmax(input_logits, dim=1)
            enhanced_preds = torch.argmax(enhanced_logits, dim=1)
            generation_validator_preds = torch.argmax(batch_gen_logits, dim=1)
            
            total_input_correct += (input_preds == labels).sum().item()
            total_reconstructed_correct += (enhanced_preds == labels).sum().item()
            total_generation_validator_correct += (generation_validator_preds == labels).sum().item()
            total_samples += batch_size
            
            # 计算重建损失 (保持原有逻辑)
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
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=True)
                        else:
                            recon_loss_i = recon_loss_i + criterion(pred, target)
                        real_count += 1
                    elif real_channels[c]:
                        # Have模态：只对真实通道计算损失
                        if hasattr(criterion, '__call__') and criterion.__class__.__name__ != 'MSELoss':
                            recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=False)
                        else:
                            recon_loss_i = recon_loss_i + criterion(pred, target)
                        real_count += 1
                
                if real_count > 0:
                    recon_loss_i = recon_loss_i / real_count
                
                loss += recon_loss_i
                recon_loss += recon_loss_i
            
            loss = loss / batch_size
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss
    
    n = len(dataloader.dataset)
    input_acc = total_input_correct / total_samples if total_samples > 0 else 0.0
    reconstructed_acc = total_reconstructed_correct / total_samples if total_samples > 0 else 0.0
    generation_validator_acc = total_generation_validator_correct / total_samples if total_samples > 0 else 0.0
    
    return total_loss / n, total_recon_loss / n, input_acc, reconstructed_acc, generation_validator_acc


def train_generation_validator_only(model, dataloader, optimizer, criterion, device, mask_indices, 
                                  accumulate_grad_batches=2, use_mixed_precision=True, scaler=None, 
                                  dataset=None, current_epoch=1):
    """
    专门训练验证生成分类器的函数
    冻结其他参数，只训练generation_validator分支
    """
    model.train()
    
    # 冻结除了验证生成分类器之外的所有参数
    for param in model.parameters():
        param.requires_grad = False
    for param in model.generation_validator.parameters():
        param.requires_grad = True
    
    # 但需要编码器+解码器参与前向传播，只是不更新梯度
    # 这样可以保证生成的特征是稳定的
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    step_count = 0
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training Generation Validator")):
        if len(batch_data) == 5:
            batch, labels, _, is_real_mask, source_datasets = batch_data
        elif len(batch_data) == 4:
            batch, labels, _, is_real_mask = batch_data
            source_datasets = ['UNKNOWN'] * batch.size(0)
        else:
            batch, labels, _, is_real_mask, source_datasets = batch_data
        
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 动态遮掩
        masked = mask_channel(batch, source_datasets, dataset)
        batch_size = batch.size(0)
        
        # 梯度累积：只在累积周期开始时清零梯度
        if step_count % accumulate_grad_batches == 0:
            optimizer.zero_grad()
        
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            with autocast():
                # 获取生成验证分类结果
                result = forward_batch_parallel(model, masked, device, use_generation_validator=True)
                if len(result) == 3:
                    _, _, gen_logits = result
                else:
                    # 回退处理
                    _, _ = result
                    gen_logits = torch.zeros((batch_size, 2), device=device)
                
                # 计算分类损失
                gen_cls_loss = nn.CrossEntropyLoss()(gen_logits, labels)
                loss = gen_cls_loss / accumulate_grad_batches
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            
            # 在累积周期结束时更新参数
            if (step_count + 1) % accumulate_grad_batches == 0:
                scaler.unscale_(optimizer)
                # 只对验证生成分类器进行梯度裁剪
                clip_grad_norm_(model.generation_validator.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            # 标准精度训练
            result = forward_batch_parallel(model, masked, device, use_generation_validator=True)
            if len(result) == 3:
                _, _, gen_logits = result
            else:
                # 回退处理
                _, _ = result
                gen_logits = torch.zeros((batch_size, 2), device=device)
            gen_cls_loss = nn.CrossEntropyLoss()(gen_logits, labels)
            loss = gen_cls_loss / accumulate_grad_batches
            loss.backward()
            
            # 在累积周期结束时更新参数
            if (step_count + 1) % accumulate_grad_batches == 0:
                clip_grad_norm_(model.generation_validator.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 统计信息
        gen_preds = torch.argmax(gen_logits, dim=1)
        total_correct += (gen_preds == labels).sum().item()
        total_samples += batch_size
        total_loss += (loss.item() * accumulate_grad_batches) * batch_size
        step_count += 1
    
    # 处理最后的不完整累积批次
    if step_count % accumulate_grad_batches != 0:
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.generation_validator.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.generation_validator.parameters(), max_norm=1.0)
            optimizer.step()
    
    n = len(dataloader.dataset)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 恢复所有参数的梯度
    for param in model.parameters():
        param.requires_grad = True
    
    return total_loss / n, acc

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
    
    # 检查标签分布
    logger.info("检查数据集标签分布...")
    if hasattr(dataset, 'analyze_dataset_label_distribution'):
        dataset.analyze_dataset_label_distribution()
    
    label_counter, all_labels = check_label_distribution(dataset)
    logger.info(f"标签分布: {dict(label_counter)}")
    logger.info(f"所有标签: {sorted(list(all_labels))}")
    
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
    
    # 创建主生成模型（TGATUNet）
    model = TGATUNet(
        in_channels=input_channels,
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=input_channels,
        num_classes=config.get('num_classes', 2)
    ).to(device)
    logger.info(f"主生成模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # V16按照V13方式：使用UNet内置分类器，不创建独立分类器
    # 合并UNet模型参数进行优化
    all_parameters = list(model.parameters())
    optimizer = Adam(all_parameters, lr=config.get('learning_rate', 1e-4), weight_decay=config.get('weight_decay', 1e-5))
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
        logger.info("使用标准MSE损失函数")
    
    # 检查是否有现有检查点可以恢复训练
    start_epoch = 1
    resume_training = config.get('resume_training', False)
    checkpoint_path = os.path.join(ckpt_dir, 'best_model.pth')
    
    if resume_training and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                start_epoch = checkpoint.get('epoch', 1) + 1
                
                if scaler is not None and 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
                logger.info(f"恢复训练从第 {start_epoch} 轮开始，最佳验证损失: {best_val_loss:.6f}")
            else:
                logger.info("检查点格式不兼容，从头开始训练")
        except Exception as e:
            logger.warning(f"恢复训练失败: {e}，从头开始训练")
    elif resume_training:
        logger.info("未找到检查点文件，从头开始训练")    # 训练主循环
    logger.info(f"开始训练，总轮数: {epochs}")
    
    # 记录关键超参数配置
    loss_config = config.get('loss_config', {})
    logger.info("=== 训练配置摘要 ===")
    logger.info(f"模型: TGATUNet, 输入通道: {input_channels}, 隐藏通道: {config.get('hidden_channels', 64)}")
    logger.info(f"批量大小: {batch_size}, 学习率: {config.get('learning_rate', 1e-4)}")
    logger.info(f"损失权重: recon={loss_config.get('recon_weight', 1.0)}, cls_improvement={loss_config.get('cls_improvement_weight', 2.0)}")
    logger.info(f"准确率奖励: scale={loss_config.get('accuracy_reward_scale', 2.0)}, threshold={loss_config.get('accuracy_threshold', 0.05)}")
    logger.info(f"动态调整: {loss_config.get('dynamic_weighting', True)}, 损失平滑: {loss_config.get('loss_smoothing', False)}")
    logger.info(f"训练策略: {config.get('training_strategy', 'mask_have')}")
    logger.info("=== 循环渐进训练策略 ===")
    logger.info("路径1: 输入数据分类 (Common真实 + Have真实 + Need初始值/上轮生成) → 基准性能")
    logger.info("路径2: 重建数据分类 (Common真实 + Have真实 + Need当前生成) → 目标性能")
    logger.info("目标: 每轮训练都让路径2的分类效果比路径1更好，实现Need通道的循序渐进补全")
    logger.info("=====================")
    for epoch in range(start_epoch, epochs + 1):
        # 在每个epoch开始前，用当前模型对训练集进行Need通道循环补全
        logger.info(f"Epoch {epoch}: 开始训练集Need通道循环补全...")
        complete_need_with_model(model, train_dataset, device)
        
        train_loss, train_recon, train_cls_improvement, train_input_acc, train_reconstructed_acc, _ = train_phased_with_grad_accumulation(
            model, train_loader, optimizer, criterion, device, [],
            accumulate_grad_batches=config.get('accumulate_grad_batches', 2),
            use_mixed_precision=AMP_AVAILABLE and config.get('use_amp', False), scaler=scaler,
            need_indices=[], training_strategy=config.get('training_strategy', 'mask_have'),
            common_indices=[], have_indices=None, 
            recon_weight=config.get('loss_config', {}).get('recon_weight', 1.0), 
            cls_improvement_weight=config.get('loss_config', {}).get('cls_improvement_weight', 2.0),
            dataset=dataset, current_epoch=epoch, loss_config=config.get('loss_config', {})
        )
        
        # 在验证前，也用当前模型对验证集进行Need通道更新，确保一致性
        logger.info(f"Epoch {epoch}: 开始验证集Need通道更新...")
        complete_need_with_model(model, val_dataset, device)
        
        # 验证阶段
        val_loss, val_recon, val_input_acc, val_reconstructed_acc = eval_loop(
            model, val_loader, criterion, device, [], need_indices=[], dataset=dataset
        )
        
        # 验证生成分类器评估（如果启用）
        gen_val_metrics = None
        if config.get('enable_generation_validator', False):
            gen_val_config = config.get('generation_validator_config', {})
            eval_frequency = gen_val_config.get('eval_frequency', 5)
            
            if epoch % eval_frequency == 0:
                logger.info(f"Epoch {epoch}: 开始验证生成分类器评估...")
                gen_val_loss, gen_val_recon, gen_input_acc, gen_reconstructed_acc, gen_validator_acc = eval_with_generation_validator(
                    model, val_loader, criterion, device, [], need_indices=[], dataset=dataset
                )
                
                # 计算各种指标
                gen_recon_vs_input = gen_reconstructed_acc - gen_input_acc
                gen_validator_vs_input = gen_validator_acc - gen_input_acc
                
                # 记录验证生成分类器指标
                logger.info(f"验证生成分类器结果:")
                logger.info(f"  输入数据准确率: {gen_input_acc:.4f}")
                logger.info(f"  重建数据准确率: {gen_reconstructed_acc:.4f}")
                logger.info(f"  生成验证准确率: {gen_validator_acc:.4f}")
                logger.info(f"  重建 vs 输入提升: {gen_recon_vs_input:+.4f}")
                logger.info(f"  生成验证 vs 输入提升: {gen_validator_vs_input:+.4f}")
                
                # 将验证生成分类器指标存储为字典，方便后续使用
                gen_val_metrics = {
                    'loss': gen_val_loss,
                    'recon_loss': gen_val_recon,
                    'input_acc': gen_input_acc,
                    'reconstructed_acc': gen_reconstructed_acc,
                    'generated_acc': gen_validator_acc,
                    'recon_vs_input': gen_recon_vs_input,
                    'gen_vs_input': gen_validator_vs_input
                }
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 日志记录
        accuracy_improvement = train_reconstructed_acc - train_input_acc
        val_accuracy_improvement = val_reconstructed_acc - val_input_acc
        
        # 记录平滑损失信息（如果启用）
        loss_config = config.get('loss_config', {})
        if loss_config.get('loss_smoothing', False) and hasattr(train_phased_with_grad_accumulation, '_loss_history'):
            smoothed_recon = train_phased_with_grad_accumulation._loss_history.get('recon_loss', train_recon)
            smoothed_cls = train_phased_with_grad_accumulation._loss_history.get('cls_improvement_loss', train_cls_improvement)
            smoothed_acc_improvement = train_phased_with_grad_accumulation._loss_history.get('accuracy_improvement', accuracy_improvement)
            
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, train_recon={train_recon:.6f} (smoothed: {smoothed_recon:.6f}), "
                       f"train_input_acc={train_input_acc:.4f}, train_reconstructed_acc={train_reconstructed_acc:.4f}, "
                       f"train_accuracy_improvement={accuracy_improvement:+.4f} (smoothed: {smoothed_acc_improvement:+.4f}), "
                       f"val_loss={val_loss:.6f}, val_recon={val_recon:.6f}, "
                       f"val_input_acc={val_input_acc:.4f}, val_reconstructed_acc={val_reconstructed_acc:.4f}, "
                       f"val_accuracy_improvement={val_accuracy_improvement:+.4f}, lr={current_lr:.6e}")
        else:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, train_recon={train_recon:.6f}, "
                       f"train_input_acc={train_input_acc:.4f}, train_reconstructed_acc={train_reconstructed_acc:.4f}, "
                       f"train_accuracy_improvement={accuracy_improvement:+.4f}, "
                       f"val_loss={val_loss:.6f}, val_recon={val_recon:.6f}, "
                       f"val_input_acc={val_input_acc:.4f}, val_reconstructed_acc={val_reconstructed_acc:.4f}, "
                       f"val_accuracy_improvement={val_accuracy_improvement:+.4f}, lr={current_lr:.6e}")
        
        # TensorBoard记录
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Recon_Loss', train_recon, epoch)
        writer.add_scalar('Train/Cls_Improvement_Loss', train_cls_improvement, epoch)
        writer.add_scalar('Train/Input_Acc', train_input_acc, epoch)  # 输入数据准确率
        writer.add_scalar('Train/Reconstructed_Acc', train_reconstructed_acc, epoch)
        writer.add_scalar('Train/Accuracy_Improvement', accuracy_improvement, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Recon_Loss', val_recon, epoch)
        writer.add_scalar('Val/Input_Acc', val_input_acc, epoch)  # 验证集输入数据准确率
        writer.add_scalar('Val/Reconstructed_Acc', val_reconstructed_acc, epoch)  # 验证集重建数据准确率
        writer.add_scalar('Val/Accuracy_Improvement', val_accuracy_improvement, epoch)  # 验证集准确率提升
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        # 记录验证生成分类器指标到TensorBoard（如果有）
        if gen_val_metrics is not None:
            writer.add_scalar('Val_GenValidator/Input_Acc', gen_val_metrics['input_acc'], epoch)
            writer.add_scalar('Val_GenValidator/Reconstructed_Acc', gen_val_metrics['reconstructed_acc'], epoch)
            writer.add_scalar('Val_GenValidator/Generated_Acc', gen_val_metrics['generated_acc'], epoch)
            writer.add_scalar('Val_GenValidator/Recon_vs_Input', gen_val_metrics['recon_vs_input'], epoch)
            writer.add_scalar('Val_GenValidator/Gen_vs_Input', gen_val_metrics['gen_vs_input'], epoch)
            writer.add_scalar('Val_GenValidator/Loss', gen_val_metrics['loss'], epoch)
            writer.add_scalar('Val_GenValidator/Recon_Loss', gen_val_metrics['recon_loss'], epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 构建完整的检查点信息（V13方式：只保存UNet模型）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'input_channels': input_channels,
                'train_loss': train_loss,
                'train_recon': train_recon,
                'train_input_acc': train_input_acc,
                'train_reconstructed_acc': train_reconstructed_acc,
                'val_input_acc': val_input_acc,
                'val_reconstructed_acc': val_reconstructed_acc,
                'accuracy_improvement': accuracy_improvement,
                'val_accuracy_improvement': val_accuracy_improvement
            }
            
            # 如果有验证生成分类器指标，也保存到检查点
            if gen_val_metrics is not None:
                checkpoint.update({
                    'gen_val_input_acc': gen_val_metrics['input_acc'],
                    'gen_val_reconstructed_acc': gen_val_metrics['reconstructed_acc'],
                    'gen_val_generated_acc': gen_val_metrics['generated_acc'],
                    'gen_val_recon_vs_input': gen_val_metrics['recon_vs_input'],
                    'gen_val_gen_vs_input': gen_val_metrics['gen_vs_input'],
                    'gen_val_loss': gen_val_metrics['loss'],
                    'gen_val_recon_loss': gen_val_metrics['recon_loss']
                })
            
            # 如果使用了混合精度训练，也保存scaler状态
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            # 保存完整检查点
            checkpoint_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # 额外保存一个只包含模型权重的文件（向后兼容）
            model_only_path = os.path.join(ckpt_dir, 'best_model_weights_only.pth')
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
                      model_only_path)
            
            logger.info(f"保存最佳模型，验证损失: {best_val_loss:.6f}")
            logger.info(f"保存完整检查点到: {checkpoint_path}")
            logger.info(f"保存模型权重到: {model_only_path}")
        else:
            patience_counter += 1
        
        # 周期性保存检查点（每10个epoch或配置指定的间隔）
        save_interval = config.get('save_interval', 10)
        if epoch % save_interval == 0:
            periodic_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'current_val_loss': val_loss,
                'config': config,
                'input_channels': input_channels
            }
            
            if scaler is not None:
                periodic_checkpoint['scaler_state_dict'] = scaler.state_dict()
                
            periodic_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(periodic_checkpoint, periodic_path)
            logger.info(f"周期性保存检查点到: {periodic_path}")
            
        # 🔥 循环学习核心机制：每个epoch结束后用当前模型更新数据集的Need通道
        if epoch % config.get('need_update_interval', 1) == 0:  # 每N个epoch更新一次Need通道
            logger.info(f"开始第 {epoch} 轮循环更新Need通道...")
            
            # 用当前模型对训练数据集进行Need通道补全（动态根据source_dataset确定Need通道）
            logger.info("更新训练数据集的Need通道（根据source_dataset动态确定）")
            complete_need_with_model(model, train_dataset, device)
            
            # 同样更新验证数据集的Need通道（可选，但建议更新以保持一致性）
            if config.get('update_val_need', True):
                logger.info("更新验证数据集的Need通道（根据source_dataset动态确定）")
                complete_need_with_model(model, val_dataset, device)
            
            logger.info(f"第 {epoch} 轮Need通道更新完成")
        
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"早停触发，在第 {epoch} 轮停止训练")
            break
    
    # 测试评估
    logger.info("开始测试集评估...")
    
    # 智能加载最佳模型
    checkpoint_path = os.path.join(ckpt_dir, 'best_model.pth')
    model_only_path = os.path.join(ckpt_dir, 'best_model_weights_only.pth')
    
    try:
        # 尝试加载完整检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 这是完整的检查点格式
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("成功加载模型状态")
            
            logger.info(f"从完整检查点加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
            logger.info(f"最佳验证损失: {checkpoint.get('best_val_loss', 'unknown')}")
            
            # 如果有训练和验证指标，也记录下来
            if 'train_input_acc' in checkpoint:
                logger.info(f"训练集 - 输入准确率: {checkpoint['train_input_acc']:.4f}, "
                           f"重建准确率: {checkpoint['train_reconstructed_acc']:.4f}")
            if 'val_input_acc' in checkpoint:
                logger.info(f"验证集 - 输入准确率: {checkpoint['val_input_acc']:.4f}, "
                           f"重建准确率: {checkpoint['val_reconstructed_acc']:.4f}")
        else:
            # 这是旧格式的state_dict
            model.load_state_dict(checkpoint)
            logger.info("从旧格式检查点加载模型")
            logger.warning("旧格式检查点不包含分类器，使用随机初始化的分类器")
            
    except Exception as e:
        logger.warning(f"加载完整检查点失败: {e}")
        
        # 尝试加载仅权重文件
        try:
            if os.path.exists(model_only_path):
                model.load_state_dict(torch.load(model_only_path, map_location=device))
                logger.info("从仅权重文件加载模型")
            else:
                logger.error("无法找到任何模型文件！")
                raise FileNotFoundError("找不到模型检查点文件")
        except Exception as e2:
            logger.error(f"加载仅权重文件也失败: {e2}")
            raise
    
    # 在测试前，用最佳模型对测试集进行Need通道更新
    logger.info("开始测试集Need通道更新...")
    complete_need_with_model(model, test_dataset, device)
    
    test_loss, test_recon, test_input_acc, test_reconstructed_acc = eval_loop(
        model, test_loader, criterion, device, [], need_indices=[], dataset=dataset
    )
    
    # 计算测试集的准确率提升
    test_accuracy_improvement = test_reconstructed_acc - test_input_acc
    
    # 测试验证生成分类器性能（如果启用）
    test_gen_val_metrics = None
    if config.get('enable_generation_validator', False):
        logger.info("开始测试集验证生成分类器评估...")
        test_gen_val_loss, test_gen_val_recon, test_gen_input_acc, test_gen_reconstructed_acc, test_gen_validator_acc = eval_with_generation_validator(
            model, test_loader, criterion, device, [], need_indices=[], dataset=dataset
        )
        
        test_gen_recon_vs_input = test_gen_reconstructed_acc - test_gen_input_acc
        test_gen_validator_vs_input = test_gen_validator_acc - test_gen_input_acc
        
        test_gen_val_metrics = {
            'loss': test_gen_val_loss,
            'recon_loss': test_gen_val_recon,
            'input_acc': test_gen_input_acc,
            'reconstructed_acc': test_gen_reconstructed_acc,
            'generated_acc': test_gen_validator_acc,
            'recon_vs_input': test_gen_recon_vs_input,
            'gen_vs_input': test_gen_validator_vs_input
        }
        
        logger.info(f"测试集验证生成分类器结果:")
        logger.info(f"  输入数据准确率: {test_gen_input_acc:.4f}")
        logger.info(f"  重建数据准确率: {test_gen_reconstructed_acc:.4f}")
        logger.info(f"  生成验证准确率: {test_gen_validator_acc:.4f}")
        logger.info(f"  重建 vs 输入提升: {test_gen_recon_vs_input:+.4f}")
        logger.info(f"  生成验证 vs 输入提升: {test_gen_validator_vs_input:+.4f}")
    
    logger.info(f"测试结果: loss={test_loss:.6f}, recon_loss={test_recon:.6f}")
    logger.info(f"测试集分类性能:")
    logger.info(f"  输入数据准确率 (Common真+Have真+Need当前值): {test_input_acc:.4f}")
    logger.info(f"  重建数据准确率 (Common真+Have真+Need生成值): {test_reconstructed_acc:.4f}")
    logger.info(f"  准确率提升: {test_accuracy_improvement:+.4f}")
      # 记录测试结果到TensorBoard
    writer.add_scalar('Test/Loss', test_loss)
    writer.add_scalar('Test/Recon_Loss', test_recon)
    writer.add_scalar('Test/Input_Acc', test_input_acc)
    writer.add_scalar('Test/Reconstructed_Acc', test_reconstructed_acc)
    writer.add_scalar('Test/Accuracy_Improvement', test_accuracy_improvement)
    
    # 记录测试验证生成分类器指标到TensorBoard（如果有）
    if test_gen_val_metrics is not None:
        writer.add_scalar('Test_GenValidator/Input_Acc', test_gen_val_metrics['input_acc'])
        writer.add_scalar('Test_GenValidator/Reconstructed_Acc', test_gen_val_metrics['reconstructed_acc'])
        writer.add_scalar('Test_GenValidator/Generated_Acc', test_gen_val_metrics['generated_acc'])
        writer.add_scalar('Test_GenValidator/Recon_vs_Input', test_gen_val_metrics['recon_vs_input'])
        writer.add_scalar('Test_GenValidator/Gen_vs_Input', test_gen_val_metrics['gen_vs_input'])
        writer.add_scalar('Test_GenValidator/Loss', test_gen_val_metrics['loss'])
        writer.add_scalar('Test_GenValidator/Recon_Loss', test_gen_val_metrics['recon_loss'])
    
    writer.close()
    logger.info("训练完成！")

if __name__ == "__main__":
    main()
