# train_refactored.py - V13多模态时间序列模型训练脚本（完全重构版本）

"""
V13 多模态时间序列模型训练脚本

核心特性：
- 支持多数据集（FM/OD/MEFAR）混合训练
- 动态遮掩have通道，按source_dataset动态处理损失
- 使用重构的数据处理流程，完全移除旧逻辑
- 支持混合精度训练和优化的数据加载
"""

import os
import sys
import yaml
import argparse
import logging
import collections
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

# 添加混合精度训练支持
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Warning: Automatic Mixed Precision not available.")

# 导入V13重构的数据处理模块
from data import create_multimodal_dataset_from_config, load_config, check_label_distribution

# 导入自定义模块
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
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def calculate_multimodal_loss(predictions, targets, have_indices_batch, source_datasets, config):
    """
    计算多模态损失函数
    
    Args:
        predictions: 模型预测结果 [B, C, T]
        targets: 目标值 [B, C, T]
        have_indices_batch: 每个样本的have通道索引列表
        source_datasets: 每个样本的源数据集列表
        config: 配置字典
    
    Returns:
        total_loss: 总损失
        loss_details: 损失详情字典
    """
    device = predictions.device
    batch_size = predictions.size(0)
    
    total_loss = 0.0
    loss_details = {
        'total': 0.0,
        'by_dataset': {},
        'by_channel_type': {'have': 0.0, 'need': 0.0}
    }
    
    mse_loss = nn.MSELoss(reduction='none')
    
    for i in range(batch_size):
        pred_i = predictions[i]  # [C, T]
        target_i = targets[i]    # [C, T]
        have_indices = have_indices_batch[i] if isinstance(have_indices_batch[i], list) else have_indices_batch[i].tolist()
        source_dataset = source_datasets[i]
        
        # 计算have通道损失（重构损失）
        if len(have_indices) > 0:
            have_pred = pred_i[have_indices]      # [H, T]
            have_target = target_i[have_indices]  # [H, T]
            have_loss = mse_loss(have_pred, have_target).mean()
            
            total_loss += have_loss
            loss_details['by_channel_type']['have'] += have_loss.item()
            
            if source_dataset not in loss_details['by_dataset']:
                loss_details['by_dataset'][source_dataset] = 0.0
            loss_details['by_dataset'][source_dataset] += have_loss.item()
    
    # 平均化损失
    if batch_size > 0:
        total_loss /= batch_size
        loss_details['total'] = total_loss.item()
        loss_details['by_channel_type']['have'] /= batch_size
        
        for dataset in loss_details['by_dataset']:
            loss_details['by_dataset'][dataset] /= batch_size
    
    return total_loss, loss_details


def train_epoch(model, train_loader, optimizer, scaler, device, config, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_loss_details = {
        'by_dataset': collections.defaultdict(list),
        'by_channel_type': {'have': [], 'need': []}
    }
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # 解析批次数据
        if len(batch_data) == 5:
            batch_x, batch_y, have_indices_batch, is_real_mask_batch, source_datasets = batch_data
        else:
            print(f"Warning: Unexpected batch format with {len(batch_data)} elements")
            continue
            
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        if AMP_AVAILABLE and scaler is not None:
            with autocast():
                # 前向传播
                outputs = model(batch_x)
                
                # 计算损失
                loss, loss_details = calculate_multimodal_loss(
                    outputs, batch_x, have_indices_batch, source_datasets, config
                )
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播
            outputs = model(batch_x)
            
            # 计算损失
            loss, loss_details = calculate_multimodal_loss(
                outputs, batch_x, have_indices_batch, source_datasets, config
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if config.get('gradient_clip_norm', 0) > 0:
                clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            
            optimizer.step()
        
        # 累积损失统计
        total_loss += loss.item()
        
        # 记录详细损失
        for dataset, loss_val in loss_details['by_dataset'].items():
            all_loss_details['by_dataset'][dataset].append(loss_val)
        
        all_loss_details['by_channel_type']['have'].append(loss_details['by_channel_type']['have'])
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg': f'{total_loss/(batch_idx+1):.6f}'
        })
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    
    # 计算各数据集的平均损失
    dataset_avg_losses = {}
    for dataset, losses in all_loss_details['by_dataset'].items():
        dataset_avg_losses[dataset] = np.mean(losses) if losses else 0.0
    
    return avg_loss, dataset_avg_losses


def validate_epoch(model, val_loader, device, config, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    all_loss_details = {
        'by_dataset': collections.defaultdict(list),
        'by_channel_type': {'have': [], 'need': []}
    }
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc=f'Validation {epoch}'):
            # 解析批次数据
            if len(batch_data) == 5:
                batch_x, batch_y, have_indices_batch, is_real_mask_batch, source_datasets = batch_data
            else:
                continue
                
            batch_x = batch_x.to(device)
            
            # 前向传播
            outputs = model(batch_x)
            
            # 计算损失
            loss, loss_details = calculate_multimodal_loss(
                outputs, batch_x, have_indices_batch, source_datasets, config
            )
            
            total_loss += loss.item()
            
            # 记录详细损失
            for dataset, loss_val in loss_details['by_dataset'].items():
                all_loss_details['by_dataset'][dataset].append(loss_val)
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    
    # 计算各数据集的平均损失
    dataset_avg_losses = {}
    for dataset, losses in all_loss_details['by_dataset'].items():
        dataset_avg_losses[dataset] = np.mean(losses) if losses else 0.0
    
    return avg_loss, dataset_avg_losses


def main():
    """主训练函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='V13 Multimodal Time Series Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'], help='Running mode')
    args = parser.parse_args()
    
    # 加载配置
    print(f"🔧 加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 设置日志
    logger = setup_logging(config.get('log_dir', 'logs'))
    logger.info(f"开始V13多模态训练，配置文件: {args.config}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    ckpt_dir = config.get('checkpoint_dir', 'Checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ====== 使用V13重构的多模态数据加载 ======
    logger.info("🚀 使用V13重构的多模态数据处理流程")
    logger.info(f"📋 配置文件中的数据文件: {config.get('data_files', [])}")
    logger.info(f"📋 数据目录: {config.get('data_dir', '')}")
    
    # 创建数据集
    dataset = create_multimodal_dataset_from_config(config, phase='encode')
    logger.info(f"✅ 数据集创建完成，滑动窗口数量: {len(dataset)}")
    
    # 检查数据集分布
    sample_sources = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        if len(sample) >= 5:
            source = sample[4]  # source_dataset
            sample_sources.append(source)
    
    source_distribution = collections.Counter(sample_sources)
    logger.info(f"🔍 样本中的数据集分布（前100个样本）: {dict(source_distribution)}")
    
    # 检查标签分布
    logger.info("📊 检查数据集标签分布:")
    check_label_distribution(dataset)
    
    # 创建数据分割
    dataset_size = len(dataset)
    train_size = int(config.get('train_ratio', 0.7) * dataset_size)
    val_size = int(config.get('val_ratio', 0.15) * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"📊 数据集划分:")
    logger.info(f"   - 总样本数: {dataset_size}")
    logger.info(f"   - 训练集: {len(train_dataset)} 样本")
    logger.info(f"   - 验证集: {len(val_dataset)} 样本") 
    logger.info(f"   - 测试集: {len(test_dataset)} 样本")
    
    # 创建数据加载器
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"✅ 数据加载器创建完成")
    
    # 创建模型
    # 注意：这里需要根据实际的模型结构进行调整
    # 暂时使用一个简单的自编码器作为示例
    from model import create_model  # 假设有这个函数
    
    # 获取特征维度
    sample_batch = next(iter(train_loader))
    input_channels = sample_batch[0].size(1)  # [B, C, T]
    sequence_length = sample_batch[0].size(2)
    
    logger.info(f"📊 模型输入维度: channels={input_channels}, sequence_length={sequence_length}")
    
    model = create_model(config, input_channels, sequence_length)
    model = model.to(device)
    
    logger.info(f"🤖 模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器
    optimizer = Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get('lr_factor', 0.5),
        patience=config.get('lr_patience', 10),
        verbose=True
    )
    
    # 混合精度训练设置
    scaler = GradScaler() if AMP_AVAILABLE and config.get('use_amp', False) else None
    
    # TensorBoard设置
    writer = SummaryWriter(config.get('tensorboard_dir', 'runs'))
    
    # 训练参数
    epochs = config.get('epochs', 100)
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 20)
    patience_counter = 0
    
    logger.info(f"🚀 开始训练，总轮数: {epochs}")
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_loss, train_dataset_losses = train_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch
        )
        
        # 验证
        val_loss, val_dataset_losses = validate_epoch(
            model, val_loader, device, config, epoch
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录各数据集的损失
        for dataset_name, loss_val in train_dataset_losses.items():
            writer.add_scalar(f'Loss/Train_{dataset_name}', loss_val, epoch)
        
        for dataset_name, loss_val in val_dataset_losses.items():
            writer.add_scalar(f'Loss/Val_{dataset_name}', loss_val, epoch)
        
        # 日志输出
        logger.info(f"训练损失: {train_loss:.6f}")
        logger.info(f"验证损失: {val_loss:.6f}")
        logger.info(f"各数据集训练损失: {train_dataset_losses}")
        logger.info(f"各数据集验证损失: {val_dataset_losses}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, best_model_path)
            
            logger.info(f"💾 保存最佳模型到: {best_model_path}")
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= patience:
            logger.info(f"🛑 早停触发，验证损失连续{patience}轮未改善")
            break
        
        # 定期保存检查点
        if epoch % config.get('save_interval', 10) == 0:
            checkpoint_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"💾 保存检查点到: {checkpoint_path}")
    
    # 训练完成
    logger.info(f"🎉 训练完成！最佳验证损失: {best_val_loss:.6f}")
    
    # 关闭TensorBoard
    writer.close()
    
    logger.info("✅ V13多模态训练流程完成")


if __name__ == "__main__":
    main()
