# train_fixed.py - V13修复版训练脚本

"""
V13 修复版多模态时间序列模型训练脚本

修复问题：
1. 确保custom_collate_fn被正确使用
2. 移除所有emoji字符避免UnicodeEncodeError
3. 优化标签分布检查性能
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
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

# 导入V13重构的数据处理模块
from data import create_multimodal_dataset_from_config, load_config


def custom_collate_fn(batch):
    """
    自定义collate函数来处理不同长度的列表
    batch中每个元素格式: (tensor, label, indices_list, is_real_mask, source_dataset)
    """
    tensors = []
    labels = []
    indices_lists = []
    is_real_masks = []
    source_datasets = []
    
    for item in batch:
        tensor, label, indices_list, is_real_mask, source_dataset = item
        tensors.append(tensor)
        labels.append(label)
        indices_lists.append(indices_list)
        is_real_masks.append(is_real_mask)
        source_datasets.append(source_dataset)
    
    # 将tensors和labels堆叠
    batched_tensors = torch.stack(tensors)
    batched_labels = torch.stack(labels)
    batched_is_real_masks = torch.stack(is_real_masks)
    
    # indices_lists和source_datasets保持为列表
    return batched_tensors, batched_labels, indices_lists, batched_is_real_masks, source_datasets


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


def fast_check_label_distribution(dataset, sample_size=1000):
    """
    快速检查数据集标签分布（采样版本）
    """
    total_size = len(dataset)
    if total_size <= sample_size:
        # 如果数据集较小，检查全部
        indices = range(total_size)
    else:
        # 随机采样
        indices = np.random.choice(total_size, size=sample_size, replace=False)
    
    label_counter = collections.Counter()
    all_labels = set()
    
    for i in indices:
        item = dataset[i]
        label = item[1]
        if hasattr(label, 'item'):
            label = label.item()
        label_counter[label] += 1
        all_labels.add(label)
    
    print(f"标签分布（采样{len(indices)}个样本）:", dict(label_counter))
    print("所有标签:", sorted(list(all_labels)))
    return label_counter, all_labels


class SimpleAutoencoder(nn.Module):
    """简单的自编码器模型"""
    def __init__(self, input_channels=32, sequence_length=320, latent_dim=128):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_channels * sequence_length, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_channels * sequence_length)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 二分类
        )
    
    def forward(self, x):
        # x: [B, C, T]
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [B, C*T]
        
        # 编码
        latent = self.encoder(x_flat)
        
        # 解码
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(batch_size, self.input_channels, self.sequence_length)
        
        # 分类
        cls_output = self.classifier(latent)
        
        return reconstructed, cls_output


def main():
    parser = argparse.ArgumentParser(description='V13 多模态训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置日志
    logger = setup_logging()
    
    print(f"加载配置文件: {args.config}")
    logger.info(f"开始V13多模态训练，配置文件: {args.config}")
    logger.info(f"使用设备: {device}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    ckpt_dir = config.get('checkpoint_dir', 'Checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ====== 使用V13重构的多模态数据加载 ======
    logger.info("使用V13重构的多模态数据处理流程")
    logger.info(f"配置文件中的数据文件: {config.get('data_files', [])}")
    logger.info(f"数据目录: {config.get('data_dir', '')}")
    
    # 创建数据集
    dataset = create_multimodal_dataset_from_config(config, phase='encode')
    logger.info(f"数据集创建完成，滑动窗口数量: {len(dataset)}")
    
    # 检查数据集分布
    sample_sources = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        if len(sample) >= 5:
            source = sample[4]  # source_dataset
            sample_sources.append(source)
    
    source_distribution = collections.Counter(sample_sources)
    logger.info(f"样本中的数据集分布（前100个样本）: {dict(source_distribution)}")
    
    # 快速检查标签分布
    logger.info("检查数据集标签分布:")
    fast_check_label_distribution(dataset, sample_size=1000)
    
    # 创建数据分割
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"数据集划分:")
    logger.info(f"   - 总样本数: {dataset_size}")
    logger.info(f"   - 训练集: {len(train_dataset)} 样本")
    logger.info(f"   - 验证集: {len(val_dataset)} 样本") 
    logger.info(f"   - 测试集: {len(test_dataset)} 样本")
    
    # 创建数据加载器（使用自定义collate函数）
    batch_size = config.get('batch_size', 32)
    num_workers = 0  # 设置为0避免多进程问题
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=custom_collate_fn  # 关键：使用自定义collate函数
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=custom_collate_fn  # 关键：使用自定义collate函数
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=custom_collate_fn  # 关键：使用自定义collate函数
    )
    
    logger.info(f"数据加载器创建完成")
    
    # 测试DataLoader
    logger.info("测试DataLoader...")
    try:
        sample_batch = next(iter(train_loader))
        logger.info(f"DataLoader测试成功！")
        logger.info(f"批次格式: {len(sample_batch)} 个元素")
        logger.info(f"张量形状: {sample_batch[0].shape}")  # batched_tensors
        logger.info(f"标签形状: {sample_batch[1].shape}")  # batched_labels
        logger.info(f"indices_lists长度: {len(sample_batch[2])}")  # indices_lists
        logger.info(f"is_real_masks形状: {sample_batch[3].shape}")  # batched_is_real_masks
        logger.info(f"source_datasets长度: {len(sample_batch[4])}")  # source_datasets
        logger.info(f"source_datasets示例: {sample_batch[4][:5]}")  # 前5个
    except Exception as e:
        logger.error(f"DataLoader测试失败: {e}")
        raise e
      # 创建模型
    # 使用实际的特征列数量
    input_channels = 32  # 强制使用32维特征
    sequence_length = config.get('window_size', 320)
    
    logger.info(f"模型输入维度: channels={input_channels}, sequence_length={sequence_length}")
    
    model = SimpleAutoencoder(
        input_channels=input_channels,
        sequence_length=sequence_length,
        latent_dim=config.get('latent_dim', 128)
    ).to(device)
    
    logger.info(f"模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器和损失函数
    optimizer = Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    criterion_recon = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    
    # TensorBoard日志
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    epochs = config.get('num_epochs', 10)
    logger.info(f"开始训练，总轮数: {epochs}")
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"{'='*50}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_cls_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Training")):
            tensors, labels, indices_lists, is_real_masks, source_datasets = batch_data
            tensors = tensors.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            reconstructed, cls_output = model(tensors)
            
            # 计算损失
            recon_loss = criterion_recon(reconstructed, tensors)
            cls_loss = criterion_cls(cls_output, labels)
            total_loss = recon_loss + 0.1 * cls_loss  # 重建损失权重更高
            
            # 反向传播
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_cls_loss += cls_loss.item()
            
            _, predicted = torch.max(cls_output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {total_loss.item():.4f}, "
                           f"Recon: {recon_loss.item():.4f}, "
                           f"Cls: {cls_loss.item():.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_cls_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation"):
                tensors, labels, indices_lists, is_real_masks, source_datasets = batch_data
                tensors = tensors.to(device)
                labels = labels.to(device)
                
                reconstructed, cls_output = model(tensors)
                
                recon_loss = criterion_recon(reconstructed, tensors)
                cls_loss = criterion_cls(cls_output, labels)
                total_loss = recon_loss + 0.1 * cls_loss
                
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_cls_loss += cls_loss.item()
                
                _, predicted = torch.max(cls_output.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算平均值
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # 记录日志
        logger.info(f"训练 - 损失: {train_loss:.4f}, 重建: {train_recon_loss:.4f}, "
                   f"分类: {train_cls_loss:.4f}, 准确率: {train_acc:.2f}%")
        logger.info(f"验证 - 损失: {val_loss:.4f}, 重建: {val_recon_loss:.4f}, "
                   f"分类: {val_cls_loss:.4f}, 准确率: {val_acc:.2f}%")
        
        # TensorBoard记录
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Recon_Loss', train_recon_loss, epoch)
        writer.add_scalar('Train/Cls_Loss', train_cls_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Recon_Loss', val_recon_loss, epoch)
        writer.add_scalar('Val/Cls_Loss', val_cls_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存模型
        if epoch % 5 == 0 or epoch == epochs - 1:
            model_path = os.path.join(ckpt_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, model_path)
            logger.info(f"模型已保存: {model_path}")
    
    writer.close()
    logger.info("训练完成！")


if __name__ == '__main__':
    main()
