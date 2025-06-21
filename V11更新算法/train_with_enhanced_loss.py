# train_with_enhanced_loss.py - 集成增强损失函数的训练示例

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from datetime import datetime
import logging
from tqdm import tqdm
import collections

# 导入现有模块
from data import create_datasets, get_data_loaders
from model import TGATUNet
from model_optimized import OptimizedTGATUNet
from simple_enhanced_loss import create_simple_enhanced_loss, create_adaptive_loss, CompatibleEnhancedLoss
from memory_optimizer import MemoryOptimizer

def setup_enhanced_training(config):
    """设置增强版训练配置"""
    
    # 损失函数配置
    loss_config = config.get('loss_config', {})
    loss_type = loss_config.get('type', 'basic')
    
    if loss_type == 'enhanced_simple':
        # 简化版增强损失
        enhanced_loss = create_simple_enhanced_loss({
            'recon_weight': loss_config.get('recon_weight', 1.0),
            'cls_weight': loss_config.get('cls_weight', 1.0),
            'l1_weight': loss_config.get('l1_weight', 0.1),
            'smooth_weight': loss_config.get('smooth_weight', 0.05)
        })
        return enhanced_loss, 'enhanced_simple'
        
    elif loss_type == 'enhanced_adaptive':
        # 自适应权重版本
        enhanced_loss = create_adaptive_loss({
            'recon_weight': loss_config.get('recon_weight', 1.0),
            'cls_weight': loss_config.get('cls_weight', 1.0),
            'l1_weight': loss_config.get('l1_weight', 0.1),
            'smooth_weight': loss_config.get('smooth_weight', 0.05)
        })
        return enhanced_loss, 'enhanced_adaptive'
        
    elif loss_type == 'compatible':
        # 兼容版本（最小改动）
        return CompatibleEnhancedLoss(), 'compatible'
        
    else:
        # 原始MSE损失
        return nn.MSELoss(), 'basic'

def train_with_enhanced_loss(
    model, dataloader, optimizer, criterion, device, mask_indices, 
    loss_type='basic', epoch=None, total_epochs=None
):
    """使用增强损失函数的训练循环"""
    
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    
    # 动态权重更新
    if loss_type == 'enhanced_adaptive' and epoch is not None and total_epochs is not None:
        criterion.update_epoch(epoch, total_epochs)
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 用于记录详细损失信息
    loss_details = collections.defaultdict(float)
    
    for batch, labels, mask_idx, is_real_mask in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        labels = labels.to(device)
        is_real_mask = is_real_mask.to(device)
        
        # Mask通道
        from train import mask_channel
        masked, mask_idx = mask_channel(batch, mask_indices)
        
        batch_size, C, T = batch.size()
        optimizer.zero_grad()
        
        if loss_type in ['enhanced_simple', 'enhanced_adaptive']:
            # 使用增强损失函数
            batch_predictions = []
            batch_logits = []
            
            # 收集所有样本的预测
            for i in range(batch_size):
                window = masked[i].t()  # [T, C]
                out, logits = model(window)  # out: [C, T], logits: [num_classes]
                batch_predictions.append(out.unsqueeze(0))  # [1, C, T]
                batch_logits.append(logits.unsqueeze(0))   # [1, num_classes]
            
            # 拼接为batch
            predictions = torch.cat(batch_predictions, dim=0)  # [B, C, T]
            logits_batch = torch.cat(batch_logits, dim=0)      # [B, num_classes]
            
            # 计算增强损失
            loss_dict = criterion(predictions, batch, logits_batch, labels, is_real_mask)
            loss = loss_dict['total_loss']
            
            # 记录详细损失
            for key, value in loss_dict.items():
                loss_details[key] += value.item() * batch_size
            
            # 计算准确率
            pred_classes = logits_batch.argmax(dim=-1)
            total_correct += (pred_classes == labels).sum().item()
            total_samples += batch_size
            
        elif loss_type == 'compatible':
            # 兼容版本：逐样本计算
            loss = 0.0
            for i in range(batch_size):
                window = masked[i].t()
                out, logits = model(window)
                
                # 重建损失（兼容版）
                real_channels = is_real_mask[i] if is_real_mask.dim() == 2 else is_real_mask
                recon_loss_i = 0.0
                real_count = 0
                
                for c in range(C):
                    if real_channels[c]:
                        target = batch[i, c, :]
                        pred = out[c, :]
                        recon_loss_i += criterion(pred, target)  # 使用增强的criterion
                        real_count += 1
                
                if real_count > 0:
                    recon_loss_i /= real_count
                
                # 分类损失
                cls_loss_i = ce_loss(logits.unsqueeze(0), labels[i].unsqueeze(0))
                loss += recon_loss_i + cls_loss_i
                
                # 统计
                total_recon_loss += recon_loss_i.item()
                total_cls_loss += cls_loss_i.item()
                
                pred_class = logits.argmax(-1).item()
                if pred_class == labels[i].item():
                    total_correct += 1
                total_samples += 1
            
            loss = loss / batch_size
            
        else:
            # 原始训练方式
            loss = 0.0
            for i in range(batch_size):
                window = masked[i].t()
                out, logits = model(window)
                
                real_channels = is_real_mask[i] if is_real_mask.dim() == 2 else is_real_mask
                recon_loss_i = 0.0
                real_count = 0
                
                for c in range(C):
                    if real_channels[c]:
                        target = batch[i, c, :]
                        pred = out[c, :]
                        recon_loss_i += criterion(pred, target)
                        real_count += 1
                
                if real_count > 0:
                    recon_loss_i /= real_count
                
                cls_loss_i = ce_loss(logits.unsqueeze(0), labels[i].unsqueeze(0))
                loss += recon_loss_i + cls_loss_i
                
                total_recon_loss += recon_loss_i.item()
                total_cls_loss += cls_loss_i.item()
                
                pred_class = logits.argmax(-1).item()
                if pred_class == labels[i].item():
                    total_correct += 1
                total_samples += 1
            
            loss = loss / batch_size
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
    
    n = len(dataloader.dataset)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 返回结果
    result = {
        'total_loss': total_loss / n,
        'accuracy': acc
    }
    
    if loss_type in ['enhanced_simple', 'enhanced_adaptive']:
        # 添加详细损失信息
        for key, value in loss_details.items():
            result[key] = value / n
    else:
        result['reconstruction_loss'] = total_recon_loss / n
        result['classification_loss'] = total_cls_loss / n
    
    return result

def compare_training_modes():
    """对比不同训练模式的效果"""
    
    print("=" * 60)
    print("增强损失函数训练对比")
    print("=" * 60)
    
    # 加载配置
    config_path = 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 测试不同损失函数配置
    loss_configs = [
        {'type': 'basic', 'name': '原始MSE损失'},
        {'type': 'compatible', 'name': '兼容增强损失（MSE+L1）'},
        {
            'type': 'enhanced_simple', 
            'name': '简化增强损失',
            'recon_weight': 1.0,
            'cls_weight': 1.0,
            'l1_weight': 0.1,
            'smooth_weight': 0.05
        },
        {
            'type': 'enhanced_adaptive',
            'name': '自适应权重损失',
            'recon_weight': 1.0,
            'cls_weight': 1.0,
            'l1_weight': 0.1,
            'smooth_weight': 0.05
        }
    ]
    
    for loss_config in loss_configs:
        print(f"\n测试配置: {loss_config['name']}")
        
        # 设置损失函数
        config['loss_config'] = loss_config
        criterion, loss_type = setup_enhanced_training(config)
        
        print(f"  损失函数类型: {loss_type}")
        print(f"  损失函数: {type(criterion).__name__}")
        
        if hasattr(criterion, 'recon_weight'):
            print(f"  重建权重: {criterion.recon_weight}")
            print(f"  分类权重: {criterion.cls_weight}")
            if hasattr(criterion, 'l1_weight'):
                print(f"  L1权重: {criterion.l1_weight}")
            if hasattr(criterion, 'smooth_weight'):
                print(f"  平滑权重: {criterion.smooth_weight}")

def create_enhanced_config():
    """创建增强版配置文件"""
    
    enhanced_config = {
        'loss_config': {
            'type': 'enhanced_adaptive',  # basic, compatible, enhanced_simple, enhanced_adaptive
            'recon_weight': 1.0,
            'cls_weight': 1.0,
            'l1_weight': 0.1,
            'smooth_weight': 0.05
        },
        'training': {
            'epochs': 200,
            'lr': 0.001,
            'batch_size': 16,
            'early_stopping_patience': 15
        },
        'logging': {
            'log_dir': './Logs',
            'save_interval': 10,
            'detailed_loss_logging': True
        }
    }
    
    with open('config_enhanced_loss.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(enhanced_config, f, default_flow_style=False, allow_unicode=True)
    
    print("增强版配置文件已保存到: config_enhanced_loss.yaml")

if __name__ == "__main__":
    print("增强损失函数集成示例")
    print("=" * 40)
    
    # 对比不同训练模式
    compare_training_modes()
    
    # 创建增强版配置
    create_enhanced_config()
    
    print("\n使用说明:")
    print("1. 在train.py中导入: from train_with_enhanced_loss import setup_enhanced_training, train_with_enhanced_loss")
    print("2. 替换损失函数设置: criterion, loss_type = setup_enhanced_training(config)")
    print("3. 替换训练循环: train_result = train_with_enhanced_loss(...)")
    print("4. 根据config_enhanced_loss.yaml调整配置参数")
