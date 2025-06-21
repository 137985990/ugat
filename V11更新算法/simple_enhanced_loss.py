# simple_enhanced_loss.py - 简化版增强损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleEnhancedLoss(nn.Module):
    """简化版增强损失函数 - 易于集成到现有代码"""
    
    def __init__(self, 
                 recon_weight: float = 1.0,
                 cls_weight: float = 1.0,
                 l1_weight: float = 0.1,
                 smooth_weight: float = 0.05):
        super().__init__()
        
        self.recon_weight = recon_weight
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.smooth_weight = smooth_weight
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss() 
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, logits, labels, real_mask=None):
        """
        计算简化版增强损失
        
        Args:
            predictions: [B, C, T] 模型预测输出
            targets: [B, C, T] 真实目标
            logits: [B, num_classes] 分类logits
            labels: [B] 分类标签
            real_mask: [B, C] 真实通道mask (可选)
        
        Returns:
            loss_dict: 包含各种损失的字典
        """
        
        # 1. 重建损失 (MSE + L1)
        if real_mask is not None:
            # 应用通道mask
            mask = real_mask.unsqueeze(-1).expand_as(predictions)
            masked_pred = predictions * mask
            masked_target = targets * mask
            
            mse_loss = F.mse_loss(masked_pred, masked_target, reduction='sum')
            mse_loss = mse_loss / (mask.sum() + 1e-8)
            
            l1_loss = F.l1_loss(masked_pred, masked_target, reduction='sum') 
            l1_loss = l1_loss / (mask.sum() + 1e-8)
        else:
            mse_loss = self.mse_loss(predictions, targets)
            l1_loss = self.l1_loss(predictions, targets)
        
        # 2. 分类损失
        cls_loss = self.ce_loss(logits, labels)
        
        # 3. 时序平滑损失
        pred_diff = torch.diff(predictions, dim=-1)  # [B, C, T-1]
        smooth_loss = (pred_diff ** 2).mean()
        
        # 4. 组合损失
        reconstruction_loss = mse_loss + self.l1_weight * l1_loss
        total_loss = (self.recon_weight * reconstruction_loss + 
                     self.cls_weight * cls_loss +
                     self.smooth_weight * smooth_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss,
            'classification_loss': cls_loss,
            'smooth_loss': smooth_loss
        }

class AdaptiveWeightLoss(SimpleEnhancedLoss):
    """带自适应权重的增强损失函数"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0
        self.total_epochs = 100
        
        # 保存初始权重
        self.init_recon_weight = self.recon_weight
        self.init_cls_weight = self.cls_weight
        
    def update_epoch(self, epoch, total_epochs=None):
        """更新训练epoch信息"""
        self.epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
            
        # 动态调整权重
        progress = epoch / self.total_epochs
        
        # 重建权重：从1.0逐渐降到0.8
        self.recon_weight = self.init_recon_weight * (1.0 - 0.2 * progress)
        
        # 分类权重：从初始值逐渐升到1.2倍
        self.cls_weight = self.init_cls_weight * (1.0 + 0.2 * progress)

def create_simple_enhanced_loss(config=None):
    """创建简化版增强损失函数"""
    if config is None:
        config = {}
    
    return SimpleEnhancedLoss(
        recon_weight=config.get('recon_weight', 1.0),
        cls_weight=config.get('cls_weight', 1.0),
        l1_weight=config.get('l1_weight', 0.1),
        smooth_weight=config.get('smooth_weight', 0.05)
    )

def create_adaptive_loss(config=None):
    """创建自适应权重损失函数"""
    if config is None:
        config = {}
    
    return AdaptiveWeightLoss(
        recon_weight=config.get('recon_weight', 1.0),
        cls_weight=config.get('cls_weight', 1.0),
        l1_weight=config.get('l1_weight', 0.1),
        smooth_weight=config.get('smooth_weight', 0.05)
    )

# 兼容接口：可以直接替换MSELoss
class CompatibleEnhancedLoss(nn.Module):
    """兼容现有代码的增强损失函数"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        """兼容MSELoss的接口"""
        # 组合MSE和L1损失
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target) 
        return mse + 0.1 * l1  # 添加少量L1正则化

if __name__ == "__main__":
    # 测试简化版损失函数
    batch_size, num_channels, seq_len = 4, 12, 100
    num_classes = 3
    
    predictions = torch.randn(batch_size, num_channels, seq_len)
    targets = torch.randn(batch_size, num_channels, seq_len)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    real_mask = torch.ones(batch_size, num_channels)
    
    # 测试简化版损失
    simple_loss = create_simple_enhanced_loss()
    loss_dict = simple_loss(predictions, targets, logits, labels, real_mask)
    
    print("简化版增强损失测试结果:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    # 测试自适应权重版本
    adaptive_loss = create_adaptive_loss()
    print(f"\n权重调整前: recon_weight={adaptive_loss.recon_weight:.3f}, cls_weight={adaptive_loss.cls_weight:.3f}")
    
    adaptive_loss.update_epoch(50, 100)  # 训练到一半
    print(f"权重调整后: recon_weight={adaptive_loss.recon_weight:.3f}, cls_weight={adaptive_loss.cls_weight:.3f}")
    
    # 测试兼容版本
    compatible_loss = CompatibleEnhancedLoss()
    simple_loss_value = compatible_loss(predictions.mean(dim=1), targets.mean(dim=1))
    print(f"\n兼容版本损失: {simple_loss_value.item():.6f}")
