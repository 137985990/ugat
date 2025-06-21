# enhanced_loss_with_common.py - 包含common_modalities的增强损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

class MultiModalLoss(nn.Module):
    """多模态损失函数 - 支持common_modalities和have_modalities的分别计算"""
    
    def __init__(self, 
                 common_modalities: List[str],
                 all_modalities: List[str],
                 recon_weight: float = 1.0,
                 cls_weight: float = 1.0,
                 common_weight: float = 1.0,
                 have_weight: float = 1.0,
                 l1_weight: float = 0.1):
        super().__init__()
        
        # 模态配置
        self.common_modalities = common_modalities
        self.all_modalities = all_modalities
        
        # 创建模态索引映射
        self.common_indices = [all_modalities.index(mod) for mod in common_modalities if mod in all_modalities]
        self.have_indices = [i for i in range(len(all_modalities)) if i not in self.common_indices]
        
        # 损失权重
        self.recon_weight = recon_weight
        self.cls_weight = cls_weight
        self.common_weight = common_weight  # common_modalities权重
        self.have_weight = have_weight      # have_modalities权重
        self.l1_weight = l1_weight
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        print(f"MultiModalLoss初始化:")
        print(f"  common_modalities: {common_modalities}")
        print(f"  common_indices: {self.common_indices}")
        print(f"  have_indices: {self.have_indices}")
        print(f"  权重配置: common_weight={common_weight}, have_weight={have_weight}")
    
    def forward(self, 
                predictions: torch.Tensor,    # [B, C, T] 模型输出
                targets: torch.Tensor,       # [B, C, T] 目标数据
                logits: torch.Tensor,        # [B, num_classes] 分类logits
                labels: torch.Tensor,        # [B] 分类标签
                have_mask: torch.Tensor      # [B, C] have模态mask
                ) -> Dict[str, torch.Tensor]:
        """
        计算多模态损失
        
        Args:
            predictions: 模型预测输出
            targets: 真实目标值
            logits: 分类logits
            labels: 分类标签
            have_mask: have模态的mask（原is_real_mask）
        """
        
        batch_size, num_channels, seq_len = predictions.shape
        device = predictions.device
        
        # 创建common_modalities的mask（始终为True）
        common_mask = torch.zeros(batch_size, num_channels, dtype=torch.bool, device=device)
        for idx in self.common_indices:
            if idx < num_channels:
                common_mask[:, idx] = True
        
        # 组合mask：common + have
        combined_mask = common_mask | have_mask.bool()
        
        loss_dict = {}
        
        # ================== 1. Common Modalities重建损失 ==================
        if len(self.common_indices) > 0:
            common_loss = self._compute_modality_loss(
                predictions, targets, common_mask, "common"
            )
            loss_dict.update(common_loss)
        else:
            loss_dict['common_mse_loss'] = torch.tensor(0.0, device=device)
            loss_dict['common_l1_loss'] = torch.tensor(0.0, device=device)
            loss_dict['common_total_loss'] = torch.tensor(0.0, device=device)
        
        # ================== 2. Have Modalities重建损失 ==================
        have_loss = self._compute_modality_loss(
            predictions, targets, have_mask.bool(), "have"
        )
        loss_dict.update(have_loss)
        
        # ================== 3. 分类损失 ==================
        cls_loss = self.ce_loss(logits, labels)
        loss_dict['classification_loss'] = cls_loss
        
        # ================== 4. 总重建损失 ==================
        total_recon_loss = (
            self.common_weight * loss_dict['common_total_loss'] +
            self.have_weight * loss_dict['have_total_loss']
        )
        loss_dict['reconstruction_loss'] = total_recon_loss
        
        # ================== 5. 总损失 ==================
        total_loss = (
            self.recon_weight * total_recon_loss +
            self.cls_weight * cls_loss
        )
        loss_dict['total_loss'] = total_loss
        
        # ================== 6. 统计信息 ==================
        loss_dict['common_channels_used'] = torch.tensor(len(self.common_indices), device=device)
        loss_dict['have_channels_used'] = have_mask.sum()
        loss_dict['total_channels_used'] = combined_mask.sum()
        
        return loss_dict
    
    def _compute_modality_loss(self, 
                              predictions: torch.Tensor,
                              targets: torch.Tensor,
                              mask: torch.Tensor,
                              modality_name: str) -> Dict[str, torch.Tensor]:
        """计算特定模态的损失"""
        
        device = predictions.device
        
        if mask.sum() == 0:
            # 没有该模态的通道
            return {
                f'{modality_name}_mse_loss': torch.tensor(0.0, device=device),
                f'{modality_name}_l1_loss': torch.tensor(0.0, device=device),
                f'{modality_name}_total_loss': torch.tensor(0.0, device=device)
            }
        
        # 扩展mask维度以匹配数据
        mask_expanded = mask.unsqueeze(-1).expand_as(predictions)  # [B, C, T]
        
        # 应用mask
        masked_pred = predictions * mask_expanded
        masked_target = targets * mask_expanded
        
        # 计算损失（只在有效位置）
        mse_losses = F.mse_loss(masked_pred, masked_target, reduction='none')  # [B, C, T]
        l1_losses = F.l1_loss(masked_pred, masked_target, reduction='none')    # [B, C, T]
        
        # 只在mask为True的位置计算平均
        mse_loss = (mse_losses * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        l1_loss = (l1_losses * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        
        # 组合损失
        total_loss = mse_loss + self.l1_weight * l1_loss
        
        return {
            f'{modality_name}_mse_loss': mse_loss,
            f'{modality_name}_l1_loss': l1_loss,
            f'{modality_name}_total_loss': total_loss
        }

class AdaptiveMultiModalLoss(MultiModalLoss):
    """自适应权重的多模态损失函数"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0
        self.total_epochs = 100
        
        # 保存初始权重
        self.init_common_weight = self.common_weight
        self.init_have_weight = self.have_weight
        self.init_cls_weight = self.cls_weight
    
    def update_epoch(self, epoch: int, total_epochs: int = None):
        """动态调整权重"""
        self.epoch = epoch
        if total_epochs is not None:
            self.total_epochs = total_epochs
        
        progress = epoch / self.total_epochs
        
        # Common modalities权重：始终保持较高（因为是真实数据）
        self.common_weight = self.init_common_weight * (1.0 + 0.1 * progress)
        
        # Have modalities权重：逐渐提高（学习补全质量）
        self.have_weight = self.init_have_weight * (0.8 + 0.4 * progress)
        
        # 分类权重：后期提高
        self.cls_weight = self.init_cls_weight * (0.7 + 0.6 * progress)

def create_multimodal_loss(config: Dict) -> MultiModalLoss:
    """创建多模态损失函数"""
    
    common_modalities = config.get('common_modalities', [])
    
    # 构建all_modalities列表
    all_modalities = common_modalities.copy()
    dataset_modalities = config.get('dataset_modalities', {})
    
    # 添加所有have和need模态
    for dataset_name, modalities in dataset_modalities.items():
        have_mods = modalities.get('have', [])
        need_mods = modalities.get('need', [])
        for mod in have_mods + need_mods:
            if mod not in all_modalities:
                all_modalities.append(mod)
    
    loss_config = config.get('loss_config', {})
    
    if loss_config.get('adaptive', False):
        return AdaptiveMultiModalLoss(
            common_modalities=common_modalities,
            all_modalities=all_modalities,
            recon_weight=loss_config.get('recon_weight', 1.0),
            cls_weight=loss_config.get('cls_weight', 1.0),
            common_weight=loss_config.get('common_weight', 1.0),
            have_weight=loss_config.get('have_weight', 1.0),
            l1_weight=loss_config.get('l1_weight', 0.1)
        )
    else:
        return MultiModalLoss(
            common_modalities=common_modalities,
            all_modalities=all_modalities,
            recon_weight=loss_config.get('recon_weight', 1.0),
            cls_weight=loss_config.get('cls_weight', 1.0),
            common_weight=loss_config.get('common_weight', 1.0),
            have_weight=loss_config.get('have_weight', 1.0),
            l1_weight=loss_config.get('l1_weight', 0.1)
        )

if __name__ == "__main__":
    # 测试多模态损失函数
    
    # 模拟配置
    config = {
        'common_modalities': ['acc_x', 'acc_y', 'acc_z', 'ppg', 'gsr', 'hr', 'skt'],
        'dataset_modalities': {
            'FM': {
                'have': ['alpha_tp9', 'beta_tp9', 'ecg'],
                'need': ['space_distance', 'pose_pca']
            }
        },
        'loss_config': {
            'recon_weight': 1.0,
            'cls_weight': 1.0,
            'common_weight': 1.2,  # common模态权重稍高
            'have_weight': 1.0,
            'l1_weight': 0.1,
            'adaptive': True
        }
    }
    
    # 创建损失函数
    loss_fn = create_multimodal_loss(config)
    
    # 模拟数据
    batch_size, num_channels, seq_len = 4, 12, 100
    num_classes = 3
    
    predictions = torch.randn(batch_size, num_channels, seq_len)
    targets = torch.randn(batch_size, num_channels, seq_len)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 模拟have_mask（FM数据集的have模态）
    have_mask = torch.zeros(batch_size, num_channels, dtype=torch.bool)
    have_mask[:, 7:10] = True  # 假设7-9通道是have模态
    
    # 计算损失
    loss_dict = loss_fn(predictions, targets, logits, labels, have_mask)
    
    print("\n多模态损失测试结果:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n权重信息:")
    print(f"  common_weight: {loss_fn.common_weight:.3f}")
    print(f"  have_weight: {loss_fn.have_weight:.3f}")
    print(f"  cls_weight: {loss_fn.cls_weight:.3f}")
    
    # 测试自适应权重
    if isinstance(loss_fn, AdaptiveMultiModalLoss):
        print(f"\n权重变化测试:")
        for epoch in [0, 50, 100]:
            loss_fn.update_epoch(epoch, 100)
            print(f"  Epoch {epoch}: common={loss_fn.common_weight:.3f}, have={loss_fn.have_weight:.3f}, cls={loss_fn.cls_weight:.3f}")
