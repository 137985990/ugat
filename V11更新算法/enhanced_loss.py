# enhanced_loss.py - 增强版损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class MultiTaskLoss(nn.Module):
    """多任务复合损失函数"""
    
    def __init__(self, 
                 recon_weight: float = 1.0,
                 cls_weight: float = 1.0, 
                 consistency_weight: float = 0.1,
                 adversarial_weight: float = 0.01,
                 temporal_weight: float = 0.1,
                 spectral_weight: float = 0.05):
        super().__init__()
        
        # 损失权重
        self.recon_weight = recon_weight
        self.cls_weight = cls_weight
        self.consistency_weight = consistency_weight
        self.adversarial_weight = adversarial_weight
        self.temporal_weight = temporal_weight
        self.spectral_weight = spectral_weight
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        
        # 权重调度器
        self.epoch = 0
        
    def forward(self, 
                predictions: torch.Tensor,      # [B, C, T] 模型输出
                targets: torch.Tensor,         # [B, C, T] 目标数据
                logits: torch.Tensor,          # [B, num_classes] 分类logits
                labels: torch.Tensor,          # [B] 分类标签
                real_mask: torch.Tensor,       # [B, C] 真实通道mask
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Returns:
            loss_dict: 包含各种损失的字典
        """
        batch_size, num_channels, seq_len = predictions.shape
        loss_dict = {}
        
        # ================== 1. 重建损失 ==================
        recon_losses = self._compute_reconstruction_loss(
            predictions, targets, real_mask
        )
        loss_dict.update(recon_losses)
        
        # ================== 2. 分类损失 ==================
        cls_loss = self.ce_loss(logits, labels)
        loss_dict['classification_loss'] = cls_loss
        
        # ================== 3. 一致性损失 ==================
        if self.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(
                predictions, targets, real_mask
            )
            loss_dict['consistency_loss'] = consistency_loss
        
        # ================== 4. 时序连续性损失 ==================
        if self.temporal_weight > 0:
            temporal_loss = self._compute_temporal_loss(predictions, real_mask)
            loss_dict['temporal_loss'] = temporal_loss
        
        # ================== 5. 频域损失 ==================
        if self.spectral_weight > 0:
            spectral_loss = self._compute_spectral_loss(
                predictions, targets, real_mask
            )
            loss_dict['spectral_loss'] = spectral_loss
        
        # ================== 6. 总损失计算 ==================
        total_loss = (
            self.recon_weight * loss_dict['reconstruction_loss'] +
            self.cls_weight * loss_dict['classification_loss']
        )
        
        if 'consistency_loss' in loss_dict:
            total_loss += self.consistency_weight * loss_dict['consistency_loss']
        
        if 'temporal_loss' in loss_dict:
            total_loss += self.temporal_weight * loss_dict['temporal_loss']
            
        if 'spectral_loss' in loss_dict:
            total_loss += self.spectral_weight * loss_dict['spectral_loss']
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _compute_reconstruction_loss(self, 
                                   predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   real_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算重建损失（支持多种度量）"""
        
        # 扩展real_mask维度以匹配数据
        if real_mask.dim() == 2:
            real_mask = real_mask.unsqueeze(-1)  # [B, C, 1]
        
        # MSE损失
        mse_losses = self.mse_loss(predictions, targets)  # [B, C, T]
        mse_losses = mse_losses * real_mask  # 只计算真实通道
        mse_loss = mse_losses.sum() / (real_mask.sum() * predictions.size(-1))
        
        # L1损失（更鲁棒）
        l1_losses = self.l1_loss(predictions, targets)  # [B, C, T]
        l1_losses = l1_losses * real_mask
        l1_loss = l1_losses.sum() / (real_mask.sum() * predictions.size(-1))
        
        # Huber损失（结合MSE和L1优点）
        huber_losses = self.huber_loss(predictions, targets)  # [B, C, T]
        huber_losses = huber_losses * real_mask
        huber_loss = huber_losses.sum() / (real_mask.sum() * predictions.size(-1))
        
        # 组合重建损失
        reconstruction_loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * huber_loss
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss, 
            'huber_loss': huber_loss
        }
    
    def _compute_consistency_loss(self,
                                predictions: torch.Tensor,
                                targets: torch.Tensor, 
                                real_mask: torch.Tensor) -> torch.Tensor:
        """计算一致性损失（不同时间窗口的一致性）"""
        
        # 时间窗口一致性：相邻时间步的变化应该平滑
        pred_diff = torch.diff(predictions, dim=-1)  # [B, C, T-1]
        target_diff = torch.diff(targets, dim=-1)    # [B, C, T-1]
        
        if real_mask.dim() == 2:
            mask_diff = real_mask.unsqueeze(-1).expand(-1, -1, pred_diff.size(-1))
        else:
            mask_diff = real_mask[..., :-1]
        
        consistency_loss = F.mse_loss(pred_diff * mask_diff, 
                                    target_diff * mask_diff,
                                    reduction='sum')
        consistency_loss = consistency_loss / (mask_diff.sum() + 1e-8)
        
        return consistency_loss
    
    def _compute_temporal_loss(self,
                             predictions: torch.Tensor,
                             real_mask: torch.Tensor) -> torch.Tensor:
        """计算时序平滑性损失"""
        
        # 二阶差分惩罚（减少不自然的突变）
        second_diff = torch.diff(predictions, n=2, dim=-1)  # [B, C, T-2]
        
        if real_mask.dim() == 2:
            mask_temporal = real_mask.unsqueeze(-1).expand(-1, -1, second_diff.size(-1))
        else:
            mask_temporal = real_mask[..., :-2]
        
        temporal_loss = (second_diff ** 2 * mask_temporal).sum()
        temporal_loss = temporal_loss / (mask_temporal.sum() + 1e-8)
        
        return temporal_loss
    
    def _compute_spectral_loss(self,
                             predictions: torch.Tensor,
                             targets: torch.Tensor,
                             real_mask: torch.Tensor) -> torch.Tensor:
        """计算频域损失（保持频域特征）"""
        
        # FFT变换到频域
        pred_fft = torch.fft.fft(predictions, dim=-1)
        target_fft = torch.fft.fft(targets, dim=-1)
        
        # 计算功率谱密度
        pred_psd = torch.abs(pred_fft) ** 2
        target_psd = torch.abs(target_fft) ** 2
        
        if real_mask.dim() == 2:
            spectral_mask = real_mask.unsqueeze(-1).expand_as(pred_psd)
        else:
            spectral_mask = real_mask
        
        # 频域MSE损失
        spectral_loss = F.mse_loss(pred_psd * spectral_mask,
                                 target_psd * spectral_mask,
                                 reduction='sum')
        spectral_loss = spectral_loss / (spectral_mask.sum() + 1e-8)
        
        return spectral_loss
    
    def update_weights(self, epoch: int, total_epochs: int):
        """动态调整损失权重"""
        self.epoch = epoch
        
        # 训练早期更注重重建，后期更注重分类
        progress = epoch / total_epochs
        
        # 重建权重：从1.0逐渐降到0.7
        self.recon_weight = 1.0 - 0.3 * progress
        
        # 分类权重：从0.5逐渐升到1.0  
        self.cls_weight = 0.5 + 0.5 * progress
        
        # 正则化项权重：中期最大
        self.consistency_weight = 0.2 * np.sin(progress * np.pi)
        self.temporal_weight = 0.15 * np.sin(progress * np.pi)

class AdversarialLoss(nn.Module):
    """对抗损失（可选）"""
    
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, generated_data: torch.Tensor, real_data: torch.Tensor):
        batch_size = generated_data.size(0)
        
        # 真实标签和假标签
        real_labels = torch.ones(batch_size, 1, device=generated_data.device)
        fake_labels = torch.zeros(batch_size, 1, device=generated_data.device)
        
        # 判别器对真实数据的输出
        real_validity = self.discriminator(real_data)
        d_real_loss = self.bce_loss(real_validity, real_labels)
        
        # 判别器对生成数据的输出
        fake_validity = self.discriminator(generated_data.detach())
        d_fake_loss = self.bce_loss(fake_validity, fake_labels)
        
        # 判别器总损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # 生成器损失（希望判别器认为生成数据是真的）
        g_validity = self.discriminator(generated_data)
        g_loss = self.bce_loss(g_validity, real_labels)
        
        return {
            'discriminator_loss': d_loss,
            'generator_loss': g_loss,
            'd_real_loss': d_real_loss,
            'd_fake_loss': d_fake_loss
        }

def create_enhanced_loss(config: Dict) -> MultiTaskLoss:
    """创建增强版损失函数"""
    
    return MultiTaskLoss(
        recon_weight=config.get('recon_weight', 1.0),
        cls_weight=config.get('cls_weight', 1.0),
        consistency_weight=config.get('consistency_weight', 0.1),
        temporal_weight=config.get('temporal_weight', 0.1),
        spectral_weight=config.get('spectral_weight', 0.05)
    )
