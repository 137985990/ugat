# magvit2_multimodal_loss.py - Magvit2专用的多模态损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F


class Magvit2MultimodalCriterion(nn.Module):
    """
    Magvit2专用的多模态损失函数，结合重建损失、VQ损失和分类损失
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 获取配置
        loss_config = config.get('loss_config', {})
        self.recon_weight = loss_config.get('recon_weight', 1.0)
        self.vq_weight = loss_config.get('vq_weight', 1.0)
        self.cls_weight = loss_config.get('cls_weight', 0.1)
        self.perceptual_weight = loss_config.get('perceptual_weight', 0.1)
        
        # 模态配置
        self.common_modalities = config.get('common_modalities', [])
        self.dataset_modalities = config.get('dataset_modalities', {})
        
        # 构建模态索引映射
        self._build_modality_mapping(config)
        
        # 损失函数
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss(reduction='none')
        
        print(f"[Magvit2MultimodalCriterion] Initialized with weights: "
              f"recon={self.recon_weight}, vq={self.vq_weight}, cls={self.cls_weight}")
    
    def _build_modality_mapping(self, config):
        """构建模态索引映射"""
        # 获取所有特征列
        all_feature_mods = self.common_modalities.copy()
        for dataset, mods in self.dataset_modalities.items():
            have = mods.get('have', [])
            need = mods.get('need', [])
            for mod in have + need:
                if mod not in all_feature_mods:
                    all_feature_mods.append(mod)
        
        # 创建索引映射
        self.modality_to_idx = {mod: i for i, mod in enumerate(all_feature_mods)}
        self.common_indices = [self.modality_to_idx[mod] for mod in self.common_modalities 
                              if mod in self.modality_to_idx]
        
        print(f"[Magvit2MultimodalCriterion] Modality mapping: {self.modality_to_idx}")
        print(f"[Magvit2MultimodalCriterion] Common indices: {self.common_indices}")
    
    def compute_reconstruction_loss(self, pred, target, is_real_mask, channel_idx=None, is_common=None):
        """
        计算重建损失
        Args:
            pred: 预测值 [seq_len] or [batch_size, seq_len]
            target: 目标值 [seq_len] or [batch_size, seq_len]
            is_real_mask: 真实数据掩码
            channel_idx: 通道索引
            is_common: 是否为common模态
        """
        # 基础MSE损失
        mse = self.mse_loss(pred, target)
        
        # 根据模态类型调整损失权重
        if is_common:
            # Common模态：更高权重
            weight = 2.0
        else:
            # Have模态：标准权重
            weight = 1.0
        
        # 应用权重
        loss = mse * weight
        
        # 如果有掩码，只计算真实数据的损失
        if is_real_mask is not None and channel_idx is not None:
            if is_real_mask.dim() == 2:
                mask = is_real_mask[:, channel_idx] if channel_idx < is_real_mask.size(1) else True
            else:
                mask = is_real_mask[channel_idx] if channel_idx < is_real_mask.size(0) else True
            
            if isinstance(mask, torch.Tensor):
                loss = loss * mask.float()
        
        return loss.mean()
    
    def compute_perceptual_loss(self, pred, target):
        """
        计算感知损失（基于特征相似性）
        """
        # 简单的L1感知损失
        return self.l1_loss(pred, target).mean()
    
    def forward(self, pred, target, vq_loss=None, classification_logits=None, classification_targets=None, 
                is_real_mask=None, channel_idx=None, is_common=None):
        """
        前向传播计算总损失
        Args:
            pred: 重建预测 [batch_size, channels, seq_len] or [channels, seq_len]
            target: 重建目标 [batch_size, channels, seq_len] or [channels, seq_len]
            vq_loss: Vector Quantization损失
            classification_logits: 分类预测 [batch_size, num_classes] or [num_classes]
            classification_targets: 分类目标 [batch_size] or scalar
            is_real_mask: 真实数据掩码
            channel_idx: 通道索引
            is_common: 是否为common模态
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 重建损失
        if pred is not None and target is not None:
            if pred.dim() == target.dim() and pred.dim() >= 2:
                # 批量处理
                recon_loss = 0.0
                batch_size = pred.size(0)
                channels = pred.size(1)
                
                for b in range(batch_size):
                    for c in range(channels):
                        # 判断是否为common模态
                        is_common_channel = c in self.common_indices
                        
                        # 计算单通道损失
                        channel_loss = self.compute_reconstruction_loss(
                            pred[b, c], target[b, c], 
                            is_real_mask[b] if is_real_mask is not None and is_real_mask.dim() == 2 else is_real_mask,
                            channel_idx=c, is_common=is_common_channel
                        )
                        recon_loss += channel_loss
                
                recon_loss = recon_loss / (batch_size * channels)
            else:
                # 单样本处理
                recon_loss = self.compute_reconstruction_loss(
                    pred, target, is_real_mask, channel_idx, is_common
                )
            
            total_loss += self.recon_weight * recon_loss
            loss_dict['recon_loss'] = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
        
        # 2. VQ损失
        if vq_loss is not None:
            total_loss += self.vq_weight * vq_loss
            loss_dict['vq_loss'] = vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
        
        # 3. 分类损失
        if classification_logits is not None and classification_targets is not None:
            cls_loss = self.ce_loss(classification_logits, classification_targets)
            total_loss += self.cls_weight * cls_loss
            loss_dict['cls_loss'] = cls_loss.item()
        
        # 4. 感知损失（可选）
        if pred is not None and target is not None and self.perceptual_weight > 0:
            perceptual_loss = self.compute_perceptual_loss(pred, target)
            total_loss += self.perceptual_weight * perceptual_loss
            loss_dict['perceptual_loss'] = perceptual_loss.item()
        
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss
    
    def __call__(self, pred, target, channel_idx=None, is_common=None, **kwargs):
        """
        兼容原始接口的调用方法
        """
        return self.forward(pred, target, channel_idx=channel_idx, is_common=is_common, **kwargs)


def create_magvit2_multimodal_criterion(config):
    """创建Magvit2多模态损失函数"""
    return Magvit2MultimodalCriterion(config)
