# loss_stabilization.py - 在V15基础上的损失稳定化修复

import torch
import torch.nn as nn

def stabilize_classification_improvement_loss(input_cls_loss, enhanced_cls_loss, 
                                            accuracy_improvement, 
                                            accuracy_reward_scale=0.5,  # 降低奖励系数
                                            min_improvement_margin=0.01,  # 降低要求
                                            current_epoch=1,
                                            use_focal_loss=False):
    """
    稳定化的分类改进损失 - 保持原有双路径思路，增加数值稳定性
    
    这个函数是对您原有损失函数的最小化改动版本
    """
    
    # 1. 基础分类监督损失（保持不变）
    base_classification_loss = input_cls_loss
    
    # 2. 循环改进损失（添加稳定性检查）
    # 避免极端情况导致的数值不稳定
    classification_improvement_loss = enhanced_cls_loss - input_cls_loss
    
    # 检查并修复NaN/Inf
    if torch.isnan(classification_improvement_loss) or torch.isinf(classification_improvement_loss):
        classification_improvement_loss = torch.tensor(0.0, device=input_cls_loss.device)
    
    # 3. 准确率奖励（使用更稳定的函数）
    # 限制accuracy_improvement范围，避免极值
    accuracy_improvement_clamped = torch.clamp(accuracy_improvement, -0.2, 0.2)  # 更保守的范围
    
    # 使用更平滑的奖励函数
    if abs(accuracy_improvement_clamped) > 0.01:  # 只对显著改进给奖励
        accuracy_reward = -torch.sign(accuracy_improvement_clamped) * torch.log1p(
            torch.abs(accuracy_improvement_clamped) * 10
        ) * accuracy_reward_scale
    else:
        accuracy_reward = torch.tensor(0.0, device=input_cls_loss.device)
    
    # 4. 自适应margin（更温和的增长）
    epoch_factor = min(current_epoch / 200.0, 1.0)  # 200个epoch内线性增长到最大
    adaptive_margin = min_improvement_margin * (1.0 + 0.5 * epoch_factor)  # 更温和的增长
    
    # 5. 组合损失（添加权重衰减）
    # 早期阶段减少改进损失的权重，让模型先学好基础
    improvement_weight = min(current_epoch / 50.0, 1.0)  # 前50个epoch逐步增加权重
    
    cls_improvement_loss = (
        classification_improvement_loss * improvement_weight + 
        accuracy_reward + 
        adaptive_margin
    )
    
    total_classification_loss = base_classification_loss + cls_improvement_loss
    
    return {
        'total_classification_loss': total_classification_loss,
        'base_classification_loss': base_classification_loss,
        'cls_improvement_loss': cls_improvement_loss,
        'accuracy_reward': accuracy_reward,
        'adaptive_margin': adaptive_margin,
        'improvement_weight': improvement_weight
    }

def stabilize_dynamic_weighting(accuracy_improvement, recon_weight, cls_improvement_weight,
                               accuracy_threshold=0.02,  # 降低阈值
                               max_weight_change=0.1):    # 限制权重变化幅度
    """
    稳定化的动态权重调整 - 避免权重剧烈变化
    """
    dynamic_recon_weight = recon_weight
    dynamic_cls_weight = cls_improvement_weight
    
    # 限制权重调整幅度，避免训练不稳定
    if accuracy_improvement > accuracy_threshold:
        # 准确率提升时，适度增加分类权重
        weight_multiplier = min(1.0 + max_weight_change, 1.0 + accuracy_improvement * 2.0)
        dynamic_cls_weight *= weight_multiplier
    elif accuracy_improvement < -accuracy_threshold:
        # 准确率下降时，适度增加重建权重
        weight_multiplier = min(1.0 + max_weight_change, 1.0 - accuracy_improvement * 2.0)
        dynamic_recon_weight *= weight_multiplier
    
    return dynamic_recon_weight, dynamic_cls_weight