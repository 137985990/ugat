# simple_multimodal_integration.py - 简单的多模态损失集成方案

"""
这个文件提供了将多模态损失集成到现有train.py的最简单方案
只需要最小的代码修改，就能让common_modalities参与损失计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class SimpleMultiModalCriterion(nn.Module):
    """简化版多模态损失函数 - 可直接替换MSELoss"""
    
    def __init__(self, common_indices: List[int], common_weight: float = 1.2):
        super().__init__()
        self.common_indices = common_indices
        self.common_weight = common_weight
        self.mse_loss = nn.MSELoss()
        
        print(f"SimpleMultiModalCriterion初始化:")
        print(f"  common_indices: {common_indices}")
        print(f"  common_weight: {common_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                channel_idx: int = None, is_common: bool = None) -> torch.Tensor:
        """
        兼容现有MSELoss接口的多模态损失
        
        Args:
            pred: 预测值 [T] 或 [B, T]
            target: 目标值 [T] 或 [B, T]
            channel_idx: 当前通道索引（可选）
            is_common: 是否为common模态（可选）
        """
        
        # 基础MSE损失
        mse = self.mse_loss(pred, target)
        
        # 如果是common模态，应用权重
        if is_common or (channel_idx is not None and channel_idx in self.common_indices):
            return self.common_weight * mse
        else:
            return mse

def create_simple_multimodal_criterion(config):
    """创建简化版多模态损失函数"""
    
    common_modalities = config.get('common_modalities', [])
    
    # 构建所有模态列表
    all_modalities = common_modalities.copy()
    dataset_modalities = config.get('dataset_modalities', {})
    
    for dataset_name, modalities in dataset_modalities.items():
        have_mods = modalities.get('have', [])
        need_mods = modalities.get('need', [])
        for mod in have_mods + need_mods:
            if mod not in all_modalities:
                all_modalities.append(mod)
    
    # 获取common模态的索引
    common_indices = []
    for i, mod in enumerate(all_modalities):
        if mod in common_modalities:
            common_indices.append(i)
    
    loss_config = config.get('loss_config', {})
    common_weight = loss_config.get('common_weight', 1.2)
    
    return SimpleMultiModalCriterion(common_indices, common_weight)

def modify_train_phased_for_multimodal():
    """提供train_phased函数的修改示例"""
    
    modified_code = '''
# 在train.py中的修改示例

# 1. 在文件开头添加导入
from simple_multimodal_integration import create_simple_multimodal_criterion

# 2. 在main函数中，替换criterion的创建
def main():
    # ... 现有代码 ...
    
    # 原来的代码：
    # criterion = MSELoss()
    
    # 替换为：
    if config.get('loss_config', {}).get('type') == 'multimodal':
        criterion = create_simple_multimodal_criterion(config)
        print("使用多模态损失函数")
    else:
        criterion = MSELoss()
        print("使用标准MSE损失函数")
    
    # ... 其他代码保持不变 ...

# 3. 修改train_phased函数中的损失计算部分
def train_phased_modified(model, dataloader, optimizer, criterion, device, mask_indices, ...):
    """修改后的训练函数 - 最小改动版本"""
    
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    total_common_loss = 0.0  # 新增：common模态损失统计
    total_have_loss = 0.0    # 新增：have模态损失统计
    total_correct = 0
    total_samples = 0
    
    # 获取common模态索引
    common_indices = getattr(criterion, 'common_indices', [])
    
    for batch, labels, mask_idx, is_real_mask in tqdm(dataloader, desc=phase.capitalize()):
        batch = batch.to(device)
        labels = labels.to(device)
        is_real_mask = is_real_mask.to(device)
        masked, mask_idx = mask_channel(batch, mask_indices)
        batch_size, C, T = batch.size()
        
        optimizer.zero_grad()
        loss = 0.0
        cls_loss = 0.0
        recon_loss = 0.0
        common_loss = 0.0  # 新增
        have_loss = 0.0    # 新增
        
        for i in range(batch_size):
            window = masked[i].t()
            out, logits = model(window)
            
            # 获取真实通道信息
            if is_real_mask.dim() == 2:
                real_channels = is_real_mask[i]
            else:
                real_channels = is_real_mask
            
            recon_loss_i = 0.0
            common_loss_i = 0.0
            have_loss_i = 0.0
            real_count = 0
            common_count = 0
            have_count = 0
            
            # 分别计算common和have模态的损失
            for c in range(C):
                target = batch[i, c, :]
                pred = out[c, :]
                
                # 判断是否为common模态
                is_common_channel = c in common_indices
                
                if is_common_channel:
                    # Common模态：始终计算损失
                    loss_c = criterion(pred, target, channel_idx=c, is_common=True)
                    common_loss_i += loss_c
                    common_count += 1
                elif real_channels[c]:
                    # Have模态：只对真实通道计算损失
                    loss_c = criterion(pred, target, channel_idx=c, is_common=False)
                    have_loss_i += loss_c
                    have_count += 1
                
                # 总重建损失（保持原有逻辑兼容性）
                if is_common_channel or real_channels[c]:
                    recon_loss_i += criterion(pred, target, channel_idx=c)
                    real_count += 1
            
            # 平均损失
            if real_count > 0:
                recon_loss_i = recon_loss_i / real_count
            if common_count > 0:
                common_loss_i = common_loss_i / common_count
            if have_count > 0:
                have_loss_i = have_loss_i / have_count
            
            # 分类损失
            cls_loss_i = ce_loss(logits.unsqueeze(0), labels[i].unsqueeze(0))
            
            # 累积损失
            loss += recon_loss_i + cls_loss_i
            recon_loss += recon_loss_i
            common_loss += common_loss_i
            have_loss += have_loss_i
            cls_loss += cls_loss_i
            
            # 准确率统计
            pred_class = logits.argmax(-1).item()
            if pred_class == labels[i].item():
                total_correct += 1
            total_samples += 1
        
        loss = loss / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_recon_loss += recon_loss
        total_common_loss += common_loss
        total_have_loss += have_loss
        total_cls_loss += cls_loss
    
    n = len(dataloader.dataset)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 返回包含详细信息的结果
    return {
        'total_loss': total_loss / n,
        'reconstruction_loss': total_recon_loss / n,
        'common_loss': total_common_loss / n,      # 新增
        'have_loss': total_have_loss / n,          # 新增
        'classification_loss': total_cls_loss / n,
        'accuracy': acc
    }

# 4. 修改日志记录部分
# 在训练循环中添加详细的损失记录
for epoch in range(num_epochs):
    train_result = train_phased_modified(...)
    
    # 详细日志
    logging.info(f"[TRAIN] Epoch {epoch}: "
                f"total_loss={train_result['total_loss']:.6f}, "
                f"common_loss={train_result['common_loss']:.6f}, "
                f"have_loss={train_result['have_loss']:.6f}, "
                f"cls_loss={train_result['classification_loss']:.6f}, "
                f"acc={train_result['accuracy']:.4f}")
    
    # TensorBoard记录
    writer.add_scalar('Train/Loss/Total', train_result['total_loss'], epoch)
    writer.add_scalar('Train/Loss/Common', train_result['common_loss'], epoch)
    writer.add_scalar('Train/Loss/Have', train_result['have_loss'], epoch)
    writer.add_scalar('Train/Loss/Classification', train_result['classification_loss'], epoch)
    writer.add_scalar('Train/Accuracy', train_result['accuracy'], epoch)
'''
    
    return modified_code

def test_simple_multimodal():
    """测试简化版多模态损失函数"""
    
    print("=" * 60)
    print("简化版多模态损失函数测试")
    print("=" * 60)
    
    # 模拟配置
    config = {
        'common_modalities': ['acc_x', 'acc_y', 'acc_z', 'ppg', 'gsr', 'hr', 'skt'],
        'dataset_modalities': {
            'FM': {'have': ['alpha_tp9', 'beta_tp9'], 'need': ['space_distance']},
            'OD': {'have': ['space_distance'], 'need': ['alpha_tp9', 'beta_tp9']}
        },
        'loss_config': {
            'type': 'multimodal',
            'common_weight': 1.2
        }
    }
    
    # 创建损失函数
    criterion = create_simple_multimodal_criterion(config)
    
    # 测试损失计算
    pred = torch.randn(100)
    target = torch.randn(100)
    
    # 测试common模态损失
    common_loss = criterion(pred, target, channel_idx=0, is_common=True)
    print(f"Common模态损失: {common_loss.item():.6f}")
    
    # 测试have模态损失
    have_loss = criterion(pred, target, channel_idx=7, is_common=False)
    print(f"Have模态损失: {have_loss.item():.6f}")
    
    # 测试权重效果
    ratio = common_loss.item() / have_loss.item()
    print(f"Common/Have损失比例: {ratio:.3f} (预期约为1.2)")

if __name__ == "__main__":
    # 运行测试
    test_simple_multimodal()
    
    # 生成修改代码
    modified_code = modify_train_phased_for_multimodal()
    
    with open('train_phased_modification.py', 'w', encoding='utf-8') as f:
        f.write(modified_code)
    
    print(f"\n" + "=" * 60)
    print("简化版多模态损失集成方案")
    print("=" * 60)
    print("\n优势：")
    print("✓ 最小代码修改，兼容现有逻辑")
    print("✓ Common模态现在参与损失计算")
    print("✓ 保持原有训练流程不变")
    print("✓ 添加详细的损失监控")
    print("\n生成文件：")
    print("- train_phased_modification.py: 详细修改代码示例")
    print("\n使用步骤：")
    print("1. 将config.yaml中的loss_config.type设为'multimodal'")
    print("2. 按照train_phased_modification.py中的示例修改train.py")
    print("3. 观察common_loss和have_loss的变化趋势")
    print("4. 监控分类性能是否因为更好的重建而提升")
