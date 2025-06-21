# train_multimodal_loss.py - 集成多模态损失函数的训练修改

"""
修改现有训练代码以支持common_modalities的损失计算
主要变更：
1. 替换损失函数为MultiModalLoss
2. 修改损失计算逻辑，分别计算common和have模态的损失
3. 添加详细的损失监控和日志
"""

import torch
import torch.nn as nn
from enhanced_loss_with_common import create_multimodal_loss
import yaml

def create_enhanced_train_function(config):
    """创建支持多模态损失的训练函数"""
    
    # 创建多模态损失函数
    multimodal_loss = create_multimodal_loss(config)
    ce_loss = nn.CrossEntropyLoss()
    
    def train_phased_multimodal(
        model, dataloader, optimizer, device, mask_indices, 
        phase="encode", epoch=None, total_epochs=None
    ):
        """使用多模态损失的训练函数"""
        
        model.train()
        
        # 更新权重（如果是自适应版本）
        if hasattr(multimodal_loss, 'update_epoch') and epoch is not None:
            multimodal_loss.update_epoch(epoch, total_epochs or 100)
        
        total_loss_dict = {}
        total_correct = 0
        total_samples = 0
        
        for batch, labels, mask_idx, is_real_mask in dataloader:
            batch = batch.to(device)
            labels = labels.to(device)
            is_real_mask = is_real_mask.to(device)
            
            # 应用mask
            from train import mask_channel
            masked, mask_idx = mask_channel(batch, mask_indices)
            
            batch_size, C, T = batch.size()
            optimizer.zero_grad()
            
            # 收集batch中所有样本的预测
            batch_predictions = []
            batch_logits = []
            
            for i in range(batch_size):
                window = masked[i].t()  # [T, C]
                out, logits = model(window)  # out: [C, T], logits: [num_classes]
                batch_predictions.append(out.unsqueeze(0))  # [1, C, T]
                batch_logits.append(logits.unsqueeze(0))   # [1, num_classes]
            
            # 拼接为batch
            predictions = torch.cat(batch_predictions, dim=0)  # [B, C, T]
            logits_batch = torch.cat(batch_logits, dim=0)      # [B, num_classes]
            
            # 使用多模态损失函数
            loss_dict = multimodal_loss(
                predictions=predictions,
                targets=batch,
                logits=logits_batch,
                labels=labels,
                have_mask=is_real_mask
            )
            
            # 反向传播
            total_loss = loss_dict['total_loss']
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积损失统计
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0.0
                total_loss_dict[key] += value.item() * batch_size
            
            # 计算准确率
            pred_classes = logits_batch.argmax(dim=-1)
            total_correct += (pred_classes == labels).sum().item()
            total_samples += batch_size
        
        # 平均损失
        n = len(dataloader.dataset)
        for key in total_loss_dict:
            total_loss_dict[key] /= n
        
        # 添加准确率
        total_loss_dict['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
        
        return total_loss_dict
    
    return train_phased_multimodal, multimodal_loss

def create_enhanced_eval_function(multimodal_loss):
    """创建支持多模态损失的评估函数"""
    
    def eval_loop_multimodal(model, dataloader, device, mask_indices):
        """使用多模态损失的评估函数"""
        
        model.eval()
        total_loss_dict = {}
        
        with torch.no_grad():
            for batch, _, _, is_real_mask in dataloader:
                batch = batch.to(device)
                is_real_mask = is_real_mask.to(device)
                
                # 应用mask
                from train import mask_channel
                masked, mask_idx = mask_channel(batch, mask_indices)
                
                batch_size, C, T = batch.size()
                
                # 收集预测
                batch_predictions = []
                batch_logits = []
                
                for i in range(batch_size):
                    window = masked[i].t()
                    out, logits = model(window)
                    batch_predictions.append(out.unsqueeze(0))
                    batch_logits.append(logits.unsqueeze(0))
                
                # 拼接
                predictions = torch.cat(batch_predictions, dim=0)
                logits_batch = torch.cat(batch_logits, dim=0)
                
                # 创建虚拟标签（评估时不需要真实标签）
                dummy_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                
                # 计算损失
                loss_dict = multimodal_loss(
                    predictions=predictions,
                    targets=batch,
                    logits=logits_batch,
                    labels=dummy_labels,
                    have_mask=is_real_mask
                )
                
                # 累积损失
                for key, value in loss_dict.items():
                    if key != 'classification_loss':  # 评估时忽略分类损失
                        if key not in total_loss_dict:
                            total_loss_dict[key] = 0.0
                        total_loss_dict[key] += value.item() * batch_size
        
        # 平均损失
        n = len(dataloader.dataset)
        for key in total_loss_dict:
            total_loss_dict[key] /= n
        
        return total_loss_dict
    
    return eval_loop_multimodal

def demonstrate_multimodal_training():
    """演示多模态损失训练的使用"""
    
    print("=" * 60)
    print("多模态损失函数训练演示")
    print("=" * 60)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 添加多模态损失配置
    config['loss_config'] = {
        'recon_weight': 1.0,
        'cls_weight': 1.0,
        'common_weight': 1.2,  # common模态权重稍高
        'have_weight': 1.0,
        'l1_weight': 0.1,
        'adaptive': True
    }
    
    # 创建增强训练函数
    train_fn, multimodal_loss = create_enhanced_train_function(config)
    eval_fn = create_enhanced_eval_function(multimodal_loss)
    
    print("配置信息:")
    print(f"  common_modalities: {config['common_modalities']}")
    print(f"  数据集模态配置: {list(config['dataset_modalities'].keys())}")
    print(f"  loss配置: {config['loss_config']}")
    
    # 显示模态映射
    print(f"\n模态映射:")
    print(f"  common_indices: {multimodal_loss.common_indices}")
    print(f"  have_indices: {multimodal_loss.have_indices}")
    print(f"  总模态数: {len(multimodal_loss.all_modalities)}")
    
    # 模拟训练过程
    print(f"\n模拟训练过程:")
    for epoch in [0, 25, 50, 75, 100]:
        if hasattr(multimodal_loss, 'update_epoch'):
            multimodal_loss.update_epoch(epoch, 100)
            print(f"  Epoch {epoch:3d}: common_w={multimodal_loss.common_weight:.3f}, "
                  f"have_w={multimodal_loss.have_weight:.3f}, cls_w={multimodal_loss.cls_weight:.3f}")

def create_integration_patch():
    """创建集成补丁代码"""
    
    patch_code = '''
# ==================== 集成多模态损失函数的代码修改 ====================

# 1. 在train.py文件开头添加导入
from enhanced_loss_with_common import create_multimodal_loss

# 2. 在main函数中，替换criterion创建部分
def main():
    # ... 现有代码 ...
    
    # 原来的代码：
    # criterion = MSELoss()
    
    # 替换为：
    # 添加多模态损失配置到config中
    if 'loss_config' not in config:
        config['loss_config'] = {
            'recon_weight': 1.0,
            'cls_weight': 1.0,
            'common_weight': 1.2,  # 给common模态更高权重
            'have_weight': 1.0,
            'l1_weight': 0.1,
            'adaptive': True  # 使用自适应权重
        }
    
    # 创建多模态损失函数
    multimodal_criterion = create_multimodal_loss(config)
    
    # ... 其他现有代码 ...

# 3. 修改train_phased函数
def train_phased_modified(
    model, dataloader, optimizer, device, mask_indices, 
    multimodal_criterion, phase="encode", epoch=None, total_epochs=None
):
    """修改后的训练函数"""
    
    model.train()
    
    # 更新自适应权重
    if hasattr(multimodal_criterion, 'update_epoch') and epoch is not None:
        multimodal_criterion.update_epoch(epoch, total_epochs or 100)
    
    total_loss_dict = {}
    total_correct = 0
    total_samples = 0
    
    for batch, labels, mask_idx, is_real_mask in tqdm(dataloader, desc=phase.capitalize()):
        batch = batch.to(device)
        labels = labels.to(device)
        is_real_mask = is_real_mask.to(device)
        
        masked, mask_idx = mask_channel(batch, mask_indices)
        batch_size, C, T = batch.size()
        
        optimizer.zero_grad()
        
        # 收集batch预测
        batch_predictions = []
        batch_logits = []
        
        for i in range(batch_size):
            window = masked[i].t()
            out, logits = model(window)
            batch_predictions.append(out.unsqueeze(0))
            batch_logits.append(logits.unsqueeze(0))
        
        predictions = torch.cat(batch_predictions, dim=0)
        logits_batch = torch.cat(batch_logits, dim=0)
        
        # 使用多模态损失
        loss_dict = multimodal_criterion(
            predictions=predictions,
            targets=batch,
            logits=logits_batch,
            labels=labels,
            have_mask=is_real_mask
        )
        
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 统计
        for key, value in loss_dict.items():
            if key not in total_loss_dict:
                total_loss_dict[key] = 0.0
            total_loss_dict[key] += value.item() * batch_size
        
        pred_classes = logits_batch.argmax(dim=-1)
        total_correct += (pred_classes == labels).sum().item()
        total_samples += batch_size
    
    # 返回平均损失
    n = len(dataloader.dataset)
    result = {}
    for key, value in total_loss_dict.items():
        result[key] = value / n
    
    result['accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
    return result

# 4. 修改训练循环调用
for epoch in range(num_epochs):
    # 原来的调用：
    # train_loss, train_recon, train_cls, train_acc = train_phased(...)
    
    # 替换为：
    train_result = train_phased_modified(
        model, train_loader, optimizer, device, mask_indices,
        multimodal_criterion, phase="encode", epoch=epoch, total_epochs=num_epochs
    )
    
    # 记录详细损失信息
    logging.info(f"[TRAIN] Epoch {epoch}:")
    logging.info(f"  Total Loss: {train_result['total_loss']:.6f}")
    logging.info(f"  Common Loss: {train_result['common_total_loss']:.6f}")
    logging.info(f"  Have Loss: {train_result['have_total_loss']:.6f}")
    logging.info(f"  Classification Loss: {train_result['classification_loss']:.6f}")
    logging.info(f"  Accuracy: {train_result['accuracy']:.4f}")
    
    # TensorBoard记录
    writer.add_scalar('Train/Loss/Total', train_result['total_loss'], epoch)
    writer.add_scalar('Train/Loss/Common', train_result['common_total_loss'], epoch)
    writer.add_scalar('Train/Loss/Have', train_result['have_total_loss'], epoch)
    writer.add_scalar('Train/Loss/Classification', train_result['classification_loss'], epoch)
    writer.add_scalar('Train/Accuracy', train_result['accuracy'], epoch)
    
    # 记录权重变化（如果是自适应版本）
    if hasattr(multimodal_criterion, 'common_weight'):
        writer.add_scalar('Weights/Common', multimodal_criterion.common_weight, epoch)
        writer.add_scalar('Weights/Have', multimodal_criterion.have_weight, epoch)
        writer.add_scalar('Weights/Classification', multimodal_criterion.cls_weight, epoch)
'''
    
    with open('multimodal_loss_integration_patch.py', 'w', encoding='utf-8') as f:
        f.write(patch_code)
    
    print("集成补丁代码已保存到: multimodal_loss_integration_patch.py")

if __name__ == "__main__":
    demonstrate_multimodal_training()
    create_integration_patch()
    
    print(f"\n" + "=" * 60)
    print("多模态损失函数集成完成！")
    print("=" * 60)
    print("\n主要改进:")
    print("✓ Common modalities现在参与损失计算")
    print("✓ 分别计算common和have模态的重建损失")
    print("✓ 支持自适应权重调整")
    print("✓ 详细的损失监控和可视化")
    print("\n使用建议:")
    print("1. Common权重设置稍高（1.2），因为是真实数据")
    print("2. 启用自适应权重，训练过程中动态优化")
    print("3. 监控common和have损失的平衡")
    print("4. 观察分类性能是否因为更好的重建质量而提升")
