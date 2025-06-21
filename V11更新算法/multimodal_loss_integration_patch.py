
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
