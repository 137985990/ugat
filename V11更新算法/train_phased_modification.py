
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
