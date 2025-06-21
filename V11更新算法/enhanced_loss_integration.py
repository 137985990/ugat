
# 在train.py中集成增强版损失函数的示例

from enhanced_loss import create_enhanced_loss

def setup_enhanced_loss(config):
    """设置增强版损失函数"""
    loss_config = {
        'recon_weight': config.get('recon_weight', 1.0),
        'cls_weight': config.get('cls_weight', 1.0),
        'consistency_weight': config.get('consistency_weight', 0.1),
        'temporal_weight': config.get('temporal_weight', 0.1),
        'spectral_weight': config.get('spectral_weight', 0.05)
    }
    return create_enhanced_loss(loss_config)

def train_with_enhanced_loss(model, dataloader, optimizer, enhanced_loss, device, mask_indices):
    """使用增强版损失函数的训练循环"""
    model.train()
    total_losses = {}
    
    for batch, labels, mask_idx, is_real_mask in dataloader:
        batch = batch.to(device)
        labels = labels.to(device)
        is_real_mask = is_real_mask.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        predictions, logits = model(batch)
        
        # 计算增强损失
        loss_dict = enhanced_loss(
            predictions=predictions,
            targets=batch, 
            logits=logits,
            labels=labels,
            real_mask=is_real_mask
        )
        
        # 反向传播
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累积损失
        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value.item()
    
    return total_losses

# 使用示例
if __name__ == "__main__":
    # 设置增强损失函数
    enhanced_loss = setup_enhanced_loss(config)
    
    # 训练循环中更新权重
    for epoch in range(num_epochs):
        enhanced_loss.update_weights(epoch, num_epochs)
        
        # 训练
        train_losses = train_with_enhanced_loss(
            model, train_loader, optimizer, enhanced_loss, device, mask_indices
        )
        
        # 日志记录
        print(f"Epoch {epoch}: Total Loss = {train_losses['total_loss']:.6f}")
        print(f"  Reconstruction: {train_losses['reconstruction_loss']:.6f}")
        print(f"  Classification: {train_losses['classification_loss']:.6f}")
