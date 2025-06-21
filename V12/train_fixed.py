# train_fixed.py - V12修复后的训练脚本

def train_phased(model, dataloader, optimizer, criterion, device, mask_indices, use_mixed_precision=True, scaler=None):
    """训练函数 - 用重建结果验证encode分类，计算两次分类的差异损失来指导生成模型"""
    model.train()
    kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')  # 用于计算两次分类结果的差异
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_consistency_loss = 0.0  # 两次分类一致性损失
    total_encode_correct = 0  # encode阶段分类正确数
    total_decode_correct = 0  # decode阶段分类正确数
    total_samples = 0
    
    # 获取common模态索引
    common_indices = getattr(criterion, 'common_indices', [])
    
    for batch_data in tqdm(dataloader, desc="Training"):
        if len(batch_data) == 4:
            batch, labels, _, is_real_mask = batch_data
        else:
            batch, labels, _, is_real_mask, _ = batch_data
        
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        is_real_mask = is_real_mask.to(device, non_blocking=True)
        
        masked, mask_idx = mask_channel(batch, mask_indices)
        batch_size, C, T = batch.size()
        
        optimizer.zero_grad()
        loss = 0.0
        recon_loss = 0.0
        cls_consistency_loss = 0.0
        
        # 混合精度训练前向传播
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            with autocast():
                for i in range(batch_size):
                    window_masked = masked[i].t()  # 被mask的输入
                    
                    # 第一步：Encode阶段 - 用被mask的数据进行分类和重建
                    out_encode, logits_encode = model(window_masked, phase="encode")
                    
                    # 第二步：用重建的数据再次进行分类
                    window_reconstructed = out_encode.t()  # 重建后的完整数据
                    _, logits_decode = model(window_reconstructed, phase="encode")  # 用重建数据分类
                    
                    # 计算重建损失
                    if is_real_mask.dim() == 2:
                        real_channels = is_real_mask[i]
                    else:
                        real_channels = is_real_mask
                    
                    recon_loss_i = 0.0
                    real_count = 0
                    
                    # 计算重建损失（相对于原始数据）
                    for c in range(C):
                        target = batch[i, c, :]  # 原始数据
                        pred = out_encode[c, :]   # 重建数据
                        
                        is_common_channel = c in common_indices
                        
                        if is_common_channel:
                            recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=True)
                            real_count += 1
                        elif real_channels[c]:
                            recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=False)
                            real_count += 1
                    
                    if real_count > 0:
                        recon_loss_i = recon_loss_i / real_count
                    
                    # 计算两次分类结果的一致性损失
                    # 目标：让重建数据的分类结果接近原始(被mask)数据的分类结果
                    logits_decode_log_softmax = torch.nn.functional.log_softmax(logits_decode, dim=0)  # 重建数据的分类
                    logits_encode_softmax = torch.nn.functional.softmax(logits_encode, dim=0)     # 原始数据的分类
                    cls_consistency_loss_i = kl_div_loss(logits_decode_log_softmax.unsqueeze(0), 
                                                        logits_encode_softmax.unsqueeze(0))
                    
                    # 总损失 = 重建损失 + 分类一致性损失
                    loss += recon_loss_i + cls_consistency_loss_i
                    recon_loss += recon_loss_i
                    cls_consistency_loss += cls_consistency_loss_i
                    
                    # 计算准确率（与真实标签比较）
                    pred_class_encode = logits_encode.argmax(-1).item()
                    pred_class_decode = logits_decode.argmax(-1).item()
                    
                    if pred_class_encode == labels[i].item():
                        total_encode_correct += 1
                    if pred_class_decode == labels[i].item():
                        total_decode_correct += 1
                    
                    total_samples += 1
                
                loss = loss / batch_size
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准精度训练
            for i in range(batch_size):
                window_masked = masked[i].t()  # 被mask的输入
                
                # 第一步：Encode阶段 - 用被mask的数据进行分类和重建
                out_encode, logits_encode = model(window_masked, phase="encode")
                
                # 第二步：用重建的数据再次进行分类
                window_reconstructed = out_encode.t()  # 重建后的完整数据
                _, logits_decode = model(window_reconstructed, phase="encode")  # 用重建数据分类
                
                # 计算重建损失
                if is_real_mask.dim() == 2:
                    real_channels = is_real_mask[i]
                else:
                    real_channels = is_real_mask
                
                recon_loss_i = 0.0
                real_count = 0
                
                # 计算重建损失（相对于原始数据）
                for c in range(C):
                    target = batch[i, c, :]  # 原始数据
                    pred = out_encode[c, :]   # 重建数据
                    
                    is_common_channel = c in common_indices
                    
                    if is_common_channel:
                        recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=True)
                        real_count += 1
                    elif real_channels[c]:
                        recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=False)
                        real_count += 1
                
                if real_count > 0:
                    recon_loss_i = recon_loss_i / real_count
                
                # 计算两次分类结果的一致性损失
                logits_decode_log_softmax = torch.nn.functional.log_softmax(logits_decode, dim=0)  # 重建数据的分类
                logits_encode_softmax = torch.nn.functional.softmax(logits_encode, dim=0)     # 原始数据的分类
                cls_consistency_loss_i = kl_div_loss(logits_decode_log_softmax.unsqueeze(0), 
                                                    logits_encode_softmax.unsqueeze(0))
                
                # 总损失 = 重建损失 + 分类一致性损失
                loss += recon_loss_i + cls_consistency_loss_i
                recon_loss += recon_loss_i
                cls_consistency_loss += cls_consistency_loss_i
                
                # 计算准确率（与真实标签比较）
                pred_class_encode = logits_encode.argmax(-1).item()
                pred_class_decode = logits_decode.argmax(-1).item()
                
                if pred_class_encode == labels[i].item():
                    total_encode_correct += 1
                if pred_class_decode == labels[i].item():
                    total_decode_correct += 1
                
                total_samples += 1
            
            loss = loss / batch_size
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_recon_loss += recon_loss
        total_cls_consistency_loss += cls_consistency_loss
    
    n = len(dataloader.dataset)
    encode_acc = total_encode_correct / total_samples if total_samples > 0 else 0.0
    decode_acc = total_decode_correct / total_samples if total_samples > 0 else 0.0
    return total_loss / n, total_recon_loss / n, total_cls_consistency_loss / n, encode_acc, decode_acc
