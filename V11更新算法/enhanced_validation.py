# enhanced_validation.py - 增强验证集策略

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import logging
import os

class ValidationTracker:
    """验证集性能跟踪器"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.history = []
        
    def update(self, score: float, epoch: int) -> bool:
        """
        更新验证分数
        Returns: True if should stop early, False otherwise
        """
        self.history.append(score)
        
        if self.mode == 'min':
            is_better = score < (self.best_score - self.min_delta)
        else:
            is_better = score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'patience': self.patience,
            'should_stop': self.counter >= self.patience,
            'improvement_rate': self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """计算改进率"""
        if len(self.history) < 10:
            return 0.0
        
        recent = np.mean(self.history[-5:])
        past = np.mean(self.history[-10:-5])
        
        if self.mode == 'min':
            return (past - recent) / past if past > 0 else 0.0
        else:
            return (recent - past) / past if past > 0 else 0.0

class EnhancedValidationLoop:
    """增强验证循环"""
    
    def __init__(self, model, criterion, device, mask_indices, 
                 enable_detailed_metrics: bool = True):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.mask_indices = mask_indices
        self.enable_detailed_metrics = enable_detailed_metrics
        
    def evaluate(self, dataloader, return_detailed: bool = False) -> Dict:
        """
        执行完整的验证评估
        
        Args:
            dataloader: 验证数据加载器
            return_detailed: 是否返回详细指标
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        
        total_losses = {}
        total_samples = 0
        predictions_list = []
        targets_list = []
        labels_list = []
        logits_list = []
        
        with torch.no_grad():
            for batch, labels, mask_idx, is_real_mask in dataloader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                is_real_mask = is_real_mask.to(self.device)
                
                # 应用mask
                from train import mask_channel
                masked, mask_idx = mask_channel(batch, self.mask_indices)
                
                batch_size, C, T = batch.size()
                batch_predictions = []
                batch_logits = []
                
                # 逐样本预测
                for i in range(batch_size):
                    window = masked[i].t()
                    out, logits = self.model(window)
                    batch_predictions.append(out.unsqueeze(0))
                    batch_logits.append(logits.unsqueeze(0))
                
                predictions = torch.cat(batch_predictions, dim=0)
                logits_batch = torch.cat(batch_logits, dim=0)
                
                # 计算损失
                if hasattr(self.criterion, 'common_indices'):
                    # 多模态损失
                    loss_dict = self._compute_multimodal_loss(
                        predictions, batch, logits_batch, labels, is_real_mask
                    )
                else:
                    # 标准损失
                    loss_dict = self._compute_standard_loss(
                        predictions, batch, logits_batch, labels, is_real_mask
                    )
                
                # 累积损失
                for key, value in loss_dict.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item() * batch_size
                
                total_samples += batch_size
                
                # 收集详细信息
                if return_detailed:
                    predictions_list.append(predictions.cpu())
                    targets_list.append(batch.cpu())
                    labels_list.append(labels.cpu())
                    logits_list.append(logits_batch.cpu())
        
        # 平均损失
        for key in total_losses:
            total_losses[key] /= total_samples
        
        # 计算准确率
        if 'classification_loss' in total_losses:
            total_losses['accuracy'] = self._compute_accuracy(
                torch.cat(logits_list) if logits_list else None,
                torch.cat(labels_list) if labels_list else None
            )
        
        if return_detailed and self.enable_detailed_metrics:
            total_losses.update(self._compute_detailed_metrics(
                predictions_list, targets_list, labels_list, logits_list
            ))
        
        return total_losses
    
    def _compute_multimodal_loss(self, predictions, targets, logits, labels, is_real_mask):
        """计算多模态损失"""
        common_indices = self.criterion.common_indices
        batch_size, C, T = predictions.shape
        
        total_loss = 0.0
        common_loss = 0.0
        have_loss = 0.0
        ce_loss = nn.CrossEntropyLoss()
        
        for i in range(batch_size):
            real_channels = is_real_mask[i]
            recon_loss_i = 0.0
            common_loss_i = 0.0
            have_loss_i = 0.0
            common_count = 0
            have_count = 0
            
            for c in range(C):
                target = targets[i, c, :]
                pred = predictions[i, c, :]
                
                is_common_channel = c in common_indices
                
                if is_common_channel:
                    loss_c = self.criterion(pred, target, channel_idx=c, is_common=True)
                    common_loss_i += loss_c
                    common_count += 1
                elif real_channels[c]:
                    loss_c = self.criterion(pred, target, channel_idx=c, is_common=False)
                    have_loss_i += loss_c
                    have_count += 1
            
            if common_count > 0:
                common_loss_i /= common_count
            if have_count > 0:
                have_loss_i /= have_count
            
            recon_loss_i = common_loss_i + have_loss_i
            cls_loss_i = ce_loss(logits[i:i+1], labels[i:i+1])
            
            total_loss += recon_loss_i + cls_loss_i
            common_loss += common_loss_i
            have_loss += have_loss_i
        
        return {
            'total_loss': total_loss / batch_size,
            'reconstruction_loss': (common_loss + have_loss) / batch_size,
            'common_loss': common_loss / batch_size,
            'have_loss': have_loss / batch_size,
            'classification_loss': ce_loss(logits, labels)
        }
    
    def _compute_standard_loss(self, predictions, targets, logits, labels, is_real_mask):
        """计算标准损失"""
        batch_size, C, T = predictions.shape
        total_recon_loss = 0.0
        
        for i in range(batch_size):
            real_channels = is_real_mask[i]
            recon_loss_i = 0.0
            real_count = 0
            
            for c in range(C):
                if real_channels[c]:
                    target = targets[i, c, :]
                    pred = predictions[i, c, :]
                    recon_loss_i += self.criterion(pred, target)
                    real_count += 1
            
            if real_count > 0:
                recon_loss_i /= real_count
            
            total_recon_loss += recon_loss_i
        
        ce_loss = nn.CrossEntropyLoss()
        cls_loss = ce_loss(logits, labels)
        
        return {
            'total_loss': total_recon_loss / batch_size + cls_loss,
            'reconstruction_loss': total_recon_loss / batch_size,
            'classification_loss': cls_loss
        }
    
    def _compute_accuracy(self, logits, labels):
        """计算分类准确率"""
        if logits is None or labels is None:
            return 0.0
        
        pred_classes = logits.argmax(dim=-1)
        return (pred_classes == labels).float().mean().item()
    
    def _compute_detailed_metrics(self, predictions_list, targets_list, labels_list, logits_list):
        """计算详细指标"""
        if not predictions_list:
            return {}
        
        # 拼接所有批次
        all_predictions = torch.cat(predictions_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        
        # 计算重建指标
        mse = torch.mean((all_predictions - all_targets) ** 2).item()
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        
        # 计算信噪比
        signal_power = torch.mean(all_targets ** 2).item()
        noise_power = torch.mean((all_predictions - all_targets) ** 2).item()
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae,
            'snr_db': snr,
            'signal_power': signal_power,
            'noise_power': noise_power
        }

def create_enhanced_validation_strategy(config):
    """创建增强验证策略"""
    
    patience = config.get('patience', 20)
    min_delta = config.get('min_delta', 1e-6)
    
    # 创建多个跟踪器
    trackers = {
        'total_loss': ValidationTracker(patience, min_delta, 'min'),
        'reconstruction_loss': ValidationTracker(patience, min_delta, 'min'),
        'classification_loss': ValidationTracker(patience, min_delta, 'min'),
        'accuracy': ValidationTracker(patience, min_delta, 'max'),
    }
    
    # 如果使用多模态损失，添加额外跟踪器
    if config.get('loss_config', {}).get('type') == 'multimodal':
        trackers.update({
            'common_loss': ValidationTracker(patience, min_delta, 'min'),
            'have_loss': ValidationTracker(patience, min_delta, 'min'),
        })
    
    return trackers

def enhanced_training_loop_with_validation(model, train_loader, val_loader, optimizer, 
                                          criterion, device, mask_indices, config):
    """增强的训练循环，包含完整验证策略"""
    
    # 创建验证组件
    val_loop = EnhancedValidationLoop(model, criterion, device, mask_indices)
    trackers = create_enhanced_validation_strategy(config)
    
    epochs = config.get('epochs', 100)
    log_dir = config.get('log_dir', 'Logs')
    ckpt_dir = config.get('ckpt_dir', 'Checkpoints')
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("🚀 开始增强验证策略训练")
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        from train import train_phased
        train_metrics = train_phased(
            model, train_loader, optimizer, criterion, device, mask_indices
        )
        
        # 验证阶段
        val_metrics = val_loop.evaluate(val_loader, return_detailed=(epoch % 10 == 0))
        
        # 更新跟踪器
        should_stop = False
        for metric_name, tracker in trackers.items():
            if metric_name in val_metrics:
                metric_should_stop = tracker.update(val_metrics[metric_name], epoch)
                if metric_name == 'total_loss':  # 主要指标
                    should_stop = metric_should_stop
        
        # 学习率调度
        scheduler.step(val_metrics['total_loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(ckpt_dir, 'best_model.pth'))
            print(f"💾 保存最佳模型 (Epoch {epoch}, Val Loss: {best_val_loss:.6f})")
        
        # 详细日志
        log_message = f"[ENHANCED] Epoch {epoch}: "
        log_message += f"train_loss={train_metrics.get('total_loss', 0):.6f}, "
        log_message += f"val_loss={val_metrics['total_loss']:.6f}, "
        log_message += f"val_acc={val_metrics.get('accuracy', 0):.4f}, "
        log_message += f"lr={current_lr:.6e}"
        
        if 'common_loss' in val_metrics:
            log_message += f", common_loss={val_metrics['common_loss']:.6f}"
            log_message += f", have_loss={val_metrics['have_loss']:.6f}"
        
        logging.info(log_message)
        print(log_message)
        
        # 早停检查
        if should_stop:
            total_tracker = trackers['total_loss']
            print(f"🛑 早停触发! 最佳epoch: {total_tracker.best_epoch}, "
                  f"最佳分数: {total_tracker.best_score:.6f}")
            break
        
        # 每10个epoch显示详细统计
        if epoch % 10 == 0:
            print(f"\n📊 Epoch {epoch} 验证统计:")
            for metric_name, tracker in trackers.items():
                stats = tracker.get_stats()
                print(f"  {metric_name}: 当前={val_metrics.get(metric_name, 0):.6f}, "
                      f"最佳={stats['best_score']:.6f} (Epoch {stats['best_epoch']}), "
                      f"停滞={stats['counter']}/{stats['patience']}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"🏆 加载最佳模型 (Val Loss: {best_val_loss:.6f})")
    
    return model, trackers

if __name__ == "__main__":
    print("增强验证策略模块")
    print("主要功能:")
    print("✅ 正确的验证集计算（不更新参数）")
    print("✅ 多指标早停策略")
    print("✅ 详细的验证指标跟踪")
    print("✅ 最佳模型自动保存")
    print("✅ 学习率调度优化")
    print("✅ 多模态损失支持")
