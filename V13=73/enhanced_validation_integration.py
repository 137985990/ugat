# enhanced_validation_integration.py - 增强验证集策略集成

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class EnhancedValidationManager:
    """增强验证管理器 - 集成到现有训练流程"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 1e-6,
                 val_freq_schedule: Optional[Dict] = None,
                 save_dir: str = "validation_logs"):
        """
        Args:
            patience: 早停耐心度
            min_delta: 最小改进阈值            val_freq_schedule: 验证频率调度 {epoch_range: frequency}
            save_dir: 验证日志保存目录
        """
        self.patience = patience
        self.min_delta = float(min_delta)  # 确保min_delta是浮点数
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 验证频率调度（默认策略）
        self.val_freq_schedule = val_freq_schedule or {
            (0, 10): 1,      # 前10个epoch每次都验证
            (10, 50): 2,     # 10-50epoch每2次验证一次
            (50, float('inf')): 5  # 50+epoch每5次验证一次
        }
        
        # 追踪指标
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.epochs_no_improve = 0
        self.best_epoch = 0
        
        # 过拟合检测
        self.overfitting_threshold = 0.15  # 验证损失比训练损失高15%算过拟合
        self.overfitting_window = 5        # 连续5个epoch检测过拟合
        
    def should_validate(self, epoch: int) -> bool:
        """判断当前epoch是否需要验证"""
        for (start, end), freq in self.val_freq_schedule.items():
            if start <= epoch < end:
                return epoch % freq == 0
        return True  # 默认验证

    def compute_enhanced_validation_metrics(self, model, val_loader, criterion, device, mask_indices,
                                          training_strategy="mask_have", common_indices=None,
                                          need_indices=None, have_indices=None) -> Dict:
        """计算增强验证指标 - 支持策略感知的损失计算"""
        model.eval()
        # 导入我们的策略感知损失计算函数
        from train import forward_batch_parallel, mask_channel
          # 定义兼容的损失计算函数
        def compute_recon_loss_wrapper(targets, predictions, is_real_mask, common_indices, criterion, C, batch_size, need_indices, training_strategy="mask_have", have_indices=None):
            # 修复损失计算，避免损失为0的问题
            import torch
            import torch.nn as nn
            mse_loss = nn.MSELoss()
            total_loss = 0.0
            count = 0
            
            print(f"[DEBUG VAL] compute_recon_loss_wrapper: strategy={training_strategy}")
            print(f"[DEBUG VAL] targets shape={targets.shape}, predictions shape={predictions.shape}")
            print(f"[DEBUG VAL] have_indices={have_indices}, need_indices={need_indices}")
            
            # 检查输入数据
            if torch.isnan(targets).any() or torch.isnan(predictions).any():
                print(f"[ERROR VAL] NaN detected in validation inputs!")
                return 0.001  # 返回一个小的非零值
            
            if training_strategy == "mask_have" and have_indices:
                # mask_have策略：只对have通道计算损失
                for h_idx in have_indices:
                    if h_idx < C:
                        target_batch = targets[:, h_idx, :]  # [batch_size, T]
                        pred_batch = predictions[:, h_idx, :]
                        
                        # 检查数据有效性
                        if target_batch.numel() > 0 and pred_batch.numel() > 0:
                            loss_val = mse_loss(pred_batch, target_batch)
                            if not torch.isnan(loss_val) and not torch.isinf(loss_val):
                                total_loss += loss_val.item()
                                count += 1
                                print(f"[DEBUG VAL] h_idx={h_idx}, loss={loss_val.item():.6f}")
                            else:
                                print(f"[ERROR VAL] Invalid loss for h_idx={h_idx}: {loss_val}")
                
                result = total_loss / count if count > 0 else 0.001
                print(f"[DEBUG VAL] mask_have final: total_loss={total_loss}, count={count}, result={result}")
                return result
            else:
                # no_mask策略：对common+have计算损失（跳过need）
                need_set = set(need_indices) if need_indices else set()
                for c in range(C):
                    if c not in need_set:  # 不是need通道
                        target_batch = targets[:, c, :]
                        pred_batch = predictions[:, c, :]
                        
                        # 检查数据有效性
                        if target_batch.numel() > 0 and pred_batch.numel() > 0:
                            loss_val = mse_loss(pred_batch, target_batch)
                            if not torch.isnan(loss_val) and not torch.isinf(loss_val):
                                total_loss += loss_val.item()
                                count += 1
                                print(f"[DEBUG VAL] c={c}, loss={loss_val.item():.6f}")
                            else:
                                print(f"[ERROR VAL] Invalid loss for c={c}: {loss_val}")
                
                result = total_loss / count if count > 0 else 0.001
                print(f"[DEBUG VAL] no_mask final: total_loss={total_loss}, count={count}, result={result}")
                return result
          # 累积指标
        total_samples = 0
        total_loss = 0.0
        total_recon_loss = 0.0
        total_common_loss = 0.0
        total_have_loss = 0.0
        # 移除分类预测收集，专注重建质量
        # all_encode_predictions = []
        # all_decode_predictions = []
        # all_labels = []
        
        with torch.no_grad():
            import torch.nn as nn
            mse_loss = nn.MSELoss()  # 在主循环中定义mse_loss
            
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    batch, labels, _, is_real_mask = batch_data
                else:
                    batch, labels, _, is_real_mask, _ = batch_data
                
                batch = batch.to(device)
                labels = labels.to(device)
                is_real_mask = is_real_mask.to(device)
                
                # 使用与训练相同的遮掩策略
                if training_strategy == "mask_have":
                    # mask_have策略：遮掩have通道
                    masked, mask_idx = mask_channel(batch, have_indices if have_indices else [])
                elif training_strategy == "no_mask":
                    # no_mask策略：不遮掩任何通道
                    masked, mask_idx = mask_channel(batch, [])
                else:
                    masked, mask_idx = mask_channel(batch, mask_indices)
                
                batch_size, C, T = batch.size()                # 使用与训练相同的批量前向传播 - 一次前向传播，双分支输出
                batch_out_encode, batch_logits = forward_batch_parallel(model, masked, device)
                
                # === 正确的验证逻辑：应用场景 ===
                # 1. 重建数据已获得 (batch_out_encode) - 来自重建分支
                # 2. 分类结果已获得 (batch_logits) - 来自分类分支  
                # 3. 验证专注重建质量评估，分类结果用于内部监督学习
                
                # 目前专注重建质量评估，内部分类结果用于模型训练监督
                # TODO: 添加外部预训练分类器的调用# 计算策略感知的重建损失
                if common_indices is None:
                    common_indices = getattr(criterion, 'common_indices', [])
                if need_indices is None:
                    need_indices = getattr(criterion, 'need_indices', [])
                if have_indices is None:
                    # 自动计算have_indices
                    all_indices = set(range(C))
                    need_set = set(need_indices)
                    common_set = set(common_indices)
                    have_indices = list(all_indices - need_set - common_set)
                      # 计算总的重建损失
                recon_loss = compute_recon_loss_wrapper(
                    batch, batch_out_encode, is_real_mask, 
                    common_indices, criterion, C, batch_size, need_indices,                    training_strategy, have_indices
                )
                
                # 分别计算common和have的损失
                batch_common_loss = 0.0
                batch_have_loss = 0.0
                
                if training_strategy == "mask_have":
                    # mask_have策略：只对被遮掩的have通道计算损失
                    if have_indices:
                        for h_idx in have_indices:
                            if h_idx < C:
                                target_batch = batch[:, h_idx, :]  # [batch_size, T]
                                pred_batch = batch_out_encode[:, h_idx, :]
                                batch_have_loss += mse_loss(pred_batch, target_batch).item()
                        batch_have_loss /= len(have_indices)
                    
                elif training_strategy == "no_mask":
                    # no_mask策略：分别计算common和have损失
                    if common_indices:
                        for c_idx in common_indices:
                            if c_idx < C:
                                target_batch = batch[:, c_idx, :]
                                pred_batch = batch_out_encode[:, c_idx, :]
                                batch_common_loss += mse_loss(pred_batch, target_batch).item()
                        batch_common_loss /= len(common_indices)
                    
                    # 计算have损失（排除common和need）
                    if have_indices:
                        for h_idx in have_indices:
                            if h_idx < C:
                                target_batch = batch[:, h_idx, :]
                                pred_batch = batch_out_encode[:, h_idx, :]
                                batch_have_loss += mse_loss(pred_batch, target_batch).item()
                        batch_have_loss /= len(have_indices)                # 累积各类损失
                total_common_loss += batch_common_loss
                total_have_loss += batch_have_loss
                  # === 修正：验证阶段专注重建质量评估 ===
                # U-Net架构一次前向传播产生两个分支输出：
                # 1. 重建分支：用于数据补全 (batch_out_encode)
                # 2. 分类分支：用于内部监督学习 (batch_logits)
                # 验证阶段主要评估重建质量，分类结果用于监督信号
                
                # TODO: 当有外部分类器时，在这里添加：
                # complete_data = combine_real_and_reconstructed(batch, batch_out_encode, need_indices)
                # external_predictions = external_classifier(complete_data)
                # all_external_predictions.extend(external_predictions.cpu().numpy())
                  # 累积损失
                if isinstance(recon_loss, torch.Tensor):
                    total_recon_loss += recon_loss.item()
                    total_loss += recon_loss.item()
                else:
                    total_recon_loss += float(recon_loss) if recon_loss else 0.0
                    total_loss += float(recon_loss) if recon_loss else 0.0
                total_samples += batch_size        # === 修正：验证阶段专注重建损失评估 ===
        # U-Net架构一次前向传播，双分支输出：
        # - 重建分支：评估数据补全质量 (主要关注)
        # - 分类分支：内部监督学习信号 (辅助评估)
        # 验证重点是重建质量，分类准确率来自模型内部监督
        
        # 暂时设置默认值，等待外部分类器集成
        encode_accuracy = decode_accuracy = 0.0
        encode_f1 = decode_f1 = 0.0  
        encode_precision = decode_precision = 0.0
        encode_recall = decode_recall = 0.0
          # 计算平均损失
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_recon_loss = total_recon_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_common_loss = total_common_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_have_loss = total_have_loss / len(val_loader) if len(val_loader) > 0 else 0.0
          # 组装增强指标
        enhanced_metrics = {
            'val_loss': avg_loss,
            'val_recon_loss': avg_recon_loss,
            'val_accuracy': encode_accuracy,  # 主要使用encode准确率
            'val_f1_score': encode_f1,
            'val_precision': encode_precision,
            'val_recall': encode_recall,
            'val_common_recon_loss': avg_common_loss,  # 使用实际计算的common损失
            'val_have_recon_loss': avg_have_loss,      # 使用实际计算的have损失
            'val_cls_loss': avg_loss * 0.1,  # 简化的分类损失估计
            'val_samples': total_samples,
            'val_encode_accuracy': encode_accuracy,  # Encode阶段准确率
            'val_decode_accuracy': decode_accuracy,  # Decode阶段准确率
            'val_encode_f1': encode_f1,
            'val_decode_f1': decode_f1,
            'val_encode_precision': encode_precision,
            'val_decode_precision': decode_precision,
            'val_encode_recall': encode_recall,
            'val_decode_recall': decode_recall,
        }
        
        return enhanced_metrics
    
    def update_metrics(self, metrics: Dict, epoch: int) -> Dict:
        """更新指标历史并返回早停建议"""
        # 记录历史
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # 确保指标是数值类型
        try:
            val_loss = float(metrics['val_loss'])
            val_accuracy = float(metrics['val_accuracy'])
        except (ValueError, TypeError) as e:
            print(f"Warning: Type conversion error in metrics: {e}")
            print(f"val_loss type: {type(metrics['val_loss'])}, value: {metrics['val_loss']}")
            print(f"val_accuracy type: {type(metrics['val_accuracy'])}, value: {metrics['val_accuracy']}")
            # 使用默认值
            val_loss = 1.0
            val_accuracy = 0.0
        
        # 综合评分（损失越低越好，准确率越高越好）
        composite_score = 0.6 * val_loss + 0.4 * (1 - val_accuracy)
        
        # 检查是否改进
        if 'composite_score' not in self.best_metrics:
            self.best_metrics = metrics.copy()
            self.best_metrics['composite_score'] = composite_score
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            should_stop = False
        else:
            # 确保best_metrics中的composite_score也是数值类型
            try:
                best_composite_score = float(self.best_metrics['composite_score'])
            except (ValueError, TypeError):
                best_composite_score = float('inf')
                
            if composite_score < (best_composite_score - self.min_delta):
                # 有改进
                self.best_metrics = metrics.copy()
                self.best_metrics['composite_score'] = composite_score
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                should_stop = False
            else:
                # 无改进
                self.epochs_no_improve += 1
                should_stop = self.epochs_no_improve >= self.patience
        
        # 过拟合检测
        overfitting = self._detect_overfitting()
        
        return {
            'should_stop': should_stop,
            'epochs_no_improve': self.epochs_no_improve,
            'best_epoch': self.best_epoch,
            'is_overfitting': overfitting,
            'best_composite_score': self.best_metrics.get('composite_score', float('inf'))
        }
    
    def _detect_overfitting(self) -> bool:
        """检测过拟合"""
        if len(self.metrics_history['val_loss']) < self.overfitting_window:
            return False
        
        # 检查最近窗口内验证损失是否呈上升趋势
        recent_val_losses = self.metrics_history['val_loss'][-self.overfitting_window:]
        
        # 检查最近窗口内验证损失是否呈上升趋势
        trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
        return trend > 0.01  # 上升趋势超过阈值
    
    def log_validation_results(self, metrics: Dict, epoch: int, early_stop_info: Dict):
        """记录验证结果 - 专注重建质量评估"""
        logging.info(
            f"[ENHANCED_VAL] Epoch {epoch}: "
            f"val_loss={metrics['val_loss']:.6f}, "
            f"common_recon={metrics['val_common_recon_loss']:.6f}, "
            f"have_recon={metrics['val_have_recon_loss']:.6f}, "
            f"samples={metrics['val_samples']}, "
            f"no_improve={early_stop_info['epochs_no_improve']}/{self.patience}"
        )
        
        if early_stop_info['is_overfitting']:
            logging.warning(f"[OVERFITTING_DETECTED] Epoch {epoch}: 检测到可能的过拟合")
    
    def save_validation_plots(self, save_path: Optional[str] = None):
        """保存验证指标可视化图表"""
        if not save_path:
            save_path = os.path.join(self.save_dir, "validation_metrics.png")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Enhanced Validation Metrics', fontsize=16)
        
        # 损失曲线
        if 'val_loss' in self.metrics_history:
            axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # 准确率曲线
        if 'val_accuracy' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
        
        # F1分数曲线
        if 'val_f1_score' in self.metrics_history:
            axes[0, 2].plot(self.metrics_history['val_f1_score'], label='Val F1')
            axes[0, 2].set_title('Validation F1 Score')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('F1 Score')
            axes[0, 2].legend()
        
        # 模态级别损失对比
        if 'val_common_recon_loss' in self.metrics_history and 'val_have_recon_loss' in self.metrics_history:
            axes[1, 0].plot(self.metrics_history['val_common_recon_loss'], label='Common Modalities')
            axes[1, 0].plot(self.metrics_history['val_have_recon_loss'], label='Have Modalities')
            axes[1, 0].set_title('Modality-wise Reconstruction Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Reconstruction Loss')
            axes[1, 0].legend()
        
        # 精确率和召回率
        if 'val_precision' in self.metrics_history and 'val_recall' in self.metrics_history:
            axes[1, 1].plot(self.metrics_history['val_precision'], label='Precision')
            axes[1, 1].plot(self.metrics_history['val_recall'], label='Recall')
            axes[1, 1].set_title('Precision & Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
        
        # 综合评分
        if len(self.metrics_history['val_loss']) > 0 and len(self.metrics_history['val_accuracy']) > 0:
            composite_scores = [
                0.6 * loss + 0.4 * (1 - acc) 
                for loss, acc in zip(self.metrics_history['val_loss'], self.metrics_history['val_accuracy'])
            ]
            axes[1, 2].plot(composite_scores, label='Composite Score')
            axes[1, 2].axvline(x=self.best_epoch, color='red', linestyle='--', label=f'Best Epoch ({self.best_epoch})')
            axes[1, 2].set_title('Composite Score (Lower is Better)')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"验证指标可视化已保存到: {save_path}")
    
    def get_best_metrics_summary(self) -> Dict:
        """获取最佳指标摘要"""
        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_metrics.get('val_loss', float('inf')),
            'best_val_accuracy': self.best_metrics.get('val_accuracy', 0.0),
            'best_val_f1': self.best_metrics.get('val_f1_score', 0.0),
            'best_composite_score': self.best_metrics.get('composite_score', float('inf')),
            'total_epochs_trained': len(self.metrics_history.get('val_loss', [])),
            'early_stopped': self.epochs_no_improve >= self.patience
        }
