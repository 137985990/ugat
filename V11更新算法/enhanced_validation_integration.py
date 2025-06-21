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
            min_delta: 最小改进阈值
            val_freq_schedule: 验证频率调度 {epoch_range: frequency}
            save_dir: 验证日志保存目录
        """
        self.patience = patience
        self.min_delta = min_delta
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
    
    def compute_enhanced_validation_metrics(self, 
                                          model, 
                                          val_loader, 
                                          criterion, 
                                          device, 
                                          mask_indices) -> Dict:
        """计算增强验证指标"""
        model.eval()
        
        # 累积指标
        total_samples = 0
        total_losses = defaultdict(float)
        all_predictions = []
        all_labels = []
        all_logits = []
        
        # 模态级别的损失
        common_losses = []
        have_losses = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 4:
                    batch, labels, _, is_real_mask = batch_data
                else:
                    batch, labels, _, is_real_mask, _ = batch_data
                
                batch = batch.to(device)
                labels = labels.to(device)
                is_real_mask = is_real_mask.to(device)
                
                # 应用mask
                from train import mask_channel
                masked, mask_idx = mask_channel(batch, mask_indices)
                
                batch_size, C, T = batch.size()
                batch_logits = []
                
                # 逐样本处理
                for i in range(batch_size):
                    window = masked[i].t()
                    out, logits = model(window)
                    batch_logits.append(logits)
                    
                    # 计算模态级别损失
                    common_indices = getattr(criterion, 'common_indices', [])
                    sample_common_loss = 0.0
                    sample_have_loss = 0.0
                    common_count = 0
                    have_count = 0
                    
                    for c in range(C):
                        target = batch[i, c, :]
                        pred = out[c, :]
                        
                        if c in common_indices:
                            sample_common_loss += nn.MSELoss()(pred, target).item()
                            common_count += 1
                        elif is_real_mask[c]:
                            sample_have_loss += nn.MSELoss()(pred, target).item()
                            have_count += 1
                    
                    if common_count > 0:
                        common_losses.append(sample_common_loss / common_count)
                    if have_count > 0:
                        have_losses.append(sample_have_loss / have_count)
                
                # 收集预测和标签用于分类指标
                batch_logits = torch.stack(batch_logits)
                predictions = torch.argmax(batch_logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(batch_logits.cpu().numpy())
                
                total_samples += batch_size
        
        # 计算分类指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # 计算标准验证损失（复用原有逻辑）
        from train import eval_loop
        val_loss, val_recon, _, _ = eval_loop(model, val_loader, criterion, device, mask_indices)
        
        # 组装增强指标
        enhanced_metrics = {
            'val_loss': val_loss,
            'val_recon_loss': val_recon,
            'val_accuracy': accuracy,
            'val_f1_score': f1,
            'val_precision': precision,
            'val_recall': recall,
            'val_common_recon_loss': np.mean(common_losses) if common_losses else 0.0,
            'val_have_recon_loss': np.mean(have_losses) if have_losses else 0.0,
            'val_samples': total_samples
        }
        
        return enhanced_metrics
    
    def update_metrics(self, metrics: Dict, epoch: int) -> Dict:
        """更新指标历史并返回早停建议"""
        # 记录历史
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # 综合评分（损失越低越好，准确率越高越好）
        composite_score = 0.6 * metrics['val_loss'] + 0.4 * (1 - metrics['val_accuracy'])
        
        # 检查是否改进
        if 'composite_score' not in self.best_metrics:
            self.best_metrics = metrics.copy()
            self.best_metrics['composite_score'] = composite_score
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            should_stop = False
        else:
            if composite_score < (self.best_metrics['composite_score'] - self.min_delta):
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
        
        # 需要训练损失历史来比较
        # 这里简化为检查验证损失是否持续增长
        recent_val_losses = self.metrics_history['val_loss'][-self.overfitting_window:]
        
        # 检查最近窗口内验证损失是否呈上升趋势
        trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
        return trend > 0.01  # 上升趋势超过阈值
    
    def log_validation_results(self, metrics: Dict, epoch: int, early_stop_info: Dict):
        """记录验证结果"""
        logging.info(
            f"[ENHANCED_VAL] Epoch {epoch}: "
            f"val_loss={metrics['val_loss']:.6f}, "
            f"val_acc={metrics['val_accuracy']:.4f}, "
            f"val_f1={metrics['val_f1_score']:.4f}, "
            f"common_recon={metrics['val_common_recon_loss']:.6f}, "
            f"have_recon={metrics['val_have_recon_loss']:.6f}, "
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


def integrate_enhanced_validation_to_training():
    """集成增强验证到现有训练流程的示例代码"""
    
    # 在train.py的主循环中替换验证部分：
    example_code = '''
    # 在训练开始前初始化增强验证管理器
    enhanced_val_manager = EnhancedValidationManager(
        patience=config.get('patience', 20),
        save_dir=os.path.join(log_dir_full, 'enhanced_validation')
    )
    
    for epoch in range(1, epochs + 1):
        # 训练步骤
        train_loss, train_recon, train_cls, train_acc = train_phased(
            model, train_loader, optimizer, criterion, device, mask_indices
        )
        
        # 检查是否需要验证
        if enhanced_val_manager.should_validate(epoch):
            # 计算增强验证指标
            val_metrics = enhanced_val_manager.compute_enhanced_validation_metrics(
                model, val_loader, criterion, device, mask_indices
            )
            
            # 更新指标并获取早停建议
            early_stop_info = enhanced_val_manager.update_metrics(val_metrics, epoch)
            
            # 记录结果
            enhanced_val_manager.log_validation_results(val_metrics, epoch, early_stop_info)
            
            # TensorBoard记录
            writer.add_scalar('Enhanced_Val/Loss', val_metrics['val_loss'], epoch)
            writer.add_scalar('Enhanced_Val/Accuracy', val_metrics['val_accuracy'], epoch)
            writer.add_scalar('Enhanced_Val/F1_Score', val_metrics['val_f1_score'], epoch)
            writer.add_scalar('Enhanced_Val/Common_Recon', val_metrics['val_common_recon_loss'], epoch)
            writer.add_scalar('Enhanced_Val/Have_Recon', val_metrics['val_have_recon_loss'], epoch)
            
            # 学习率调度
            scheduler.step(val_metrics['val_loss'])
            
            # 保存最佳模型
            if epoch == enhanced_val_manager.best_epoch:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                logging.info(f"Saved best model at epoch {epoch}")
            
            # 早停检查
            if early_stop_info['should_stop']:
                logging.info(f"Enhanced early stopping triggered at epoch {epoch}")
                break
        
        # 常规日志记录
        logging.info(f"[TRAIN] Epoch {epoch}: train_loss={train_loss:.6f}, train_acc={train_acc:.4f}")
    
    # 训练结束后保存可视化
    enhanced_val_manager.save_validation_plots()
    best_summary = enhanced_val_manager.get_best_metrics_summary()
    logging.info(f"Training completed. Best metrics: {best_summary}")
    '''
    
    return example_code

if __name__ == "__main__":
    print("Enhanced Validation Integration 增强验证集策略集成")
    print("=" * 60)
    print("主要功能:")
    print("1. 智能验证频率调度")
    print("2. 多维度验证指标计算")
    print("3. 过拟合检测")
    print("4. 综合评分早停策略")
    print("5. 可视化验证指标")
    print("6. 模态级别性能分析")
    print("=" * 60)
    print("使用方法：在train.py中导入并替换验证部分即可")
