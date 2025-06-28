# monitoring_enhancements.py - 增强监控和调试功能

import torch
import logging
import numpy as np
from collections import defaultdict

class V15TrainingMonitor:
    """
    V15训练监控器 - 在不改变训练逻辑的情况下增加监控功能
    """
    
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.loss_history = defaultdict(list)
        self.accuracy_history = defaultdict(list)
        self.step_count = 0
        
    def log_batch_metrics(self, metrics_dict, epoch, batch_idx):
        """记录批次指标"""
        self.step_count += 1
        
        # 记录损失历史
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.loss_history[key].append(value)
        
        # 每隔一定步数打印详细信息
        if batch_idx % self.log_interval == 0:
            self.print_detailed_metrics(metrics_dict, epoch, batch_idx)
    
    def print_detailed_metrics(self, metrics, epoch, batch_idx):
        """打印详细指标"""
        print(f"\n[Epoch {epoch}, Batch {batch_idx}] 详细指标:")
        
        # 基础损失
        if 'recon_loss' in metrics:
            print(f"  重建损失: {metrics['recon_loss']:.6f}")
        if 'total_classification_loss' in metrics:
            print(f"  总分类损失: {metrics['total_classification_loss']:.6f}")
        
        # 准确率
        if 'input_accuracy' in metrics:
            print(f"  输入数据准确率: {metrics['input_accuracy']:.4f}")
        if 'enhanced_accuracy' in metrics:
            print(f"  增强数据准确率: {metrics['enhanced_accuracy']:.4f}")
        if 'accuracy_improvement' in metrics:
            print(f"  准确率改进: {metrics['accuracy_improvement']:.4f}")
        
        # 权重信息
        if 'dynamic_recon_weight' in metrics:
            print(f"  动态重建权重: {metrics['dynamic_recon_weight']:.4f}")
        if 'dynamic_cls_weight' in metrics:
            print(f"  动态分类权重: {metrics['dynamic_cls_weight']:.4f}")
    
    def check_training_health(self):
        """检查训练健康状况"""
        warnings = []
        
        # 检查损失是否发散
        if 'recon_loss' in self.loss_history:
            recent_losses = self.loss_history['recon_loss'][-10:]
            if len(recent_losses) >= 10:
                if recent_losses[-1] > recent_losses[0] * 2:
                    warnings.append("⚠️ 重建损失可能在发散")
        
        # 检查准确率是否停滞
        if 'input_accuracy' in self.loss_history:
            recent_acc = self.loss_history['input_accuracy'][-20:]
            if len(recent_acc) >= 20:
                acc_std = np.std(recent_acc)
                if acc_std < 0.01:
                    warnings.append("⚠️ 准确率可能陷入停滞")
        
        # 检查梯度爆炸
        if 'total_loss' in self.loss_history:
            recent_losses = self.loss_history['total_loss'][-5:]
            if len(recent_losses) >= 5:
                if any(loss > 100 for loss in recent_losses):
                    warnings.append("🚨 可能出现梯度爆炸！")
        
        if warnings:
            print("\n" + "="*50)
            print("训练健康检查:")
            for warning in warnings:
                print(warning)
            print("="*50)
        
        return len(warnings) == 0
    
    def suggest_adjustments(self):
        """基于监控数据建议调整"""
        suggestions = []
        
        if 'accuracy_improvement' in self.loss_history:
            recent_improvements = self.loss_history['accuracy_improvement'][-20:]
            if len(recent_improvements) >= 20:
                avg_improvement = np.mean(recent_improvements)
                
                if avg_improvement < -0.02:
                    suggestions.append("建议降低cls_improvement_weight到0.3")
                elif avg_improvement > 0.05:
                    suggestions.append("建议增加cls_improvement_weight到1.0")
        
        if 'recon_loss' in self.loss_history:
            recent_losses = self.loss_history['recon_loss'][-10:]
            if len(recent_losses) >= 10:
                if all(loss > 1.0 for loss in recent_losses):
                    suggestions.append("建议增加learning rate到0.001")
                elif all(loss < 0.01 for loss in recent_losses):
                    suggestions.append("建议降低learning rate到0.0003")
        
        if suggestions:
            print("\n📋 训练优化建议:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")

# 使用示例
"""
在您的train.py中添加：

# 在训练循环开始前
monitor = V15TrainingMonitor(log_interval=10)

# 在每个batch的损失计算后
metrics = {
    'recon_loss': recon_loss,
    'total_classification_loss': total_classification_loss,
    'input_accuracy': input_accuracy,
    'enhanced_accuracy': enhanced_accuracy,
    'accuracy_improvement': accuracy_improvement,
    'dynamic_recon_weight': dynamic_recon_weight,
    'dynamic_cls_weight': dynamic_cls_weight,
}
monitor.log_batch_metrics(metrics, epoch, batch_idx)

# 每隔一定epoch检查训练健康状况
if epoch % 10 == 0:
    is_healthy = monitor.check_training_health()
    if not is_healthy:
        monitor.suggest_adjustments()
"""