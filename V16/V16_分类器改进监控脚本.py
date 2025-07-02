# V16分类器改进效果监控脚本

"""
监控V16分类器改进后的性能表现，与V13进行对比分析
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from pathlib import Path

class ClassifierPerformanceMonitor:
    """V16分类器性能监控器"""
    
    def __init__(self, save_dir="V16/monitoring"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 监控指标
        self.metrics = {
            'epoch': [],
            'input_acc': [],
            'reconstructed_acc': [],
            'acc_improvement': [],
            'input_loss': [],
            'reconstructed_loss': [],
            'feature_quality': []
        }
        
    def log_epoch_metrics(self, epoch, input_acc, reconstructed_acc, 
                         input_loss=None, reconstructed_loss=None):
        """记录每个epoch的性能指标"""
        self.metrics['epoch'].append(epoch)
        self.metrics['input_acc'].append(input_acc)
        self.metrics['reconstructed_acc'].append(reconstructed_acc)
        self.metrics['acc_improvement'].append(reconstructed_acc - input_acc)
        
        if input_loss is not None:
            self.metrics['input_loss'].append(input_loss)
        if reconstructed_loss is not None:
            self.metrics['reconstructed_loss'].append(reconstructed_loss)
    
    def analyze_feature_quality(self, original_features, enhanced_features, labels, epoch):
        """分析特征质量改进效果"""
        
        # 1. 特征维度分析
        orig_dim = original_features.shape[1]
        enhanced_dim = enhanced_features.shape[1]
        
        print(f"Epoch {epoch} 特征维度分析:")
        print(f"  原始特征维度: {orig_dim}")
        print(f"  增强特征维度: {enhanced_dim}")
        print(f"  维度扩展倍数: {enhanced_dim / orig_dim:.1f}x")
        
        # 2. 特征分布可视化
        if epoch % 10 == 0:  # 每10个epoch可视化一次
            self._visualize_feature_distributions(
                original_features, enhanced_features, labels, epoch
            )
        
        # 3. 类别分离度分析
        orig_separability = self._compute_class_separability(original_features, labels)
        enhanced_separability = self._compute_class_separability(enhanced_features, labels)
        
        print(f"  类别分离度提升: {enhanced_separability/orig_separability:.2f}x")
        
        self.metrics['feature_quality'].append({
            'epoch': epoch,
            'orig_separability': orig_separability,
            'enhanced_separability': enhanced_separability,
            'improvement_ratio': enhanced_separability / orig_separability
        })
        
        return enhanced_separability / orig_separability
    
    def _compute_class_separability(self, features, labels):
        """计算类别分离度（基于类内类间距离比）"""
        unique_labels = torch.unique(labels)
        
        if len(unique_labels) < 2:
            return 0.0
        
        # 计算类内距离
        intra_class_dist = 0.0
        for label in unique_labels:
            mask = (labels == label)
            class_features = features[mask]
            if len(class_features) > 1:
                center = class_features.mean(dim=0)
                distances = torch.norm(class_features - center, dim=1)
                intra_class_dist += distances.mean().item()
        
        intra_class_dist /= len(unique_labels)
        
        # 计算类间距离
        centers = []
        for label in unique_labels:
            mask = (labels == label)
            center = features[mask].mean(dim=0)
            centers.append(center)
        
        inter_class_dist = 0.0
        count = 0
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = torch.norm(centers[i] - centers[j]).item()
                inter_class_dist += dist
                count += 1
        
        if count > 0:
            inter_class_dist /= count
        
        # 分离度 = 类间距离 / 类内距离
        if intra_class_dist > 0:
            return inter_class_dist / intra_class_dist
        else:
            return inter_class_dist
    
    def _visualize_feature_distributions(self, orig_features, enhanced_features, labels, epoch):
        """可视化特征分布"""
        
        # 使用t-SNE降维到2D
        if len(orig_features) > 1000:
            # 随机采样1000个样本以加速可视化
            indices = torch.randperm(len(orig_features))[:1000]
            orig_vis = orig_features[indices]
            enhanced_vis = enhanced_features[indices]
            labels_vis = labels[indices]
        else:
            orig_vis = orig_features
            enhanced_vis = enhanced_features
            labels_vis = labels
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(orig_vis)-1))
        
        orig_2d = tsne.fit_transform(orig_vis.cpu().numpy())
        enhanced_2d = tsne.fit_transform(enhanced_vis.cpu().numpy())
        
        # 绘制对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始特征分布
        scatter1 = ax1.scatter(orig_2d[:, 0], orig_2d[:, 1], 
                              c=labels_vis.cpu().numpy(), cmap='viridis', alpha=0.7)
        ax1.set_title(f'原始特征分布 (Epoch {epoch})')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # 增强特征分布
        scatter2 = ax2.scatter(enhanced_2d[:, 0], enhanced_2d[:, 1], 
                              c=labels_vis.cpu().numpy(), cmap='viridis', alpha=0.7)
        ax2.set_title(f'增强特征分布 (Epoch {epoch})')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'feature_distribution_epoch_{epoch}.png', dpi=300)
        plt.close()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.metrics['epoch']:
            print("没有可绘制的数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 准确率曲线
        axes[0,0].plot(self.metrics['epoch'], self.metrics['input_acc'], 
                      label='输入数据准确率', marker='o', alpha=0.7)
        axes[0,0].plot(self.metrics['epoch'], self.metrics['reconstructed_acc'], 
                      label='重建数据准确率', marker='s', alpha=0.7)
        axes[0,0].set_title('分类准确率对比')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('准确率')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 准确率改进
        axes[0,1].plot(self.metrics['epoch'], self.metrics['acc_improvement'], 
                      color='red', marker='d', alpha=0.7)
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0,1].set_title('准确率改进 (重建 - 输入)')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('准确率差值')
        axes[0,1].grid(True, alpha=0.3)
        
        # 损失对比（如果有）
        if self.metrics['input_loss']:
            axes[1,0].plot(self.metrics['epoch'], self.metrics['input_loss'], 
                          label='输入数据损失', marker='o', alpha=0.7)
            axes[1,0].plot(self.metrics['epoch'], self.metrics['reconstructed_loss'], 
                          label='重建数据损失', marker='s', alpha=0.7)
            axes[1,0].set_title('分类损失对比')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('损失值')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 特征质量改进（如果有）
        if self.metrics['feature_quality']:
            quality_data = self.metrics['feature_quality']
            epochs = [item['epoch'] for item in quality_data]
            ratios = [item['improvement_ratio'] for item in quality_data]
            
            axes[1,1].plot(epochs, ratios, color='green', marker='*', alpha=0.7)
            axes[1,1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            axes[1,1].set_title('特征质量改进比率')
            axes[1,1].set_xlabel('Epoch')  
            axes[1,1].set_ylabel('改进比率')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300)
        plt.show()
    
    def generate_performance_report(self, v13_baseline=None):
        """生成性能报告"""
        if not self.metrics['epoch']:
            return
        
        # 计算关键统计数据
        final_input_acc = self.metrics['input_acc'][-1]
        final_reconstructed_acc = self.metrics['reconstructed_acc'][-1]
        final_improvement = self.metrics['acc_improvement'][-1]
        
        max_input_acc = max(self.metrics['input_acc'])
        max_reconstructed_acc = max(self.metrics['reconstructed_acc'])
        max_improvement = max(self.metrics['acc_improvement'])
        
        report = f"""
# V16分类器改进效果报告

## 最终性能
- 输入数据分类准确率: {final_input_acc:.4f}
- 重建数据分类准确率: {final_reconstructed_acc:.4f}
- 准确率改进: {final_improvement:.4f}

## 最佳性能
- 最高输入数据准确率: {max_input_acc:.4f}
- 最高重建数据准确率: {max_reconstructed_acc:.4f}
- 最大准确率改进: {max_improvement:.4f}

## 改进分析
- 特征维度扩展: 4倍 (均值+标准差+最大值+最小值)
- 网络结构升级: 3层MLP + BatchNorm + Dropout
"""
        
        if v13_baseline:
            report += f"""
## 与V13对比
- V13基线准确率: {v13_baseline:.4f}
- V16改进后准确率: {final_input_acc:.4f}
- 相对V13的提升: {(final_input_acc - v13_baseline):.4f}
"""
        
        if self.metrics['feature_quality']:
            avg_quality_improvement = np.mean([
                item['improvement_ratio'] for item in self.metrics['feature_quality']
            ])
            report += f"""
## 特征质量分析
- 平均类别分离度改进: {avg_quality_improvement:.2f}x
- 特征空间优化效果: {'显著' if avg_quality_improvement > 1.5 else '中等' if avg_quality_improvement > 1.2 else '轻微'}
"""
        
        # 保存报告
        with open(self.save_dir / 'performance_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

# 使用示例
def monitor_training_with_enhanced_features():
    """在训练过程中使用性能监控器"""
    monitor = ClassifierPerformanceMonitor()
    
    # 在训练循环中调用
    # for epoch in range(epochs):
    #     # ... 训练代码 ...
    #     
    #     # 提取特征进行分析
    #     with torch.no_grad():
    #         original_features = batch.mean(dim=2)  # 原始池化特征
    #         enhanced_features = torch.cat([
    #             batch.mean(dim=2), batch.std(dim=2), 
    #             batch.max(dim=2)[0], batch.min(dim=2)[0]
    #         ], dim=1)  # 增强特征
    #         
    #         # 分析特征质量
    #         monitor.analyze_feature_quality(
    #             original_features, enhanced_features, labels, epoch
    #         )
    #     
    #     # 记录性能指标
    #     monitor.log_epoch_metrics(
    #         epoch, input_acc, reconstructed_acc, 
    #         input_loss, reconstructed_loss
    #     )
    
    # 生成最终报告
    # monitor.plot_training_curves()
    # report = monitor.generate_performance_report(v13_baseline=0.85)
    # print(report)
    
    pass

if __name__ == "__main__":
    print("V16分类器改进效果监控脚本已就绪")
    print("请在训练代码中集成ClassifierPerformanceMonitor来监控改进效果")
