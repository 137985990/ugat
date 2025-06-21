# attention_visualizer.py - 注意力权重可视化模块

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

class AttentionExtractor(nn.Module):
    """注意力权重提取器"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        self.register_hooks()
    
    def register_hooks(self):
        """注册钩子函数来捕获注意力权重"""
        def get_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attention_weights') or len(output) > 1:
                    # 如果模块返回注意力权重
                    if isinstance(output, tuple) and len(output) > 1:
                        self.attention_maps[name] = output[1].detach().cpu()
                    elif hasattr(module, 'attention_weights'):
                        self.attention_maps[name] = module.attention_weights.detach().cpu()
            return hook
        
        # 为所有GAT层注册钩子
        for name, module in self.model.named_modules():
            if 'gat' in name.lower() or 'attention' in name.lower():
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, x):
        """前向传播并收集注意力权重"""
        self.attention_maps.clear()
        output = self.model(x)
        return output, self.attention_maps

class AttentionVisualizer:
    """注意力可视化器"""
    
    def __init__(self, save_dir: str = "attention_plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_attention_matrix(self, attention_weights: torch.Tensor, 
                            layer_name: str, head_idx: int = 0,
                            save_path: Optional[str] = None) -> None:
        """
        绘制注意力权重矩阵热力图
        
        Args:
            attention_weights: [num_heads, num_edges] or [num_edges] 注意力权重
            layer_name: 层名称
            head_idx: 注意力头索引
            save_path: 保存路径
        """
        if attention_weights.dim() == 2:
            # 多头注意力
            weights = attention_weights[head_idx].numpy()
        else:
            # 单头注意力
            weights = attention_weights.numpy()
        
        # 重构为矩阵形式（这里需要根据图结构调整）
        seq_len = int(np.sqrt(len(weights))) if len(weights) > 0 else 1
        if seq_len * seq_len != len(weights):
            # 如果不是完全图，使用稀疏表示
            matrix = np.zeros((seq_len, seq_len))
            # 简化处理：将权重分布到对角线附近
            for i, w in enumerate(weights[:seq_len]):
                if i < seq_len:
                    matrix[i, i] = w
        else:
            matrix = weights.reshape(seq_len, seq_len)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, 
                   annot=True if seq_len <= 10 else False,
                   fmt='.3f',
                   cmap='Blues',
                   cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f'{layer_name} - Head {head_idx} Attention Map', fontsize=14, fontweight='bold')
        plt.xlabel('Target Position', fontsize=12)
        plt.ylabel('Source Position', fontsize=12)
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f'{layer_name}_head{head_idx}_attention.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 注意力图已保存: {save_path}")
    
    def plot_attention_distribution(self, attention_maps: Dict[str, torch.Tensor],
                                  save_path: Optional[str] = None) -> None:
        """
        绘制所有层的注意力分布统计
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        layer_names = list(attention_maps.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
        
        # 1. 注意力权重分布直方图
        axes[0].set_title('Attention Weight Distribution', fontsize=12, fontweight='bold')
        for i, (layer_name, weights) in enumerate(attention_maps.items()):
            if weights.numel() > 0:
                flat_weights = weights.flatten().numpy()
                axes[0].hist(flat_weights, bins=30, alpha=0.6, 
                           label=layer_name, color=colors[i], density=True)
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 注意力权重方差（衡量注意力集中度）
        axes[1].set_title('Attention Concentration (Variance)', fontsize=12, fontweight='bold')
        variances = []
        for layer_name, weights in attention_maps.items():
            if weights.numel() > 0:
                var = weights.var().item()
                variances.append(var)
            else:
                variances.append(0)
        
        bars = axes[1].bar(range(len(layer_names)), variances, color=colors)
        axes[1].set_xticks(range(len(layer_names)))
        axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[1].set_ylabel('Variance')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{var:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 注意力权重最大值（衡量峰值强度）
        axes[2].set_title('Maximum Attention Weight', fontsize=12, fontweight='bold')
        max_weights = []
        for layer_name, weights in attention_maps.items():
            if weights.numel() > 0:
                max_w = weights.max().item()
                max_weights.append(max_w)
            else:
                max_weights.append(0)
        
        bars = axes[2].bar(range(len(layer_names)), max_weights, color=colors)
        axes[2].set_xticks(range(len(layer_names)))
        axes[2].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[2].set_ylabel('Max Weight')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # 4. 注意力熵（衡量注意力分散程度）
        axes[3].set_title('Attention Entropy', fontsize=12, fontweight='bold')
        entropies = []
        for layer_name, weights in attention_maps.items():
            if weights.numel() > 0:
                # 计算熵
                probs = torch.softmax(weights.flatten(), dim=0)
                entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                entropies.append(entropy)
            else:
                entropies.append(0)
        
        bars = axes[3].bar(range(len(layer_names)), entropies, color=colors)
        axes[3].set_xticks(range(len(layer_names)))
        axes[3].set_xticklabels(layer_names, rotation=45, ha='right')
        axes[3].set_ylabel('Entropy')
        axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'attention_analysis_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 注意力分析图已保存: {save_path}")
    
    def plot_temporal_attention(self, attention_weights: torch.Tensor, 
                              time_steps: List[int], layer_name: str,
                              save_path: Optional[str] = None) -> None:
        """
        绘制时间步注意力模式
        """
        if attention_weights.numel() == 0:
            return
        
        plt.figure(figsize=(12, 6))
        
        # 假设attention_weights是时间步之间的注意力
        if attention_weights.dim() == 2:
            # 多头：取平均
            weights = attention_weights.mean(dim=0).numpy()
        else:
            weights = attention_weights.numpy()
        
        # 创建时间步注意力图
        seq_len = len(time_steps)
        if len(weights) >= seq_len:
            # 重新整理为时间步矩阵
            time_attention = weights[:seq_len]
            
            plt.subplot(1, 2, 1)
            plt.plot(time_steps, time_attention, 'b-o', linewidth=2, markersize=6)
            plt.title(f'{layer_name} - Temporal Attention Pattern')
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
            
            # 注意力焦点分析
            plt.subplot(1, 2, 2)
            focus_window = 5  # 注意力焦点窗口
            smoothed = np.convolve(time_attention, np.ones(focus_window)/focus_window, mode='valid')
            plt.plot(time_steps[focus_window-1:], smoothed, 'r-', linewidth=2, label='Smoothed')
            plt.plot(time_steps, time_attention, 'b-', alpha=0.5, label='Original')
            plt.title(f'{layer_name} - Attention Focus')
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f'{layer_name}_temporal_attention.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⏰ 时间注意力图已保存: {save_path}")

def analyze_model_attention(model, sample_input: torch.Tensor, 
                          save_dir: str = "attention_analysis") -> Dict:
    """
    完整的模型注意力分析
    
    Args:
        model: 要分析的模型
        sample_input: 样本输入
        save_dir: 保存目录
    
    Returns:
        attention_info: 注意力分析结果
    """
    model.eval()
    
    # 创建注意力提取器
    extractor = AttentionExtractor(model)
    visualizer = AttentionVisualizer(save_dir)
    
    try:
        with torch.no_grad():
            # 提取注意力权重
            output, attention_maps = extractor(sample_input)
            
            print(f"🔍 提取到 {len(attention_maps)} 个注意力层的权重")
            
            # 生成各种可视化
            # 1. 整体分析
            visualizer.plot_attention_distribution(attention_maps)
            
            # 2. 各层详细热力图
            for layer_name, weights in attention_maps.items():
                if weights.numel() > 0:
                    if weights.dim() == 2:  # 多头
                        for head_idx in range(min(weights.size(0), 4)):  # 最多显示4个头
                            visualizer.plot_attention_matrix(weights, layer_name, head_idx)
                    else:  # 单头
                        visualizer.plot_attention_matrix(weights, layer_name)
            
            # 3. 时间序列注意力（如果适用）
            if hasattr(sample_input, 'shape') and len(sample_input.shape) >= 2:
                time_steps = list(range(sample_input.shape[0]))  # 假设第一维是时间
                for layer_name, weights in attention_maps.items():
                    if weights.numel() > 0:
                        visualizer.plot_temporal_attention(weights, time_steps, layer_name)
            
            # 统计信息
            attention_info = {
                'num_layers': len(attention_maps),
                'layer_stats': {}
            }
            
            for layer_name, weights in attention_maps.items():
                if weights.numel() > 0:
                    attention_info['layer_stats'][layer_name] = {
                        'shape': list(weights.shape),
                        'mean': weights.mean().item(),
                        'std': weights.std().item(),
                        'max': weights.max().item(),
                        'min': weights.min().item()
                    }
            
            print(f"📊 注意力分析完成！结果保存在: {save_dir}")
            return attention_info
            
    finally:
        # 清理钩子
        extractor.remove_hooks()
