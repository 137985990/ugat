# visualization.py - 图表可视化模块

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
import warnings

# 设置图表样式
plt.style.use('seaborn-v0_8')  # 使用seaborn样式
sns.set_palette("husl")

class ChartGenerator:
    """图表生成器，解决中文显示问题"""
    
    def __init__(self):
        self.setup_chinese_fonts()
        self.setup_style()
    
    def setup_chinese_fonts(self):
        """设置中文字体支持"""
        # 尝试多种中文字体
        chinese_fonts = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',          # 黑体
            'SimSun',          # 宋体
            'KaiTi',           # 楷体
            'DejaVu Sans',     # 备用字体
            'Arial Unicode MS' # macOS字体
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        font_found = False
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                font_found = True
                print(f"✅ 使用字体: {font}")
                break
        
        if not font_found:
            warnings.warn("⚠️ 未找到中文字体，图表中文可能显示为方框")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False
    
    def setup_style(self):
        """设置图表样式"""
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 100
    
    def plot_performance_comparison(self, seq_lengths: List[int], 
                                  results: Dict[str, List[float]], 
                                  save_path: str = 'performance_comparison',
                                  use_chinese: bool = True) -> None:
        """绘制性能对比图"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 图1: 执行时间对比
        if use_chinese:
            ax1.plot(seq_lengths, results['standard'], 'r-o', 
                    label='标准图构建', linewidth=2.5, markersize=8)
            ax1.plot(seq_lengths, results['cached'], 'b-o', 
                    label='缓存图构建', linewidth=2.5, markersize=8)
            ax1.set_xlabel('序列长度', fontsize=12, fontweight='bold')
            ax1.set_ylabel('构建时间 (秒)', fontsize=12, fontweight='bold')
            ax1.set_title('图构建性能对比', fontsize=14, fontweight='bold')
        else:
            ax1.plot(seq_lengths, results['standard'], 'r-o', 
                    label='Standard Building', linewidth=2.5, markersize=8)
            ax1.plot(seq_lengths, results['cached'], 'b-o', 
                    label='Cached Building', linewidth=2.5, markersize=8)
            ax1.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Build Time (seconds)', fontsize=12, fontweight='bold')
            ax1.set_title('Graph Building Performance', fontsize=14, fontweight='bold')
        
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_yscale('log')  # 使用对数坐标
        
        # 图2: 加速比
        speedups = [std/cache if cache > 0 else 1 
                   for std, cache in zip(results['standard'], results['cached'])]
        
        bars = ax2.bar(range(len(seq_lengths)), speedups, 
                      color=['lightcoral' if s < 2 else 'lightgreen' if s < 4 else 'gold' 
                            for s in speedups], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        if use_chinese:
            ax2.set_xlabel('序列长度', fontsize=12, fontweight='bold')
            ax2.set_ylabel('加速比', fontsize=12, fontweight='bold') 
            ax2.set_title('缓存加速效果', fontsize=14, fontweight='bold')
        else:
            ax2.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Speedup Ratio', fontsize=12, fontweight='bold')
            ax2.set_title('Cache Speedup Effect', fontsize=14, fontweight='bold')
        
        ax2.set_xticks(range(len(seq_lengths)))
        ax2.set_xticklabels(seq_lengths)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 在柱状图上添加数值标签
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{speedup:.1f}x', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存多种格式
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
        plt.savefig(f'{save_path}.svg', bbox_inches='tight')  # 矢量图
        
        print(f"📊 图表已保存: {save_path}.png/.pdf/.svg")
        plt.show()
    
    def plot_memory_usage(self, metrics: Dict, save_path: str = 'memory_usage') -> None:
        """绘制内存使用图表"""
        
        categories = list(metrics.keys())
        memory_deltas = [metrics[cat]['memory_delta'] for cat in categories]
        gpu_memory_deltas = [metrics[cat]['gpu_memory_delta'] for cat in categories]
        durations = [metrics[cat]['duration'] for cat in categories]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 内存使用增量
        ax1.bar(categories, memory_deltas, color='skyblue', alpha=0.7)
        ax1.set_title('CPU Memory Usage', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Memory Delta (GB)', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # GPU内存使用增量
        ax2.bar(categories, gpu_memory_deltas, color='lightgreen', alpha=0.7)
        ax2.set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
        ax2.set_ylabel('GPU Memory Delta (GB)', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        
        # 执行时间
        ax3.bar(categories, durations, color='lightcoral', alpha=0.7)
        ax3.set_title('Execution Time', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Duration (seconds)', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        
        # 效率对比 (时间vs内存)
        scatter = ax4.scatter(memory_deltas, durations, 
                            c=gpu_memory_deltas, s=100, alpha=0.7, cmap='viridis')
        ax4.set_xlabel('Memory Delta (GB)', fontsize=10)
        ax4.set_ylabel('Duration (seconds)', fontsize=10)
        ax4.set_title('Efficiency Analysis', fontsize=12, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('GPU Memory (GB)', fontsize=9)
        
        # 为每个点添加标签
        for i, cat in enumerate(categories):
            ax4.annotate(cat, (memory_deltas[i], durations[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
        
        print(f"📊 内存使用图表已保存: {save_path}.png/.pdf")
        plt.show()
    
    def plot_model_comparison(self, model_metrics: Dict, save_path: str = 'model_comparison') -> None:
        """绘制模型对比图表"""
        
        models = list(model_metrics.keys())
        metrics = ['duration', 'memory_delta', 'gpu_memory_delta']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, metric in enumerate(metrics):
            values = [model_metrics[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=colors[:len(models)], 
                              alpha=0.8, edgecolor='black', linewidth=1)
            
            # 设置标题和标签
            if metric == 'duration':
                axes[i].set_title('Inference Time', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Time (seconds)', fontsize=10)
            elif metric == 'memory_delta':
                axes[i].set_title('CPU Memory Usage', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Memory (GB)', fontsize=10)
            else:
                axes[i].set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('GPU Memory (GB)', fontsize=10)
            
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
        
        print(f"📊 模型对比图表已保存: {save_path}.png/.pdf")
        plt.show()

# 全局图表生成器实例
chart_generator = ChartGenerator()

def safe_plot_with_fallback(plot_func, *args, use_chinese=True, **kwargs):
    """安全的图表绘制函数，如果中文失败则使用英文"""
    try:
        if use_chinese:
            plot_func(*args, use_chinese=True, **kwargs)
        else:
            plot_func(*args, use_chinese=False, **kwargs)
    except Exception as e:
        print(f"⚠️ 图表生成失败: {e}")
        print("🔄 尝试使用英文重新生成...")
        try:
            plot_func(*args, use_chinese=False, **kwargs)
        except Exception as e2:
            print(f"❌ 图表生成完全失败: {e2}")
            print("💡 建议检查matplotlib和字体安装")
