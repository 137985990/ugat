# curriculum_learning.py - 课程学习模块

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from torch.utils.data import DataLoader, Subset
import random
from enum import Enum

class DifficultyMetric(Enum):
    """难度度量方式"""
    SEQUENCE_LENGTH = "seq_length"      # 序列长度
    MISSING_RATIO = "missing_ratio"     # 缺失比例
    NOISE_LEVEL = "noise_level"         # 噪声水平
    LABEL_COMPLEXITY = "label_complex"  # 标签复杂度
    RECONSTRUCTION_ERROR = "recon_error" # 重建误差

class CurriculumScheduler:
    """课程学习调度器"""
    
    def __init__(self, total_epochs: int, 
                 difficulty_metric: DifficultyMetric = DifficultyMetric.MISSING_RATIO,
                 curriculum_type: str = "linear"):
        """
        Args:
            total_epochs: 总训练轮数
            difficulty_metric: 难度度量方式
            curriculum_type: 课程类型 ["linear", "exponential", "step", "adaptive"]
        """
        self.total_epochs = total_epochs
        self.difficulty_metric = difficulty_metric
        self.curriculum_type = curriculum_type
        self.current_epoch = 0
        
        # 难度级别定义
        self.difficulty_levels = {
            DifficultyMetric.SEQUENCE_LENGTH: [64, 128, 192, 256, 320],  # 从短到长
            DifficultyMetric.MISSING_RATIO: [0.1, 0.3, 0.5, 0.7, 0.9], # 从少缺失到多缺失
            DifficultyMetric.NOISE_LEVEL: [0.0, 0.1, 0.2, 0.3, 0.5],   # 从无噪声到高噪声
            DifficultyMetric.LABEL_COMPLEXITY: [0, 1, 2, 3, 4],         # 从简单到复杂
        }
        
        # 性能历史记录
        self.performance_history = []
        self.difficulty_history = []
    
    def get_current_difficulty(self) -> float:
        """获取当前难度级别 (0-1)"""
        progress = self.current_epoch / self.total_epochs
        
        if self.curriculum_type == "linear":
            return progress
        elif self.curriculum_type == "exponential":
            return progress ** 2
        elif self.curriculum_type == "step":
            # 分阶段提升
            if progress < 0.3:
                return 0.2
            elif progress < 0.6:
                return 0.5
            elif progress < 0.8:
                return 0.8
            else:
                return 1.0
        elif self.curriculum_type == "adaptive":
            # 根据性能自适应调整
            return self._adaptive_difficulty()
        else:
            return progress
    
    def _adaptive_difficulty(self) -> float:
        """自适应难度调整"""
        if len(self.performance_history) < 3:
            return 0.2  # 初始简单
        
        # 检查最近3轮的性能趋势
        recent_performance = self.performance_history[-3:]
        avg_performance = np.mean(recent_performance)
        
        # 如果性能稳定且较好，增加难度
        if avg_performance > 0.8 and np.std(recent_performance) < 0.1:
            current_difficulty = self.difficulty_history[-1] if self.difficulty_history else 0.2
            return min(1.0, current_difficulty + 0.1)
        
        # 如果性能下降，降低难度
        elif len(self.performance_history) >= 2 and self.performance_history[-1] < self.performance_history[-2] - 0.1:
            current_difficulty = self.difficulty_history[-1] if self.difficulty_history else 0.5
            return max(0.1, current_difficulty - 0.1)
        
        # 否则保持当前难度
        return self.difficulty_history[-1] if self.difficulty_history else 0.2
    
    def update_performance(self, performance: float):
        """更新性能记录"""
        self.performance_history.append(performance)
        self.difficulty_history.append(self.get_current_difficulty())
        
        # 保持历史记录长度
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
            self.difficulty_history.pop(0)
    
    def step_epoch(self):
        """更新epoch"""
        self.current_epoch += 1

class CurriculumDataset:
    """课程学习数据集"""
    
    def __init__(self, base_dataset, curriculum_scheduler: CurriculumScheduler):
        self.base_dataset = base_dataset
        self.scheduler = curriculum_scheduler
        self.sample_difficulties = self._compute_sample_difficulties()
    
    def _compute_sample_difficulties(self) -> List[float]:
        """计算每个样本的难度"""
        difficulties = []
        
        for i in range(len(self.base_dataset)):
            try:
                sample = self.base_dataset[i]
                difficulty = self._compute_single_difficulty(sample, i)
                difficulties.append(difficulty)
            except:
                difficulties.append(0.5)  # 默认中等难度
        
        return difficulties
    
    def _compute_single_difficulty(self, sample, idx: int) -> float:
        """计算单个样本难度"""
        metric = self.scheduler.difficulty_metric
        
        if metric == DifficultyMetric.SEQUENCE_LENGTH:
            # 基于序列长度
            if hasattr(sample, '__len__'):
                seq_len = len(sample[0]) if isinstance(sample, (tuple, list)) else len(sample)
            else:
                seq_len = sample[0].shape[-1] if hasattr(sample[0], 'shape') else 320
            
            max_len = max(self.scheduler.difficulty_levels[metric])
            return min(1.0, seq_len / max_len)
        
        elif metric == DifficultyMetric.MISSING_RATIO:
            # 基于缺失比例（如果样本包含mask信息）
            if isinstance(sample, (tuple, list)) and len(sample) >= 4:
                mask_info = sample[3]  # is_real_mask
                if hasattr(mask_info, 'sum'):
                    missing_ratio = 1.0 - (mask_info.sum().float() / mask_info.numel())
                    return missing_ratio.item()
            
            # 随机分配缺失比例
            return random.random()
        
        elif metric == DifficultyMetric.NOISE_LEVEL:
            # 基于数据噪声水平
            if isinstance(sample, (tuple, list)):
                data = sample[0]
                if hasattr(data, 'std'):
                    noise_level = data.std().item()
                    return min(1.0, noise_level / 2.0)  # 归一化
            return random.random()
        
        elif metric == DifficultyMetric.LABEL_COMPLEXITY:
            # 基于标签复杂度
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                label = sample[1]
                if hasattr(label, 'item'):
                    label_val = label.item()
                    return label_val / 4.0  # 假设最大标签值为4
            return random.random()
        
        return 0.5  # 默认中等难度
    
    def get_curriculum_subset(self) -> Subset:
        """获取当前课程难度对应的数据子集"""
        current_difficulty = self.scheduler.get_current_difficulty()
        
        # 选择难度不超过当前水平的样本
        valid_indices = []
        for i, difficulty in enumerate(self.sample_difficulties):
            if difficulty <= current_difficulty + 0.1:  # 允许小幅超出
                valid_indices.append(i)
        
        # 确保至少有一些样本
        if len(valid_indices) < 10:
            easy_indices = [i for i, d in enumerate(self.sample_difficulties) if d <= 0.5]
            valid_indices = easy_indices[:max(10, len(easy_indices)//2)]
        
        return Subset(self.base_dataset, valid_indices)

class CurriculumTrainer:
    """课程学习训练器"""
    
    def __init__(self, model: nn.Module, base_dataset, 
                 curriculum_scheduler: CurriculumScheduler,
                 batch_size: int = 32):
        self.model = model
        self.base_dataset = base_dataset
        self.scheduler = curriculum_scheduler
        self.batch_size = batch_size
        
        # 创建课程数据集
        self.curriculum_dataset = CurriculumDataset(base_dataset, curriculum_scheduler)
        
        # 性能追踪
        self.training_log = {
            'epochs': [],
            'difficulties': [],
            'train_losses': [],
            'subset_sizes': []
        }
    
    def get_current_dataloader(self) -> DataLoader:
        """获取当前难度的数据加载器"""
        current_subset = self.curriculum_dataset.get_curriculum_subset()
        
        return DataLoader(
            current_subset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
    
    def train_epoch(self, optimizer, criterion, device) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 获取当前课程数据
        dataloader = self.get_current_dataloader()
        
        total_loss = 0.0
        total_samples = 0
        
        current_difficulty = self.scheduler.get_current_difficulty()
        subset_size = len(dataloader.dataset)
        
        print(f"📚 Epoch {self.scheduler.current_epoch}: "
              f"难度 {current_difficulty:.2f}, "
              f"样本数 {subset_size}")
        
        for batch_idx, batch_data in enumerate(dataloader):
            # 解包批次数据
            if isinstance(batch_data, (tuple, list)):
                inputs = batch_data[0].to(device)
                labels = batch_data[1].to(device) if len(batch_data) > 1 else None
            else:
                inputs = batch_data.to(device)
                labels = None
            
            optimizer.zero_grad()
            
            # 前向传播
            if labels is not None:
                outputs, logits = self.model(inputs)
                
                # 计算损失（重建 + 分类）
                recon_loss = criterion(outputs, inputs)
                if logits is not None and labels is not None:
                    ce_loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), labels.unsqueeze(0))
                    loss = recon_loss + ce_loss
                else:
                    loss = recon_loss
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_samples += inputs.size(0)
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
        
        # 更新课程调度器
        # 这里用负损失作为性能指标（损失越低，性能越好）
        performance = max(0.0, 1.0 - avg_loss)  # 简单的性能转换
        self.scheduler.update_performance(performance)
        self.scheduler.step_epoch()
        
        # 记录训练日志
        self.training_log['epochs'].append(self.scheduler.current_epoch)
        self.training_log['difficulties'].append(current_difficulty)
        self.training_log['train_losses'].append(avg_loss)
        self.training_log['subset_sizes'].append(subset_size)
        
        return {
            'loss': avg_loss,
            'difficulty': current_difficulty,
            'subset_size': subset_size,
            'performance': performance
        }
    
    def plot_curriculum_progress(self, save_path: str = "curriculum_progress.png"):
        """绘制课程学习进度"""
        if not self.training_log['epochs']:
            return
        
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_log['epochs']
        
        # 1. 难度进度
        ax1.plot(epochs, self.training_log['difficulties'], 'b-o', markersize=4)
        ax1.set_title('Curriculum Difficulty Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Difficulty Level')
        ax1.grid(True, alpha=0.3)
        
        # 2. 训练损失
        ax2.plot(epochs, self.training_log['train_losses'], 'r-o', markersize=4)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        # 3. 数据集大小变化
        ax3.plot(epochs, self.training_log['subset_sizes'], 'g-o', markersize=4)
        ax3.set_title('Training Subset Size')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Number of Samples')
        ax3.grid(True, alpha=0.3)
        
        # 4. 难度vs性能散点图
        if len(self.scheduler.performance_history) > 0:
            ax4.scatter(self.training_log['difficulties'], 
                       self.scheduler.performance_history[-len(epochs):],
                       c=epochs, cmap='viridis', alpha=0.7)
            ax4.set_title('Difficulty vs Performance')
            ax4.set_xlabel('Difficulty Level')
            ax4.set_ylabel('Performance')
            ax4.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('Epoch')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 课程学习进度图已保存: {save_path}")

def create_curriculum_trainer(model, dataset, config: Dict) -> CurriculumTrainer:
    """创建课程学习训练器的工厂函数"""
    
    # 解析配置
    total_epochs = config.get('epochs', 100)
    difficulty_metric = DifficultyMetric(config.get('curriculum_metric', 'missing_ratio'))
    curriculum_type = config.get('curriculum_type', 'linear')
    batch_size = config.get('batch_size', 32)
    
    # 创建调度器
    scheduler = CurriculumScheduler(
        total_epochs=total_epochs,
        difficulty_metric=difficulty_metric,
        curriculum_type=curriculum_type
    )
    
    # 创建训练器
    trainer = CurriculumTrainer(
        model=model,
        base_dataset=dataset,
        curriculum_scheduler=scheduler,
        batch_size=batch_size
    )
    
    print(f"📚 课程学习训练器已创建:")
    print(f"  - 总轮数: {total_epochs}")
    print(f"  - 难度度量: {difficulty_metric.value}")
    print(f"  - 课程类型: {curriculum_type}")
    print(f"  - 数据集大小: {len(dataset)}")
    
    return trainer
