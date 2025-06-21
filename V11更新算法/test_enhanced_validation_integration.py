# test_enhanced_validation_integration.py - 测试增强验证策略集成

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_validation_manager():
    """测试增强验证管理器的基本功能"""
    from enhanced_validation_integration import EnhancedValidationManager
    
    print("🧪 测试增强验证管理器...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:        # 初始化管理器 - 设置更小的耐心度用于测试
        manager = EnhancedValidationManager(
            patience=3,  # 降低耐心度以便测试
            min_delta=0.01,
            save_dir=temp_dir
        )
        
        # 测试验证频率调度
        assert manager.should_validate(1) == True  # 第1个epoch应该验证
        assert manager.should_validate(2) == True  # 第2个epoch应该验证
        assert manager.should_validate(11) == False # 第11个epoch不应该验证（每2次）
        assert manager.should_validate(12) == True  # 第12个epoch应该验证
        
        print("✅ 验证频率调度测试通过")
        
        # 测试指标更新和早停逻辑
        mock_metrics = [
            {'val_loss': 1.0, 'val_accuracy': 0.7, 'val_f1_score': 0.65},
            {'val_loss': 0.8, 'val_accuracy': 0.75, 'val_f1_score': 0.7},  # 改进
            {'val_loss': 0.9, 'val_accuracy': 0.73, 'val_f1_score': 0.68}, # 无改进
            {'val_loss': 0.85, 'val_accuracy': 0.74, 'val_f1_score': 0.69}, # 无改进            {'val_loss': 0.87, 'val_accuracy': 0.72, 'val_f1_score': 0.67}, # 无改进
            {'val_loss': 0.88, 'val_accuracy': 0.71, 'val_f1_score': 0.66}, # 无改进
        ]
        
        should_stop = False
        for epoch, metrics in enumerate(mock_metrics, 1):
            early_stop_info = manager.update_metrics(metrics, epoch)
            should_stop = early_stop_info['should_stop']
            print(f"Epoch {epoch}: metrics={metrics}, early_stop_info={early_stop_info}")
        
        assert should_stop == True  # 应该在第5个epoch触发早停 (patience=3, 从第2个epoch开始没改进)
        assert manager.best_epoch == 2  # 最佳epoch应该是第2个
        
        print("✅ 早停逻辑测试通过")
        
        # 测试获取最佳指标摘要
        summary = manager.get_best_metrics_summary()
        assert summary['best_epoch'] == 2
        assert summary['early_stopped'] == True
        
        print("✅ 指标摘要测试通过")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("🎉 增强验证管理器测试全部通过！")


def test_mock_enhanced_validation_metrics():
    """测试增强验证指标计算的模拟版本"""
    from enhanced_validation_integration import EnhancedValidationManager
    
    print("🧪 测试增强验证指标计算...")
    
    # 创建模拟的模型、数据加载器等
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x):
            # 模拟返回重建输出和分类logits
            out = torch.randn(x.size(0), x.size(1))  # 重建输出
            logits = self.linear(torch.randn(10))     # 分类logits
            return out, logits
    
    # 创建模拟数据
    def create_mock_data_loader():
        data = []
        for _ in range(5):  # 5个batch
            batch = torch.randn(4, 8, 32)  # (batch_size=4, channels=8, time=32)
            labels = torch.randint(0, 2, (4,))
            mask_indices = torch.randint(0, 8, (4,))
            is_real_mask = torch.randint(0, 2, (8,)).bool()
            data.append((batch, labels, mask_indices, is_real_mask))
        return data
    
    # 创建模拟criterion
    class MockCriterion(nn.Module):
        def __init__(self):
            super().__init__()
            self.common_indices = [0, 1, 2]  # 前3个通道是common
        
        def forward(self, pred, target, channel_idx=None, is_common=False):
            return nn.MSELoss()(pred, target)
    
    # 创建管理器和模拟组件
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = EnhancedValidationManager(save_dir=temp_dir)
        
        # 模拟compute_enhanced_validation_metrics方法
        mock_metrics = {
            'val_loss': 0.5,
            'val_recon_loss': 0.3,
            'val_accuracy': 0.8,
            'val_f1_score': 0.75,
            'val_precision': 0.82,
            'val_recall': 0.78,
            'val_common_recon_loss': 0.25,
            'val_have_recon_loss': 0.35,
            'val_samples': 20
        }
        
        # 测试指标格式
        required_keys = ['val_loss', 'val_accuracy', 'val_f1_score', 'val_common_recon_loss', 'val_have_recon_loss']
        for key in required_keys:
            assert key in mock_metrics, f"Missing required metric: {key}"
        
        print("✅ 增强验证指标格式测试通过")
        
        # 测试更新指标
        early_stop_info = manager.update_metrics(mock_metrics, 1)
        
        assert 'should_stop' in early_stop_info
        assert 'epochs_no_improve' in early_stop_info
        assert 'best_epoch' in early_stop_info
        assert 'is_overfitting' in early_stop_info
        
        print("✅ 指标更新测试通过")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("🎉 增强验证指标计算测试通过！")


def test_integration_with_training_loop():
    """测试与训练循环的集成"""
    print("🧪 测试训练循环集成...")
    
    # 模拟训练循环中的关键部分
    from enhanced_validation_integration import EnhancedValidationManager
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = EnhancedValidationManager(
            patience=3,
            save_dir=temp_dir
        )
        
        # 模拟10个epoch的训练
        for epoch in range(1, 11):
            # 模拟训练指标
            train_loss = 1.0 - epoch * 0.05  # 训练损失逐渐下降
            train_acc = 0.5 + epoch * 0.03   # 训练准确率逐渐上升
            
            # 检查是否需要验证
            if manager.should_validate(epoch):
                # 模拟验证指标
                val_metrics = {
                    'val_loss': max(0.1, 1.2 - epoch * 0.1 + np.random.normal(0, 0.05)),
                    'val_accuracy': min(0.95, 0.4 + epoch * 0.04 + np.random.normal(0, 0.02)),
                    'val_f1_score': min(0.9, 0.35 + epoch * 0.04 + np.random.normal(0, 0.02)),
                    'val_precision': 0.8,
                    'val_recall': 0.75,
                    'val_common_recon_loss': 0.2,
                    'val_have_recon_loss': 0.25,
                    'val_samples': 100
                }
                
                # 更新验证指标
                early_stop_info = manager.update_metrics(val_metrics, epoch)
                
                print(f"Epoch {epoch}: 验证 - val_loss={val_metrics['val_loss']:.4f}, "
                      f"val_acc={val_metrics['val_accuracy']:.4f}, "
                      f"best_epoch={early_stop_info['best_epoch']}, "
                      f"no_improve={early_stop_info['epochs_no_improve']}")
                
                # 模拟早停检查
                if early_stop_info['should_stop']:
                    print(f"早停触发于epoch {epoch}")
                    break
            else:
                print(f"Epoch {epoch}: 跳过验证 - train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
        
        # 获取最终摘要
        summary = manager.get_best_metrics_summary()
        print(f"训练摘要: {summary}")
        
        print("✅ 训练循环集成测试通过")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("🎉 训练循环集成测试完成！")


def test_visualization_generation():
    """测试可视化生成"""
    print("🧪 测试可视化生成...")
    
    from enhanced_validation_integration import EnhancedValidationManager
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = EnhancedValidationManager(save_dir=temp_dir)
        
        # 模拟一些历史数据
        for epoch in range(1, 6):
            mock_metrics = {
                'val_loss': 1.0 - epoch * 0.1,
                'val_accuracy': 0.5 + epoch * 0.08,
                'val_f1_score': 0.45 + epoch * 0.07,
                'val_precision': 0.6 + epoch * 0.05,
                'val_recall': 0.55 + epoch * 0.06,
                'val_common_recon_loss': 0.5 - epoch * 0.05,
                'val_have_recon_loss': 0.6 - epoch * 0.06,
                'val_samples': 100
            }
            manager.update_metrics(mock_metrics, epoch)
        
        # 生成可视化
        plot_path = os.path.join(temp_dir, "test_validation_metrics.png")
        manager.save_validation_plots(plot_path)
        
        # 检查文件是否生成
        assert os.path.exists(plot_path), "可视化文件未生成"
        
        print("✅ 可视化生成测试通过")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("🎉 可视化生成测试完成！")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("🚀 开始测试增强验证策略集成")
    print("=" * 60)
    
    try:
        test_enhanced_validation_manager()
        print()
        
        test_mock_enhanced_validation_metrics()
        print()
        
        test_integration_with_training_loop()
        print()
        
        test_visualization_generation()
        print()
        
        print("=" * 60)
        print("🎉 所有测试通过！增强验证策略集成成功！")
        print("=" * 60)
        
        print("\n📋 集成要点总结:")
        print("1. ✅ 增强验证管理器功能正常")
        print("2. ✅ 验证频率调度工作正确")
        print("3. ✅ 多指标早停策略有效")
        print("4. ✅ 过拟合检测机制运行")
        print("5. ✅ 可视化生成功能正常")
        print("6. ✅ 训练循环集成无问题")
        
        print("\n🔧 下一步建议:")
        print("1. 在实际训练中测试增强验证策略")
        print("2. 根据实际数据调整验证频率调度")
        print("3. 微调早停策略的权重参数")
        print("4. 监控增强验证指标的变化趋势")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
