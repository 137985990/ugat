# test_v15_fixes.py - 测试V15优化修复

import torch
import yaml
import sys
import os

def test_loss_stabilization():
    """测试损失稳定化模块"""
    print("🧪 测试损失稳定化...")
    
    try:
        from loss_stabilization import stabilize_classification_improvement_loss, stabilize_dynamic_weighting
        
        # 模拟损失值
        input_cls_loss = torch.tensor(0.6)
        enhanced_cls_loss = torch.tensor(0.5)
        accuracy_improvement = torch.tensor(0.1)
        
        result = stabilize_classification_improvement_loss(
            input_cls_loss, enhanced_cls_loss, accuracy_improvement,
            current_epoch=10
        )
        
        print(f"  ✅ 损失计算正常: {result['total_classification_loss']:.4f}")
        
        # 测试权重调整
        dynamic_recon_weight, dynamic_cls_weight = stabilize_dynamic_weighting(
            accuracy_improvement, 1.0, 0.5
        )
        
        print(f"  ✅ 权重调整正常: recon={dynamic_recon_weight:.3f}, cls={dynamic_cls_weight:.3f}")
        return True
        
    except Exception as e:
        print(f"  ❌ 损失稳定化测试失败: {e}")
        return False

def test_monitoring():
    """测试监控模块"""
    print("📊 测试监控功能...")
    
    try:
        from monitoring_enhancements import V15TrainingMonitor
        
        monitor = V15TrainingMonitor(log_interval=5)
        
        # 模拟一些指标
        metrics = {
            'recon_loss': 0.5,
            'total_classification_loss': 0.3,
            'input_accuracy': 0.7,
            'enhanced_accuracy': 0.75,
            'accuracy_improvement': 0.05
        }
        
        monitor.log_batch_metrics(metrics, epoch=1, batch_idx=0)
        
        # 添加更多数据点
        for i in range(15):
            metrics['recon_loss'] = 0.5 - i * 0.02
            metrics['input_accuracy'] = 0.7 + i * 0.01
            monitor.log_batch_metrics(metrics, epoch=1, batch_idx=i)
        
        # 检查健康状况
        is_healthy = monitor.check_training_health()
        monitor.suggest_adjustments()
        
        print(f"  ✅ 监控功能正常, 训练健康: {is_healthy}")
        return True
        
    except Exception as e:
        print(f"  ❌ 监控测试失败: {e}")
        return False

def test_data_optimizations():
    """测试数据优化模块"""
    print("🔧 测试数据优化...")
    
    try:
        from data_optimizations import add_data_validation, enhance_batch_normalization
        
        # 创建测试数据
        test_batch = torch.randn(4, 32, 320)  # [batch, channels, time]
        
        # 测试验证
        is_valid = add_data_validation(test_batch, "test_batch")
        print(f"  ✅ 数据验证: {is_valid}")
        
        # 测试标准化
        normalized = enhance_batch_normalization(test_batch, method='zscore')
        print(f"  ✅ 标准化完成: {normalized.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据优化测试失败: {e}")
        return False

def test_config_loading():
    """测试配置加载"""
    print("⚙️ 测试配置文件...")
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置项
        key_configs = ['batch_size', 'lr', 'epochs', 'loss_config']
        missing = [key for key in key_configs if key not in config]
        
        if missing:
            print(f"  ⚠️ 缺少配置项: {missing}")
        else:
            print(f"  ✅ 配置文件加载正常")
            print(f"    - batch_size: {config['batch_size']}")
            print(f"    - lr: {config['lr']}")
            print(f"    - epochs: {config['epochs']}")
        
        return len(missing) == 0
        
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("V15 优化修复测试")
    print("=" * 60)
    
    tests = [
        ("损失稳定化", test_loss_stabilization),
        ("监控功能", test_monitoring),
        ("数据优化", test_data_optimizations),
        ("配置加载", test_config_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！可以开始使用优化功能。")
        print("\n📋 下一步建议:")
        print("1. 在train.py中集成loss_stabilization模块")
        print("2. 添加monitoring_enhancements监控")
        print("3. 使用修改后的config.yaml配置")
        print("4. 运行几个epoch观察效果")
    else:
        print("⚠️ 部分测试失败，请检查相关模块。")
    
    print("=" * 60)

if __name__ == "__main__":
    main()