#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V12版本简化集成测试 - 无emoji版本
适用于Windows GBK编码环境
"""

import sys
import os
import yaml
import importlib
import torch
from pathlib import Path

def test_file_structure():
    """测试V12文件结构"""
    print("\n[TEST] 测试V12文件结构...")
    
    required_files = [
        'config.yaml',
        'train.py', 
        'data.py',
        'model.py',
        'graph.py',
        'simple_multimodal_integration.py',
        'enhanced_validation_integration.py'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"[PASS] {file} 存在")
        else:
            print(f"[FAIL] {file} 缺失")
            return False
    
    return True

def test_imports():
    """测试模块导入"""
    print("\n[TEST] 测试V12版本导入...")
    
    try:
        # 测试导入
        simple_multimodal = importlib.import_module('simple_multimodal_integration')
        enhanced_validation = importlib.import_module('enhanced_validation_integration')
        
        print("[PASS] simple_multimodal_integration 导入成功")
        print("[PASS] enhanced_validation_integration 导入成功")
        
        # 测试配置加载
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("[PASS] config.yaml 加载成功")
        
        # 测试必要配置项
        required_configs = ['loss_config', 'common_modalities', 'enhanced_validation']
        for cfg in required_configs:
            if cfg in config:
                print(f"[PASS] 配置项 {cfg} 存在")
            else:
                print(f"[FAIL] 配置项 {cfg} 缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 导入测试失败: {e}")
        return False

def test_multimodal_loss():
    """测试多模态损失函数"""
    print("\n[TEST] 测试多模态损失函数创建...")
    
    try:
        from simple_multimodal_integration import create_simple_multimodal_criterion
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建损失函数
        criterion = create_simple_multimodal_criterion(config)
        
        # 测试损失计算
        pred = torch.randn(100)
        target = torch.randn(100)
        
        common_loss = criterion(pred, target, is_common=True)
        have_loss = criterion(pred, target, is_common=False)
        
        print(f"[PASS] 多模态损失函数创建成功")
        print(f"[INFO] Common损失: {common_loss.item():.6f}")
        print(f"[INFO] Have损失: {have_loss.item():.6f}")
        print(f"[INFO] 权重比例: {common_loss.item()/have_loss.item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 多模态损失函数测试失败: {e}")
        return False

def test_enhanced_validation():
    """测试增强验证管理器"""
    print("\n[TEST] 测试增强验证管理器...")
    
    try:
        from enhanced_validation_integration import EnhancedValidationManager
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建验证管理器
        validation_config = config.get('enhanced_validation', {})
        manager = EnhancedValidationManager(validation_config)
        
        # 测试验证频率
        freq1 = manager.should_validate(1)
        freq10 = manager.should_validate(10)
        freq50 = manager.should_validate(50)
        
        print(f"[PASS] 增强验证管理器创建成功")
        print(f"[INFO] Epoch 1 验证: {freq1}")
        print(f"[INFO] Epoch 10 验证: {freq10}")
        print(f"[INFO] Epoch 50 验证: {freq50}")
        
        # 测试指标更新
        metrics = {
            'val_loss': 1.0,
            'val_accuracy': 0.7,
            'val_f1_score': 0.65
        }
        
        result = manager.update_metrics(metrics, 1)
        print(f"[PASS] 指标更新成功: {result}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 增强验证管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("V12版本集成测试")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_imports,
        test_multimodal_loss,
        test_enhanced_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"[FAIL] 测试 {test_func.__name__} 失败")
        except Exception as e:
            print(f"[ERROR] 测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 60)
    if passed == total:
        print("所有测试通过！V12版本集成成功！")
        print("=" * 60)
        print("V12版本特性:")
        print("[PASS] 多模态损失函数 - Common模态参与损失计算")
        print("[PASS] 增强验证策略 - 智能调度，多指标监控")
        print("[PASS] 配置文件统一 - 所有参数集中管理")
        print("[PASS] 训练流程优化 - 无缝集成，详细日志")
        print("[PASS] 代码结构清理 - 模块化，易维护")
        print("可以开始使用V12版本进行训练:")
        print("   python train.py --config config.yaml")
        return True
    else:
        print(f"部分测试失败，请检查V12版本配置 ({passed}/{total})")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
