#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试优化配置是否正常工作
"""

import yaml
import torch
import sys
import os

def test_configuration():
    print("🔧 测试优化配置...")
    
    # 检查CUDA
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n📊 优化配置参数:")
    print(f"  - Batch Size: {config.get('batch_size', 'N/A')}")
    print(f"  - Learning Rate: {config.get('lr', 'N/A')}")
    print(f"  - Num Workers: {config.get('num_workers', 'N/A')}")
    print(f"  - Mixed Precision: {config.get('use_mixed_precision', 'N/A')}")
    print(f"  - Pin Memory: {config.get('pin_memory', 'N/A')}")
    print(f"  - Prefetch Factor: {config.get('prefetch_factor', 'N/A')}")
    
    # 检查混合精度支持
    try:
        from torch.cuda.amp import GradScaler, autocast
        print(f"  - AMP支持: ✅ 可用")
        scaler = GradScaler()
        print(f"  - GradScaler创建: ✅ 成功")
    except ImportError as e:
        print(f"  - AMP支持: ❌ 不可用 ({e})")
    
    # 测试数据加载器配置
    print(f"\n🚀 数据加载器测试:")
    try:
        from torch.utils.data import DataLoader, TensorDataset
        # 创建测试数据
        test_data = torch.randn(100, 32, 320)
        test_labels = torch.randint(0, 2, (100,))
        test_dataset = TensorDataset(test_data, test_labels)
        
        # 创建高性能DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get('batch_size', 64),
            num_workers=min(config.get('num_workers', 8), 4),  # 限制测试用的worker数
            pin_memory=config.get('pin_memory', True),
            prefetch_factor=config.get('prefetch_factor', 4),
            persistent_workers=True,
            shuffle=True
        )
        
        # 测试一个batch
        for batch_data in test_loader:
            batch_x, batch_y = batch_data
            print(f"  - 测试batch形状: {batch_x.shape}")
            print(f"  - 标签形状: {batch_y.shape}")
            break
        
        print(f"  - DataLoader创建: ✅ 成功")
        
    except Exception as e:
        print(f"  - DataLoader测试: ❌ 失败 ({e})")
    
    print(f"\n✨ 配置测试完成!")

if __name__ == "__main__":
    test_configuration()
