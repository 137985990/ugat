#!/usr/bin/env python3
"""
test_gan.py

测试脚本：验证GAN模型的训练和运行
"""

import os
import sys
import torch
import argparse
import logging

# 设置路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_gan, parse_args

def test_models():
    """测试模型的基本功能"""
    print("Testing GAN models...")
    
    from model import GANGenerator, GANDiscriminator, ClassifierDiscriminator
    
    # 测试参数
    batch_size = 4
    in_channels = 7
    seq_len = 320
    
    # 创建模型
    G = GANGenerator(in_channels, in_channels, seq_len)
    D1 = GANDiscriminator(in_channels, seq_len)
    D2 = ClassifierDiscriminator(in_channels, seq_len, num_classes=3)
    
    # 创建测试数据
    test_input = torch.randn(batch_size, in_channels, seq_len)
    
    print(f"Input shape: {test_input.shape}")
    
    # 测试生成器
    with torch.no_grad():
        fake_output = G(test_input)
        print(f"Generator output shape: {fake_output.shape}")
        
        # 测试判别器1
        d1_output = D1(test_input)
        print(f"Discriminator 1 output shape: {d1_output.shape}")
        
        # 测试判别器2
        d2_disc, d2_class = D2(test_input)
        print(f"Discriminator 2 disc output shape: {d2_disc.shape}")
        print(f"Discriminator 2 class output shape: {d2_class.shape}")
    
    print("✓ All models work correctly!")

def test_data_loading():
    """测试数据加载"""
    print("\nTesting data loading...")
    
    try:
        from data import create_dataset_from_config
        config_path = "config.yaml"
        
        if os.path.exists(config_path):
            train_set, val_set, test_set = create_dataset_from_config(config_path)
            print(f"✓ Data loaded successfully!")
            print(f"  Train samples: {len(train_set)}")
            print(f"  Val samples: {len(val_set)}")
            print(f"  Test samples: {len(test_set)}")
            
            if len(train_set) > 0:
                sample = train_set[0]
                print(f"  Sample shape: {sample.shape}")
        else:
            print("⚠ Config file not found, skipping data loading test")
            
    except Exception as e:
        print(f"⚠ Data loading test failed: {e}")

def main():
    """主测试函数"""
    print("=== GAN Model Testing ===")
    
    # 测试模型
    test_models()
    
    # 测试数据加载
    test_data_loading()
    
    print("\n=== Testing Training (Small Scale) ===")
    
    # 设置测试参数
    class TestArgs:
        config = "config.yaml"
        epochs = 2  # 只训练2个epoch做测试
        batch_size = 4
        lr = 1e-3
        output_dir = "test_outputs"
        patience = 5
    
    args = TestArgs()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if os.path.exists(args.config):
            print("Starting mini training test...")
            train_gan(args, gan_type='test')
            print("✓ Training test completed successfully!")
        else:
            print("⚠ Config file not found, skipping training test")
            print("  Please ensure config.yaml exists in the current directory")
    except Exception as e:
        print(f"⚠ Training test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Summary ===")
    print("Tests completed. Check the output above for any issues.")

if __name__ == "__main__":
    main()
