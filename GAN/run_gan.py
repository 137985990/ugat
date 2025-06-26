#!/usr/bin/env python3
"""
run_gan.py

运行脚本：启动GAN训练
使用方法：
    python run_gan.py --config config.yaml --epochs 100 --batch_size 16
"""

import argparse
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_gan

def main():
    parser = argparse.ArgumentParser(description="Train GAN with GAT-based generator and dual discriminators")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save logs and models")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在!")
        print("请确保config.yaml文件在当前目录中")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 启动GAN训练 ===")
    print(f"配置文件: {args.config}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"输出目录: {args.output_dir}")
    print(f"早停耐心值: {args.patience}")
    print()
    
    # 启动训练
    try:
        train_gan(args, gan_type='gan')
        print("\n=== 训练完成 ===")
        print(f"模型和日志已保存到: {args.output_dir}")
    except Exception as e:
        print(f"\n错误: 训练失败 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
