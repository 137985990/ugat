#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V12 多模态时序算法启动脚本
快速开始训练和测试
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n=> {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 执行成功")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ 执行失败")
            if result.stderr:
                print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 执行出错: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="V12多模态时序算法启动脚本")
    parser.add_argument('--action', choices=['test', 'train', 'all'], default='all',
                      help='执行动作: test(仅测试), train(仅训练), all(测试+训练)')
    parser.add_argument('--config', default='config.yaml',
                      help='配置文件路径')
    
    args = parser.parse_args()
    
    print("🎯 V12多模态时序算法启动器")
    print("=" * 60)
    print(f"📁 当前目录: {os.getcwd()}")
    print(f"⚙️  配置文件: {args.config}")
    print(f"🎬 执行动作: {args.action}")
    
    # 检查必要文件
    required_files = [
        'config.yaml',
        'train.py',
        'simple_multimodal_integration.py',
        'enhanced_validation_integration.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ 缺少必要文件: {missing_files}")
        print("请确保在V12目录中运行此脚本")
        return 1
    
    success = True
    
    # 运行测试
    if args.action in ['test', 'all']:
        print(f"\n🧪 开始运行测试套件")
        print("=" * 60)
        
        tests = [
            ('python test_v12_integration.py', 'V12集成测试'),
            ('python test_multimodal_modifications.py', '多模态损失测试'),
            ('python test_enhanced_validation_integration.py', '增强验证测试')
        ]
        
        for cmd, desc in tests:
            if not run_command(cmd, desc):
                success = False
                break
    
    # 运行训练
    if args.action in ['train', 'all'] and success:
        print(f"\n🏃 开始训练模型")
        print("=" * 60)
        
        train_cmd = f"python train.py --config {args.config}"
        run_command(train_cmd, "模型训练")
    
    if success:
        print(f"\n🎉 V12启动完成！")
        print("=" * 60)
        print("📋 后续步骤:")
        print("1. 检查训练日志和TensorBoard")
        print("2. 监控验证指标变化")
        print("3. 调整配置参数优化性能")
        print("4. 使用可视化工具分析结果")
    else:
        print(f"\n❌ 执行过程中出现错误")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
