#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的训练启动脚本 - 用于测试
"""

import sys
import os

print("🚀 启动V12优化训练...")
print(f"Python版本: {sys.version}")
print(f"当前目录: {os.getcwd()}")
print(f"参数: {sys.argv}")

# 检查文件是否存在
train_file = "train.py"
config_file = "config.yaml"

if os.path.exists(train_file):
    print(f"✅ {train_file} 文件存在")
else:
    print(f"❌ {train_file} 文件不存在")

if os.path.exists(config_file):
    print(f"✅ {config_file} 文件存在")
else:
    print(f"❌ {config_file} 文件不存在")

# 尝试导入主要模块
try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")

try:
    import yaml
    print("✅ PyYAML可用")
except ImportError as e:
    print(f"❌ PyYAML导入失败: {e}")

# 加载配置
try:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✅ 配置加载成功，batch_size: {config.get('batch_size')}")
except Exception as e:
    print(f"❌ 配置加载失败: {e}")

print("\n🎯 准备启动真实训练...")

# 导入并运行训练脚本
try:
    exec(open(train_file).read())
except Exception as e:
    print(f"❌ 训练脚本执行失败: {e}")
    import traceback
    traceback.print_exc()
