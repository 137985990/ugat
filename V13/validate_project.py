#!/usr/bin/env python3
"""
validate_project.py - V13项目完整性验证脚本

检查运行V13项目所需的所有必备元素是否齐全。
"""

import os
import sys
import yaml
import importlib.util
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING")
        return False

def check_directory_writable(dirpath, description):
    """检查目录是否存在且可写"""
    if os.path.exists(dirpath):
        if os.access(dirpath, os.W_OK):
            print(f"✓ {description}: {dirpath} (writable)")
            return True
        else:
            print(f"⚠ {description}: {dirpath} (exists but not writable)")
            return False
    else:
        try:
            os.makedirs(dirpath, exist_ok=True)
            print(f"✓ {description}: {dirpath} (created)")
            return True
        except Exception as e:
            print(f"✗ {description}: {dirpath} - Cannot create ({e})")
            return False

def check_python_module(module_name, description):
    """检查Python模块是否可导入"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError:
        print(f"✗ {description}: {module_name} - MISSING")
        return False

def check_config_validity(config_path):
    """检查配置文件有效性"""
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查必需的配置项
        required_keys = ['data_files', 'label_col', 'common_modalities', 'dataset_modalities']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"✗ Config missing required keys: {missing_keys}")
            return False
        
        # 检查数据集配置
        datasets = config.get('dataset_modalities', {})
        if not datasets:
            print("✗ No dataset_modalities defined in config")
            return False
        
        print(f"✓ Config file valid with {len(datasets)} datasets: {list(datasets.keys())}")
        return True
        
    except yaml.YAMLError as e:
        print(f"✗ Config file YAML error: {e}")
        return False
    except Exception as e:
        print(f"✗ Config file error: {e}")
        return False

def main():
    """主验证流程"""
    print("=" * 60)
    print("V13 Multi-modal Time Series Project Validation")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    all_checks_passed = True
    
    print("\n1. 核心代码文件检查:")
    core_files = [
        ('train.py', '主训练脚本'),
        ('data.py', '数据处理模块'),
        ('model.py', '模型定义'),
        ('graph.py', '图构建模块'),
        ('config.yaml', '配置文件'),
    ]
    
    for filename, description in core_files:
        filepath = project_root / filename
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    print("\n2. 数据文件检查:")
    # 首先读取配置文件获取数据路径
    config_path = project_root / 'config.yaml'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            data_files = config.get('data_files', [])
            for data_file in data_files:
                # 解析相对路径
                if data_file.startswith('../'):
                    data_path = project_root.parent / data_file[3:]
                else:
                    data_path = project_root / data_file
                
                if not check_file_exists(data_path, f"数据文件"):
                    all_checks_passed = False
        except Exception as e:
            print(f"✗ 无法读取配置文件检查数据路径: {e}")
            all_checks_passed = False
    
    print("\n3. 输出目录检查:")
    output_dirs = [
        ('Checkpoints', '模型检查点目录'),
        ('Logs', '日志目录'),
        ('runs', 'TensorBoard日志目录'),
    ]
    
    for dirname, description in output_dirs:
        dirpath = project_root / dirname
        if not check_directory_writable(dirpath, description):
            all_checks_passed = False
    
    print("\n4. Python依赖检查:")
    required_modules = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'TQDM'),
    ]
    
    for module_name, description in required_modules:
        if not check_python_module(module_name, description):
            all_checks_passed = False
    
    print("\n5. 可选依赖检查:")
    optional_modules = [
        ('torch_geometric', 'PyTorch Geometric'),
        ('tensorboard', 'TensorBoard'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    for module_name, description in optional_modules:
        check_python_module(module_name, f"{description} (可选)")
    
    print("\n6. 配置文件验证:")
    if not check_config_validity(config_path):
        all_checks_passed = False
    
    print("\n7. GPU支持检查:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"✓ CUDA available: {gpu_count} GPU(s), current: {gpu_name}")
        else:
            print("⚠ CUDA not available, will use CPU training")
    except:
        print("✗ Cannot check CUDA availability")
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ 所有必备组件检查通过！项目已准备就绪。")
        print("\n启动训练:")
        print("  python train.py")
        print("\n查看帮助:")
        print("  python train.py --help")
        return 0
    else:
        print("✗ 项目验证失败，请检查缺失的组件。")
        print("\n安装依赖:")
        print("  pip install -r requirements.txt")
        print("\n检查数据文件路径和配置文件设置。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
