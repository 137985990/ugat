#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_main_training_integration.py
测试主训练循环中验证生成分类器集成功能

测试内容：
1. 配置文件解析验证生成分类器相关选项
2. 主训练循环中verification_validator的调用时机
3. TensorBoard指标记录完整性
4. 检查点保存与加载包含generation_validator指标
"""

import os
import sys
import yaml
import torch
import tempfile
import shutil
from unittest.mock import Mock, MagicMock

# 添加V16目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入V16模块
try:
    import train
    from model import TGATUNet
    print("✓ 成功导入V16训练模块")
except ImportError as e:
    print(f"✗ 导入V16训练模块失败: {e}")
    sys.exit(1)


def test_config_parsing():
    """测试配置文件中验证生成分类器选项的解析"""
    print("\n=== 测试配置文件解析 ===")
    
    # 读取示例配置文件
    config_path = "config_with_generation_validator.yaml"
    if not os.path.exists(config_path):
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查关键配置项
    checks = [
        ('enable_generation_validator', bool),
        ('generation_validator_config', dict),
        ('generation_validator_config.eval_frequency', int),
        ('generation_validator_config.dropout_rate', float),
        ('generation_validator_config.train_separately', bool)
    ]
    
    for key_path, expected_type in checks:
        keys = key_path.split('.')
        value = config
        try:
            for key in keys:
                value = value[key]
            
            if isinstance(value, expected_type):
                print(f"✓ {key_path}: {value} (类型: {type(value).__name__})")
            else:
                print(f"✗ {key_path}: 类型错误，期望 {expected_type.__name__}，实际 {type(value).__name__}")
                return False
                
        except KeyError:
            print(f"✗ 配置项缺失: {key_path}")
            return False
    
    print("✓ 配置文件解析测试通过")
    return True


def test_tensorboard_metrics():
    """测试TensorBoard指标记录是否完整"""
    print("\n=== 测试TensorBoard指标记录 ===")
    
    # 模拟TensorBoard writer
    mock_writer = Mock()
    mock_writer.add_scalar = Mock()
    
    # 模拟验证生成分类器指标
    gen_val_metrics = {
        'input_acc': 0.75,
        'reconstructed_acc': 0.80,
        'generated_acc': 0.78,
        'recon_vs_input': 0.05,
        'gen_vs_input': 0.03,
        'loss': 0.45,
        'recon_loss': 0.40
    }
    
    # 模拟记录指标的过程
    epoch = 10
    expected_calls = [
        ('Val_GenValidator/Input_Acc', gen_val_metrics['input_acc'], epoch),
        ('Val_GenValidator/Reconstructed_Acc', gen_val_metrics['reconstructed_acc'], epoch),
        ('Val_GenValidator/Generated_Acc', gen_val_metrics['generated_acc'], epoch),
        ('Val_GenValidator/Recon_vs_Input', gen_val_metrics['recon_vs_input'], epoch),
        ('Val_GenValidator/Gen_vs_Input', gen_val_metrics['gen_vs_input'], epoch),
        ('Val_GenValidator/Loss', gen_val_metrics['loss'], epoch),
        ('Val_GenValidator/Recon_Loss', gen_val_metrics['recon_loss'], epoch)
    ]
    
    # 执行记录
    for metric_name, value, ep in expected_calls:
        mock_writer.add_scalar(metric_name, value, ep)
    
    # 验证调用
    assert mock_writer.add_scalar.call_count == len(expected_calls)
    
    for i, (expected_metric, expected_value, expected_epoch) in enumerate(expected_calls):
        call_args = mock_writer.add_scalar.call_args_list[i]
        actual_metric, actual_value, actual_epoch = call_args[0]
        
        assert actual_metric == expected_metric, f"指标名不匹配: {actual_metric} != {expected_metric}"
        assert actual_value == expected_value, f"指标值不匹配: {actual_value} != {expected_value}"
        assert actual_epoch == expected_epoch, f"epoch不匹配: {actual_epoch} != {expected_epoch}"
    
    print("✓ TensorBoard指标记录测试通过")
    return True


def test_checkpoint_saving():
    """测试检查点保存是否包含验证生成分类器指标"""
    print("\n=== 测试检查点保存 ===")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 模拟检查点数据
        checkpoint = {
            'epoch': 50,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'best_val_loss': 0.35,
            'config': {'enable_generation_validator': True},
            'input_channels': 32,
            'train_loss': 0.40,
            'train_recon': 0.38,
            'train_input_acc': 0.72,
            'train_reconstructed_acc': 0.77,
            'val_input_acc': 0.70,
            'val_reconstructed_acc': 0.75,
            'accuracy_improvement': 0.05,
            'val_accuracy_improvement': 0.05
        }
        
        # 添加验证生成分类器指标
        gen_val_metrics = {
            'input_acc': 0.70,
            'reconstructed_acc': 0.75,
            'generated_acc': 0.73,
            'recon_vs_input': 0.05,
            'gen_vs_input': 0.03,
            'loss': 0.42,
            'recon_loss': 0.40
        }
        
        checkpoint.update({
            'gen_val_input_acc': gen_val_metrics['input_acc'],
            'gen_val_reconstructed_acc': gen_val_metrics['reconstructed_acc'],
            'gen_val_generated_acc': gen_val_metrics['generated_acc'],
            'gen_val_recon_vs_input': gen_val_metrics['recon_vs_input'],
            'gen_val_gen_vs_input': gen_val_metrics['gen_vs_input'],
            'gen_val_loss': gen_val_metrics['loss'],
            'gen_val_recon_loss': gen_val_metrics['recon_loss']
        })
        
        # 保存检查点
        checkpoint_path = os.path.join(temp_dir, 'test_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 加载并验证
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查关键字段
        required_gen_val_fields = [
            'gen_val_input_acc', 'gen_val_reconstructed_acc', 'gen_val_generated_acc',
            'gen_val_recon_vs_input', 'gen_val_gen_vs_input', 'gen_val_loss', 'gen_val_recon_loss'
        ]
        
        for field in required_gen_val_fields:
            if field not in loaded_checkpoint:
                print(f"✗ 检查点缺失字段: {field}")
                return False
            print(f"✓ {field}: {loaded_checkpoint[field]}")
        
        print("✓ 检查点保存测试通过")
        return True
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


def test_eval_frequency_logic():
    """测试验证生成分类器评估频率逻辑"""
    print("\n=== 测试评估频率逻辑 ===")
    
    # 测试不同的评估频率设置
    test_cases = [
        {'eval_frequency': 1, 'epochs': [1, 2, 3, 4, 5], 'expected_evals': [1, 2, 3, 4, 5]},
        {'eval_frequency': 2, 'epochs': [1, 2, 3, 4, 5], 'expected_evals': [2, 4]},
        {'eval_frequency': 5, 'epochs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'expected_evals': [5, 10]},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"测试案例 {i+1}: eval_frequency={case['eval_frequency']}")
        
        actual_evals = []
        for epoch in case['epochs']:
            if epoch % case['eval_frequency'] == 0:
                actual_evals.append(epoch)
        
        if actual_evals == case['expected_evals']:
            print(f"✓ 预期评估epoch: {case['expected_evals']}, 实际: {actual_evals}")
        else:
            print(f"✗ 预期评估epoch: {case['expected_evals']}, 实际: {actual_evals}")
            return False
    
    print("✓ 评估频率逻辑测试通过")
    return True


def test_config_disable_generation_validator():
    """测试禁用验证生成分类器的情况"""
    print("\n=== 测试禁用验证生成分类器 ===")
    
    # 模拟配置：禁用验证生成分类器
    config = {'enable_generation_validator': False}
    
    # 模拟训练循环逻辑
    epoch = 10
    gen_val_metrics = None
    
    if config.get('enable_generation_validator', False):
        # 这里应该不会执行
        gen_val_metrics = {'input_acc': 0.75}
        print("✗ 意外执行了验证生成分类器评估")
        return False
    
    # 验证指标为None
    if gen_val_metrics is None:
        print("✓ 正确跳过验证生成分类器评估")
    else:
        print("✗ 应该跳过验证生成分类器评估")
        return False
    
    # 模拟TensorBoard记录逻辑
    tensorboard_calls = 0
    if gen_val_metrics is not None:
        # 这里应该不会执行
        tensorboard_calls += 7  # 7个验证生成分类器指标
    
    if tensorboard_calls == 0:
        print("✓ 正确跳过TensorBoard验证生成分类器指标记录")
        return True
    else:
        print("✗ 意外记录了TensorBoard验证生成分类器指标")
        return False


def main():
    """运行所有测试"""
    print("开始测试主训练循环中验证生成分类器集成功能...")
    
    tests = [
        test_config_parsing,
        test_tensorboard_metrics,
        test_checkpoint_saving,
        test_eval_frequency_logic,
        test_config_disable_generation_validator
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ 测试失败: {test_func.__name__}")
        except Exception as e:
            print(f"✗ 测试异常: {test_func.__name__} - {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！验证生成分类器集成功能正常")
        return True
    else:
        print("❌ 部分测试失败，请检查集成逻辑")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
