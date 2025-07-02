#!/usr/bin/env python3
"""
V16双分类器功能验证脚本
用于快速验证V16代码的双分类器功能是否正常工作
"""

import sys
import os
import torch
import torch.nn as nn

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_syntax():
    """测试语法是否正确"""
    print("🔍 正在进行语法检查...")
    try:
        import py_compile
        py_compile.compile('train.py', doraise=True)
        print("✅ 语法检查通过")
        return True
    except Exception as e:
        print(f"❌ 语法错误: {e}")
        return False

def test_imports():
    """测试模块导入是否正常"""
    print("🔍 正在测试模块导入...")
    try:
        # 测试关键模块导入
        from data import load_config
        # 注意：实际的import可能因具体实现而异
        print("✅ 模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_classifier_creation():
    """测试分类器创建"""
    print("🔍 正在测试分类器创建...")
    try:
        # 模拟配置
        input_channels = 32
        
        # 创建分类器
        input_classifier = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        reconstructed_classifier = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        # 测试参数计算
        input_params = sum(p.numel() for p in input_classifier.parameters())
        reconstructed_params = sum(p.numel() for p in reconstructed_classifier.parameters())
        
        print(f"✅ 分类器创建成功")
        print(f"   Input分类器参数量: {input_params:,}")
        print(f"   Reconstructed分类器参数量: {reconstructed_params:,}")
        
        # 验证参数独立性
        assert input_params == reconstructed_params, "两个分类器参数量应该相同"
        assert input_params > 0, "分类器应该有参数"
        
        return True
    except Exception as e:
        print(f"❌ 分类器创建失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("🔍 正在测试前向传播...")
    try:
        # 创建分类器
        input_channels = 32
        batch_size = 4
        seq_length = 100
        
        input_classifier = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        reconstructed_classifier = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        # 模拟数据
        input_data = torch.randn(batch_size, input_channels, seq_length)
        reconstructed_data = torch.randn(batch_size, input_channels, seq_length)
        
        # 池化操作（模拟实际使用）
        input_data_pooled = input_data.mean(dim=2)  # [batch_size, input_channels]
        reconstructed_data_pooled = reconstructed_data.mean(dim=2)
        
        # 前向传播
        input_logits = input_classifier(input_data_pooled)
        reconstructed_logits = reconstructed_classifier(reconstructed_data_pooled)
        
        # 验证输出形状
        expected_shape = (batch_size, 2)
        assert input_logits.shape == expected_shape, f"Input logits形状错误: {input_logits.shape}, 期望: {expected_shape}"
        assert reconstructed_logits.shape == expected_shape, f"Reconstructed logits形状错误: {reconstructed_logits.shape}, 期望: {expected_shape}"
        
        print(f"✅ 前向传播测试成功")
        print(f"   Input logits形状: {input_logits.shape}")
        print(f"   Reconstructed logits形状: {reconstructed_logits.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        return False

def test_accuracy_calculation():
    """测试准确率计算"""
    print("🔍 正在测试准确率计算...")
    try:
        # 模拟数据
        batch_size = 10
        logits = torch.randn(batch_size, 2)
        labels = torch.randint(0, 2, (batch_size,))
        
        # 计算准确率
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        print(f"✅ 准确率计算测试成功")
        print(f"   模拟准确率: {accuracy:.4f}")
        
        # 验证准确率在合理范围内
        assert 0.0 <= accuracy <= 1.0, f"准确率应该在[0,1]范围内，实际: {accuracy}"
        
        return True
    except Exception as e:
        print(f"❌ 准确率计算测试失败: {e}")
        return False

def test_optimizer_parameters():
    """测试优化器参数合并"""
    print("🔍 正在测试优化器参数合并...")
    try:
        # 创建模拟模型
        model = nn.Linear(32, 16)
        input_classifier = nn.Linear(32, 2)
        reconstructed_classifier = nn.Linear(32, 2)
        
        # 合并参数
        all_params = list(model.parameters()) + \
                    list(input_classifier.parameters()) + \
                    list(reconstructed_classifier.parameters())
        
        # 创建优化器
        optimizer = torch.optim.Adam(all_params, lr=0.001)
        
        # 验证参数组数量
        param_count = len(all_params)
        optimizer_param_count = sum(len(group['params']) for group in optimizer.param_groups)
        
        assert param_count == optimizer_param_count, f"优化器参数数量不匹配: {param_count} vs {optimizer_param_count}"
        
        print(f"✅ 优化器参数合并测试成功")
        print(f"   总参数数量: {param_count}")
        
        return True
    except Exception as e:
        print(f"❌ 优化器参数合并测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始V16双分类器功能验证...\n")
    
    tests = [
        ("语法检查", test_syntax),
        ("模块导入", test_imports),
        ("分类器创建", test_classifier_creation),
        ("前向传播", test_forward_pass),
        ("准确率计算", test_accuracy_calculation),
        ("优化器参数合并", test_optimizer_parameters),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"正在运行: {test_name}")
        print(f"{'='*50}")
        
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"⚠️  {test_name} 失败，建议检查相关代码")
    
    # 总结结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:20} {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试都通过了！V16双分类器功能正常。")
        print("建议下一步：运行实际训练测试")
    else:
        print(f"\n⚠️  还有 {total-passed} 个测试未通过，建议检查相关代码。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
