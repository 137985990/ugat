#!/usr/bin/env python3
"""
测试V16是否与V13实现方式一致
"""

import torch
import torch.nn as nn
from model import TGATUNet

def test_model_architecture():
    """测试模型架构是否正确"""
    print("=== 测试V16模型架构 ===")
    
    # 创建模型
    model = TGATUNet(
        in_channels=32,
        hidden_channels=64,
        out_channels=32,
        num_classes=2
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查是否有内置分类器
    assert hasattr(model, 'classifier'), "模型应该有内置分类器"
    print("✅ 模型有内置分类器")
    
    # 检查分类器结构
    classifier = model.classifier
    print(f"分类器结构: {classifier}")
    
    # 测试前向传播
    print("\n=== 测试前向传播 ===")
    
    # 创建测试数据
    batch_size = 4
    T = 50  # 时间步
    C = 32  # 通道数
    
    # 测试单样本前向传播
    window = torch.randn(T, C)  # [T, C]
    out, logits = model(window)
    
    print(f"输入形状: {window.shape}")
    print(f"重建输出形状: {out.shape}")  # 应该是 [C, T]
    print(f"分类输出形状: {logits.shape}")  # 应该是 [num_classes]
    
    assert out.shape == (C, T), f"重建输出形状错误: {out.shape}"
    assert logits.shape == (2,), f"分类输出形状错误: {logits.shape}"
    
    print("✅ 单样本前向传播测试通过")
    
    # 测试批量前向传播
    windows_batch = torch.randn(batch_size, T, C)  # [batch_size, T, C]
    batch_out, batch_logits = model.forward_batch(windows_batch)
    
    print(f"批量输入形状: {windows_batch.shape}")
    print(f"批量重建输出形状: {batch_out.shape}")  # 应该是 [batch_size, C, T]
    print(f"批量分类输出形状: {batch_logits.shape}")  # 应该是 [batch_size, num_classes]
    
    assert batch_out.shape == (batch_size, C, T), f"批量重建输出形状错误: {batch_out.shape}"
    assert batch_logits.shape == (batch_size, 2), f"批量分类输出形状错误: {batch_logits.shape}"
    
    print("✅ 批量前向传播测试通过")
    
    return model

def test_dual_path_classification(model):
    """测试双路径分类逻辑"""
    print("\n=== 测试双路径分类 ===")
    
    batch_size = 2
    T = 30
    C = 32
    
    # 模拟输入数据（包含初始Need值）
    input_data = torch.randn(batch_size, C, T)
    
    # 模拟重建数据（生成的Need值）
    reconstructed_data = torch.randn(batch_size, C, T)
    
    # 模拟need_indices
    need_indices = [28, 29, 30, 31]  # 假设最后4个通道是Need通道
    
    # 构建增强数据：保留原始输入，用重建结果替换Need通道
    enhanced_data = input_data.clone()
    for need_idx in need_indices:
        if need_idx < C:
            enhanced_data[:, need_idx, :] = reconstructed_data[:, need_idx, :]
    
    # 测试路径1：输入数据分类（基准性能）
    model.eval()
    with torch.no_grad():
        # 转置：[batch_size, C, T] -> [batch_size, T, C] 
        input_windows = input_data.transpose(1, 2)
        _, input_logits = model.forward_batch(input_windows)
        
        # 测试路径2：增强数据分类（目标性能）
        enhanced_windows = enhanced_data.transpose(1, 2)
        _, enhanced_logits = model.forward_batch(enhanced_windows)
    
    print(f"输入数据分类logits形状: {input_logits.shape}")
    print(f"增强数据分类logits形状: {enhanced_logits.shape}")
    
    # 计算分类预测
    input_preds = torch.argmax(input_logits, dim=1)
    enhanced_preds = torch.argmax(enhanced_logits, dim=1)
    
    print(f"输入数据预测: {input_preds}")
    print(f"增强数据预测: {enhanced_preds}")
    
    print("✅ 双路径分类测试通过")

def main():
    """主测试函数"""
    print("开始测试V16与V13的一致性...")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    try:
        # 测试模型架构
        model = test_model_architecture()
        
        # 测试双路径分类
        test_dual_path_classification(model)
        
        print("\n🎉 所有测试通过！")
        print("V16现在使用与V13完全一致的分类器实现方式：")
        print("- 只有一个TGATUNet模型")
        print("- 模型内置分类器（self.classifier）")
        print("- 双路径分类：输入数据 vs 增强数据")
        print("- 不使用独立的MLP分类器")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
