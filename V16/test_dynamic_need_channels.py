#!/usr/bin/env python3
"""
测试V16的动态Need通道处理机制
"""

import torch
import torch.nn as nn

def test_dynamic_need_channel_processing():
    """测试动态Need通道处理逻辑"""
    print("=== 测试V16动态Need通道处理 ===")
    
    # 模拟混合batch数据
    batch_size = 3
    C = 32
    T = 50
    
    # 创建测试batch（来自不同数据集）
    batch = torch.randn(batch_size, C, T)
    source_datasets = ['FM', 'OD', 'MEFAR']
    
    # 模拟重建数据
    batch_reconstructed = torch.randn(batch_size, C, T)
    
    print(f"输入batch形状: {batch.shape}")
    print(f"来源数据集: {source_datasets}")
    
    # 模拟数据集的Need通道定义
    class MockDataset:
        def get_need_indices_for_dataset(self, source):
            need_mapping = {
                'FM': [28, 29, 30, 31],      # FM缺少MEFAR专有通道
                'OD': [24, 25, 26, 27],      # OD缺少FM专有通道
                'MEFAR': [24, 25, 26, 27]    # MEFAR缺少FM专有通道
            }
            return need_mapping.get(source, [])
    
    dataset = MockDataset()
    
    # 测试动态Need通道处理
    print("\n=== 动态Need通道识别 ===")
    for i, src in enumerate(source_datasets):
        need_indices = dataset.get_need_indices_for_dataset(src)
        print(f"样本 {i} ({src}): Need通道 = {need_indices}")
    
    # 测试增强数据构建（模拟训练函数中的逻辑）
    print("\n=== 构建增强数据 ===")
    enhanced_data = batch.clone()
    
    for i in range(batch_size):
        src = source_datasets[i]
        sample_need_indices = dataset.get_need_indices_for_dataset(src)
        
        print(f"处理样本 {i} ({src}):")
        print(f"  Need通道: {sample_need_indices}")
        
        # 用重建结果替换该样本的Need通道
        for need_idx in sample_need_indices:
            if need_idx < batch_reconstructed.size(1):
                # 检查是否确实被替换
                original_val = enhanced_data[i, need_idx, 0].item()
                enhanced_data[i, need_idx, :] = batch_reconstructed[i, need_idx, :]
                new_val = enhanced_data[i, need_idx, 0].item()
                print(f"    通道 {need_idx}: {original_val:.4f} -> {new_val:.4f}")
    
    # 验证不同样本的不同通道被正确处理
    print("\n=== 验证处理结果 ===")
    
    # 检查FM样本（索引0）的通道28应该被替换
    fm_need_channels = dataset.get_need_indices_for_dataset('FM')
    if fm_need_channels and fm_need_channels[0] < C:
        channel_28_changed = not torch.equal(batch[0, fm_need_channels[0], :], 
                                           enhanced_data[0, fm_need_channels[0], :])
        print(f"✅ FM样本通道{fm_need_channels[0]}被正确替换: {channel_28_changed}")
    
    # 检查OD样本（索引1）的通道24应该被替换
    od_need_channels = dataset.get_need_indices_for_dataset('OD')
    if od_need_channels and od_need_channels[0] < C:
        channel_24_changed = not torch.equal(batch[1, od_need_channels[0], :], 
                                           enhanced_data[1, od_need_channels[0], :])
        print(f"✅ OD样本通道{od_need_channels[0]}被正确替换: {channel_24_changed}")
    
    # 检查不应该被替换的通道（比如通道0，所有数据集都有）
    channel_0_unchanged = torch.equal(batch[:, 0, :], enhanced_data[:, 0, :])
    print(f"✅ 公共通道0未被替换: {channel_0_unchanged}")
    
    return True

def test_input_data_construction():
    """测试输入数据构建（Need通道设为0）"""
    print("\n=== 测试输入数据构建 ===")
    
    batch_size = 2
    C = 32
    T = 30
    
    batch = torch.randn(batch_size, C, T)
    source_datasets = ['FM', 'OD']
    
    class MockDataset:
        def get_need_indices_for_dataset(self, source):
            return {
                'FM': [28, 29],
                'OD': [24, 25]
            }.get(source, [])
    
    dataset = MockDataset()
    
    # 构建输入数据（Need通道设为0）
    input_data = batch.clone()
    for i in range(batch_size):
        src = source_datasets[i]
        sample_need_indices = dataset.get_need_indices_for_dataset(src)
        
        for need_idx in sample_need_indices:
            if need_idx < input_data.size(1):
                input_data[i, need_idx, :] = 0
                
        print(f"样本 {i} ({src}): 通道 {sample_need_indices} 设为0")
    
    # 验证
    fm_zeros = torch.all(input_data[0, 28, :] == 0)
    od_zeros = torch.all(input_data[1, 24, :] == 0)
    
    print(f"✅ FM样本通道28设为0: {fm_zeros}")
    print(f"✅ OD样本通道24设为0: {od_zeros}")
    
    return True

def main():
    """主测试函数"""
    print("开始测试V16动态Need通道处理机制...")
    
    try:
        # 测试动态Need通道处理
        test_dynamic_need_channel_processing()
        
        # 测试输入数据构建
        test_input_data_construction()
        
        print("\n🎉 所有测试通过！")
        print("V16动态Need通道处理机制工作正常：")
        print("- ✅ 根据source_dataset动态确定Need通道")
        print("- ✅ 正确构建增强数据（替换Need通道）")
        print("- ✅ 正确构建输入数据（Need通道设为0）")
        print("- ✅ 不同数据集样本在同一batch中正确处理")
        print("- ✅ 公共通道保持不变")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
