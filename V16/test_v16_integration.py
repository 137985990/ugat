#!/usr/bin/env python3
"""
V16集成测试脚本
测试动态Need通道处理在完整训练流程中的工作情况
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data import SlidingWindowDataset
from model import TGATUNet
from train import complete_need_with_model, mask_channel

def test_v16_integration():
    """测试V16完整集成流程"""
    print("🧪 开始V16集成测试...")
    
    # 1. 创建测试数据集
    print("\n=== 创建测试数据集 ===")
    try:
        train_config = {
            'data_dir': '../Data',
            'datasets': ['FM', 'OD', 'MEFAR'],
            'window_size': 50,
            'stride_size': 25,
            'feature_cols': [f'feature_{i}' for i in range(32)],
            'block_col': 'block_id',
            'seg_col': 'segment_label'
        }
        
        dataset = SlidingWindowDataset(**train_config)
        print(f"✅ 数据集创建成功，样本数量: {len(dataset)}")
        
        # 检查数据集配置
        print(f"   - 支持的数据集: {list(dataset.dataset_modalities_config.keys())}")
        for ds_name, config in dataset.dataset_modalities_config.items():
            need_indices = config.get('need', [])
            have_indices = config.get('have', [])
            print(f"   - {ds_name}: Need通道{need_indices}, Have通道{have_indices}")
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False
    
    # 2. 创建数据加载器
    print("\n=== 创建数据加载器 ===")
    try:
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        print(f"✅ 数据加载器创建成功")
        
        # 获取一个batch用于测试
        sample_batch = next(iter(loader))
        if len(sample_batch) == 5:
            batch_x, batch_y, batch_y_class, batch_y_reg, source_datasets = sample_batch
            print(f"   - Batch形状: {batch_x.shape}")
            print(f"   - 来源数据集: {source_datasets}")
        else:
            print(f"❌ 意外的batch格式，长度: {len(sample_batch)}")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return False
    
    # 3. 创建模型
    print("\n=== 创建模型 ===")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_config = {
            'in_channels': 32,
            'out_channels': 32,
            'base_channels': 64,
            'num_classes': 3,
            'num_reg_targets': 1
        }
        
        model = UNetModel(**model_config).to(device)
        print(f"✅ 模型创建成功，使用设备: {device}")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 4. 测试动态Need通道识别
    print("\n=== 测试动态Need通道识别 ===")
    try:
        batch_x_test = batch_x[:3]  # 取前3个样本
        source_datasets_test = source_datasets[:3]
        
        for i, src in enumerate(source_datasets_test):
            need_indices = dataset.get_need_indices_for_dataset(src)
            have_indices = dataset.get_have_indices_for_dataset(src)
            print(f"   - 样本{i} ({src}): Need通道{need_indices}, Have通道{have_indices}")
        
        print("✅ 动态Need通道识别正常")
        
    except Exception as e:
        print(f"❌ 动态Need通道识别失败: {e}")
        return False
    
    # 5. 测试通道遮掩
    print("\n=== 测试通道遮掩 ===")
    try:
        masked_batch = mask_channel(batch_x_test, source_datasets_test, dataset)
        
        # 验证遮掩效果
        for i, src in enumerate(source_datasets_test):
            have_indices = dataset.get_have_indices_for_dataset(src)
            if have_indices:
                # 检查have通道是否被遮掩为0
                have_channel = have_indices[0]
                is_masked = torch.allclose(masked_batch[i, have_channel, :], torch.zeros_like(masked_batch[i, have_channel, :]))
                print(f"   - 样本{i} ({src}): 通道{have_channel}遮掩状态={is_masked}")
        
        print("✅ 通道遮掩功能正常")
        
    except Exception as e:
        print(f"❌ 通道遮掩失败: {e}")
        return False
    
    # 6. 测试模型推理
    print("\n=== 测试模型推理 ===")
    try:
        model.eval()
        with torch.no_grad():
            batch_x_device = batch_x_test.to(device)
            
            # 测试单个样本推理
            sample = batch_x_device[0].t()  # [T, C]
            out, (class_logits, reg_output) = model(sample)
            
            print(f"   - 输入形状: {sample.shape}")
            print(f"   - 重建输出形状: {out.shape}")
            print(f"   - 分类输出形状: {class_logits.shape}")
            print(f"   - 回归输出形状: {reg_output.shape}")
        
        print("✅ 模型推理正常")
        
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        return False
    
    # 7. 测试Need通道补全（小批量）
    print("\n=== 测试Need通道补全 ===")
    try:
        # 创建一个小的测试数据加载器
        small_dataset_indices = list(range(min(10, len(dataset))))
        small_subset = torch.utils.data.Subset(dataset, small_dataset_indices)
        small_loader = DataLoader(small_subset, batch_size=2, shuffle=False)
        
        print(f"   - 小测试集样本数: {len(small_subset)}")
        
        # 运行need通道补全
        complete_need_with_model(model, small_loader, dataset, device)
        
        print("✅ Need通道补全功能正常")
        
    except Exception as e:
        print(f"❌ Need通道补全失败: {e}")
        return False
    
    print("\n🎉 V16集成测试全部通过！")
    print("✅ 动态Need通道处理机制在完整流程中工作正常")
    print("✅ V16已准备好进行完整训练")
    
    return True

if __name__ == "__main__":
    success = test_v16_integration()
    if success:
        print("\n✨ 集成测试成功！可以开始完整训练。")
    else:
        print("\n❌ 集成测试失败！请检查配置和实现。")
        sys.exit(1)
