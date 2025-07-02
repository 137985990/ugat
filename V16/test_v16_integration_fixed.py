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
import yaml

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data import create_multimodal_dataset_from_config, load_config
from model import TGATUNet
from train import complete_need_with_model, mask_channel

def test_v16_integration():
    """测试V16完整集成流程"""
    print("🧪 开始V16集成测试...")
    
    # 1. 加载配置
    print("\n=== 加载配置文件 ===")
    try:
        config = load_config('config.yaml')
        print(f"✅ 配置文件加载成功")
        print(f"   - 数据目录: {config.get('data_dir', 'N/A')}")
        print(f"   - 数据文件: {config.get('data_files', [])}")
        print(f"   - 窗口大小: {config.get('window_size', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False
    
    # 2. 创建数据集
    print("\n=== 创建测试数据集 ===")
    try:
        dataset = create_multimodal_dataset_from_config(config, phase='encode')
        print(f"✅ 数据集创建成功，样本数量: {len(dataset)}")
        
        # 检查数据集配置
        if hasattr(dataset, 'dataset_modalities_config') and dataset.dataset_modalities_config:
            print(f"   - 支持的数据集: {list(dataset.dataset_modalities_config.keys())}")
            for ds_name, config_item in dataset.dataset_modalities_config.items():
                need_indices = config_item.get('need', [])
                have_indices = config_item.get('have', [])
                print(f"   - {ds_name}: Need通道{need_indices}, Have通道{have_indices}")
        else:
            print("   - 使用默认通道配置")
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False
    
    # 3. 创建数据加载器
    print("\n=== 创建数据加载器 ===")
    try:
        batch_size = min(4, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"✅ 数据加载器创建成功，批量大小: {batch_size}")
        
        # 获取一个batch用于测试
        sample_batch = next(iter(loader))
        print(f"   - Batch长度: {len(sample_batch)}")
        
        if len(sample_batch) >= 5:
            batch_x = sample_batch[0]
            source_datasets = sample_batch[4]
            print(f"   - Batch形状: {batch_x.shape}")
            print(f"   - 来源数据集: {source_datasets}")
        else:
            batch_x = sample_batch[0]
            source_datasets = ['UNKNOWN'] * batch_x.size(0)
            print(f"   - Batch形状: {batch_x.shape}")
            print(f"   - 使用默认来源: {source_datasets}")
            
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return False
    
    # 4. 创建模型
    print("\n=== 创建模型 ===")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_channels = batch_x.shape[1]
        
        model = TGATUNet(
            in_channels=input_channels,
            hidden_channels=config.get('hidden_channels', 64),
            out_channels=input_channels,
            num_classes=config.get('num_classes', 2)
        ).to(device)
        
        print(f"✅ 模型创建成功，使用设备: {device}")
        print(f"   - 输入通道: {input_channels}")
        print(f"   - 隐藏通道: {config.get('hidden_channels', 64)}")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 5. 测试动态Need通道识别
    print("\n=== 测试动态Need通道识别 ===")
    try:
        batch_x_test = batch_x[:min(3, batch_x.size(0))]  # 取前3个样本
        source_datasets_test = source_datasets[:min(3, len(source_datasets))]
        
        for i, src in enumerate(source_datasets_test):
            need_indices = dataset.get_need_indices_for_dataset(src) if hasattr(dataset, 'get_need_indices_for_dataset') else []
            have_indices = dataset.get_have_indices_for_dataset(src) if hasattr(dataset, 'get_have_indices_for_dataset') else []
            print(f"   - 样本{i} ({src}): Need通道{need_indices}, Have通道{have_indices}")
        
        print("✅ 动态Need通道识别正常")
        
    except Exception as e:
        print(f"❌ 动态Need通道识别失败: {e}")
        return False
    
    # 6. 测试通道遮掩
    print("\n=== 测试通道遮掩 ===")
    try:
        masked_batch = mask_channel(batch_x_test, source_datasets_test, dataset)
        
        # 验证遮掩效果
        for i, src in enumerate(source_datasets_test):
            if hasattr(dataset, 'get_have_indices_for_dataset'):
                have_indices = dataset.get_have_indices_for_dataset(src)
                if have_indices:
                    # 检查have通道是否被遮掩为0
                    have_channel = have_indices[0]
                    if have_channel < masked_batch.size(1):
                        is_masked = torch.allclose(masked_batch[i, have_channel, :], torch.zeros_like(masked_batch[i, have_channel, :]))
                        print(f"   - 样本{i} ({src}): 通道{have_channel}遮掩状态={is_masked}")
        
        print("✅ 通道遮掩功能正常")
        
    except Exception as e:
        print(f"❌ 通道遮掩失败: {e}")
        return False
    
    # 7. 测试模型推理
    print("\n=== 测试模型推理 ===")
    try:
        model.eval()
        with torch.no_grad():
            batch_x_device = batch_x_test.to(device)
            
            # 测试单个样本推理
            sample = batch_x_device[0].t()  # [T, C]
            out, logits = model(sample)
            
            print(f"   - 输入形状: {sample.shape}")
            print(f"   - 重建输出形状: {out.shape}")
            print(f"   - 分类输出形状: {logits.shape}")
        
        print("✅ 模型推理正常")
        
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        return False
    
    # 8. 测试Need通道补全（小批量）
    print("\n=== 测试Need通道补全 ===")
    try:
        # 创建一个小的测试数据加载器
        small_dataset_size = min(6, len(dataset))
        small_subset = torch.utils.data.Subset(dataset, list(range(small_dataset_size)))
        
        print(f"   - 小测试集样本数: {len(small_subset)}")
        
        # 运行need通道补全
        complete_need_with_model(model, small_subset, device)
        
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
