#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_generation_validator.py

测试验证生成分类器功能的脚本
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import TGATUNet
from data import create_multimodal_dataset_from_config, load_config
from train import (
    forward_batch_parallel, 
    eval_with_generation_validator, 
    train_generation_validator_only,
    custom_collate_fn
)

def test_generation_validator():
    """测试验证生成分类器的基本功能"""
    print("=" * 60)
    print("测试验证生成分类器功能")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    try:
        config = load_config('config.yaml')
        print("✓ 配置文件加载成功")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False
    
    # 创建数据集
    try:
        dataset = create_multimodal_dataset_from_config(config, phase='encode')
        print(f"✓ 数据集创建成功，样本数: {len(dataset)}")
        
        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, 
                               collate_fn=custom_collate_fn, num_workers=0)
        print("✓ DataLoader创建成功")
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        return False
    
    # 创建模型
    try:
        # 从数据集获取输入通道数
        sample_batch = next(iter(dataloader))
        input_channels = sample_batch[0].size(1)
        
        model = TGATUNet(
            in_channels=input_channels,
            hidden_channels=64,
            out_channels=input_channels,
            num_classes=2
        ).to(device)
        print(f"✓ 模型创建成功，输入通道: {input_channels}")
        
        # 检查验证生成分类器是否存在
        if hasattr(model, 'generation_validator'):
            print("✓ 验证生成分类器模块存在")
        else:
            print("✗ 验证生成分类器模块不存在")
            return False
            
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False
    
    # 测试前向传播
    try:
        print("\n--- 测试前向传播 ---")
        model.eval()
        
        with torch.no_grad():
            # 测试标准前向传播
            batch_x, batch_y, _, _, _ = sample_batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            print(f"输入batch形状: {batch_x.shape}")
            
            # 测试不使用验证生成分类器
            result1 = forward_batch_parallel(model, batch_x, device, use_generation_validator=False)
            print(f"标准前向传播结果: {len(result1)}个输出")
            print(f"  - 重建输出形状: {result1[0].shape}")
            print(f"  - 主分类器输出形状: {result1[1].shape}")
            
            # 测试使用验证生成分类器
            result2 = forward_batch_parallel(model, batch_x, device, use_generation_validator=True)
            print(f"验证生成分类器前向传播结果: {len(result2)}个输出")
            print(f"  - 重建输出形状: {result2[0].shape}")
            print(f"  - 主分类器输出形状: {result2[1].shape}")
            if len(result2) >= 3:
                print(f"  - 验证生成分类器输出形状: {result2[2].shape}")
            
            print("✓ 前向传播测试成功")
            
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试分类性能对比
    try:
        print("\n--- 测试分类性能对比 ---")
        
        # 计算主分类器准确率
        main_preds = torch.argmax(result1[1], dim=1)
        main_acc = (main_preds == batch_y).float().mean().item()
        print(f"主分类器准确率: {main_acc:.4f}")
        
        # 计算验证生成分类器准确率
        if len(result2) >= 3:
            gen_preds = torch.argmax(result2[2], dim=1)
            gen_acc = (gen_preds == batch_y).float().mean().item()
            print(f"验证生成分类器准确率: {gen_acc:.4f}")
            
            # 分析差异
            diff = gen_acc - main_acc
            print(f"准确率差异: {diff:+.4f}")
            
            if abs(diff) < 0.1:
                print("✓ 两个分类器性能相近")
            else:
                print(f"⚠ 两个分类器性能差异较大")
        else:
            print("✗ 验证生成分类器输出缺失")
            
    except Exception as e:
        print(f"✗ 分类性能对比失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试梯度控制
    try:
        print("\n--- 测试梯度控制 ---")
        
        # 测试冻结验证生成分类器
        model.freeze_generation_validator()
        gen_grad_before = any(p.requires_grad for p in model.generation_validator.parameters())
        main_grad_before = any(p.requires_grad for p in model.classifier.parameters())
        
        print(f"冻结后 - 验证生成分类器梯度: {gen_grad_before}")
        print(f"冻结后 - 主分类器梯度: {main_grad_before}")
        
        # 测试解冻所有参数
        model.unfreeze_all()
        gen_grad_after = any(p.requires_grad for p in model.generation_validator.parameters())
        main_grad_after = any(p.requires_grad for p in model.classifier.parameters())
        
        print(f"解冻后 - 验证生成分类器梯度: {gen_grad_after}")
        print(f"解冻后 - 主分类器梯度: {main_grad_after}")
        
        if not gen_grad_before and gen_grad_after and main_grad_after:
            print("✓ 梯度控制测试成功")
        else:
            print("✗ 梯度控制测试失败")
            
    except Exception as e:
        print(f"✗ 梯度控制测试失败: {e}")
    
    # 测试分阶段训练
    try:
        print("\n--- 测试分阶段训练 ---")
        
        # 创建优化器（只优化验证生成分类器）
        gen_params = model.get_generation_validator_params()
        optimizer = torch.optim.Adam(gen_params, lr=1e-4)
        criterion = nn.MSELoss()
        
        print(f"验证生成分类器参数数量: {sum(p.numel() for p in gen_params)}")
        
        # 运行一个训练步骤
        model.train()
        
        # 只取少量数据测试
        small_dataloader = DataLoader(dataset, batch_size=4, shuffle=False, 
                                     collate_fn=custom_collate_fn, num_workers=0)
        
        # 保存训练前的参数
        gen_params_before = [p.clone() for p in model.generation_validator.parameters()]
        main_params_before = [p.clone() for p in model.classifier.parameters()]
        
        # 执行一次训练
        train_loss, train_acc = train_generation_validator_only(
            model, small_dataloader, optimizer, criterion, device, [],
            accumulate_grad_batches=1, use_mixed_precision=False, 
            dataset=dataset, current_epoch=1
        )
        
        print(f"验证生成分类器训练 - 损失: {train_loss:.6f}, 准确率: {train_acc:.4f}")
        
        # 检查参数是否更新
        gen_params_after = list(model.generation_validator.parameters())
        main_params_after = list(model.classifier.parameters())
        
        gen_updated = any(not torch.equal(before, after) for before, after in zip(gen_params_before, gen_params_after))
        main_updated = any(not torch.equal(before, after) for before, after in zip(main_params_before, main_params_after))
        
        print(f"验证生成分类器参数更新: {gen_updated}")
        print(f"主分类器参数更新: {main_updated}")
        
        if gen_updated and not main_updated:
            print("✓ 分阶段训练测试成功")
        else:
            print("⚠ 分阶段训练测试结果异常")
            
    except Exception as e:
        print(f"✗ 分阶段训练测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("验证生成分类器测试完成")
    print("=" * 60)
    
    return True

def test_eval_with_generation_validator():
    """测试带验证生成分类器的评估函数"""
    print("\n--- 测试增强评估函数 ---")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 加载配置和数据
        config = load_config('config.yaml')
        dataset = create_multimodal_dataset_from_config(config, phase='encode')
        
        # 创建小数据集用于测试
        from torch.utils.data import Subset
        test_indices = list(range(min(20, len(dataset))))
        test_dataset = Subset(dataset, test_indices)
        
        dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
                               collate_fn=custom_collate_fn, num_workers=0)
        
        # 创建模型
        sample_batch = next(iter(dataloader))
        input_channels = sample_batch[0].size(1)
        
        model = TGATUNet(
            in_channels=input_channels,
            hidden_channels=32,  # 小一些以便快速测试
            out_channels=input_channels,
            num_classes=2
        ).to(device)
        
        # 创建损失函数
        criterion = nn.MSELoss()
        
        # 运行增强评估
        # 处理Subset包装的数据集
        try:
            # 尝试获取原始数据集（如果是Subset）
            original_dataset = dataset.dataset
        except AttributeError:
            # 这是原始数据集
            original_dataset = dataset
            
        results = eval_with_generation_validator(
            model, dataloader, criterion, device, [], need_indices=[], dataset=original_dataset
        )
        
        total_loss, total_recon_loss, input_acc, reconstructed_acc, generation_validator_acc = results
        
        print(f"评估结果:")
        print(f"  - 总损失: {total_loss:.6f}")
        print(f"  - 重建损失: {total_recon_loss:.6f}")
        print(f"  - 输入数据准确率: {input_acc:.4f}")
        print(f"  - 重建数据准确率: {reconstructed_acc:.4f}")
        print(f"  - 验证生成分类器准确率: {generation_validator_acc:.4f}")
        
        # 分析结果
        main_vs_gen_diff = generation_validator_acc - reconstructed_acc
        input_vs_recon_diff = reconstructed_acc - input_acc
        
        print(f"分析:")
        print(f"  - 验证生成 vs 主分类器差异: {main_vs_gen_diff:+.4f}")
        print(f"  - 重建 vs 输入数据差异: {input_vs_recon_diff:+.4f}")
        
        print("✓ 增强评估函数测试成功")
        
    except Exception as e:
        print(f"✗ 增强评估函数测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行测试
    success = test_generation_validator()
    if success:
        test_eval_with_generation_validator()
    else:
        print("基础测试失败，跳过增强评估测试")
