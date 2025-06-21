# loss_analysis.py - 损失函数分析和对比

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from enhanced_loss import MultiTaskLoss, create_enhanced_loss
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def analyze_current_loss():
    """分析当前损失函数的特点"""
    
    print("=" * 60)
    print("当前损失函数分析")
    print("=" * 60)
    
    print("\n1. 损失函数组成:")
    print("   - 重建损失: MSE Loss")
    print("   - 分类损失: CrossEntropy Loss")
    print("   - 总损失: reconstruction_loss + classification_loss")
    
    print("\n2. 计算特点:")
    print("   - 逐样本计算重建损失")
    print("   - 只对真实通道计算重建损失（通过is_real_mask）")
    print("   - 固定权重（1:1）")
    print("   - 梯度裁剪（max_norm=1.0）")
    
    print("\n3. 优点:")
    print("   ✓ 多任务学习（重建+分类）")
    print("   ✓ 通道级masking避免噪声")
    print("   ✓ 简单稳定的训练过程")
    
    print("\n4. 局限性:")
    print("   ✗ 缺乏正则化约束")
    print("   ✗ 固定权重无法适应训练阶段")
    print("   ✗ 没有时序连续性考虑")
    print("   ✗ 缺乏频域特征保持")

def demonstrate_enhanced_loss():
    """演示增强版损失函数的使用"""
    
    print("\n" + "=" * 60)
    print("增强版损失函数演示")
    print("=" * 60)
    
    # 配置参数
    config = {
        'recon_weight': 1.0,
        'cls_weight': 1.0,
        'consistency_weight': 0.1,
        'temporal_weight': 0.1,
        'spectral_weight': 0.05
    }
    
    # 创建增强损失函数
    enhanced_loss = create_enhanced_loss(config)
    
    # 模拟数据
    batch_size, num_channels, seq_len = 8, 12, 100
    num_classes = 3
    
    predictions = torch.randn(batch_size, num_channels, seq_len)
    targets = torch.randn(batch_size, num_channels, seq_len)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    real_mask = torch.ones(batch_size, num_channels)
    
    # 计算损失
    loss_dict = enhanced_loss(predictions, targets, logits, labels, real_mask)
    
    print("\n增强版损失函数组件:")
    for key, value in loss_dict.items():
        print(f"   {key}: {value.item():.6f}")
    
    print(f"\n总损失: {loss_dict['total_loss'].item():.6f}")

def compare_loss_functions():
    """对比当前损失函数和增强版损失函数"""
    
    print("\n" + "=" * 60)
    print("损失函数对比")
    print("=" * 60)
    
    # 模拟数据
    batch_size, num_channels, seq_len = 4, 12, 100
    num_classes = 3
    
    predictions = torch.randn(batch_size, num_channels, seq_len)
    targets = torch.randn(batch_size, num_channels, seq_len)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    real_mask = torch.ones(batch_size, num_channels)
    
    # 当前损失函数
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    current_recon = mse_loss(predictions, targets)
    current_cls = ce_loss(logits, labels)
    current_total = current_recon + current_cls
    
    print(f"\n当前损失函数:")
    print(f"   重建损失: {current_recon.item():.6f}")
    print(f"   分类损失: {current_cls.item():.6f}")
    print(f"   总损失: {current_total.item():.6f}")
    
    # 增强版损失函数
    config = {
        'recon_weight': 1.0,
        'cls_weight': 1.0,
        'consistency_weight': 0.1,
        'temporal_weight': 0.1,
        'spectral_weight': 0.05
    }
    enhanced_loss = create_enhanced_loss(config)
    loss_dict = enhanced_loss(predictions, targets, logits, labels, real_mask)
    
    print(f"\n增强版损失函数:")
    print(f"   重建损失: {loss_dict['reconstruction_loss'].item():.6f}")
    print(f"   分类损失: {loss_dict['classification_loss'].item():.6f}")
    print(f"   一致性损失: {loss_dict['consistency_loss'].item():.6f}")
    print(f"   时序损失: {loss_dict['temporal_loss'].item():.6f}")
    print(f"   频域损失: {loss_dict['spectral_loss'].item():.6f}")
    print(f"   总损失: {loss_dict['total_loss'].item():.6f}")

def visualize_weight_scheduling():
    """可视化损失权重调度"""
    
    print("\n" + "=" * 60)
    print("损失权重调度可视化")
    print("=" * 60)
    
    enhanced_loss = MultiTaskLoss()
    epochs = 100
    
    recon_weights = []
    cls_weights = []
    consistency_weights = []
    temporal_weights = []
    
    for epoch in range(epochs):
        enhanced_loss.update_weights(epoch, epochs)
        recon_weights.append(enhanced_loss.recon_weight)
        cls_weights.append(enhanced_loss.cls_weight)
        consistency_weights.append(enhanced_loss.consistency_weight)
        temporal_weights.append(enhanced_loss.temporal_weight)
    
    # 绘制权重变化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(recon_weights, label='Reconstruction Weight', linewidth=2)
    plt.plot(cls_weights, label='Classification Weight', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Main Task Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(consistency_weights, label='Consistency Weight', linewidth=2, color='green')
    plt.plot(temporal_weights, label='Temporal Weight', linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Regularization Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, (3, 4))
    plt.plot(recon_weights, label='Reconstruction', linewidth=2)
    plt.plot(cls_weights, label='Classification', linewidth=2)
    plt.plot(consistency_weights, label='Consistency', linewidth=2)
    plt.plot(temporal_weights, label='Temporal', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('All Weights Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('e:/NEW/V11更新算法/loss_weights_scheduling.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("权重调度图已保存到: loss_weights_scheduling.png")

def loss_function_recommendations():
    """损失函数优化建议"""
    
    print("\n" + "=" * 60)
    print("损失函数优化建议")
    print("=" * 60)
    
    print("\n1. 短期优化（保持当前架构）:")
    print("   ✓ 添加梯度监控和异常检测")
    print("   ✓ 引入动态权重调整")
    print("   ✓ 添加时序连续性约束")
    print("   ✓ 实现损失可视化监控")
    
    print("\n2. 中期改进（部分采用增强功能）:")
    print("   ✓ 集成一致性损失")
    print("   ✓ 添加L1正则化减少过拟合")
    print("   ✓ 实现curriculum learning权重调度")
    print("   ✓ 引入频域损失保持频域特征")
    
    print("\n3. 长期重构（完全增强版）:")
    print("   ✓ 多重建损失组合（MSE+L1+Huber）")
    print("   ✓ 自适应权重调整机制")
    print("   ✓ 对抗训练（可选）")
    print("   ✓ 元学习优化损失权重")
    
    print("\n4. 实施步骤:")
    print("   Step 1: 替换当前MSE为MultiTaskLoss")
    print("   Step 2: 配置适当的权重参数")
    print("   Step 3: 监控训练稳定性")
    print("   Step 4: 根据结果调整权重")

def create_integration_example():
    """创建集成示例代码"""
    
    integration_code = '''
# 在train.py中集成增强版损失函数的示例

from enhanced_loss import create_enhanced_loss

def setup_enhanced_loss(config):
    """设置增强版损失函数"""
    loss_config = {
        'recon_weight': config.get('recon_weight', 1.0),
        'cls_weight': config.get('cls_weight', 1.0),
        'consistency_weight': config.get('consistency_weight', 0.1),
        'temporal_weight': config.get('temporal_weight', 0.1),
        'spectral_weight': config.get('spectral_weight', 0.05)
    }
    return create_enhanced_loss(loss_config)

def train_with_enhanced_loss(model, dataloader, optimizer, enhanced_loss, device, mask_indices):
    """使用增强版损失函数的训练循环"""
    model.train()
    total_losses = {}
    
    for batch, labels, mask_idx, is_real_mask in dataloader:
        batch = batch.to(device)
        labels = labels.to(device)
        is_real_mask = is_real_mask.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        predictions, logits = model(batch)
        
        # 计算增强损失
        loss_dict = enhanced_loss(
            predictions=predictions,
            targets=batch, 
            logits=logits,
            labels=labels,
            real_mask=is_real_mask
        )
        
        # 反向传播
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累积损失
        for key, value in loss_dict.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value.item()
    
    return total_losses

# 使用示例
if __name__ == "__main__":
    # 设置增强损失函数
    enhanced_loss = setup_enhanced_loss(config)
    
    # 训练循环中更新权重
    for epoch in range(num_epochs):
        enhanced_loss.update_weights(epoch, num_epochs)
        
        # 训练
        train_losses = train_with_enhanced_loss(
            model, train_loader, optimizer, enhanced_loss, device, mask_indices
        )
        
        # 日志记录
        print(f"Epoch {epoch}: Total Loss = {train_losses['total_loss']:.6f}")
        print(f"  Reconstruction: {train_losses['reconstruction_loss']:.6f}")
        print(f"  Classification: {train_losses['classification_loss']:.6f}")
'''
    
    with open('e:/NEW/V11更新算法/enhanced_loss_integration.py', 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print("\n集成示例代码已保存到: enhanced_loss_integration.py")

if __name__ == "__main__":
    # 运行所有分析
    analyze_current_loss()
    demonstrate_enhanced_loss()
    compare_loss_functions()
    visualize_weight_scheduling()
    loss_function_recommendations()
    create_integration_example()
    
    print("\n" + "=" * 60)
    print("损失函数分析完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("- loss_weights_scheduling.png: 权重调度可视化")
    print("- enhanced_loss_integration.py: 集成示例代码")
    print("\n建议:")
    print("1. 查看权重调度图了解动态权重变化")
    print("2. 参考集成示例代码进行实际集成")
    print("3. 根据具体任务调整损失权重参数")
