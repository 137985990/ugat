# loss_integration_demo.py - 损失函数集成演示（无复杂依赖）

import torch
import torch.nn as nn
import yaml
from simple_enhanced_loss import create_simple_enhanced_loss, create_adaptive_loss, CompatibleEnhancedLoss

def demo_loss_function_replacement():
    """演示如何替换现有的损失函数"""
    
    print("=" * 60)
    print("损失函数替换演示")
    print("=" * 60)
    
    # 模拟数据
    batch_size, num_channels, seq_len = 8, 12, 100
    num_classes = 3
    
    predictions = torch.randn(batch_size, num_channels, seq_len)
    targets = torch.randn(batch_size, num_channels, seq_len)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    real_mask = torch.ones(batch_size, num_channels)
    
    print(f"模拟数据形状:")
    print(f"  predictions: {predictions.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  logits: {logits.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  real_mask: {real_mask.shape}")
    
    # 1. 原始MSE损失（逐样本计算方式）
    print(f"\n1. 原始MSE损失计算:")
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    
    total_loss_original = 0.0
    for i in range(batch_size):
        # 重建损失
        recon_loss = 0.0
        real_count = 0
        for c in range(num_channels):
            if real_mask[i, c]:
                pred = predictions[i, c, :]
                target = targets[i, c, :]
                recon_loss += mse_criterion(pred, target)
                real_count += 1
        if real_count > 0:
            recon_loss /= real_count
        
        # 分类损失
        cls_loss = ce_criterion(logits[i:i+1], labels[i:i+1])
        
        total_loss_original += recon_loss + cls_loss
    
    total_loss_original /= batch_size
    print(f"  原始总损失: {total_loss_original.item():.6f}")
    
    # 2. 兼容版增强损失（最小改动）
    print(f"\n2. 兼容版增强损失:")
    compatible_criterion = CompatibleEnhancedLoss()
    
    total_loss_compatible = 0.0
    for i in range(batch_size):
        # 重建损失（使用增强criterion）
        recon_loss = 0.0
        real_count = 0
        for c in range(num_channels):
            if real_mask[i, c]:
                pred = predictions[i, c, :]
                target = targets[i, c, :]
                recon_loss += compatible_criterion(pred, target)  # MSE + L1
                real_count += 1
        if real_count > 0:
            recon_loss /= real_count
        
        # 分类损失
        cls_loss = ce_criterion(logits[i:i+1], labels[i:i+1])
        
        total_loss_compatible += recon_loss + cls_loss
    
    total_loss_compatible /= batch_size
    print(f"  兼容版总损失: {total_loss_compatible.item():.6f}")
    print(f"  改进: {((total_loss_original - total_loss_compatible) / total_loss_original * 100):.2f}%")
    
    # 3. 简化版增强损失
    print(f"\n3. 简化版增强损失:")
    simple_criterion = create_simple_enhanced_loss()
    loss_dict = simple_criterion(predictions, targets, logits, labels, real_mask)
    
    print(f"  损失组件:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value.item():.6f}")
    
    # 4. 自适应权重版本
    print(f"\n4. 自适应权重版本:")
    adaptive_criterion = create_adaptive_loss()
    
    print(f"  训练初期 (epoch 0):")
    adaptive_criterion.update_epoch(0, 100)
    loss_dict_early = adaptive_criterion(predictions, targets, logits, labels, real_mask)
    print(f"    权重: recon={adaptive_criterion.recon_weight:.3f}, cls={adaptive_criterion.cls_weight:.3f}")
    print(f"    总损失: {loss_dict_early['total_loss'].item():.6f}")
    
    print(f"  训练中期 (epoch 50):")
    adaptive_criterion.update_epoch(50, 100)
    loss_dict_mid = adaptive_criterion(predictions, targets, logits, labels, real_mask)
    print(f"    权重: recon={adaptive_criterion.recon_weight:.3f}, cls={adaptive_criterion.cls_weight:.3f}")
    print(f"    总损失: {loss_dict_mid['total_loss'].item():.6f}")
    
    print(f"  训练后期 (epoch 90):")
    adaptive_criterion.update_epoch(90, 100)
    loss_dict_late = adaptive_criterion(predictions, targets, logits, labels, real_mask)
    print(f"    权重: recon={adaptive_criterion.recon_weight:.3f}, cls={adaptive_criterion.cls_weight:.3f}")
    print(f"    总损失: {loss_dict_late['total_loss'].item():.6f}")

def create_integration_guide():
    """创建集成指南"""
    
    print(f"\n" + "=" * 60)
    print("集成指南")
    print("=" * 60)
    
    guide = """
## 三种集成方式

### 方式1: 兼容版本（最小改动，推荐开始）
```python
# 在train.py中，替换这一行:
criterion = MSELoss()

# 改为:
from simple_enhanced_loss import CompatibleEnhancedLoss
criterion = CompatibleEnhancedLoss()

# 其他代码无需修改！
```

### 方式2: 简化增强版本（中等改动）
```python
# 1. 导入增强损失
from simple_enhanced_loss import create_simple_enhanced_loss

# 2. 创建损失函数
criterion = create_simple_enhanced_loss({
    'recon_weight': 1.0,
    'cls_weight': 1.0,
    'l1_weight': 0.1,
    'smooth_weight': 0.05
})

# 3. 修改训练循环使用batch-wise计算
# （参考train_with_enhanced_loss.py中的示例）
```

### 方式3: 自适应权重版本（完整功能）
```python
# 1. 导入自适应损失
from simple_enhanced_loss import create_adaptive_loss

# 2. 创建损失函数
criterion = create_adaptive_loss(config)

# 3. 在每个epoch开始时更新权重
for epoch in range(num_epochs):
    criterion.update_epoch(epoch, num_epochs)
    # ... 训练代码 ...
```

## 配置建议

### 保守配置（稳定优先）
```yaml
loss_config:
  type: "compatible"
```

### 平衡配置（性能与稳定性平衡）
```yaml
loss_config:
  type: "enhanced_simple"
  recon_weight: 1.0
  cls_weight: 1.0
  l1_weight: 0.05    # 较小的L1权重
  smooth_weight: 0.02 # 较小的平滑权重
```

### 激进配置（性能优先）
```yaml
loss_config:
  type: "enhanced_adaptive"
  recon_weight: 1.0
  cls_weight: 1.0
  l1_weight: 0.1
  smooth_weight: 0.1
```
"""
    
    print(guide)
    
    # 保存到文件
    with open('LOSS_INTEGRATION_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("\n集成指南已保存到: LOSS_INTEGRATION_GUIDE.md")

def create_config_examples():
    """创建配置示例"""
    
    # 兼容版配置
    compatible_config = {
        'loss_config': {
            'type': 'compatible'
        }
    }
    
    # 简化增强版配置
    simple_config = {
        'loss_config': {
            'type': 'enhanced_simple',
            'recon_weight': 1.0,
            'cls_weight': 1.0,
            'l1_weight': 0.05,
            'smooth_weight': 0.02
        }
    }
    
    # 自适应版配置
    adaptive_config = {
        'loss_config': {
            'type': 'enhanced_adaptive',
            'recon_weight': 1.0,
            'cls_weight': 1.0,
            'l1_weight': 0.1,
            'smooth_weight': 0.05
        }
    }
    
    # 保存配置文件
    configs = [
        (compatible_config, 'config_compatible.yaml'),
        (simple_config, 'config_simple_enhanced.yaml'),
        (adaptive_config, 'config_adaptive_enhanced.yaml')
    ]
    
    for config, filename in configs:
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"配置文件已保存: {filename}")

def performance_comparison():
    """性能对比测试"""
    
    print(f"\n" + "=" * 60)
    print("性能对比测试")
    print("=" * 60)
    
    import time
    
    # 测试数据
    batch_size, num_channels, seq_len = 16, 12, 200
    num_classes = 3
    
    predictions = torch.randn(batch_size, num_channels, seq_len)
    targets = torch.randn(batch_size, num_channels, seq_len)
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    real_mask = torch.ones(batch_size, num_channels)
    
    # 测试不同损失函数的计算时间
    test_configs = [
        ("MSE原始", nn.MSELoss()),
        ("兼容增强", CompatibleEnhancedLoss()),
        ("简化增强", create_simple_enhanced_loss()),
        ("自适应增强", create_adaptive_loss())
    ]
    
    num_runs = 100
    
    for name, criterion in test_configs:
        start_time = time.time()
        
        for _ in range(num_runs):
            if name == "MSE原始":
                # 模拟原始计算方式
                loss = 0
                for i in range(batch_size):
                    for c in range(num_channels):
                        if real_mask[i, c]:
                            loss += criterion(predictions[i, c], targets[i, c])
            elif name == "兼容增强":
                # 兼容方式
                loss = 0
                for i in range(batch_size):
                    for c in range(num_channels):
                        if real_mask[i, c]:
                            loss += criterion(predictions[i, c], targets[i, c])
            else:
                # 增强方式
                ce_loss = nn.CrossEntropyLoss()
                cls_loss = ce_loss(logits, labels)
                if hasattr(criterion, '__call__') and len(criterion.__code__.co_varnames) > 4:
                    loss_dict = criterion(predictions, targets, logits, labels, real_mask)
                    loss = loss_dict['total_loss']
                else:
                    loss = criterion(predictions.mean(), targets.mean())
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        
        print(f"{name:12s}: {avg_time:.3f} ms/batch (相对speedup: {20/avg_time:.2f}x)")

if __name__ == "__main__":
    # 运行所有演示
    demo_loss_function_replacement()
    create_integration_guide()
    create_config_examples()
    performance_comparison()
    
    print(f"\n" + "=" * 60)
    print("损失函数集成演示完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("- LOSS_INTEGRATION_GUIDE.md: 详细集成指南")
    print("- config_compatible.yaml: 兼容版配置")
    print("- config_simple_enhanced.yaml: 简化增强版配置")
    print("- config_adaptive_enhanced.yaml: 自适应增强版配置")
    print("\n推荐实施步骤:")
    print("1. 先使用兼容版本测试稳定性")
    print("2. 然后尝试简化增强版本")
    print("3. 最后考虑自适应权重版本")
