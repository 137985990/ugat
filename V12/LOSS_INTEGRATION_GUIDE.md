
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
