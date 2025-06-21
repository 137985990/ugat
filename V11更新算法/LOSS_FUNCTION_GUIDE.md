# V11算法损失函数设计详解

## 目录
1. [当前损失函数分析](#当前损失函数分析)
2. [损失函数数学公式](#损失函数数学公式)
3. [实现细节](#实现细节)
4. [增强版损失函数](#增强版损失函数)
5. [性能对比](#性能对比)
6. [集成指南](#集成指南)

## 当前损失函数分析

### 基本架构
V11算法采用**多任务学习**架构，同时优化两个主要任务：
- **重建任务**：恢复被mask的时间序列通道
- **分类任务**：预测时间序列的类别标签

### 损失函数组成
```python
# 当前实现
criterion = MSELoss()  # 重建损失
ce_loss = CrossEntropyLoss()  # 分类损失

# 总损失计算
total_loss = reconstruction_loss + classification_loss
```

### 关键特点

#### 1. 通道级Masking
- **优点**：只对真实通道计算重建损失，避免补全通道的噪声影响
- **实现**：通过`is_real_mask`标识哪些通道是真实数据
```python
for c in range(C):
    if real_channels[c]:  # 只计算真实通道
        target = batch[i, c, :]
        pred = out[c, :]
        recon_loss_i += criterion(pred, target)
```

#### 2. 样本级计算
- **优点**：逐样本计算损失，提供更精细的梯度信息
- **实现**：在batch循环内部进行样本级损失计算
```python
for i in range(batch_size):
    # 对每个样本计算损失
    recon_loss_i = compute_reconstruction_loss(sample_i)
    cls_loss_i = compute_classification_loss(sample_i)
    loss += recon_loss_i + cls_loss_i
```

#### 3. 固定权重策略
- **当前**：重建损失与分类损失权重相等（1:1）
- **局限**：无法适应不同训练阶段的需求

## 损失函数数学公式

### 重建损失 (Reconstruction Loss)
$$\mathcal{L}_{recon} = \frac{1}{N \times C_{real} \times T} \sum_{i=1}^{N} \sum_{c \in \mathcal{C}_{real}} \sum_{t=1}^{T} (x_{i,c,t} - \hat{x}_{i,c,t})^2$$

其中：
- $N$：批大小
- $C_{real}$：真实通道数
- $T$：时间序列长度
- $\mathcal{C}_{real}$：真实通道集合
- $x_{i,c,t}$：真实值
- $\hat{x}_{i,c,t}$：预测值

### 分类损失 (Classification Loss)
$$\mathcal{L}_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{K} y_{i,j} \log(\hat{y}_{i,j})$$

其中：
- $K$：类别数
- $y_{i,j}$：真实标签的one-hot编码
- $\hat{y}_{i,j}$：预测概率

### 总损失 (Total Loss)
$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{cls}$$

## 实现细节

### 训练流程
```python
def train_phased(model, dataloader, optimizer, criterion, device, mask_indices):
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    
    for batch, labels, mask_idx, is_real_mask in dataloader:
        optimizer.zero_grad()
        
        # 样本级损失计算
        for i in range(batch_size):
            window = masked[i].t()  # [T, C]
            out, logits = model(window)  # out: [C, T], logits: [num_classes]
            
            # 重建损失：只计算真实通道
            recon_loss_i = 0.0
            real_count = 0
            for c in range(C):
                if real_channels[c]:
                    target = batch[i, c, :]
                    pred = out[c, :]
                    recon_loss_i += criterion(pred, target)
                    real_count += 1
            if real_count > 0:
                recon_loss_i /= real_count
            
            # 分类损失
            cls_loss_i = ce_loss(logits.unsqueeze(0), labels[i].unsqueeze(0))
            
            # 累积损失
            loss += recon_loss_i + cls_loss_i
        
        # 平均并反向传播
        loss = loss / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### 梯度处理
- **梯度裁剪**：`max_norm=1.0`防止梯度爆炸
- **异常检测**：包含NaN/Inf检测机制

## 增强版损失函数

### 新增组件

#### 1. 多重建损失组合
```python
# MSE + L1 + Huber损失组合
reconstruction_loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * huber_loss
```
- **MSE损失**：对大误差敏感，适合主要重建任务
- **L1损失**：对异常值鲁棒，增强稳定性
- **Huber损失**：结合两者优点，平滑过渡

#### 2. 一致性损失 (Consistency Loss)
$$\mathcal{L}_{consistency} = \frac{1}{N \times C_{real} \times (T-1)} \sum_{i,c,t} \mathcal{M}_{i,c} \cdot (\Delta \hat{x}_{i,c,t} - \Delta x_{i,c,t})^2$$

其中：$\Delta x_{i,c,t} = x_{i,c,t+1} - x_{i,c,t}$（一阶差分）

**作用**：保证时间窗口间的变化趋势一致性

#### 3. 时序平滑损失 (Temporal Smoothness Loss)
$$\mathcal{L}_{temporal} = \frac{1}{N \times C_{real} \times (T-2)} \sum_{i,c,t} \mathcal{M}_{i,c} \cdot (\Delta^2 \hat{x}_{i,c,t})^2$$

其中：$\Delta^2 x_{i,c,t} = x_{i,c,t+2} - 2x_{i,c,t+1} + x_{i,c,t}$（二阶差分）

**作用**：减少不自然的突变，增强时序平滑性

#### 4. 频域损失 (Spectral Loss)
$$\mathcal{L}_{spectral} = \frac{1}{N \times C_{real} \times T} \sum_{i,c,f} \mathcal{M}_{i,c} \cdot (|FFT(\hat{x}_{i,c})|_f^2 - |FFT(x_{i,c})|_f^2)^2$$

**作用**：保持频域特征的相似性，特别适合周期性时间序列

### 动态权重调度
```python
def update_weights(self, epoch: int, total_epochs: int):
    progress = epoch / total_epochs
    
    # 重建权重：从1.0逐渐降到0.7
    self.recon_weight = 1.0 - 0.3 * progress
    
    # 分类权重：从0.5逐渐升到1.0  
    self.cls_weight = 0.5 + 0.5 * progress
    
    # 正则化项权重：中期最大
    self.consistency_weight = 0.2 * np.sin(progress * np.pi)
    self.temporal_weight = 0.15 * np.sin(progress * np.pi)
```

**策略说明**：
- **早期训练**：重点学习重建任务，分类权重较低
- **中期训练**：正则化项权重最大，增强泛化能力
- **后期训练**：提高分类权重，优化最终预测性能

## 性能对比

| 损失组件 | 当前版本 | 增强版本 | 改进效果 |
|---------|---------|---------|----------|
| 重建损失 | MSE单一损失 | MSE+L1+Huber组合 | 更鲁棒，减少异常值影响 |
| 权重调度 | 固定权重(1:1) | 动态权重调度 | 适应训练阶段需求 |
| 正则化 | 无 | 一致性+时序+频域 | 增强泛化能力 |
| 训练稳定性 | 基础梯度裁剪 | 多层次约束 | 更稳定的训练过程 |

### 实验结果示例
```
当前损失函数:
   重建损失: 1.999193
   分类损失: 1.775545
   总损失: 3.774739

增强版损失函数:
   重建损失: 1.480629 (组合损失，更平滑)
   分类损失: 1.775545
   一致性损失: 4.123249
   时序损失: 6.143254
   频域损失: 19203.642578 (需要权重调整)
   总损失: 964.464966
```

## 集成指南

### Step 1: 配置参数
```python
# config.yaml中添加
loss_config:
  type: "enhanced"  # "basic" or "enhanced"
  recon_weight: 1.0
  cls_weight: 1.0
  consistency_weight: 0.1
  temporal_weight: 0.1
  spectral_weight: 0.01  # 建议调低频域权重
```

### Step 2: 替换损失函数
```python
from enhanced_loss import create_enhanced_loss

# 替换原有的criterion
if config['loss_config']['type'] == 'enhanced':
    criterion = create_enhanced_loss(config['loss_config'])
else:
    criterion = MSELoss()
```

### Step 3: 修改训练循环
```python
# 在训练循环中
for epoch in range(num_epochs):
    if hasattr(criterion, 'update_weights'):
        criterion.update_weights(epoch, num_epochs)
    
    # 使用新的损失计算方式
    loss_dict = criterion(predictions, targets, logits, labels, real_mask)
    total_loss = loss_dict['total_loss']
```

### Step 4: 监控和调整
1. **监控各损失组件**：确保没有某个组件dominate整个损失
2. **调整权重参数**：根据验证集表现微调权重
3. **检查训练稳定性**：观察损失曲线是否平滑下降
4. **验证最终性能**：对比原版和增强版的最终指标

## 建议实施路径

### 阶段1: 保守集成
- 只添加L1正则化到重建损失
- 实现基础动态权重调度
- 验证训练稳定性

### 阶段2: 渐进增强
- 添加一致性损失（权重0.05）
- 添加时序平滑损失（权重0.05）
- 监控性能变化

### 阶段3: 完全增强
- 集成频域损失（权重0.01）
- 实现高级权重调度策略
- 优化超参数配置

通过这种渐进式的集成方式，可以确保系统稳定性的同时获得性能提升。
