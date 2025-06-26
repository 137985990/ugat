# GAN-based Time Series Completion with GAT

这是一个基于生成对抗网络(GAN)和图注意力网络(GAT)的时间序列补全模型，替换了原有的U-Net架构。

## 架构特点

### 1. 生成器 (GANGenerator)
- **基于GAT的编码器-解码器结构**
- 使用Graph Attention Network处理时间序列的时空关系
- Transformer bottleneck增强特征表示
- 输入: mask后的时间序列 `[batch, channels, time]`
- 输出: 补全的时间序列 `[batch, channels, time]`

### 2. 双判别器架构

#### 主判别器 (GANDiscriminator)
- **功能**: 传统GAN判别器，判断输入是真实数据还是生成数据
- **结构**: 卷积层 + 全连接层
- **输出**: 单个概率值 (真/假)

#### 分类判别器 (ClassifierDiscriminator)
- **功能**: 原U-Net中的分类模型，作为辅助判别器
- **双重任务**: 
  1. 判别真假 (与主判别器相同)
  2. 多类别分类 (额外的监督信号)
- **输出**: 判别概率 + 分类logits

## 文件结构

```
GAN/
├── model.py           # GAN模型定义
├── train.py           # 训练脚本
├── data.py            # 数据加载
├── graph.py           # 图构建工具
├── config.yaml        # 配置文件
├── run_gan.py         # 运行脚本
├── test_gan.py        # 测试脚本
└── README.md          # 说明文档
```

## 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install torch torch-geometric pandas numpy pyyaml tqdm tensorboard
```

### 2. 配置文件

编辑 `config.yaml`:

```yaml
data_dir: Data
data_files:
  - ../Data/FM_original.csv
  - ../Data/OD_original.csv
  - ../Data/MEFAR_original.csv
block_col: block
feature_cols: [acc_x, acc_y, acc_z, ppg, gsr, hr, skt]
sample_rate: 32
window_sec: 10.0
step_sec: 3.0
window_size: 320
step_size: 96
norm_method: zscore
train_split: 0.8
batch_size: 16
epochs: 200
lr: 0.0005
patience: 20
```

### 3. 训练模型

```bash
# 使用默认参数
python run_gan.py

# 自定义参数
python run_gan.py --config config.yaml --epochs 100 --batch_size 32 --lr 0.001
```

### 4. 测试模型

```bash
# 运行测试脚本
python test_gan.py
```

## 训练流程

### 1. 数据预处理
- 滑动窗口切分时间序列
- 随机mask部分通道 (默认30%)
- 窗口内Z-score归一化

### 2. 对抗训练
每个训练步骤包括:

1. **训练主判别器 D1**:
   - 真实数据 → 标签1
   - 生成数据 → 标签0
   - 优化二元交叉熵损失

2. **训练分类判别器 D2**:
   - 真实数据 → 判别损失 + 分类损失
   - 生成数据 → 判别损失
   - 多任务优化

3. **训练生成器 G**:
   - 欺骗两个判别器 (对抗损失)
   - mask区域重建 (重建损失)
   - 总损失 = 对抗损失1 + 对抗损失2 + 10×重建损失

### 3. 损失函数

- **对抗损失**: Binary Cross Entropy
- **分类损失**: Cross Entropy
- **重建损失**: Mean Squared Error (仅在mask区域)

## 关键改进

### 相比原U-Net架构:

1. **生成对抗训练**: 提升生成质量和真实性
2. **双判别器**: 增强判别能力和训练稳定性
3. **GAT编码器**: 更好地建模时间序列的图结构关系
4. **多任务学习**: 分类任务提供额外监督信号

### 训练技巧:

1. **梯度裁剪**: 防止梯度爆炸
2. **早停机制**: 基于重建损失
3. **学习率调度**: 自适应调整
4. **标签平滑**: 稳定对抗训练

## 输出结果

### 训练过程:
- **日志文件**: `outputs/train_gan_*.log`
- **TensorBoard**: `outputs/` (损失曲线)
- **模型检查点**: `outputs/best_gan_*.pth`

### 评估指标:
- **生成器损失**: 对抗损失 + 重建损失
- **判别器损失**: D1损失 + D2损失
- **重建MSE**: 仅在mask区域计算
- **测试MSE**: 最终评估指标

## 模型加载

```python
import torch
from model import GANGenerator

# 加载训练好的模型
checkpoint = torch.load('outputs/best_gan_gan.pth')
generator = GANGenerator(in_channels=7, out_channels=7, seq_len=320)
generator.load_state_dict(checkpoint['generator'])
generator.eval()

# 使用模型进行预测
with torch.no_grad():
    completed = generator(masked_data)
```

## 注意事项

1. **数据格式**: 确保CSV文件包含指定的特征列
2. **内存使用**: 大批次可能需要更多GPU内存
3. **训练时间**: 根据数据量和epoch数调整
4. **超参数**: 可能需要根据具体数据调优

## 故障排除

### 常见问题:

1. **ImportError**: 确保安装了torch-geometric
2. **CUDA错误**: 检查GPU内存和CUDA版本
3. **数据加载失败**: 检查文件路径和格式
4. **训练不收敛**: 调整学习率和损失权重

### 调试模式:

```bash
# 运行小规模测试
python test_gan.py
```

这将运行2个epoch的小规模训练来验证模型正确性。
