# Magvit2 VAE 多模态时序数据处理

本项目实现了基于Magvit2的VAE模型，用于多模态时序数据的重建和分类任务。

## 主要文件

### 核心模型文件
- `magvit2_model.py` - Magvit2 VAE主模型实现
- `magvit2_multimodal_loss.py` - 专用多模态损失函数
- `train.py` - 修改后的训练脚本
- `config_magvit2.yaml` - Magvit2专用配置文件

### 模型特点

#### Magvit2 VAE架构
1. **编码器 (Magvit2Encoder)**
   - 残差块 + 自注意力机制
   - 下采样层用于特征压缩
   - 支持1D时序数据

2. **向量量化 (VectorQuantizer)**
   - 离散潜在表示
   - 可调节的commitment cost
   - 支持梯度直通估计

3. **解码器 (Magvit2Decoder)**
   - 对称上采样结构
   - 残差连接和注意力机制
   - 高质量重建

4. **分类头**
   - 基于量化特征的分类
   - 全局平均池化
   - 多层MLP

#### 多模态损失函数
1. **重建损失** - MSE/L1损失，根据模态类型加权
2. **VQ损失** - 向量量化正则化
3. **分类损失** - 交叉熵分类损失
4. **感知损失** - 基于特征相似性的损失

## 使用方法

### 1. 训练模型
```bash
python train.py --config config_magvit2.yaml
```

### 2. 评估模型
```bash
# 修改config中的mode为'eval'
python train.py --config config_magvit2.yaml
```

### 3. 配置说明

#### 模型参数
- `in_channels`: 输入特征维度
- `hidden_channels`: 隐藏层维度
- `latent_dim`: 潜在空间维度
- `num_embeddings`: 码本大小
- `commitment_cost`: VQ损失权重

#### 损失函数权重
- `recon_weight`: 重建损失权重
- `vq_weight`: VQ损失权重  
- `cls_weight`: 分类损失权重
- `perceptual_weight`: 感知损失权重

#### 多模态配置
- `common_modalities`: 所有数据集共有的模态
- `dataset_modalities`: 各数据集特有的模态配置

## 主要改进

### 相比原始T-GAT-UNet
1. **更强的表示能力** - 离散潜在表示
2. **更好的生成质量** - 向量量化 + 感知损失
3. **更稳定的训练** - 梯度累积 + 混合精度
4. **更快的推理** - 批量处理优化

### 训练特点
1. **梯度累积** - 支持大批量训练
2. **混合精度** - 节省显存，加速训练
3. **动态need补全** - 训练过程中动态更新缺失模态
4. **增强验证** - 复合指标早停策略

## 性能优化

### 显存优化
- 批量处理减少循环开销
- 及时清理中间变量
- 混合精度训练

### 计算优化
- 向量化批量操作
- GPU并行计算
- 编译优化（torch.compile）

## 监控和日志

### TensorBoard指标
- 训练损失分解（重建/VQ/分类）
- 验证指标（准确率/F1/精确率/召回率）
- 学习率变化
- 模态特定损失

### 早停策略
- 复合指标评估
- 过拟合检测
- 最佳模型保存

## 注意事项

1. **数据格式** - 确保CSV文件包含正确的列名
2. **显存管理** - 根据GPU显存调整batch_size
3. **模态配置** - 正确配置common和dataset特定模态
4. **路径设置** - 确保数据文件路径正确

## 故障排除

### 常见问题
1. **显存不足** - 减少batch_size或hidden_channels
2. **训练不稳定** - 调整学习率或损失权重
3. **收敛缓慢** - 增加模型容量或调整VQ参数

### 调试建议
1. 使用小数据集测试
2. 检查损失函数权重平衡
3. 监控各组件损失变化
