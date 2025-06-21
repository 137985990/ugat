# V11性能优化指南

## 🚀 性能优化总览

本次优化主要针对三个性能瓶颈：

1. **图构建开销** → 缓存机制 + 预计算
2. **内存占用** → 自适应批处理 + 内存监控
3. **计算复杂度** → 轻量化架构 + 梯度检查点

## 📊 优化效果预期

| 优化项目 | 性能提升 | 内存节省 | 参数减少 |
|---------|---------|---------|---------|
| 图构建缓存 | 2-5x | 10-20% | - |
| 轻量化模型 | 1.5-3x | 30-50% | 40-60% |
| 内存优化器 | - | 20-40% | - |
| 梯度检查点 | 0.8-1.2x | 40-60% | - |

## 🔧 使用方法

### 1. 启用所有优化（推荐）

```yaml
# config.yaml
use_optimized_model: true        # 使用优化版模型
use_memory_optimizer: true       # 使用内存优化器
use_gradient_checkpoint: true    # 使用梯度检查点
batch_size: 16                   # 适中的批大小
encoder_layers: 2                # 减少编码器层数
decoder_layers: 2                # 减少解码器层数
transformer_layers: 1            # 减少Transformer层数
attention_heads: 2               # 减少注意力头数
```

### 2. 根据硬件配置调整

#### 低内存环境 (< 8GB)
```yaml
batch_size: 8
hidden_channels: 32
encoder_layers: 1
decoder_layers: 1
transformer_layers: 1
attention_heads: 1
use_gradient_checkpoint: true
```

#### 中等配置 (8-16GB)
```yaml
batch_size: 16
hidden_channels: 64
encoder_layers: 2
decoder_layers: 2
transformer_layers: 1
attention_heads: 2
use_gradient_checkpoint: true
```

#### 高配置 (> 16GB)
```yaml
batch_size: 32
hidden_channels: 128
encoder_layers: 3
decoder_layers: 3
transformer_layers: 2
attention_heads: 4
use_gradient_checkpoint: false  # 可选
```

### 3. 运行性能测试

```bash
# 运行基准测试
python performance_test.py

# 查看详细性能报告
# 会生成 graph_benchmark.png 图表
```

## 🔍 监控指标

### 训练过程监控
- **内存使用率**: 应保持在 80% 以下
- **GPU利用率**: 目标 70-90%
- **训练速度**: epoch/min 或 samples/sec

### 模型性能监控
- **推理延迟**: 单样本推理时间
- **吞吐量**: batch/sec
- **内存峰值**: 训练/推理时的最大内存占用

## ⚠️ 注意事项

1. **梯度检查点**: 会增加计算时间但显著减少内存，适合内存紧张的环境
2. **缓存清理**: 长时间训练后建议清理图缓存 `rm -rf graph_cache/`
3. **批大小调整**: 系统会根据内存自动调整，无需手动修改
4. **数据泄露**: 动态need更新仅在训练集上进行，避免测试集信息泄露

## 🐛 故障排除

### 内存不足 (OOM)
1. 降低 `batch_size` 到 4 或 8
2. 启用 `use_gradient_checkpoint: true`
3. 减少 `hidden_channels` 到 32
4. 重启 Python 进程清理内存碎片

### 训练速度慢
1. 关闭 `use_gradient_checkpoint` (如果内存充足)
2. 增加 `batch_size` 到 32 或 64
3. 使用更少的 `transformer_layers`
4. 检查是否有内存交换 (swap)

### 图构建缓存失效
1. 删除 `graph_cache/` 目录
2. 检查 `time_k` 和 `window_size` 参数是否改变
3. 确保磁盘空间充足

## 📈 性能调优建议

### 阶段1: 基础优化
- 启用所有优化选项
- 使用推荐的超参数设置
- 运行性能基准测试

### 阶段2: 硬件适配
- 根据实际硬件调整批大小和模型大小
- 监控内存和GPU使用率
- 微调学习率和其他超参数

### 阶段3: 高级优化
- 考虑使用混合精度训练 (FP16)
- 尝试模型并行或数据并行
- 使用专业的性能分析工具 (如 nvidia-smi, htop)

## 🎯 最佳实践

1. **开始时使用保守配置**: 先确保能跑通，再逐步优化
2. **监控关键指标**: 特别关注内存使用和训练速度
3. **分阶段优化**: 一次只改变一个优化选项，观察效果
4. **保存优化配置**: 找到最佳配置后保存为模板
5. **定期清理**: 清理缓存文件和临时数据

---

💡 **提示**: 如果遇到问题，可以先运行 `performance_test.py` 来诊断性能瓶颈！
