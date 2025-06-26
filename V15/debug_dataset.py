#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import create_multimodal_dataset_from_config
import yaml

# 加载配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建数据集
dataset = create_multimodal_dataset_from_config(config, phase='encode')
print(f"数据集大小: {len(dataset)}")

# 检查前几个样本，特别关注数据形状一致性
for i in range(5):
    sample = dataset[i]
    print(f"样本 {i}:")
    print(f"  类型: {type(sample)}")
    print(f"  长度: {len(sample)}")
    for j, item in enumerate(sample):
        shape_info = getattr(item, 'shape', getattr(item, '__len__', str(item)))
        print(f"    元素 {j}: 类型={type(item)}, 形状/值={shape_info}")
    print()

# 特别检查第一个元素（tensor）的形状一致性
print("检查tensor形状一致性:")
shapes = []
for i in range(min(10, len(dataset))):
    sample = dataset[i]
    tensor_shape = sample[0].shape
    shapes.append(tensor_shape)
    print(f"样本 {i}: tensor shape = {tensor_shape}")

# 检查是否所有tensor形状都相同
unique_shapes = set(shapes)
print(f"\n唯一的tensor形状: {unique_shapes}")
if len(unique_shapes) > 1:
    print("❌ 发现不同的tensor形状！这可能是collate函数失败的原因。")
else:
    print("✅ 所有tensor形状都相同。")

# 检查其他元素
print("\n检查其他元素的类型一致性:")
for element_idx in range(1, 5):  # 检查元素1-4
    types = []
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        if element_idx < len(sample):
            item_type = type(sample[element_idx])
            types.append(item_type)
    
    unique_types = set(types)
    print(f"元素 {element_idx}: 类型 = {unique_types}")
    if len(unique_types) > 1:
        print(f"  ❌ 元素 {element_idx} 有不同的类型！")
    else:
        print(f"  ✅ 元素 {element_idx} 类型一致。")
