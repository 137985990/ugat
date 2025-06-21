import yaml
import torch
from torch.utils.data import random_split
from data import create_dataset_from_config
from collections import Counter
import random

config_path = 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 加载数据集
full_dataset = create_dataset_from_config(config_path)
total = len(full_dataset)
train_split = config.get('train_split', 0.6)
val_split = config.get('val_split', 0.2)
train_len = round(train_split * total)
val_len = round(val_split * total)
test_len = total - train_len - val_len
if test_len < 0:
    test_len = 0
while train_len + val_len + test_len < total:
    test_len += 1
while train_len + val_len + test_len > total:
    if test_len > 0:
        test_len -= 1
    elif val_len > 0:
        val_len -= 1
    else:
        train_len -= 1
assert train_len + val_len + test_len == total

train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])

for name, ds in zip(['train', 'val', 'test'], [train_ds, val_ds, test_ds]):
    labels = []
    all1_count = 0
    idx_1 = []
    idx_0 = []
    for i in range(len(ds)):
        # 获取原始dataset的窗口索引
        idx = ds.indices[i] if hasattr(ds, 'indices') else i
        b_idx, start, seg_label = full_dataset.indices[idx]
        block = full_dataset.blocks[b_idx]
        window_df = block.iloc[start:start + full_dataset.window_size]
        window_labels = window_df[full_dataset.label_col].values
        if (window_labels == 1).all():
            all1_count += 1
        _, label, _, _ = ds[i]
        label = int(label)
        labels.append(label)
        if label == 1:
            idx_1.append(i)
        else:
            idx_0.append(i)
    print(f'{name} 原始标签分布: {Counter(labels)}')
    print(f'{name} 原始窗口全为1的数量: {all1_count}，占比: {all1_count/len(ds):.4f}')
    # 欠采样1类
    n0 = len(idx_0)
    n1 = len(idx_1)
    if n1 > n0 and n0 > 0:
        idx_1_sampled = random.sample(idx_1, n0)
        # 交错排列
        balanced_indices = []
        for a, b in zip(idx_0, idx_1_sampled):
            balanced_indices.append(b)
            balanced_indices.append(a)
        # 若数量不等，补齐
        # 此处n1==n0，已平衡
        labels_bal = [labels[i] for i in balanced_indices]
        print(f'{name} 欠采样后标签分布: {Counter(labels_bal)}')
        print(f'{name} 欠采样后窗口全为1的数量: {labels_bal.count(1)}，占比: {labels_bal.count(1)/len(labels_bal):.4f}')
        print(f'{name} 欠采样后前20个标签: {labels_bal[:20]}')
    else:
        print(f'{name} 无需采样，标签已平衡或无0类')
    # 检查连续区段
    last = None
    segs = []
    count = 0
    for l in labels:
        if l == last:
            count += 1
        else:
            if last is not None:
                segs.append((last, count))
            last = l
            count = 1
    if last is not None:
        segs.append((last, count))
    print(f'{name} 连续区段统计:')
    for v, c in segs[:10]:
        print(f'  标签{v} 连续{c}个')
    print(f'  ... 共{len(segs)}段, 最大段长: {max(c for _,c in segs)}')
