import random
import numpy as np
import torch
import yaml
import os
import pandas as pd
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from model import TGATUNet
from data import load_config, load_data, SlidingWindowDataset

# 1. 加载配置
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# 2. 加载模型
ckpt_path = config.get('model_path', 'Checkpoints/best_model.pth')
ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path

in_channels = config['in_channels']
hidden_channels = config['hidden_channels']
out_channels = config['out_channels']
num_classes = config.get('num_classes', 2)

model = TGATUNet(in_channels, hidden_channels, out_channels, num_classes=num_classes)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

# 3. 加载数据
file_paths = [os.path.join(os.path.dirname(__file__), f) if not os.path.isabs(f) else f for f in config['data_files']]
data = load_data(file_paths)

# 统一modalities，补全缺失列为0
all_have = []
all_need = []
for ds in config['dataset_modalities'].values():
    all_have += ds.get('have', [])
    all_need += ds.get('need', [])
all_modalities = list(dict.fromkeys(config['common_modalities'] + all_have + all_need))
block_col = config['block_col']
label_col = config.get('label_col', 'F')
for col in all_modalities:
    if col not in data.columns:
        data[col] = 0.0
feature_cols = [col for col in all_modalities if col != label_col]

window_size = config['window_size']
step_size = config['step_size']
sampling_rate = config.get('sample_rate', 1)
normalize = config.get('norm_method', None)

# 保证分割一致：设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 构建完整数据集
full_dataset = SlidingWindowDataset(
    data=data,
    block_col=block_col,
    feature_cols=feature_cols,
    window_size=window_size,
    step_size=step_size,
    sampling_rate=sampling_rate,
    normalize=normalize,
    phase="encode"
)
# 按训练比例分割
train_split = config.get('train_split', 0.6)
val_split = config.get('val_split', 0.2)
total = len(full_dataset)
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

# 只对测试集推理
for i in range(min(10, len(test_ds))):
    idx = test_ds.indices[i]
    x, label, mask_idx, is_real_mask = test_ds.dataset[idx]
    x_t = x.t()  # [T, C]
    with torch.no_grad():
        out, logits = model(x_t)
    print(f"Test Sample {i}: label={label}, output shape={out.shape}, logits={logits}")

# === 修正：对每个原始数据文件分别补全并保存 ===
for file_path in file_paths:
    dataset_name = os.path.splitext(os.path.basename(file_path))[0].replace('_original', '')
    # 单独加载每个原始数据文件
    single_data = load_data([file_path])
    # 补全缺失列
    for col in all_modalities:
        if col not in single_data.columns:
            single_data[col] = 0.0
    feature_cols = [col for col in all_modalities if col != label_col]
    # 构建滑窗数据集
    single_dataset = SlidingWindowDataset(
        data=single_data,
        block_col=block_col,
        feature_cols=feature_cols,
        window_size=window_size,
        step_size=step_size,
        sampling_rate=sampling_rate,
        normalize=normalize,
        phase="encode"
    )
    num_rows = len(single_data)
    num_modalities = len(feature_cols)
    completed_sum = np.zeros((num_rows, num_modalities), dtype=np.float32)
    completed_count = np.zeros((num_rows, num_modalities), dtype=np.float32)
    loader = DataLoader(single_dataset, batch_size=32)
    row_idx = 0
    with torch.no_grad():
        for batch_idx, (batch, label, mask_idx, is_real_mask) in enumerate(tqdm(loader, desc=f"Complete ({dataset_name})")):
            batch_size, C, T = batch.size()
            for i in range(batch_size):
                window = batch[i].t()  # [T, C]
                out, _ = model(window)
                # === 调试：打印输入和输出的均值和标准差 ===
                # print(f"[DEBUG] window mean: {window.mean().item():.6f}, std: {window.std().item():.6f}")
                # print(f"[DEBUG] model out mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
                out_np = out.cpu().numpy().T  # [T, C]
                start = row_idx
                end = row_idx + T
                if end > num_rows:
                    break
                completed_sum[start:end, :] += out_np
                completed_count[start:end, :] += 1
                row_idx += 1
    completed_count[completed_count == 0] = 1
    completed_avg = completed_sum / completed_count
    # 只为原始数据缺失的整列补全并插入，已有列完全保留原始观测
    completed_df = single_data.copy()
    # 读取原始csv，记录原始缺失的整列特征
    orig_df = pd.read_csv(file_path)
    orig_cols = set(orig_df.columns)
    missing_cols = [col for col in feature_cols if col not in orig_cols]
    added_cols = []
    for idx, col in enumerate(feature_cols):
        if col in missing_cols:
            completed_df[col] = completed_avg[:, idx]
            added_cols.append(col)
    save_path = os.path.join(os.path.dirname(__file__), f"{dataset_name}_completed.csv")
    completed_df.to_csv(save_path, index=False)
    print(f"{dataset_name}_completed.csv 已保存，新增补全列: {added_cols}")
