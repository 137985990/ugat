def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 保证部分库的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import collections

def check_label_distribution(dataset):
    """
    检查并输出数据集标签分布和所有标签种类
    """
    label_counter = collections.Counter()
    all_labels = set()
    for i in range(len(dataset)):
        item = dataset[i]
        label = item[1]
        if hasattr(label, 'item'):
            label = label.item()
        label_counter[label] += 1
        all_labels.add(label)
    print("标签分布:", dict(label_counter))
    print("所有标签:", sorted(list(all_labels)))
    return label_counter, all_labels

def complete_need_with_model(model, dataset, device):
    """
    用模型对整个数据集的need通道进行补全，并写回dataset（动态need更新）。
    只更新need_indices指定的通道。
    """
    model.eval()
    from torch.utils.data import DataLoader
    import torch
    # 支持联合数据集：每个样本need_indices可能不同
    from torch.utils.data import DataLoader
    import torch
    loader = DataLoader(dataset, batch_size=32)
    all_preds = []
    sample_base_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Complete need (dynamic)"):
            # 兼容联合数据集 __getitem__ 返回 (x, label, mask_idx, is_real_mask, source) 或 (x, ...)
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                batch_x = batch[0]
            else:
                batch_x = batch
            batch_x = batch_x.to(device)
            batch_size, C, T = batch_x.size()
            for i in range(batch_size):
                # 获取每个样本的need_indices
                if hasattr(dataset, 'need_indices') and isinstance(dataset.need_indices, list) and len(dataset.need_indices) == len(dataset):
                    need_idx = dataset.need_indices[sample_base_idx + i]
                else:
                    need_idx = getattr(dataset, 'need_indices', None)
                if not need_idx or len(need_idx) == 0:
                    # 兼容无need通道的样本
                    all_preds.append(torch.empty((0, T), device=batch_x.device))
                    continue
                masked = batch_x[i].clone()
                for idx in need_idx:
                    masked[idx, :] = 0
                window = masked.t()  # [T, C]
                out, _ = model(window)
                out_need = out[need_idx, :]  # [num_need, T]
                all_preds.append(out_need)
            sample_base_idx += batch_size
    # all_preds: list of [num_need, T]，每个样本独立
    if hasattr(dataset, 'update_need'):
        dataset.update_need(all_preds)

import yaml
import numpy as np
import pandas as pd
import os
import argparse
import yaml
import logging
from datetime import datetime

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

from torch.utils.tensorboard import SummaryWriter

from data import create_dataset_from_config
from model import TGATUNet


def complete_dataset_to_csv(model, dataset, mask_indices, device, save_path, dataset_modalities):
    """
    用模型对数据集进行模态补全，保存补全后的数据为CSV
    """
    model.eval()
    # 1. 读取原始csv，保留block和label列，补全modalities
    import pandas as pd
    import os
    # 推断原始文件路径，优先用config.yaml里的data_files
    config_path = 'e:/NEW/V5/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 取data_files绝对路径
    data_files = cfg.get('data_files', [])
    # 兼容不同命名
    def find_file(keyword):
        for path in data_files:
            if keyword.lower() in os.path.basename(path).lower():
                # 支持相对路径
                abs_path = os.path.abspath(os.path.join(os.path.dirname(config_path), path))
                if os.path.exists(abs_path):
                    return abs_path
        # 兜底：尝试V5/Data/下
        fallback = os.path.abspath(os.path.join(os.path.dirname(config_path), 'Data', f'{keyword}_original.csv'))
        if os.path.exists(fallback):
            return fallback
        # 兜底：尝试NEW/Data/下
        fallback2 = os.path.abspath(os.path.join('e:/NEW/Data', f'{keyword}_original.csv'))
        if os.path.exists(fallback2):
            return fallback2
        raise FileNotFoundError(f'找不到原始csv: {keyword}')
    if 'FM' in save_path:
        orig_path = find_file('FM')
    elif 'OD' in save_path:
        orig_path = find_file('OD')
    elif 'MEFAR' in save_path:
        orig_path = find_file('MEFAR')
    else:
        raise ValueError('无法推断原始csv路径')
    orig_df = pd.read_csv(orig_path)
    # 统一列名小写、去空格
    # orig_df.columns = [c.strip().lower() for c in orig_df.columns]  # 保留原始大写F
    block_col = 'block'
    label_col = 'f' if 'f' in orig_df.columns else orig_df.columns[1]  # 默认第2列为label
    # 保证所有modalities都在
    for m in dataset_modalities:
        if m not in orig_df.columns:
            orig_df[m] = 0.0
    # 保证顺序：block, label, 32个modalities
    out_cols = [block_col, label_col] + list(dataset_modalities)
    orig_df = orig_df[out_cols]

    # 2. 用模型滑动窗口补全modalities部分（对每个时间点做重叠平均）
    from torch.utils.data import DataLoader
    import torch
    window_size = dataset[0][0].shape[-1]  # 假设dataset每个样本为 [C, T]
    stride = 1  # 滑动步长，可根据实际情况调整
    num_rows = len(orig_df)
    num_modalities = len(dataset_modalities)
    # 用于累加补全结果和计数
    completed_sum = np.zeros((num_rows, num_modalities), dtype=np.float32)
    completed_count = np.zeros((num_rows, num_modalities), dtype=np.float32)

    # 构造滑窗索引（假设dataset顺序与原始csv一致，且每个样本为一个滑窗）
    loader = DataLoader(dataset, batch_size=32)
    row_idx = 0
    with torch.no_grad():
        for batch_idx, (batch, label, mask_idx, is_real_mask) in enumerate(tqdm(loader, desc="Complete (sliding window)")):
            batch = batch.to(device)
            masked, mask_idx = mask_channel(batch, mask_indices)
            batch_size, C, T = batch.size()
            for i in range(batch_size):
                window = masked[i].t()  # [T, C]
                out, _ = model(window)  # out: [C, T]
                out_np = out.cpu().numpy().T  # [T, C] -> [T, C]
                start = row_idx
                end = row_idx + T
                if end > num_rows:
                    break
                completed_sum[start:end, :] += out_np
                completed_count[start:end, :] += 1
                row_idx += stride
            # 注意：这里假设dataset是顺序滑窗，若不是需根据实际索引调整

    # 防止除0
    completed_count[completed_count == 0] = 1
    completed_avg = completed_sum / completed_count
    # 替换orig_df中modalities部分
    orig_df.loc[:, dataset_modalities] = completed_avg[:num_rows, :]
    orig_df.to_csv(save_path, index=False)
    print(f"滑动窗口补全结果已保存到: {save_path}")
# src/train.py

"""
train.py

Module to train the T-GAT-UNet model on sliding-window time series dataset with self-supervised masking.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train T-GAT-UNet on time series data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    return args


import random
def mask_channel(x, mask_indices, mask_ratio=0.2):
    """
    有目的性遮掩：只遮掩实际需要补全的模态（mask_indices指定），每个样本随机遮掩一个需补全通道。
    Args:
        x: Tensor of shape [batch, channels, time]
        mask_indices: list of channel indices需要被遮掩的（即实际缺失的）模态
    Returns:
        x_masked, mask_idx
    """
    batch, C, T = x.size()
    # 遮掩所有 have 通道（即 mask_indices 指定的所有通道），用 common_modalities 预测 have 通道
    if len(mask_indices) == 0:
        # 没有需要遮掩的通道，直接返回原数据
        return x, torch.tensor([-1]*batch)
    x_masked = x.clone()
    for i in range(batch):
        for idx in mask_indices:
            x_masked[i, idx, :] = 0
    # 返回所有被遮掩的通道索引
    return x_masked, torch.tensor(mask_indices)



# =========================
# 分阶段训练主循环
# =========================
def train_phased(
    model, dataloader, optimizer, criterion, device,
    mask_indices, phase="encode", classifier=None, gen_optimizer=None, disc_optimizer=None,
    dynamic_need_indices=None
):
    """
    phase: "train"
    - train: 训练编码器+解码器+分类器（端到端）
    """
    model.train()
    ce_loss = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch, labels, mask_idx, is_real_mask in tqdm(dataloader, desc=phase.capitalize()):
        batch = batch.to(device)
        labels = labels.to(device)
        is_real_mask = is_real_mask.to(device)
        masked, mask_idx = mask_channel(batch, mask_indices)
        batch_size, C, T = batch.size()
        # === Debug: 检查输入是否有 NaN/Inf ===
        if torch.isnan(batch).any() or torch.isinf(batch).any():
            print("[DEBUG] Input batch contains NaN or Inf!")
        if torch.isnan(masked).any() or torch.isinf(masked).any():
            print("[DEBUG] Masked input contains NaN or Inf!")
        loss = 0.0
        cls_loss = 0.0
        recon_loss = 0.0
        optimizer.zero_grad()
        for i in range(batch_size):
            window = masked[i].t()  # [T, C]
            out, logits = model(window)  # out: [C, T], logits: [num_classes]
            # === Debug: 检查模型输出是否有 NaN/Inf ===
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"[DEBUG] Model output contains NaN or Inf at sample {i}!")
            # 只对真实通道计算重建损失
            if is_real_mask.dim() == 2:
                real_channels = is_real_mask[i]
            else:
                real_channels = is_real_mask
            recon_loss_i = 0.0
            real_count = 0
            for c in range(C):
                if real_channels[c]:
                    target = batch[i, c, :]
                    pred = out[c, :]
                    recon_loss_i = recon_loss_i + criterion(pred, target)
                    real_count += 1
            if real_count > 0:
                recon_loss_i = recon_loss_i / real_count
            cls_loss_i = ce_loss(logits.unsqueeze(0), labels[i].unsqueeze(0))
            loss += recon_loss_i + cls_loss_i
            recon_loss += recon_loss_i
            cls_loss += cls_loss_i
            pred_class = logits.argmax(-1).item()
            if pred_class == labels[i].item():
                total_correct += 1
            total_samples += 1
        loss = loss / batch_size
        # === Debug: 检查 loss 是否有 NaN/Inf ===
        # 避免重复包裹 tensor，直接用 float/torch API 检查
        if isinstance(loss, torch.Tensor):
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("[DEBUG] Loss is NaN or Inf after batch!")
        else:
            import math
            if math.isnan(loss) or math.isinf(loss):
                print("[DEBUG] Loss is NaN or Inf after batch!")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_size
        total_recon_loss += recon_loss
        total_cls_loss += cls_loss
    n = len(dataloader.dataset)
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return total_loss / n, total_recon_loss / n, total_cls_loss / n, acc


def eval_loop(model, dataloader, criterion, device, mask_indices):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    with torch.no_grad():
        for batch, _, _, is_real_mask in tqdm(dataloader, desc="Eval"):
            batch = batch.to(device)
            is_real_mask = is_real_mask.to(device)
            masked, mask_idx = mask_channel(batch, mask_indices)
            batch_size, C, T = batch.size()
            loss = 0.0
            recon_loss = 0.0
            for i in range(batch_size):
                window = masked[i].t()
                out, _ = model(window)
                # 只对真实通道计算重建损失
                real_channels = is_real_mask
                recon_loss_i = 0.0
                real_count = 0
                for c in range(C):
                    if real_channels[c]:
                        target = batch[i, c, :]
                        pred = out[c, :]
                        recon_loss_i = recon_loss_i + criterion(pred, target)
                        real_count += 1
                if real_count > 0:
                    recon_loss_i = recon_loss_i / real_count
                loss += recon_loss_i
                recon_loss += recon_loss_i
            loss = loss / batch_size
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss
    n = len(dataloader.dataset)
    return total_loss / n, total_recon_loss / n, 0.0, 0.0


# =========================
# 新主循环入口
# =========================
def main():
    set_seed(42)
    # 设置 logging 使 INFO 级别输出到控制台
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    # Load config.yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 参数
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 100)
    lr = config.get('lr', 1e-3)
    early_stop_patience = config.get('patience', 10)
    log_dir = config.get('log_dir', 'Logs')
    ckpt_dir = config.get('ckpt_dir', 'Checkpoints')
    mode = config.get('mode', 'train').lower()
    model_path = config.get('model_path', os.path.join(ckpt_dir, 'best_model.pth'))
    common_modalities = config.get('common_modalities', [])
    dataset_modalities_cfg = config.get('dataset_modalities', {})
    # ====== 多数据集联合加载 ======
    # 加载所有 data_files，推断每个文件的 source dataset，并为每个样本保留其 need/have 信息
    from data import load_data, SlidingWindowDataset
    data_files = config.get('data_files', [])
    data_dir = config.get('data_dir', '')
    # 拼接路径
    data_files = [os.path.join(data_dir, f) if not os.path.isabs(f) and not os.path.exists(f) else f for f in data_files]
    all_dfs = []
    all_sources = []
    for f in data_files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        # 推断source
        fname = os.path.basename(f).lower()
        if 'fm' in fname:
            source = 'FM'
        elif 'od' in fname:
            source = 'OD'
        elif 'mefar' in fname:
            source = 'MEFAR'
        else:
            source = 'UNKNOWN'
        df['__source__'] = source
        all_dfs.append(df)
        all_sources.append(source)
    data = pd.concat(all_dfs, ignore_index=True)

    # 统一modalities
    common_modalities = config.get('common_modalities', [])
    dataset_modalities_cfg = config.get('dataset_modalities', {})
    all_have = []
    all_need = []
    for dsname, modcfg in dataset_modalities_cfg.items():
        all_have.extend(modcfg.get('have', []))
        all_need.extend(modcfg.get('need', []))
    all_modalities = list(dict.fromkeys([m.strip().lower() for m in (common_modalities + all_have + all_need)]))
    block_col = config['block_col'].strip().lower()
    label_col = config.get('label_col', 'F').strip().lower()
    for col in all_modalities:
        if col not in data.columns:
            data[col] = 0.0
    feature_cols = [col for col in all_modalities if col != label_col]
    cols_to_keep = [block_col, label_col] + feature_cols + ['__source__']
    seen2 = set()
    cols_to_keep = [x for x in cols_to_keep if not (x in seen2 or seen2.add(x))]
    data = data[[col for col in cols_to_keep if col in data.columns]]
    data = data.reset_index(drop=True)  # Ensure 0-based integer index

    # Set model input/output/channel parameters after feature_cols is defined
    # (feature_cols is defined after data loading below)

    # Set model input/output/channel parameters after feature_cols is defined
    in_channels = config.get('in_channels', len(feature_cols))
    hidden_channels = config.get('hidden_channels', 64)
    out_channels = config.get('out_channels', in_channels)
    num_classes = config.get('num_classes', 2)

    # 为每个样本分配need_indices（每个source不同）
    need_indices_list = []
    for i, row in data.iterrows():
        source = row['__source__']
        have_list = common_modalities + dataset_modalities_cfg.get(source, {}).get('have', [])
        need_list = dataset_modalities_cfg.get(source, {}).get('need', [])
        if need_list is None:
            need_list = []
        # need_indices: 在feature_cols中的索引
        need_indices = [feature_cols.index(m) for m in need_list if m in feature_cols]
        if need_indices is None:
            need_indices = []
        need_indices_list.append(need_indices)
    # SlidingWindowDataset expects a DataFrame and global need_indices, so we subclass to support per-sample need_indices
    class MultiDatasetSlidingWindow(SlidingWindowDataset):
        def __init__(self, *args, need_indices_list=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.need_indices_list = need_indices_list
        def __getitem__(self, idx):
            b_idx, start, seg_label = self.indices[idx]
            block = self.blocks[b_idx]
            window_df = block.iloc[start:start + self.window_size]
            # 自动填充缺失值
            window_df = window_df.fillna(0)
            data_array = window_df[self.feature_cols].values[::self.sampling_rate]
            # 断言原始数据无 NaN/Inf
            assert not (np.isnan(data_array).any() or np.isinf(data_array).any()), f"[断言失败] 原始data_array存在NaN/Inf! idx={idx}"
            # Normalize
            if self.normalize == 'zscore':
                mean = data_array.mean(axis=0, keepdims=True)
                std = data_array.std(axis=0, keepdims=True)
                data_array = (data_array - mean) / (std + 1e-6)
            elif self.normalize == 'minmax':
                min_v = data_array.min(axis=0, keepdims=True)
                max_v = data_array.max(axis=0, keepdims=True)
                data_array = (data_array - min_v) / (max_v - min_v + 1e-6)
            # 断言归一化后无 NaN/Inf
            assert not (np.isnan(data_array).any() or np.isinf(data_array).any()), f"[断言失败] 归一化后data_array存在NaN/Inf! idx={idx}"
            tensor = torch.from_numpy(data_array.T).float()  # [C, T]
            label = int(seg_label)
            label = torch.as_tensor(label, dtype=torch.long)
            # 通道可信mask
            is_real_mask = torch.tensor(self.get_is_real_mask(), dtype=torch.bool)
            # 分阶段遮掩/补全逻辑
            idx_in_data = block.index[start]
            if idx_in_data >= len(self.need_indices_list) or idx_in_data < 0:
                need_indices = []
            else:
                need_indices = self.need_indices_list[idx_in_data]
            if need_indices is None:
                need_indices = []
            if self.phase == "encode":
                if len(need_indices) == 0:
                    mask_idx = -1
                    tensor_masked = tensor
                else:
                    mask_idx = np.random.choice(need_indices)
                    tensor_masked = tensor.clone()
                    tensor_masked[mask_idx, :] = 0
                return tensor_masked, label, mask_idx, is_real_mask
            elif self.phase == "decode":
                if self.dynamic_need and len(need_indices) > 0:
                    need_idx = np.random.choice(need_indices)
                elif len(need_indices) > 0:
                    need_idx = need_indices[0]
                else:
                    need_idx = -1
                tensor_masked = tensor.clone()
                if need_idx != -1:
                    tensor_masked[need_idx, :] = 0
                return tensor_masked, label, need_idx, is_real_mask
            else:
                return tensor, label, -1, is_real_mask

    dataset = MultiDatasetSlidingWindow(
        data=data,
        block_col=block_col,
        feature_cols=feature_cols,
        window_size=config['window_size'],
        step_size=config['step_size'],
        sampling_rate=config.get('sampling_rate', 1),
        normalize=config.get('norm_method'),
        label_col=label_col,
        phase="encode",
        need_indices=[],  # not used, per-sample instead
        dynamic_need=False,
        need_indices_list=need_indices_list
    )
    total = len(dataset)
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
    print(f"[数据集分配] (多数据集融合) 总数: {total}, 训练: {train_len}, 验证: {val_len}, 测试: {test_len}")
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 输出训练集标签分布
    check_label_distribution(train_ds)

    dataset_modalities = feature_cols
    # 遮掩逻辑修正：遮掩所有 have 通道，用 common_modalities 预测 have 通道（以FM/OD/MEFAR全集为准）
    have_list_only = all_have
    mask_indices = [i for i, m in enumerate(dataset_modalities) if m in have_list_only]
    if len(mask_indices) == 0:
        logging.warning("当前数据集无 have 通道可用于遮掩，训练时不会遮掩任何通道。")

    # ========== 模型结构适配 ==========
    # 这里假设 TGATUNet 支持 encode/decode 阶段切换，后续可细化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGATUNet(in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_classes=num_classes)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    criterion = MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    import glob
    latest_model = None
    if os.path.exists(model_path):
        latest_model = model_path
    else:
        model_files = sorted(glob.glob(os.path.join(ckpt_dir, 'best_model.pth')), reverse=True)
        if model_files:
            latest_model = model_files[0]


    # ========== TensorBoard 日志器初始化 ========== 
    log_dir_full = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir_full, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir_full)

    # ========== 单阶段端到端训练主循环 ========== 
    if mode == 'train':
        for epoch in range(1, epochs + 1):
            train_loss, train_recon, train_cls, train_acc = train_phased(
                model, train_loader, optimizer, criterion, device, mask_indices
            )
            val_loss, val_recon, _, _ = train_phased(
                model, val_loader, optimizer, criterion, device, mask_indices
            )
            scheduler.step(val_loss)
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"[TRAIN] Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, train_recon={train_recon:.6f}, val_recon={val_recon:.6f}, train_cls={train_cls:.6f}, train_acc={train_acc:.4f}, lr={current_lr:.6e}")
            print(f"Epoch {epoch}: learning rate = {current_lr:.6e}")
            writer.add_scalar('Train/Loss/train', train_loss, epoch)
            writer.add_scalar('Train/Loss/val', val_loss, epoch)
            writer.add_scalar('Train/Recon/train', train_recon, epoch)
            writer.add_scalar('Train/Recon/val', val_recon, epoch)
            writer.add_scalar('Train/Cls/train', train_cls, epoch)
            writer.add_scalar('Train/Acc/train', train_acc, epoch)
            writer.add_scalar('Train/LR', current_lr, epoch)

            # === 动态need补全 ===
            if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'update_need'):
                # 如果是Subset/RandomSplit包装，取原始dataset
                complete_need_with_model(model, train_loader.dataset.dataset, device)
            elif hasattr(train_loader.dataset, 'update_need'):
                complete_need_with_model(model, train_loader.dataset, device)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved best model at epoch {epoch} to {ckpt_dir}/best_model.pth and {model_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break
        latest_model = model_path if os.path.exists(model_path) else os.path.join(ckpt_dir, 'best_model.pth')
        # 自动评估
        if latest_model and os.path.exists(latest_model):
            model.load_state_dict(torch.load(latest_model))
            test_loss, test_recon, _, _ = train_phased(
                model, test_loader, optimizer, criterion, device, mask_indices
            )
            logging.info(f"[Train End] Test Loss: {test_loss:.6f}, Recon Loss: {test_recon:.6f}")
            writer.add_scalar('Loss/test', test_loss)
            writer.add_scalar('Recon/test', test_recon)
        else:
            logging.warning(f"未找到可用的模型进行测试评估！(查找路径: {latest_model})")
    elif mode == 'eval':
        if latest_model and os.path.exists(latest_model):
            model.load_state_dict(torch.load(latest_model))
            logging.info(f"[Eval Mode] 加载模型: {latest_model}")
            test_loss, test_recon, _, _ = train_phased(
                model, test_loader, optimizer, criterion, device, mask_indices, phase="decode"
            )
            logging.info(f"[Eval Mode] Test Loss: {test_loss:.6f}, Recon Loss: {test_recon:.6f}")
            writer.add_scalar('Loss/test', test_loss)
            writer.add_scalar('Recon/test', test_recon)
            # ========== 评估模式下不再做滑动窗口补全和保存csv ========== 
        else:
            logging.warning(f"未找到可用的模型进行测试评估！(查找路径: {latest_model})")
    else:
        logging.error(f"未知模式: {mode}，请在config.yaml中设置 mode: train 或 mode: eval")
    writer.close()

if __name__ == '__main__':
    main()

