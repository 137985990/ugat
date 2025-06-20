# src/data.py

"""
data.py

Module for:  # 模块功能：
- loading multivariate time series data from CSV files
- creating sliding-window PyTorch datasets
- normalization of windows
"""
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the config YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_data(file_paths: list[str], parse_dates: list[str] | None = None) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files into a single DataFrame.

    Args:
        file_paths (list[str]): List of CSV file paths.
        parse_dates (list[str] | None): Columns to parse as dates.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path, parse_dates=parse_dates)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Detect missing values
    if data.isnull().any().any():
        print("Warning: Missing values detected. Filling forward.")
        data = data.fillna(method='ffill').fillna(method='bfill')

    return data


class SlidingWindowDataset(Dataset):
    def get_is_real_mask(self):
        """
        返回每个通道的可信性mask: 1=真实（common_modalities+have），0=补全（need）
        """
        # 获取当前主数据集名
        import os
        dataset_name = None
        if hasattr(self, 'dataset_name') and self.dataset_name:
            dataset_name = self.dataset_name
        else:
            # 尝试从feature_cols和config结构推断
            # 这里假设feature_cols顺序为 common_modalities + have + need
            # 直接用self.need_indices区分
            mask = [1]*len(self.feature_cols)
            for idx in (self.need_indices if self.need_indices else []):
                mask[idx] = 0
            return mask
        # 若有更复杂需求可扩展
        return [1]*len(self.feature_cols)
    def __init__(self,
                 data,
                 block_col,
                 feature_cols,
                 window_size,
                 step_size,
                 sampling_rate=1,
                 normalize=None,
                 label_col='F',
                 phase="encode",
                 need_indices=None,
                 dynamic_need=False):
        super().__init__()
        self.data = data.copy()
        self.block_col = block_col
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.label_col = label_col
        self.phase = phase
        self.need_indices = need_indices if need_indices is not None else []
        self.dynamic_need = dynamic_need

        # Prepare blocks
        self.blocks = [df for _, df in self.data.groupby(self.block_col)]

        # 新增：只在标签连续区间内滑窗，并直接记录区间标签
        self.indices = []  # List of tuples (block_idx, start_idx, seg_label)
        for b_idx, block in enumerate(self.blocks):
            labels = block[self.label_col].values
            for start in range(0, len(labels) - self.window_size + 1, self.step_size):
                window_labels = labels[start:start + self.window_size]
                # 只有全为1才为1，否则为0
                seg_label = 1 if (window_labels == 1).all() else 0
                self.indices.append((b_idx, start, seg_label))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b_idx, start, seg_label = self.indices[idx]
        block = self.blocks[b_idx]
        window_df = block.iloc[start:start + self.window_size]
        data_array = window_df[self.feature_cols].values[::self.sampling_rate]  # [T, C]

        # Normalize
        if self.normalize == 'zscore':
            mean = data_array.mean(axis=0, keepdims=True)
            std = data_array.std(axis=0, keepdims=True)
            data_array = (data_array - mean) / (std + 1e-6)
        elif self.normalize == 'minmax':
            min_v = data_array.min(axis=0, keepdims=True)
            max_v = data_array.max(axis=0, keepdims=True)
            data_array = (data_array - min_v) / (max_v - min_v + 1e-6)

        # 统计特征增强
        # data_array: [T, C]
        feats = []
        for c in range(data_array.shape[1]):
            x = data_array[:, c]
            # mean = np.mean(x)  # 删除
            # std = np.std(x)   # 删除
            energy = np.sum(x ** 2)
            # FFT主频幅值
            fft = np.fft.fft(x)
            fft_mag = np.abs(fft[:len(fft)//2])
            main_freq = np.max(fft_mag)
            # 极值点位置（归一化到[0,1]）
            max_pos = np.argmax(x) / (len(x)-1) if len(x) > 1 else 0.0
            min_pos = np.argmin(x) / (len(x)-1) if len(x) > 1 else 0.0
            feats.extend([energy, main_freq, max_pos, min_pos])
        feats = np.array(feats, dtype=np.float32)  # [C*4]
        # 拼接到原始特征后
        tensor = torch.from_numpy(data_array.T).float()  # [C, T]
        feats_tensor = torch.from_numpy(feats).float()   # [C*4]

        # Get label: 用最大值法标记窗口标签（适用于0/1标签）
        label = int(seg_label)
        label = torch.tensor(label, dtype=torch.long)

        # 通道可信mask
        is_real_mask = torch.tensor(self.get_is_real_mask(), dtype=torch.bool)

        # 分阶段遮掩/补全逻辑
        if self.phase == "encode":
            if len(self.need_indices) == 0:
                mask_idx = -1
                tensor_masked = tensor
            else:
                mask_idx = np.random.choice(self.need_indices)
                tensor_masked = tensor.clone()
                tensor_masked[mask_idx, :] = 0
            return tensor_masked, feats_tensor, label, mask_idx, is_real_mask
        elif self.phase == "decode":
            if self.dynamic_need and len(self.need_indices) > 0:
                need_idx = np.random.choice(self.need_indices)
            elif len(self.need_indices) > 0:
                need_idx = self.need_indices[0]
            else:
                need_idx = -1
            tensor_masked = tensor.clone()
            if need_idx != -1:
                tensor_masked[need_idx, :] = 0
            return tensor_masked, feats_tensor, label, need_idx, is_real_mask
        else:
            return tensor, feats_tensor, label, -1, is_real_mask


# 调试：打印前50个窗口的F列内容和标签
        if idx < 50:
            print(f"窗口索引: {idx}, F列: {f_vals}, 标签: {label}")

def create_dataset_from_config(cfg_path: str, phase: str = "encode", need_indices: list[int] = None, dynamic_need: bool = False) -> SlidingWindowDataset:
    """
    Factory function to create SlidingWindowDataset from config file.
    支持分阶段、动态need。
    Args:
        cfg_path (str): Path to the YAML config file.
        phase (str): "encode" or "decode"
        need_indices (list[int]): 需要补全的通道索引
        dynamic_need (bool): decode阶段是否动态指定need
    Returns:
        SlidingWindowDataset: Initialized dataset.
    """
    import os
    cfg = load_config(cfg_path)
    data_dir = cfg.get('data_dir', '')
    data_files = cfg.get('data_files', [])
    # 拼接路径：如果不是绝对路径且文件不存在，则用 data_dir 拼接
    data_files = [os.path.join(data_dir, f) if not os.path.isabs(f) and not os.path.exists(f) else f for f in data_files]
    data = load_data(data_files, parse_dates=cfg.get('parse_dates'))
    data.columns = [c.strip().lower() for c in data.columns]
    # 自动推断modalities: common_modalities + dataset_modalities[主数据集]['have'+'need']
    common_modalities = cfg.get('common_modalities', [])
    dataset_modalities_cfg = cfg.get('dataset_modalities', {})
    dataset_name = cfg.get('dataset', None)
    if dataset_name is None:
        # 自动推断主数据集名（与train.py一致）
        data_files_cfg = cfg.get('data_files', [])
        found = None
        import os
        for f in data_files_cfg:
            fname = os.path.basename(f).lower()
            if 'fm' in fname:
                found = 'FM'
                break
            elif 'od' in fname:
                found = 'OD'
                break
            elif 'mefar' in fname:
                found = 'MEFAR'
                break
        dataset_name = found
    have_list = dataset_modalities_cfg.get(dataset_name, {}).get('have', [])
    need_list = dataset_modalities_cfg.get(dataset_name, {}).get('need', [])
    modalities_raw = [m.strip().lower() for m in (common_modalities + have_list + need_list)]
    from collections import Counter
    counter = Counter(modalities_raw)
    dups = [k for k, v in counter.items() if v > 1]
    if dups:
        print('重复项:', dups)
    all_modalities = list(dict.fromkeys(modalities_raw))
    # 保证所有modalities、label_col、block_col都小写
    all_modalities = [m.lower() for m in modalities_raw]
    block_col = cfg['block_col'].strip().lower()
    label_col = cfg.get('label_col', 'F').strip().lower()
    for col in all_modalities:
        if col not in data.columns:
            data[col] = 0.0
    feature_cols = [col for col in all_modalities if col != label_col]
    cols_to_keep = [block_col, label_col] + feature_cols
    seen2 = set()
    cols_to_keep = [x for x in cols_to_keep if not (x in seen2 or seen2.add(x))]
    data = data[[col for col in cols_to_keep if col in data.columns]]
    # 自动推断need_indices
    if need_indices is None:
        # 尝试从config.yaml结构推断
        # 支持 common_modalities + dataset_modalities 结构
        common_modalities = cfg.get('common_modalities', [])
        dataset_modalities_cfg = cfg.get('dataset_modalities', {})
        dataset_name = cfg.get('dataset', None)
        if dataset_name is None:
            import os
            cfg_base = os.path.basename(cfg_path).lower()
            if 'fm' in cfg_base:
                dataset_name = 'FM'
            elif 'od' in cfg_base:
                dataset_name = 'OD'
            elif 'mefar' in cfg_base:
                dataset_name = 'MEFAR'
        have_list = common_modalities + dataset_modalities_cfg.get(dataset_name, {}).get('have', [])
        need_list = dataset_modalities_cfg.get(dataset_name, {}).get('need', [])
        all_mods = have_list + need_list
        # need_indices: 在feature_cols中的索引
        need_indices = [feature_cols.index(m) for m in need_list if m in feature_cols]
    ds = SlidingWindowDataset(
        data=data,
        block_col=block_col,
        feature_cols=feature_cols,
        window_size=cfg['window_size'],
        step_size=cfg['step_size'],
        sampling_rate=cfg.get('sampling_rate', 1),
        normalize=cfg.get('norm_method'),
        label_col=label_col,
        phase=phase,
        need_indices=need_indices,
        dynamic_need=dynamic_need
    )
    return ds

def check_label_distribution(dataset):
    """
    检查并输出数据集标签分布和所有标签种类
    """
    import collections
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
