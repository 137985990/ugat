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
    with open(config_path, 'r') as f:
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
        # 统一所有列名为小写，防止大小写不一致
        df.columns = [c.lower() for c in df.columns]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Detect missing values
    if data.isnull().any().any():
        print("Warning: Missing values detected. Filling forward.")
        data = data.fillna(method='ffill').fillna(method='bfill')

    return data


class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for sliding-window on time series data grouped by blocks.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 block_col: str,
                 feature_cols: list[str],
                 window_size: int,
                 step_size: int,
                 sampling_rate: int = 1,
                 normalize: str | None = None):
        """
        Args:
            data (pd.DataFrame): Input data containing time series.
            block_col (str): Column name to group data into separate series (e.g., sensor ID).
            time_col (str): Column name for time index.
            feature_cols (list[str]): List of feature column names.
            window_size (int): Number of time steps per window.
            step_size (int): Step between windows in time steps.
            sampling_rate (int): Down-sampling rate; only take every nth sample.
            normalize (str | None): 'zscore' or 'minmax' normalization.
        """
        super().__init__()
        self.data = data.copy()
        self.block_col = block_col
        # 统一特征名为小写
        self.feature_cols = [c.lower() for c in feature_cols]
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        # self.time_col = time_col  # Remove time_col, not needed

        # Prepare blocks
        self.blocks = [
            df for _, df in self.data.groupby(self.block_col)
        ]

        # Precompute window start indices for each block
        self.indices = []  # List of tuples (block_idx, start_idx)
        for b_idx, block in enumerate(self.blocks):
            length = len(block)
            for start in range(0, length - window_size + 1, step_size):
                self.indices.append((b_idx, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        b_idx, start = self.indices[idx]
        block = self.blocks[b_idx]

        # Sample window
        window_df = block.iloc[start:start + self.window_size]
        # Debug: check columns before selecting features
        try:
            data_array = window_df[self.feature_cols].values[::self.sampling_rate]
        except KeyError as e:
            print(f"[DEBUG] KeyError selecting features: {e}\nAvailable columns: {list(window_df.columns)}\nRequested: {self.feature_cols}\nBlock idx: {b_idx}, Window start: {start}")
            raise

        # Normalize
        if self.normalize == 'zscore':
            mean = data_array.mean(axis=0, keepdims=True)
            std = data_array.std(axis=0, keepdims=True)
            data_array = (data_array - mean) / (std + 1e-6)
        elif self.normalize == 'minmax':
            min_v = data_array.min(axis=0, keepdims=True)
            max_v = data_array.max(axis=0, keepdims=True)
            data_array = (data_array - min_v) / (max_v - min_v + 1e-6)

        # Convert to tensor with shape (channels, time_steps)
        tensor = torch.from_numpy(data_array.T).float()
        return tensor



def create_dataset_from_config(cfg_path: str):
    """
    Factory function to create train/val/test datasets from config file.
    - Each file滑窗，window内归一化
    - 所有窗口拼接后，按60/20/20划分
    Returns:
        train_set, val_set, test_set: list[Tensor]
    """
    cfg = load_config(cfg_path)
    all_windows = []
    for file_path in cfg['data_files']:
        data = load_data([file_path], parse_dates=cfg.get('parse_dates'))
        ds = SlidingWindowDataset(
            data=data,
            block_col=cfg['block_col'],
            feature_cols=cfg['feature_cols'],
            window_size=cfg['window_size'],
            step_size=cfg['step_size'],
            sampling_rate=cfg.get('sample_rate', 1),
            normalize=cfg.get('norm_method')
        )
        all_windows.extend([ds[i] for i in range(len(ds))])
    total = len(all_windows)
    train_len = int(0.6 * total)
    val_len = int(0.2 * total)
    test_len = total - train_len - val_len
    train_set = all_windows[:train_len]
    val_set = all_windows[train_len:train_len+val_len]
    test_set = all_windows[train_len+val_len:]
    return train_set, val_set, test_set
