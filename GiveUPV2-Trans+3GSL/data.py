# src/data.py

"""
data.py

Module for:
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

    if data.isnull().any().any():
        print("Warning: Missing values detected. Filling forward.")
        data = data.fillna(method='ffill').fillna(method='bfill')

    return data


class Normalizer:
    """
    Handles data normalization by fitting on training data and transforming all data.
    """
    def __init__(self, method: str | None = 'zscore'):
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray):
        """
        Fit the normalizer on the data.
        Args:
            data (np.ndarray): A numpy array of shape (n_samples, n_features).
        """
        if self.method is None:
            return
        if self.method == 'zscore':
            self.mean = data.mean(axis=0, keepdims=True)
            self.std = data.std(axis=0, keepdims=True)
        elif self.method == 'minmax':
            self.min = data.min(axis=0, keepdims=True)
            self.max = data.max(axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted parameters.
        Args:
            data (np.ndarray): Data to transform.
        Returns:
            np.ndarray: Normalized data.
        """
        if self.method is None:
            return data
        if self.method == 'zscore':
            if self.mean is None or self.std is None:
                raise RuntimeError("Normalizer has not been fitted yet.")
            return (data - self.mean) / (self.std + 1e-8)
        elif self.method == 'minmax':
            if self.min is None or self.max is None:
                raise RuntimeError("Normalizer has not been fitted yet.")
            return (data - self.min) / (self.max - self.min + 1e-8)
        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


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
                 normalizer: Normalizer | None = None,
                 sampling_rate: int = 1):
        """
        Args:
            data (pd.DataFrame): Input data containing time series for a specific split (e.g., train).
            block_col (str): Column name to group data into separate series.
            feature_cols (list[str]): List of feature column names.
            window_size (int): Number of time steps per window.
            step_size (int): Step between windows in time steps.
            normalizer (Normalizer | None): A (pre-fitted) Normalizer object.
            sampling_rate (int): Down-sampling rate; only take every nth sample.
        """
        super().__init__()
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.step_size = step_size
        self.sampling_rate = sampling_rate
        self.normalizer = normalizer

        # Normalize features before creating windows
        features = data[self.feature_cols].to_numpy()
        if self.normalizer:
            features = self.normalizer.transform(features)
        
        # Create a new DataFrame with normalized features
        self.processed_data = data[[block_col]].copy()
        self.processed_data[self.feature_cols] = features

        self.blocks = [
            df[self.feature_cols].to_numpy() for _, df in self.processed_data.groupby(block_col)
        ]

        self.indices = []
        for b_idx, block_array in enumerate(self.blocks):
            length = len(block_array)
            for start in range(0, length - window_size + 1, step_size):
                self.indices.append((b_idx, start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        b_idx, start = self.indices[idx]
        block_array = self.blocks[b_idx]

        window_array = block_array[start:start + self.window_size]
        
        # Apply down-sampling if specified
        if self.sampling_rate > 1:
            window_array = window_array[::self.sampling_rate]

        # Convert to tensor with shape (channels, time_steps)
        tensor = torch.from_numpy(window_array.T.copy()).float()
        return tensor


# The factory function create_dataset_from_config needs to be removed or adapted
# as dataset creation will now depend on the data split (train/val/test) and a fitted Normalizer.
# This logic will be moved to the main training script.