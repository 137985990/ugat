# src/data.py

"""
data.py

V13 多模态时间序列数据处理模块

核心功能：
- 多数据集（FM/OD/MEFAR）联合加载，统一特征列到32维
- 动态滑动窗口生成，确保所有数据集的block都被采样
- 支持每个样本动态获取source_dataset和对应have/need通道定义
- 训练时动态遮掩have通道，损失函数按source_dataset区分处理
"""
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import List, Dict, Tuple, Optional


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


def load_and_merge_multimodal_datasets(data_files: List[str], feature_cols: List[str], 
                                      dataset_modalities_config: Optional[Dict] = None, 
                                      balancing_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    加载并合并多个多模态数据集，确保所有数据集都有统一的32特征列
    
    Args:
        data_files: 数据文件路径列表
        feature_cols: 目标特征列列表（32列）
        dataset_modalities_config: 数据集模态配置
        balancing_config: 标签配平配置
    
    Returns:
        pd.DataFrame: 合并后的数据集，包含统一的32特征列和source列
    """
    all_dfs = []
    all_sources = []
    
    print(f"开始加载{len(data_files)}个多模态数据集...")
    
    for i, file_path in enumerate(data_files):
        print(f"\n处理文件 {i+1}/{len(data_files)}: {file_path}")
        
        # 读取原始数据
        df = pd.read_csv(file_path)
        print(f"   - 原始形状: {df.shape}")
        print(f"   - 原始列数: {len(df.columns)}")
        
        # 推断数据集类型
        fname = os.path.basename(file_path).lower()
        if 'fm' in fname:
            source = 'FM'
            block_offset = 0
        elif 'od' in fname:
            source = 'OD'
            block_offset = 1000
        elif 'mefar' in fname:
            source = 'MEFAR'
            block_offset = 2000
        else:
            source = 'UNKNOWN'
            block_offset = 9000
        
        print(f"   - 数据集类型: {source}")        # 标准化列名：保留关键列原样，其他转小写
        original_cols = df.columns.tolist()
        new_cols = []
        for col in original_cols:
            if col.lower() in ['block', 'id', 'session']:
                new_cols.append(col.lower())  # 这些列转小写
            elif col in ['F']:  # 保持F列的原始大写
                new_cols.append(col)  # 保持原样
            else:
                new_cols.append(col.strip().lower())
        df.columns = new_cols
        
        # 处理block偏移避免冲突
        if 'block' in df.columns:
            original_block_count = df['block'].nunique()
            original_range = f"{df['block'].min()}-{df['block'].max()}"
            df['block'] = df['block'] + block_offset
            new_range = f"{df['block'].min()}-{df['block'].max()}"
            print(f"   - Block处理: {original_block_count}个 ({original_range}) → ({new_range})")
        else:
            print(f"   - 警告: 未找到block列")
        
        # 确保所有目标特征列存在，缺失的补0
        missing_cols = []
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                missing_cols.append(col)
        
        if missing_cols:
            print(f"   - 补充缺失列({len(missing_cols)}个): {missing_cols}")
        
        # 添加source列
        df['source'] = source
        
        # 数据类型转换
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)        # 确保标签列存在且为数值型
        label_col = 'F'  # 默认标签列
        if 'F' in df.columns:  # 保持原始大写F列
            df['F'] = pd.to_numeric(df['F'], errors='coerce').fillna(0)
            label_col = 'F'
        elif 'f' in df.columns:  # 如果有小写f列
            df['f'] = pd.to_numeric(df['f'], errors='coerce').fillna(0)
            label_col = 'f'
        
        # 对FM数据集进行标签配平
        if (balancing_config and 
            balancing_config.get('enabled', False) and 
            source in balancing_config.get('target_datasets', []) and 
            label_col in df.columns):
            print(f"   - 开始{source}数据集标签配平...")
            strategy = balancing_config.get('strategy', 'undersample')
            target_ratio = balancing_config.get('target_ratio', None)
            df = balance_fm_labels(df, source, label_col, strategy, target_ratio)
        
        print(f"   - 处理后形状: {df.shape}")
        print(f"   - 特征列完整性: {len([c for c in feature_cols if c in df.columns])}/{len(feature_cols)}")
        
        all_dfs.append(df)
        all_sources.append(source)
    
    # 合并所有数据集
    print(f"\n合并{len(all_dfs)}个数据集...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 最终统计
    print(f"\n合并完成统计:")
    print(f"   - 总记录数: {len(combined_df):,}")
    print(f"   - 总列数: {len(combined_df.columns)}")
    print(f"   - 总block数: {combined_df['block'].nunique()}")
    
    # 各数据集分布
    print(f"   - 数据集分布:")
    for source in combined_df['source'].unique():
        subset = combined_df[combined_df['source'] == source]
        count = len(subset)
        blocks = subset['block'].nunique()
        block_range = f"{subset['block'].min()}-{subset['block'].max()}"
        print(f"     * {source}: {count:,} 条记录, {blocks} 个block (范围: {block_range})")
    
    # 检查特征列完整性
    missing_in_combined = [col for col in feature_cols if col not in combined_df.columns]
    if missing_in_combined:
        print(f"   - 警告: 合并后仍缺失特征列: {missing_in_combined}")
    else:
        print(f"   - 所有{len(feature_cols)}个特征列完整")
    
    return combined_df



class SlidingWindowDataset(Dataset):
    """
    V13 多模态滑动窗口数据集
    
    核心特性：
    1. 支持多数据集混合训练（FM/OD/MEFAR）
    2. 每个样本动态获取source_dataset和对应have/need通道定义
    3. 训练时按source_dataset动态遮掩have通道
    4. 滑动窗口保证在同一block的连续标签区间内
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 block_col: str,
                 feature_cols: List[str],
                 window_size: int,
                 step_size: int,
                 sampling_rate: int = 1,
                 normalize: Optional[str] = None,
                 label_col: str = 'f',
                 phase: str = "encode",
                 need_indices: Optional[List[int]] = None,
                 dynamic_need: bool = False,
                 common_indices: Optional[List[int]] = None,
                 training_strategy: str = "mask_have",
                 dataset_modalities_config: Optional[Dict] = None):
        """
        初始化滑动窗口数据集
        
        Args:
            data: 包含source列的合并数据集
            block_col: block列名
            feature_cols: 特征列列表（32维）
            window_size: 滑动窗口大小（320）
            step_size: 滑动步长（96）
            sampling_rate: 采样率
            normalize: 归一化方法
            label_col: 标签列名
            phase: 阶段（encode/decode）
            need_indices: need通道索引（兼容性参数）
            dynamic_need: 是否动态need
            common_indices: common通道索引
            training_strategy: 训练策略            dataset_modalities_config: 数据集模态配置
        """
        super().__init__()
        self.data = data.copy()
        self.combined_df = data  # 保存原始合并数据用于调试
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
        self.common_indices = common_indices if common_indices is not None else []
        self.training_strategy = training_strategy
        self.dataset_modalities_config = dataset_modalities_config
        
        # 初始化数据集模态配置（硬编码版本作为fallback）
        self._init_dataset_config()
        
        # 准备blocks
        self.blocks = []
        self.block_sources = []  # 记录每个block的source
        for block_id, block_df in self.data.groupby(self.block_col):
            self.blocks.append(block_df)
            # 获取该block的source（假设一个block内所有记录的source相同）
            source = block_df['source'].iloc[0] if 'source' in block_df.columns else 'UNKNOWN'
            self.block_sources.append(source)
        
        # 生成滑动窗口索引
        self._generate_window_indices()
        
    def _init_dataset_config(self):
        """初始化数据集配置"""
        self._dataset_config = {
            'FM': {
                'have': ['alpha_tp9', 'alpha_af7', 'alpha_af8', 'alpha_tp10', 
                        'beta_tp9', 'beta_af7', 'beta_af8', 'beta_tp10',
                        'delta_tp9', 'delta_af7', 'delta_af8', 'delta_tp10',
                        'gamma_tp9', 'gamma_af7', 'gamma_af8', 'gamma_tp10',
                        'theta_tp9', 'theta_af7', 'theta_af8', 'theta_tp10',
                        'ecg', 'breathing'],
                'need': ['space_distance', 'distance_to_eye_center', 'pose_pca']
            },
            'OD': {
                'have': ['space_distance', 'distance_to_eye_center', 'pose_pca'],
                'need': ['alpha_tp9', 'alpha_af7', 'alpha_af8', 'alpha_tp10',
                        'beta_tp9', 'beta_af7', 'beta_af8', 'beta_tp10',
                        'delta_tp9', 'delta_af7', 'delta_af8', 'delta_tp10',
                        'gamma_tp9', 'gamma_af7', 'gamma_af8', 'gamma_tp10',
                        'theta_tp9', 'theta_af7', 'theta_af8', 'theta_tp10',
                        'ecg', 'breathing']
            },
            'MEFAR': {
                'have': [],
                'need': ['alpha_tp9', 'alpha_af7', 'alpha_af8', 'alpha_tp10',
                        'beta_tp9', 'beta_af7', 'beta_af8', 'beta_tp10',
                        'delta_tp9', 'delta_af7', 'delta_af8', 'delta_tp10',
                        'gamma_tp9', 'gamma_af7', 'gamma_af8', 'gamma_tp10',
                        'theta_tp9', 'theta_af7', 'theta_af8', 'theta_tp10',
                        'ecg', 'breathing', 'space_distance', 'distance_to_eye_center', 'pose_pca']
            }
        }
        
    def _generate_window_indices(self):
        """生成滑动窗口索引，确保所有数据集都被采样"""
        self.indices = []  # List of tuples (block_idx, start_idx, seg_label, source_dataset)
        total_segments = 0
        total_windows = 0
        
        print(f"开始生成滑动窗口，共{len(self.blocks)}个blocks...")        
        for b_idx, block in enumerate(self.blocks):
            source_dataset = self.block_sources[b_idx]
            actual_block_id = block[self.block_col].iloc[0] if self.block_col in block.columns else b_idx
            
            # 获取标签并找到连续区间
            labels = block[self.label_col].values
            change_points = np.where(np.diff(labels) != 0)[0] + 1
            seg_starts = np.concatenate(([0], change_points))
            seg_ends = np.concatenate((change_points, [len(labels)]))
            
            block_windows = 0
            for seg_start, seg_end in zip(seg_starts, seg_ends):
                seg_label = labels[seg_start]
                seg_len = seg_end - seg_start
                total_segments += 1
                
                if seg_len < self.window_size:
                    continue
                
                # 在区间内生成滑动窗口
                windows_in_segment = 0
                for start in range(seg_start, seg_end - self.window_size + 1, self.step_size):
                    self.indices.append((b_idx, start, seg_label, source_dataset))
                    windows_in_segment += 1
                    block_windows += 1
                    total_windows += 1        
        # 统计各数据集的窗口分布
        source_stats = {}
        for b_idx, start, seg_label, source_dataset in self.indices:
            if source_dataset not in source_stats:
                source_stats[source_dataset] = {'windows': 0, 'blocks': set()}
            source_stats[source_dataset]['windows'] += 1
            source_stats[source_dataset]['blocks'].add(b_idx)
        
        print(f"\n各数据集窗口分布:")
        for source, stats in source_stats.items():
            print(f"   - {source}: {stats['windows']} 个窗口, {len(stats['blocks'])} 个block")
        
        # 随机打乱所有窗口索引
        print(f"\n随机打乱{len(self.indices)}个窗口样本...")
        np.random.shuffle(self.indices)
        print(f"样本打乱完成")
        print()
        
    def get_have_indices_for_dataset(self, source_dataset: str) -> List[int]:
        """根据源数据集返回对应的have通道索引"""
        if self.dataset_modalities_config and source_dataset in self.dataset_modalities_config:
            have_modalities = self.dataset_modalities_config[source_dataset].get('have', [])
        else:
            have_modalities = self._dataset_config.get(source_dataset, {}).get('have', [])
        
        have_indices = []
        for i, col in enumerate(self.feature_cols):
            if col in have_modalities:
                have_indices.append(i)
        return have_indices
    
    def get_need_indices_for_dataset(self, source_dataset: str) -> List[int]:
        """根据源数据集返回对应的need通道索引"""
        if self.dataset_modalities_config and source_dataset in self.dataset_modalities_config:
            need_modalities = self.dataset_modalities_config[source_dataset].get('need', [])
        else:
            need_modalities = self._dataset_config.get(source_dataset, {}).get('need', [])
        
        need_indices = []
        for i, col in enumerate(self.feature_cols):
            if col in need_modalities:
                need_indices.append(i)
        return need_indices
        
    def get_is_real_mask_for_dataset(self, source_dataset: str) -> List[int]:
        """根据源数据集返回对应的可信性mask: 1=真实（common+have），0=需要补全（need）"""
        need_modalities = []
        if self.dataset_modalities_config and source_dataset in self.dataset_modalities_config:
            need_modalities = self.dataset_modalities_config[source_dataset].get('need', [])
        else:
            need_modalities = self._dataset_config.get(source_dataset, {}).get('need', [])
        
        mask = [1] * len(self.feature_cols)
        for i, col in enumerate(self.feature_cols):
            if col in need_modalities:
                mask[i] = 0  # need通道标记为0（需要补全）
        return mask

    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        b_idx, start, seg_label, source_dataset = self.indices[idx]
        block = self.blocks[b_idx]
        
        # 提取窗口数据
        window_df = block.iloc[start:start + self.window_size]
        data_array = window_df[self.feature_cols].values[::self.sampling_rate]
        
        # 确保数据是数值型
        df_window = pd.DataFrame(data_array, columns=self.feature_cols)
        for col in df_window.columns:
            df_window[col] = pd.to_numeric(df_window[col], errors='coerce').fillna(0.0)
        data_array = df_window.values.astype(np.float32)
        
        # 确保时间步长固定 - 计算期望的时间步数
        expected_time_steps = self.window_size // self.sampling_rate
        actual_time_steps = data_array.shape[0]
        
        if actual_time_steps != expected_time_steps:
            # 如果长度不匹配，进行填充或截断
            if actual_time_steps < expected_time_steps:
                # 填充零到期望长度
                padding = np.zeros((expected_time_steps - actual_time_steps, data_array.shape[1]), dtype=np.float32)
                data_array = np.vstack([data_array, padding])
            else:
                # 截断到期望长度
                data_array = data_array[:expected_time_steps]

        # 归一化
        if self.normalize == 'zscore':
            mean = data_array.mean(axis=0, keepdims=True)
            std = data_array.std(axis=0, keepdims=True)
            data_array = (data_array - mean) / (std + 1e-6)
        elif self.normalize == 'minmax':
            min_v = data_array.min(axis=0, keepdims=True)
            max_v = data_array.max(axis=0, keepdims=True)
            data_array = (data_array - min_v) / (max_v - min_v + 1e-6)

        tensor = torch.from_numpy(data_array.T).float()  # [C, T]
        label = torch.tensor(int(seg_label), dtype=torch.long)
        
        # 获取该数据集的可信性mask
        is_real_mask = torch.tensor(self.get_is_real_mask_for_dataset(source_dataset), dtype=torch.bool)        # 根据训练策略和阶段处理数据
        if self.phase == "encode":
            if self.training_strategy == "mask_have":
                # encode阶段：只遮掩have通道，need通道保持初始状态（已经是0，不额外遮掩）
                have_indices = self.get_have_indices_for_dataset(source_dataset)
                tensor_masked = tensor.clone()
                
                # 只遮掩have通道
                for h_idx in have_indices:
                    if h_idx < tensor_masked.size(0):
                        tensor_masked[h_idx, :] = 0  # 遮掩have通道
                
                # need通道保持初始状态（已经是0，因为缺失数据补0），不额外遮掩
                
                # 返回遮掩的have索引（保持兼容性）
                return tensor_masked, label, have_indices, is_real_mask, source_dataset
            else:
                # 不遮掩策略
                return tensor, label, [], is_real_mask, source_dataset
                
        elif self.phase == "decode":
            # decode阶段：补全need通道
            need_indices_for_dataset = self.get_need_indices_for_dataset(source_dataset)
            if self.dynamic_need and len(need_indices_for_dataset) > 0:
                need_idx = np.random.choice(need_indices_for_dataset)
            elif len(need_indices_for_dataset) > 0:
                need_idx = need_indices_for_dataset[0]
            else:
                need_idx = -1
            
            tensor_masked = tensor.clone()
            if need_idx != -1 and need_idx < tensor_masked.size(0):
                tensor_masked[need_idx, :] = 0
            return tensor_masked, label, [need_idx] if need_idx != -1 else [], is_real_mask, source_dataset
        else:
            # 默认返回原始数据
            return tensor, label, [], is_real_mask, source_dataset

    def update_need_channels(self, all_need_predictions: List[Dict], need_indices: List[int]):
        """
        批量更新数据集中need通道的值
        
        Args:
            all_need_predictions: 每个样本的need通道预测结果列表
            need_indices: 全局need通道索引列表
        """
        print(f"开始批量更新need通道...")
        print(f"   - 预测结果数量: {len(all_need_predictions)}")
        print(f"   - need通道索引: {need_indices}")
        
        # 为了防止索引不匹配，我们按样本索引顺序更新
        updated_count = 0
        for sample_idx, need_pred in enumerate(all_need_predictions):
            if sample_idx < len(self.indices):
                b_idx, start_idx, seg_label, source_dataset = self.indices[sample_idx]
                
                # 获取对应的block
                block = self.blocks[b_idx]
                
                # 更新该窗口对应的数据行的need通道
                end_idx = start_idx + self.window_size
                
                for need_idx, pred_values in need_pred.items():
                    if need_idx < len(self.feature_cols):
                        feature_col = self.feature_cols[need_idx]
                        if feature_col in block.columns:
                            # 直接更新原始数据
                            block_row_start = block.index[start_idx]
                            block_row_end = block.index[min(end_idx, len(block)-1)]
                            
                            # 更新need通道值
                            self.data.loc[block_row_start:block_row_end, feature_col] = pred_values[:end_idx-start_idx].numpy()
                            updated_count += 1
        
        # 重新更新blocks引用
        self.blocks = []
        for block_id, block_df in self.data.groupby(self.block_col):
            self.blocks.append(block_df)
        
        print(f"Need通道更新完成，共更新了{updated_count}个样本窗口")
    
    def update_need(self, sample_idx: int, need_pred: Dict):
        """
        更新单个样本的need通道
        
        Args:
            sample_idx: 样本索引
            need_pred: 该样本的need通道预测结果 {channel_idx: values}
        """
        if sample_idx < len(self.indices):
            b_idx, start_idx, seg_label, source_dataset = self.indices[sample_idx]
            
            # 获取对应的block
            block = self.blocks[b_idx]
            
            # 更新该窗口对应的数据行的need通道
            end_idx = start_idx + self.window_size
            
            for need_idx, pred_values in need_pred.items():
                if need_idx < len(self.feature_cols):
                    feature_col = self.feature_cols[need_idx]
                    if feature_col in block.columns:
                        # 直接更新原始数据
                        block_row_start = block.index[start_idx]
                        block_row_end = block.index[min(end_idx, len(block)-1)]
                        
                        # 更新need通道值
                        self.data.loc[block_row_start:block_row_end, feature_col] = pred_values[:end_idx-start_idx].numpy()
    
    def get_need_channels_status(self, source_dataset: str) -> Dict:
        """
        获取指定数据集need通道的当前状态统计
        
        Args:
            source_dataset: 数据集名称
            
        Returns:
            Dict: need通道状态统计
        """
        need_indices = self.get_need_indices_for_dataset(source_dataset)
        status = {}
        
        for need_idx in need_indices:
            if need_idx < len(self.feature_cols):
                feature_col = self.feature_cols[need_idx]
                # 统计该need通道的值分布
                col_values = self.data[feature_col]
                status[feature_col] = {
                    'mean': col_values.mean(),
                    'std': col_values.std(),
                    'min': col_values.min(),
                    'max': col_values.max(),
                    'zero_ratio': (col_values == 0).mean()
                }
        
        return status

    def analyze_dataset_label_distribution(self):
        """
        分析各数据集的标签分布
        """
        print("=" * 60)
        print("数据集标签分布分析")
        print("=" * 60)
        
        # 统计总体分布
        all_labels = [self.indices[i][2] for i in range(len(self.indices))]
        all_sources = [self.indices[i][3] for i in range(len(self.indices))]
        
        import collections
        total_counter = collections.Counter(all_labels)
        source_counter = collections.Counter(all_sources)
        
        print(f"总体标签分布: {dict(total_counter)}")
        print(f"数据集分布: {dict(source_counter)}")
        
        # 各数据集的标签分布
        for source in set(all_sources):
            source_labels = [label for label, src in zip(all_labels, all_sources) if src == source]
            source_label_counter = collections.Counter(source_labels)
            print(f"{source}数据集标签分布: {dict(source_label_counter)}")
        
        print("=" * 60)
        
        return total_counter, source_counter

def create_multimodal_dataset_from_config(config: Dict, 
                                         data_files: Optional[List[str]] = None,
                                         phase: str = "encode") -> SlidingWindowDataset:
    """
    从配置创建多模态数据集
    
    Args:
        config: 配置字典
        data_files: 数据文件列表（可选，从config中获取）
        phase: 训练阶段
        
    Returns:
        SlidingWindowDataset: 初始化的数据集
    """    # 获取数据文件
    if data_files is None:
        data_files_from_config = config.get('data_files', [])
        data_dir = config.get('data_dir', '')
        data_files = [os.path.join(data_dir, f) if not os.path.isabs(f) and not os.path.exists(f) else f 
                     for f in data_files_from_config]
    
    # 构建特征列列表
    common_modalities = config.get('common_modalities', [])
    dataset_modalities = config.get('dataset_modalities', {})
    
    all_feature_mods = common_modalities.copy()
    for dataset, mods in dataset_modalities.items():
        have = mods.get('have', [])
        need = mods.get('need', [])
        for mod in have + need:
            if mod not in all_feature_mods:
                all_feature_mods.append(mod)
    
    feature_cols = all_feature_mods
    
    # 加载和合并数据
    balancing_config = config.get('label_balancing', {})
    combined_df = load_and_merge_multimodal_datasets(data_files, feature_cols, dataset_modalities, balancing_config)
      # 创建数据集
    dataset = SlidingWindowDataset(
        data=combined_df,
        block_col=config.get('block_col', 'block'),
        feature_cols=feature_cols,
        window_size=config.get('window_size', 320),
        step_size=config.get('step_size', 96),
        sampling_rate=config.get('sampling_rate', 1),
        normalize=config.get('norm_method'),
        label_col=config.get('label_col', 'F'),  
        phase=phase,
        training_strategy=config.get('training_strategy', 'mask_have'),
        dataset_modalities_config=dataset_modalities
    )
    
    return dataset


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


def balance_fm_labels(df, source, label_col='F', strategy='undersample', target_ratio=None):
    """
    对FM数据集进行标签配平
    
    Args:
        df: 数据框
        source: 数据集来源
        label_col: 标签列名
        strategy: 配平策略 ('undersample', 'oversample', 'smote')
        target_ratio: 目标比例，None表示完全平衡
    
    Returns:
        pd.DataFrame: 配平后的数据框
    """
    if source != 'FM':
        print(f"   - 跳过{source}数据集的标签配平")
        return df
    
    if label_col not in df.columns:
        print(f"   - 警告: 未找到标签列{label_col}，跳过配平")
        return df
    
    # 检查原始标签分布
    original_counts = df[label_col].value_counts().sort_index()
    print(f"   - FM原始标签分布: {dict(original_counts)}")
    
    if len(original_counts) < 2:
        print(f"   - FM只有单一标签，无需配平")
        return df
    
    # 获取各标签的数据
    label_groups = {}
    for label in original_counts.index:
        label_groups[label] = df[df[label_col] == label].copy()
    
    if strategy == 'undersample':
        # 下采样到最小类别的数量
        min_count = original_counts.min()
        if target_ratio:
            # 如果指定了目标比例，计算目标数量
            target_count = int(min_count / min(target_ratio.values()) * max(target_ratio.values()))
            min_count = min(target_count, min_count)
        
        balanced_dfs = []
        for label, group_df in label_groups.items():
            if len(group_df) > min_count:
                # 确保按block均匀采样，而不是随机采样
                sampled_df = sample_by_blocks(group_df, min_count, 'block')
                balanced_dfs.append(sampled_df)
                print(f"     * 标签{label}: {len(group_df)} -> {len(sampled_df)}")
            else:
                balanced_dfs.append(group_df)
                print(f"     * 标签{label}: {len(group_df)} (保持不变)")
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif strategy == 'oversample':
        # 上采样到最大类别的数量
        max_count = original_counts.max()
        if target_ratio:
            target_count = int(max_count / max(target_ratio.values()) * min(target_ratio.values()))
            max_count = max(target_count, max_count)
        
        balanced_dfs = []
        for label, group_df in label_groups.items():
            if len(group_df) < max_count:
                # 重复采样，优先在不同block间重复
                oversampled_df = oversample_by_blocks(group_df, max_count, 'block')
                balanced_dfs.append(oversampled_df)
                print(f"     * 标签{label}: {len(group_df)} -> {len(oversampled_df)}")
            else:
                balanced_dfs.append(group_df)
                print(f"     * 标签{label}: {len(group_df)} (保持不变)")
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif strategy == 'smote':
        print(f"   - SMOTE策略暂未实现，使用下采样替代")
        return balance_fm_labels(df, source, label_col, 'undersample', target_ratio)
    
    else:
        print(f"   - 未知配平策略{strategy}，跳过配平")
        return df
    
    # 输出配平后的分布
    final_counts = balanced_df[label_col].value_counts().sort_index()
    print(f"   - FM配平后标签分布: {dict(final_counts)}")
    print(f"   - 总记录数: {len(df)} -> {len(balanced_df)}")
    
    return balanced_df


def sample_by_blocks(df, target_count, block_col):
    """按block均匀下采样"""
    if len(df) <= target_count:
        return df
    
    # 获取所有blocks
    blocks = df[block_col].unique()
    samples_per_block = max(1, target_count // len(blocks))
    
    sampled_dfs = []
    remaining_count = target_count
    
    for block_id in blocks:
        if remaining_count <= 0:
            break
        
        block_df = df[df[block_col] == block_id]
        sample_size = min(len(block_df), samples_per_block, remaining_count)
        
        if sample_size > 0:
            sampled_block = block_df.sample(n=sample_size, random_state=42)
            sampled_dfs.append(sampled_block)
            remaining_count -= sample_size
    
    return pd.concat(sampled_dfs, ignore_index=True)


def oversample_by_blocks(df, target_count, block_col):
    """按block均匀上采样"""
    if len(df) >= target_count:
        return df
    
    # 重复整个数据集直到达到目标数量
    repeat_times = (target_count // len(df)) + 1
    oversampled_df = pd.concat([df] * repeat_times, ignore_index=True)
    
    # 随机采样到精确的目标数量
    if len(oversampled_df) > target_count:
        oversampled_df = oversampled_df.sample(n=target_count, random_state=42).reset_index(drop=True)
    
    return oversampled_df



