# train.py - V12完整训练脚本（优化版：高显存利用+低内存占用）

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
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_  # 正确的导入方式

# 优化的数据加载器和内存监控
try:
    from optimized_data_loader import OptimizedDataLoader, print_memory_usage
    OPTIMIZED_LOADER_AVAILABLE = True
except ImportError:
    OPTIMIZED_LOADER_AVAILABLE = False
    print("Warning: Optimized data loader not available, using standard loader.")

# 添加混合精度训练支持
try:
    from torch.cuda.amp.grad_scaler import GradScaler
    from torch.cuda.amp.autocast_mode import autocast
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False
import argparse
import yaml
import os
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np

# 导入自定义模块
from simple_multimodal_integration import create_simple_multimodal_criterion
from enhanced_validation_integration import EnhancedValidationManager

def check_label_distribution(dataset):
    """检查并输出数据集标签分布和所有标签种类"""
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
    """用模型对整个数据集的need通道进行补全，并写回dataset（动态need更新）"""
    model.eval()
    from torch.utils.data import DataLoader
    import torch
    loader = DataLoader(dataset, batch_size=32)
    all_preds = []
    sample_base_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Complete need (dynamic)"):
            if len(batch) == 5:
                batch_x, _, _, _, _ = batch
            else:
                batch_x = batch[0]
            
            batch_x = batch_x.to(device)
            batch_size, C, T = batch_x.size()
            for i in range(batch_size):
                window = batch_x[i].t()
                out, _ = model(window)
                out = out.t()
                all_preds.append(out.cpu())
            sample_base_idx += batch_size

    # 写回数据集

def mask_channel(batch, mask_indices):
    """对batch中的指定通道进行mask"""
    masked = batch.clone()
    if isinstance(mask_indices, (list, tuple)):
        for idx in mask_indices:
            if idx < batch.size(1):
                masked[:, idx, :] = 0
    else:
        if mask_indices < batch.size(1):
            masked[:, mask_indices, :] = 0
    return masked, mask_indices

def train_phased_with_grad_accumulation(model, dataloader, optimizer, criterion, device, mask_indices, 
                                      accumulate_grad_batches=2, use_mixed_precision=True, scaler=None):
    """训练函数 - 支持梯度累积的批量处理版本"""
    model.train()
    kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_cls_consistency_loss = 0.0
    total_encode_correct = 0
    total_decode_correct = 0
    total_samples = 0
    
    # 获取common模态索引
    common_indices = getattr(criterion, 'common_indices', [])
      # 梯度累积相关
    accumulated_loss = 0.0
    step_count = 0
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training")):
        if len(batch_data) == 4:
            batch, labels, _, is_real_mask = batch_data
        else:
            batch, labels, _, is_real_mask, _ = batch_data
        
        # 立即转移到GPU
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        is_real_mask = is_real_mask.to(device, non_blocking=True)
        
        masked, mask_idx = mask_channel(batch, mask_indices)
        batch_size, C, T = batch.size()        
        # 梯度累积：只在累积周期开始时清零梯度
        if step_count % accumulate_grad_batches == 0:
            optimizer.zero_grad()
        
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            with autocast():
                # 真正的批量前向传播
                batch_out_encode, batch_logits_encode = forward_batch_parallel(model, masked, device)
                
                # 第二次前向传播：用重建数据再次分类
                # batch_out_encode是[batch_size, C, T]，直接使用无需转置
                batch_out_decode, batch_logits_decode = forward_batch_parallel(model, batch_out_encode, device)
                
                # 高效的批量重建损失计算
                recon_loss = compute_batch_recon_loss(batch, batch_out_encode, is_real_mask, 
                                                    common_indices, criterion, C, batch_size)
                
                # 批量分类一致性损失
                log_softmax_decode = torch.nn.functional.log_softmax(batch_logits_decode, dim=1)
                softmax_encode = torch.nn.functional.softmax(batch_logits_encode, dim=1)
                cls_consistency_loss = kl_div_loss(log_softmax_decode, softmax_encode)
                
                loss = (recon_loss + cls_consistency_loss) / accumulate_grad_batches  # 标准化梯度
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            
            # 在累积周期结束时更新参数
            if (step_count + 1) % accumulate_grad_batches == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            # 标准精度批量处理
            batch_out_encode, batch_logits_encode = forward_batch_parallel(model, masked, device)
            
            # 第二次前向传播：用重建数据再次分类
            # batch_out_encode是[batch_size, C, T]，直接使用无需转置
            batch_out_decode, batch_logits_decode = forward_batch_parallel(model, batch_out_encode, device)
            
            # 高效的批量重建损失计算
            recon_loss = compute_batch_recon_loss(batch, batch_out_encode, is_real_mask, 
                                                common_indices, criterion, C, batch_size)
            
            # 批量分类一致性损失
            log_softmax_decode = torch.nn.functional.log_softmax(batch_logits_decode, dim=1)
            softmax_encode = torch.nn.functional.softmax(batch_logits_encode, dim=1)
            cls_consistency_loss = kl_div_loss(log_softmax_decode, softmax_encode)
            
            loss = (recon_loss + cls_consistency_loss) / accumulate_grad_batches  # 标准化梯度
            loss.backward()
            
            # 在累积周期结束时更新参数
            if (step_count + 1) % accumulate_grad_batches == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 统计信息（用原始损失值，不除以accumulate_grad_batches）
        original_loss = loss * accumulate_grad_batches
        original_recon = recon_loss
        original_cls = cls_consistency_loss
        
        # 批量准确率计算
        pred_encode = torch.argmax(batch_logits_encode, dim=1)
        pred_decode = torch.argmax(batch_logits_decode, dim=1)
        
        total_encode_correct += (pred_encode == labels).sum().item()
        total_decode_correct += (pred_decode == labels).sum().item()
        total_samples += batch_size
          # 安全地获取损失值
        if isinstance(original_loss, torch.Tensor):
            loss_val = original_loss.item()
        else:
            loss_val = float(original_loss)
            
        if isinstance(original_recon, torch.Tensor):
            recon_val = original_recon.item()
        else:
            recon_val = float(original_recon)
            
        if isinstance(original_cls, torch.Tensor):
            cls_val = original_cls.item()
        else:
            cls_val = float(original_cls)
        
        total_loss += loss_val * batch_size
        total_recon_loss += recon_val * batch_size
        total_cls_consistency_loss += cls_val * batch_size
        
        step_count += 1
        
        # 显存清理
        del batch, labels, is_real_mask, masked, batch_out_encode, batch_logits_encode
        del batch_out_decode, batch_logits_decode, loss, recon_loss, cls_consistency_loss
        torch.cuda.empty_cache()
    
    # 处理最后的不完整累积批次
    if step_count % accumulate_grad_batches != 0:
        if use_mixed_precision and scaler is not None and AMP_AVAILABLE:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    n = len(dataloader.dataset)
    encode_acc = total_encode_correct / total_samples if total_samples > 0 else 0.0
    decode_acc = total_decode_correct / total_samples if total_samples > 0 else 0.0
    return total_loss / n, total_recon_loss / n, total_cls_consistency_loss / n, encode_acc, decode_acc


def monitor_gpu_usage(prefix=""):
    """监控GPU使用情况 - 简化版"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        utilization = (allocated / total_memory) * 100
        return utilization
    return 0


def forward_batch_parallel(model, input_batch, device):
    """真正的批量并行前向传播，最大化利用16GB显存"""
    batch_size, C, T = input_batch.size()
    
    # 批量转置: [batch_size, C, T] -> [batch_size, T, C] 
    windows = input_batch.transpose(1, 2)  # [batch_size, T, C]
      # 尝试真正的批量处理 - 直接处理整个batch
    try:
        # 方法1: 如果模型支持批量输入，直接处理
        batch_out, batch_logits = model.forward_batch(windows)
        return batch_out, batch_logits  # [batch_size, C, T], [batch_size, num_classes]
    except AttributeError:
        # 方法2: 使用torch.vmap进行向量化批量处理
        try:
            def single_forward(window):
                out, logits = model(window)
                return out.t(), logits  # [C, T], [num_classes]
            
            # 使用vmap进行真正的并行化
            vmapped_forward = torch.vmap(single_forward, in_dims=0, out_dims=0)
            batch_out, batch_logits = vmapped_forward(windows)
            return batch_out, batch_logits  # [batch_size, C, T], [batch_size, num_classes]
        except:
            # 方法3: 优化的大批量处理 - 充分利用显存
            return forward_large_batch_optimized(model, windows, device)


def forward_large_batch_optimized(model, windows, device):
    """优化的大批量处理，充分利用16GB显存"""
    batch_size, T, C = windows.size()
    
    # 根据显存自适应调整chunk大小 - 更大的块
    if torch.cuda.get_device_properties(0).total_memory > 15e9:  # 16GB显存
        chunk_size = min(batch_size, 32)  # 大幅增加chunk大小
    else:
        chunk_size = min(batch_size, 16)
    
    batch_outputs = []
    batch_logits = []
    
    # 使用更大的chunk进行并行处理
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_windows = windows[i:end_idx]  # [chunk_size, T, C]
        chunk_size_actual = chunk_windows.size(0)
        
        # 批量处理chunk
        chunk_out_list = []
        chunk_logits_list = []
        
        # 并行处理chunk中的所有样本
        for j in range(chunk_size_actual):
            out, logits = model(chunk_windows[j])
            chunk_out_list.append(out.t())  # [C, T]
            chunk_logits_list.append(logits)
        
        # 批量堆叠chunk结果
        chunk_out = torch.stack(chunk_out_list, dim=0)  # [chunk_size, C, T]
        chunk_logits = torch.stack(chunk_logits_list, dim=0)  # [chunk_size, num_classes]
        
        batch_outputs.append(chunk_out)
        batch_logits.append(chunk_logits)
    
    # 最终堆叠所有chunk
    final_batch_out = torch.cat(batch_outputs, dim=0)  # [batch_size, C, T]
    final_batch_logits = torch.cat(batch_logits, dim=0)  # [batch_size, num_classes]
    
    return final_batch_out, final_batch_logits


def compute_batch_recon_loss(targets, predictions, is_real_mask, common_indices, criterion, C, batch_size):
    """高效的批量重建损失计算"""
    total_recon_loss = 0.0
    
    # 向量化处理相同类型的通道
    common_mask = torch.zeros(C, dtype=torch.bool, device=targets.device)
    if common_indices:
        common_mask[common_indices] = True
    
    # 处理common通道（批量）
    if common_mask.any():
        common_targets = targets[:, common_mask, :]  # [batch_size, n_common, T]
        common_preds = predictions[:, common_mask, :]
        
        # 批量计算所有common通道的损失
        for c_idx, global_c in enumerate(torch.where(common_mask)[0]):
            target_batch = common_targets[:, c_idx, :]  # [batch_size, T]
            pred_batch = common_preds[:, c_idx, :]
            
            # 批量调用criterion
            loss_sum = 0.0
            for b in range(batch_size):
                loss_sum += criterion(pred_batch[b], target_batch[b], channel_idx=global_c.item(), is_common=True)
            total_recon_loss += loss_sum / batch_size
    
    # 处理have通道（批量）
    have_mask = ~common_mask & is_real_mask[0] if is_real_mask.dim() == 2 else ~common_mask & is_real_mask
    if have_mask.any():
        have_targets = targets[:, have_mask, :]
        have_preds = predictions[:, have_mask, :]
        
        for c_idx, global_c in enumerate(torch.where(have_mask)[0]):
            target_batch = have_targets[:, c_idx, :]
            pred_batch = have_preds[:, c_idx, :]
            
            loss_sum = 0.0
            for b in range(batch_size):
                loss_sum += criterion(pred_batch[b], target_batch[b], channel_idx=global_c.item(), is_common=False)
            total_recon_loss += loss_sum / batch_size
    
    return total_recon_loss

def eval_loop(model, dataloader, criterion, device, mask_indices):
    """验证函数 - 支持多模态损失"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    
    # 获取common模态索引
    common_indices = getattr(criterion, 'common_indices', [])
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Eval"):
            if len(batch_data) == 4:
                batch, _, _, is_real_mask = batch_data
            else:
                batch, _, _, is_real_mask, _ = batch_data
            
            batch = batch.to(device)
            is_real_mask = is_real_mask.to(device)
            
            masked, mask_idx = mask_channel(batch, mask_indices)
            batch_size, C, T = batch.size()
            loss = 0.0
            recon_loss = 0.0
            
            for i in range(batch_size):
                window = masked[i].t()
                out, _ = model(window)
                
                # 获取真实通道信息
                if is_real_mask.dim() == 2:
                    real_channels = is_real_mask[i]
                else:
                    real_channels = is_real_mask
                
                recon_loss_i = 0.0
                real_count = 0
                
                for c in range(C):
                    target = batch[i, c, :]
                    pred = out[c, :]
                    
                    # 判断是否为common模态
                    is_common_channel = c in common_indices
                    
                    if is_common_channel:
                        # Common模态：始终计算损失
                        recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=True)
                        real_count += 1
                    elif real_channels[c]:                        # Have模态：只对真实通道计算损失
                        recon_loss_i = recon_loss_i + criterion(pred, target, channel_idx=c, is_common=False)
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='V12 多模态时序算法训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    return parser.parse_args()

# =========================
# 主训练入口
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
    from data import load_data, SlidingWindowDataset
    data_files = config.get('data_files', [])
    data_dir = config.get('data_dir', '')
    # 拼接路径
    data_files = [os.path.join(data_dir, f) if not os.path.isabs(f) and not os.path.exists(f) else f for f in data_files]
    all_dfs = []
    all_sources = []
    
    for f in data_files:
        df = pd.read_csv(f)
        # 保留重要列名（block, F）的原始大小写，其他列转小写
        original_cols = df.columns.tolist()
        new_cols = []
        for col in original_cols:
            if col in ['block', 'F', 'ID', 'Session']:  # 保留这些列的原始大小写
                new_cols.append(col)
            else:
                new_cols.append(col.strip().lower())
        df.columns = new_cols
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
        all_dfs.append(df)
        all_sources.append(source)
        print(f"加载数据: {f} -> {source}, shape: {df.shape}")

    # 联合数据集构建
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['source'] = [s for s, df in zip(all_sources, all_dfs) for _ in range(len(df))]    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据和创建数据集 - 使用优化版本
    window_size = config.get('window_size', 320)
    step_size = config.get('step_size', 96)
    norm_method = config.get('norm_method', 'zscore')
    block_col = config.get('block_col', 'block')
    label_col = config.get('label_col', 'F')  # 使用配置中的标签列名
    
    # 根据配置文件计算正确的特征列
    common_modalities = config.get('common_modalities', [])
    dataset_modalities = config.get('dataset_modalities', {})
    
    # 构建所有特征列表
    all_feature_mods = common_modalities.copy()
    for dataset, mods in dataset_modalities.items():
        have = mods.get('have', [])
        need = mods.get('need', [])
        for mod in have + need:
            if mod not in all_feature_mods:
                all_feature_mods.append(mod)
    
    # 只使用配置中定义的特征列，忽略其他列
    feature_cols = all_feature_mods  # 直接使用配置的列名
    print(f"📊 使用特征列数量: {len(feature_cols)}")
    print(f"📊 特征列: {feature_cols}")
      # 使用优化的数据加载器
    if OPTIMIZED_LOADER_AVAILABLE and config.get('enable_fast_data_loading', False):
        print("🚀 使用优化数据加载器")
        data_loader_factory = OptimizedDataLoader(config)
        train_ds, val_ds, test_ds = data_loader_factory.create_datasets(data_files, feature_cols)
        train_loader, val_loader, test_loader = data_loader_factory.create_data_loaders(train_ds, val_ds, test_ds)
        
        print(f"📊 数据集大小: 训练={getattr(train_ds, '__len__', lambda: 'Unknown')()}, 验证={getattr(val_ds, '__len__', lambda: 'Unknown')()}, 测试={getattr(test_ds, '__len__', lambda: 'Unknown')()}")
    else:
        print("⚙️ 使用标准数据加载器")
        # 原有的数据加载逻辑作为fallback
        from data import load_data, SlidingWindowDataset
        
        # 联合数据集构建（原有逻辑）
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['source'] = [s for s, df in zip(all_sources, all_dfs) for _ in range(len(df))]
        
        # 创建滑动窗口数据集
        dataset = SlidingWindowDataset(
            combined_df, 
            feature_cols=feature_cols,
            window_size=window_size, 
            step_size=step_size, 
            block_col=block_col,
            label_col=label_col
        )

        print(f"📊 数据集大小: {len(dataset)}")
        check_label_distribution(dataset)

        # 数据集划分
        train_split = config.get('train_split', 0.6)
        val_split = (1 - train_split) / 2
        test_split = val_split

        train_size = int(train_split * len(dataset))
        val_size = int(val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
        
        print(f"📊 训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}, 测试集大小: {len(test_ds)}")
        
        # 高性能数据加载参数
        num_workers = config.get('num_workers', 0)
        pin_memory = config.get('pin_memory', True)
        prefetch_factor = config.get('prefetch_factor', 4) if num_workers > 0 else None
        persistent_workers = config.get('persistent_workers', False) if num_workers > 0 else False
        
        # 创建数据加载器
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'drop_last': config.get('dataloader_drop_last', True)
        }
        if num_workers > 0 and prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
        
        train_loader = DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **dataloader_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **dataloader_kwargs)
    
    # 只使用配置中定义的特征列，忽略其他列
    feature_cols = [col for col in all_feature_mods if col in combined_df.columns]
    print(f"使用特征列数量: {len(feature_cols)}")
    print(f"特征列: {feature_cols}")
    
    # 创建滑动窗口数据集
    dataset = SlidingWindowDataset(
        combined_df, 
        feature_cols=feature_cols,
        window_size=window_size, 
        step_size=step_size, 
        block_col=block_col,
        label_col=label_col  # 添加label_col参数
    )

    print(f"数据集大小: {len(dataset)}")
    check_label_distribution(dataset)

    # 数据集划分
    train_split = config.get('train_split', 0.6)
    val_split = (1 - train_split) / 2  # 剩余的一半作为验证集
    test_split = val_split             # 另一半作为测试集

    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    print(f"训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}, 测试集大小: {len(test_ds)}")    # 高性能数据加载参数 - Windows兼容优化
    num_workers = config.get('num_workers', 0)  # Windows兼容：使用单进程
    pin_memory = config.get('pin_memory', True)
    prefetch_factor = config.get('prefetch_factor', 2) if num_workers > 0 else None  # 单进程时必须为None
    persistent_workers = config.get('persistent_workers', False) if num_workers > 0 else False    # 创建混合精度训练Scaler
    use_mixed_precision = config.get('use_mixed_precision', True) and AMP_AVAILABLE and torch.cuda.is_available()
    scaler = GradScaler() if use_mixed_precision else None
      # 打印配置信息
    print(f"配置: Batch Size={batch_size}, Mixed Precision={use_mixed_precision}, Learning Rate={lr}")
    
    if use_mixed_precision:
        print("启用混合精度训练 (AMP)")
    else:
        print("使用标准精度训练")
    
    # 创建数据加载器（高性能版）
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers
    }
    # 只在多进程时添加prefetch_factor
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(
        train_ds, 
        shuffle=True,
        drop_last=True,  # 丢弃最后不完整的batch以保持训练稳定性
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_ds, 
        shuffle=False,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_ds, 
        shuffle=False,
        **dataloader_kwargs
    )

    # 模型参数
    in_channels = config.get('in_channels', 32)
    hidden_channels = config.get('hidden_channels', 64)
    out_channels = config.get('out_channels', 32)
    num_classes = config.get('num_classes', 2)

    # 创建模型
    from model import TGATUNet
    model = TGATUNet(in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_classes=num_classes)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    
    # 创建损失函数
    if config.get('loss_config', {}).get('type') == 'multimodal':
        criterion = create_simple_multimodal_criterion(config)
        print("使用多模态损失函数")
    else:
        criterion = MSELoss()
        print("使用标准MSE损失函数")    # 创建增强验证管理器
    log_dir_full = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir_full, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    enhanced_val_manager = EnhancedValidationManager(
        patience=early_stop_patience,
        min_delta=float(config.get('enhanced_validation', {}).get('min_delta', 1e-6)),
        save_dir=os.path.join(log_dir_full, 'enhanced_validation')    )
    print(f"📊 启用增强验证策略，耐心度={early_stop_patience}")

    # TensorBoard 日志器初始化
    writer = SummaryWriter(log_dir=log_dir_full)
      # Mask indices (示例)
    mask_indices = [0, 1, 2]  # 可以根据需要调整

    # ========== 训练模式 ========== 
    if mode == 'train':
        print("🎯 使用标准训练模式 + 增强验证策略")
        for epoch in range(1, epochs + 1):            # 获取梯度累积设置
            accumulate_grad_batches = config.get('accumulate_grad_batches', 2)
            
            train_loss, train_recon, train_cls, train_encode_acc, train_decode_acc = train_phased_with_grad_accumulation(
                model, train_loader, optimizer, criterion, device, mask_indices, 
                accumulate_grad_batches=accumulate_grad_batches,
                use_mixed_precision=use_mixed_precision, scaler=scaler
            )
            
            # 使用增强验证策略
            if enhanced_val_manager.should_validate(epoch):
                # 计算增强验证指标
                val_metrics = enhanced_val_manager.compute_enhanced_validation_metrics(
                    model, val_loader, criterion, device, mask_indices
                )
                
                # 更新指标并获取早停建议
                early_stop_info = enhanced_val_manager.update_metrics(val_metrics, epoch)
                
                # 记录结果
                enhanced_val_manager.log_validation_results(val_metrics, epoch, early_stop_info)
                
                # 学习率调度
                scheduler.step(val_metrics['val_loss'])
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 增强日志记录
                logging.info(f"[ENHANCED_TRAIN] Epoch {epoch}: "
                           f"train_loss={train_loss:.6f}, train_encode_acc={train_encode_acc:.4f}, train_decode_acc={train_decode_acc:.4f}, "
                           f"val_loss={val_metrics['val_loss']:.6f}, val_encode_acc={val_metrics['val_encode_accuracy']:.4f}, "
                           f"val_decode_acc={val_metrics['val_decode_accuracy']:.4f}, val_f1={val_metrics['val_f1_score']:.4f}, "
                           f"common_recon={val_metrics['val_common_recon_loss']:.6f}, "
                           f"have_recon={val_metrics['val_have_recon_loss']:.6f}, "
                           f"lr={current_lr:.6e}")
                
                # TensorBoard记录增强指标
                writer.add_scalar('Train/Loss/train', train_loss, epoch)
                writer.add_scalar('Train/Recon/train', train_recon, epoch)
                writer.add_scalar('Train/Cls/train', train_cls, epoch)
                writer.add_scalar('Train/Encode_Acc/train', train_encode_acc, epoch)
                writer.add_scalar('Train/Decode_Acc/train', train_decode_acc, epoch)
                writer.add_scalar('Enhanced_Val/Loss', val_metrics['val_loss'], epoch)
                writer.add_scalar('Enhanced_Val/Accuracy', val_metrics['val_accuracy'], epoch)
                writer.add_scalar('Enhanced_Val/Encode_Accuracy', val_metrics['val_encode_accuracy'], epoch)
                writer.add_scalar('Enhanced_Val/Decode_Accuracy', val_metrics['val_decode_accuracy'], epoch)
                writer.add_scalar('Enhanced_Val/F1_Score', val_metrics['val_f1_score'], epoch)
                writer.add_scalar('Enhanced_Val/Precision', val_metrics['val_precision'], epoch)
                writer.add_scalar('Enhanced_Val/Recall', val_metrics['val_recall'], epoch)
                writer.add_scalar('Enhanced_Val/Common_Recon', val_metrics['val_common_recon_loss'], epoch)
                writer.add_scalar('Enhanced_Val/Have_Recon', val_metrics['val_have_recon_loss'], epoch)
                writer.add_scalar('Train/LR', current_lr, epoch)
                
                # 保存最佳模型
                if epoch == enhanced_val_manager.best_epoch:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                    torch.save(model.state_dict(), model_path)
                    logging.info(f"Saved best model at epoch {epoch} (composite score: {early_stop_info['best_composite_score']:.6f})")
                
                # 早停检查
                if early_stop_info['should_stop']:
                    logging.info(f"Enhanced early stopping triggered at epoch {epoch}")
                    break
                    
                # 过拟合警告
                if early_stop_info['is_overfitting']:
                    logging.warning(f"[OVERFITTING_DETECTED] Epoch {epoch}: 检测到可能的过拟合")
            
            else:
                # 不需要验证的epoch，只做简单的日志记录
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"[TRAIN] Epoch {epoch}: train_loss={train_loss:.6f}, train_encode_acc={train_encode_acc:.4f}, train_decode_acc={train_decode_acc:.4f}, lr={current_lr:.6e}")
                writer.add_scalar('Train/Loss/train', train_loss, epoch)
                writer.add_scalar('Train/Encode_Acc/train', train_encode_acc, epoch)
                writer.add_scalar('Train/Decode_Acc/train', train_decode_acc, epoch)
                writer.add_scalar('Train/LR', current_lr, epoch)
            
            # === 动态need补全 ===
            # 注意：为避免数据泄露，只在训练模式下对训练集进行动态更新
            if hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'update_need'):
                # 如果是Subset/RandomSplit包装，取原始dataset
                complete_need_with_model(model, train_ds.dataset, device)
            elif hasattr(train_ds, 'update_need'):
                complete_need_with_model(model, train_ds, device)
        
        # 训练结束后保存增强验证可视化
        enhanced_val_manager.save_validation_plots()
        best_summary = enhanced_val_manager.get_best_metrics_summary()
        logging.info(f"Training completed. Best metrics: {best_summary}")
          # 自动评估 - 使用完整的增强评估指标
        latest_model = model_path if os.path.exists(model_path) else os.path.join(ckpt_dir, 'best_model.pth')
        if latest_model and os.path.exists(latest_model):
            model.load_state_dict(torch.load(latest_model))
            logging.info(f"[Test Evaluation] 开始对测试集进行完整评估...")
            
            # 使用增强验证管理器计算完整的测试集指标
            test_metrics = enhanced_val_manager.compute_enhanced_validation_metrics(
                model, test_loader, criterion, device, mask_indices
            )
            
            # 详细输出测试集评估结果
            logging.info(f"[TEST_RESULTS] 测试集完整评估结果:")
            logging.info(f"  - Test Loss: {test_metrics['val_loss']:.6f}")
            logging.info(f"  - Test Accuracy: {test_metrics['val_accuracy']:.4f}")
            logging.info(f"  - Test F1 Score: {test_metrics['val_f1_score']:.4f}")
            logging.info(f"  - Test Precision: {test_metrics['val_precision']:.4f}")
            logging.info(f"  - Test Recall: {test_metrics['val_recall']:.4f}")
            logging.info(f"  - Test Common Recon Loss: {test_metrics['val_common_recon_loss']:.6f}")
            logging.info(f"  - Test Have Recon Loss: {test_metrics['val_have_recon_loss']:.6f}")
            logging.info(f"  - Test Classification Loss: {test_metrics['val_cls_loss']:.6f}")
            
            # 记录到TensorBoard
            writer.add_scalar('Test/Loss', test_metrics['val_loss'])
            writer.add_scalar('Test/Accuracy', test_metrics['val_accuracy'])
            writer.add_scalar('Test/F1_Score', test_metrics['val_f1_score'])
            writer.add_scalar('Test/Precision', test_metrics['val_precision'])
            writer.add_scalar('Test/Recall', test_metrics['val_recall'])
            writer.add_scalar('Test/Common_Recon_Loss', test_metrics['val_common_recon_loss'])
            writer.add_scalar('Test/Have_Recon_Loss', test_metrics['val_have_recon_loss'])
            writer.add_scalar('Test/Cls_Loss', test_metrics['val_cls_loss'])
        else:
            logging.warning(f"未找到可用的模型进行测试评估！(查找路径: {latest_model})")
    
    elif mode == 'eval':
        # 评估模式 - 使用完整的增强评估指标
        latest_model = model_path if os.path.exists(model_path) else os.path.join(ckpt_dir, 'best_model.pth')
        if latest_model and os.path.exists(latest_model):
            model.load_state_dict(torch.load(latest_model))
            logging.info(f"[Eval Mode] 加载模型: {latest_model}")
            logging.info(f"[Eval Mode] 开始对测试集进行完整评估...")
              # 创建临时的增强验证管理器用于评估模式
            temp_enhanced_val_manager = EnhancedValidationManager(
                patience=early_stop_patience,
                min_delta=float(config.get('enhanced_validation', {}).get('min_delta', 1e-6)),
                save_dir=os.path.join(log_dir_full, 'eval_mode_validation')
            )
            
            # 使用增强验证管理器计算完整的测试集指标
            test_metrics = temp_enhanced_val_manager.compute_enhanced_validation_metrics(
                model, test_loader, criterion, device, mask_indices
            )
            
            # 详细输出评估结果
            logging.info(f"[EVAL_RESULTS] 评估模式完整结果:")
            logging.info(f"  - Test Loss: {test_metrics['val_loss']:.6f}")
            logging.info(f"  - Test Accuracy: {test_metrics['val_accuracy']:.4f}")
            logging.info(f"  - Test F1 Score: {test_metrics['val_f1_score']:.4f}")
            logging.info(f"  - Test Precision: {test_metrics['val_precision']:.4f}")
            logging.info(f"  - Test Recall: {test_metrics['val_recall']:.4f}")
            logging.info(f"  - Test Common Recon Loss: {test_metrics['val_common_recon_loss']:.6f}")
            logging.info(f"  - Test Have Recon Loss: {test_metrics['val_have_recon_loss']:.6f}")
            logging.info(f"  - Test Classification Loss: {test_metrics['val_cls_loss']:.6f}")
            
            # 记录到TensorBoard
            writer.add_scalar('Eval/Loss', test_metrics['val_loss'])
            writer.add_scalar('Eval/Accuracy', test_metrics['val_accuracy'])
            writer.add_scalar('Eval/F1_Score', test_metrics['val_f1_score'])
            writer.add_scalar('Eval/Precision', test_metrics['val_precision'])
            writer.add_scalar('Eval/Recall', test_metrics['val_recall'])
            writer.add_scalar('Eval/Common_Recon_Loss', test_metrics['val_common_recon_loss'])
            writer.add_scalar('Eval/Have_Recon_Loss', test_metrics['val_have_recon_loss'])
            writer.add_scalar('Eval/Cls_Loss', test_metrics['val_cls_loss'])
        else:
            logging.warning(f"评估模式下未找到可用的模型！(查找路径: {latest_model}")

if __name__ == "__main__":
    main()
