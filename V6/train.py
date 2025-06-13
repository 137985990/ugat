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
    orig_df.columns = [c.strip().lower() for c in orig_df.columns]
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
    # 每个样本只在mask_indices中随机选一个通道遮掩
    if len(mask_indices) == 0:
        # 没有需要补全的模态，直接返回原数据
        return x, torch.tensor([-1]*batch)
    mask_idx = torch.tensor(random.choices(mask_indices, k=batch))
    x_masked = x.clone()
    for i in range(batch):
        x_masked[i, mask_idx[i], :] = 0
    return x_masked, mask_idx



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
    for batch, labels, mask_idx, is_real_mask in tqdm(dataloader, desc="Train"):
        batch = batch.to(device)
        labels = labels.to(device)
        is_real_mask = is_real_mask.to(device)
        masked, mask_idx = mask_channel(batch, mask_indices)
        batch_size, C, T = batch.size()
        loss = 0.0
        cls_loss = 0.0
        recon_loss = 0.0
        optimizer.zero_grad()
        for i in range(batch_size):
            window = masked[i].t()  # [T, C]
            out, logits = model(window)  # out: [C, T], logits: [num_classes]
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
    dataset_name = config.get('dataset', None)
    if dataset_name is None:
        # 优先从 data_files 路径推断
        data_files = config.get('data_files', [])
        found = None
        for f in data_files:
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
    if dataset_name is None:
        # 兜底：再尝试 config 文件名
        cfg_base = os.path.basename(args.config).lower()
        if 'fm' in cfg_base:
            dataset_name = 'FM'
        elif 'od' in cfg_base:
            dataset_name = 'OD'
        elif 'mefar' in cfg_base:
            dataset_name = 'MEFAR'
    if dataset_name is None:
        raise ValueError('无法自动推断主数据集类型（FM/OD/MEFAR），请在config.yaml中添加dataset字段')
    have_list = common_modalities + dataset_modalities_cfg.get(dataset_name, {}).get('have', [])
    need_list = dataset_modalities_cfg.get(dataset_name, {}).get('need', [])
    modalities = have_list + need_list
    in_channels = config.get('in_channels', len(modalities))
    hidden_channels = config.get('hidden_channels', 64)
    out_channels = config.get('out_channels', in_channels)
    num_classes = config.get('num_classes', 2)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 只加载当前主数据集的数据文件，保证验证/测试集不混合其它数据集
    # 推断主数据集名
    dataset_name = config.get('dataset', None)
    if dataset_name is None:
        data_files = config.get('data_files', [])
        found = None
        for f in data_files:
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
    # 只选当前主数据集的数据文件
    data_files = config.get('data_files', [])
    main_file = None
    for f in data_files:
        if dataset_name and dataset_name.lower() in os.path.basename(f).lower():
            main_file = f
            break
    if main_file is None:
        raise ValueError(f"未找到主数据集 {dataset_name} 的数据文件，请检查 config.yaml 的 data_files 配置")
    # 构造只包含主数据集的数据文件的临时 config
    config_single = dict(config)
    config_single['data_files'] = [main_file]
    # 创建数据集
    import tempfile
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as tmpf:
        yaml.safe_dump(config_single, tmpf)
        tmp_cfg_path = tmpf.name
    dataset = create_dataset_from_config(tmp_cfg_path)
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
    print(f"[数据集分配] (仅{dataset_name}) 总数: {total}, 训练: {train_len}, 验证: {val_len}, 测试: {test_len}")
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    dataset_modalities = modalities
    mask_indices = [i for i, m in enumerate(dataset_modalities) if m in need_list]
    if len(mask_indices) == 0:
        logging.warning("当前数据集无需补全的模态，训练时不会遮掩任何通道。")

    # ========== 模型结构适配 ==========
    # 这里假设 TGATUNet 支持 encode/decode 阶段切换，后续可细化
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
            logging.info(f"[TRAIN] Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, train_recon={train_recon:.6f}, val_recon={val_recon:.6f}, train_cls={train_cls:.6f}, train_acc={train_acc:.4f}")
            writer.add_scalar('Train/Loss/train', train_loss, epoch)
            writer.add_scalar('Train/Loss/val', val_loss, epoch)
            writer.add_scalar('Train/Recon/train', train_recon, epoch)
            writer.add_scalar('Train/Recon/val', val_recon, epoch)
            writer.add_scalar('Train/Cls/train', train_cls, epoch)
            writer.add_scalar('Train/Acc/train', train_acc, epoch)
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
