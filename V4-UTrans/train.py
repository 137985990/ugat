# src/train.py

"""
train.py

Module to train the T-GAT-UNet model on sliding-window time series dataset with self-supervised masking.
"""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train T-GAT-UNet on time series data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    return args


def mask_channel(x, mask_ratio=0.2):
    """
    Randomly mask one channel per sample for self-supervised training.

    Args:
        x: Tensor of shape [batch, channels, time]
        mask_ratio: Fraction of channels to mask (default 0.2)
    Returns:
        x_masked, mask_idx
    """
    batch, C, T = x.size()
    num_masks = max(1, int(C * mask_ratio))
    # For simplicity, mask exactly one channel
    mask_idx = torch.randint(0, C, (batch,))
    x_masked = x.clone()
    for i in range(batch):
        x_masked[i, mask_idx[i], :] = 0
    return x_masked, mask_idx


def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    ce_loss = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch, labels in tqdm(dataloader, desc="Train"):
        # batch: [batch, C, T], labels: [batch]
        batch = batch.to(device)
        labels = labels.to(device)
        masked, mask_idx = mask_channel(batch)
        batch_size, C, T = batch.size()

        loss = 0.0
        cls_loss = 0.0
        recon_loss = 0.0
        optimizer.zero_grad()
        for i in range(batch_size):
            window = masked[i].t()  # [T, C]
            out, logits = model(window)  # out: [C, T], logits: [num_classes]
            # Only compute loss on masked channel
            target = batch[i, mask_idx[i], :]
            pred = out[mask_idx[i], :]
            recon_loss_i = criterion(pred, target)
            cls_loss_i = ce_loss(logits.unsqueeze(0), labels[i].unsqueeze(0))
            loss += recon_loss_i + cls_loss_i
            recon_loss += recon_loss_i.item()
            cls_loss += cls_loss_i.item()
            # 分类准确率统计
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


def eval_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc="Eval"):
            batch = batch.to(device)
            masked, mask_idx = mask_channel(batch)
            batch_size, C, T = batch.size()
            loss = 0.0
            recon_loss = 0.0
            for i in range(batch_size):
                window = masked[i].t()
                out, _ = model(window)
                target = batch[i, mask_idx[i], :]
                pred = out[mask_idx[i], :]
                recon_loss_i = criterion(pred, target)
                loss += recon_loss_i
                recon_loss += recon_loss_i.item()
            loss = loss / batch_size
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss
    n = len(dataloader.dataset)
    return total_loss / n, total_recon_loss / n, 0.0, 0.0


def main():

    args = parse_args()
    # Load config.yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Get parameters from config
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 100)
    lr = config.get('lr', 1e-3)
    early_stop_patience = config.get('patience', 10)
    log_dir = config.get('log_dir', 'Logs')
    ckpt_dir = config.get('ckpt_dir', 'Checkpoints')
    mode = config.get('mode', 'train').lower()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Logging
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Dataset
    dataset = create_dataset_from_config(args.config)
    total = len(dataset)
    train_split = config.get('train_split', 0.6)
    val_split = config.get('val_split', 0.2)
    # 严格按比例分配，保证三者之和等于total
    train_len = round(train_split * total)
    val_len = round(val_split * total)
    test_len = total - train_len - val_len
    # 若因四舍五入导致test为负或0，优先补到test
    if test_len < 0:
        test_len = 0
    # 若有误差，补到test
    while train_len + val_len + test_len < total:
        test_len += 1
    while train_len + val_len + test_len > total:
        if test_len > 0:
            test_len -= 1
        elif val_len > 0:
            val_len -= 1
        else:
            train_len -= 1
    assert train_len + val_len + test_len == total, f"Split error: {train_len}+{val_len}+{test_len}!={total}"
    print(f"[数据集分配] 总数: {total}, 训练: {train_len}, 验证: {val_len}, 测试: {test_len}")
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Model, optimizer, scheduler
    in_channels = dataset[0][0].shape[0]  # dataset returns (window, label)
    out_channels = in_channels
    model = TGATUNet(in_channels, hidden_channels=64, out_channels=out_channels, num_classes=2)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    criterion = MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0


    # 选择是否跳过训练直接读取最新模型
    import glob
    import sys
    latest_model = None
    model_files = sorted(glob.glob(os.path.join(ckpt_dir, 'best_model.pth')), reverse=True)
    if model_files:
        latest_model = model_files[0]

    if mode == 'train':
        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss, train_recon, train_cls, train_acc = train_loop(model, train_loader, optimizer, criterion, device)
            val_loss, val_recon, _, _ = eval_loop(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            logging.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, train_recon={train_recon:.6f}, val_recon={val_recon:.6f}, train_cls={train_cls:.6f}, train_acc={train_acc:.4f}")
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Recon/train', train_recon, epoch)
            writer.add_scalar('Recon/val', val_recon, epoch)
            writer.add_scalar('Cls/train', train_cls, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                logging.info(f"Saved best model at epoch {epoch}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break
        latest_model = os.path.join(ckpt_dir, 'best_model.pth')

        # 自动评估
        if latest_model and os.path.exists(latest_model):
            model.load_state_dict(torch.load(latest_model))
            test_loss, test_recon, _, _ = eval_loop(model, test_loader, criterion, device)
            logging.info(f"[Train End] Test Loss: {test_loss:.6f}, Recon Loss: {test_recon:.6f}")
            writer.add_scalar('Loss/test', test_loss)
            writer.add_scalar('Recon/test', test_recon)
        else:
            logging.warning("未找到可用的模型进行测试评估！")

    elif mode == 'eval':
        # 只做评估
        if latest_model and os.path.exists(latest_model):
            model.load_state_dict(torch.load(latest_model))
            test_loss, test_recon, _, _ = eval_loop(model, test_loader, criterion, device)
            logging.info(f"[Eval Mode] Test Loss: {test_loss:.6f}, Recon Loss: {test_recon:.6f}")
            writer.add_scalar('Loss/test', test_loss)
            writer.add_scalar('Recon/test', test_recon)
        else:
            logging.warning("未找到可用的模型进行测试评估！")

    else:
        logging.error(f"未知模式: {mode}，请在config.yaml中设置 mode: train 或 mode: eval")

    writer.close()

if __name__ == '__main__':
    main()
