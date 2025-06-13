# src/train.py

"""
train.py

Module to train the T-GAT-UNet model on sliding-window time series dataset with self-supervised masking.
"""
import os
import argparse
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
# GAN imports
from model import GANGenerator, GANDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description="Train T-GAT-UNet on time series data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save logs and models")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (overrides config.yaml if set)")
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
    # 对每个样本随机mask num_masks个通道
    mask_idx = []
    x_masked = x.clone()
    for i in range(batch):
        idx = torch.randperm(C)[:num_masks]
        mask_idx.append(idx)
        x_masked[i, idx, :] = 0
    return x_masked, mask_idx


def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train"):
        # batch: tensor of shape [batch, C, T]
        batch = batch.to(device)
        masked, mask_idx = mask_channel(batch)
        batch_size, C, T = batch.size()

        recon = []
        loss = 0.0
        optimizer.zero_grad()
        for i in range(batch_size):
            window = masked[i].t()  # [T, C]
            out = model(window)  # [C, T]
            # Only compute loss on masked channel
            target = batch[i, mask_idx[i], :]
            pred = out[mask_idx[i], :]
            loss += criterion(pred, target)
        loss = loss / batch_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_size
    return total_loss / len(dataloader.dataset)


def eval_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            batch = batch.to(device)
            masked, mask_idx = mask_channel(batch)
            batch_size, C, T = batch.size()
            loss = 0.0
            for i in range(batch_size):
                window = masked[i].t()
                out = model(window)
                target = batch[i, mask_idx[i], :]
                pred = out[mask_idx[i], :]
                loss += criterion(pred, target)
            loss = loss / batch_size
            total_loss += loss.item() * batch_size
    return total_loss / len(dataloader.dataset)




def train_gan(args, gan_type='gan'):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"train_gan_{gan_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    writer = SummaryWriter(log_dir=args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 用train_set+val_set做训练集，test_set做测试集，实现80/20划分
    train_set, val_set, test_set = create_dataset_from_config(args.config)
    train_data = train_set + val_set
    test_data = test_set
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    in_channels = train_data[0].shape[0]
    seq_len = train_data[0].shape[1]
    # 生成器现在是自编码器结构，输入输出都是[batch, channels, time]
    G = GANGenerator(in_channels, in_channels, seq_len).to(device)
    g_optimizer = Adam(G.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss(reduction='mean')

    import yaml
    best_g_loss = float('inf')
    # 优先命令行参数，其次config.yaml，否则默认10
    patience = args.patience
    if patience is None:
        # 从config.yaml读取
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        patience = cfg.get('patience', 10)
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        G.train()
        g_loss_epoch = 0.0
        for real in tqdm(train_loader, desc=f"Train AE"):
            real = real.to(device)
            batch_size = real.size(0)
            mask_ratio = 0.3
            real_masked, mask_idx = mask_channel(real, mask_ratio=mask_ratio)
            recon = G(real_masked)
            loss_recon = 0.0
            total_mask = 0
            for i in range(batch_size):
                for idx in mask_idx[i]:
                    loss_recon += mse_loss(recon[i, idx, :], real[i, idx, :])
                    total_mask += 1
            if total_mask > 0:
                loss_recon = loss_recon / total_mask
            g_optimizer.zero_grad()
            loss_recon.backward()
            g_optimizer.step()
            g_loss_epoch += loss_recon.item() * batch_size
        g_loss_epoch /= len(train_loader.dataset)
        logging.info(f"Epoch {epoch}: AE_loss={g_loss_epoch:.6f}")
        writer.add_scalar(f'{gan_type}/AE_loss', g_loss_epoch, epoch)
        # Save generator
        if g_loss_epoch < best_g_loss:
            best_g_loss = g_loss_epoch
            torch.save(G.state_dict(), os.path.join(args.output_dir, f'best_gan_{gan_type}.pth'))
            logging.info(f"Saved best generator at epoch {epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"No improvement in AE_loss. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch} due to no improvement in AE_loss for {patience} epochs.")
            break

    # ====== 测试集MSE评估 ======
    G.eval()
    mse_loss = torch.nn.MSELoss(reduction='mean')
    mse_list = []
    with torch.no_grad():
        for real in tqdm(test_loader, desc="Test MSE"):
            real = real.to(device)
            batch_size = real.size(0)
            # Mask test data in the same way as training
            mask_ratio = 0.3
            real_masked, mask_idx = mask_channel(real, mask_ratio=mask_ratio)
            recon = G(real_masked)
            # Compute MSE only on masked channels
            total_mse = 0.0
            total_mask = 0
            for i in range(batch_size):
                for idx in mask_idx[i]:
                    total_mse += mse_loss(recon[i, idx, :], real[i, idx, :]).item()
                    total_mask += 1
            mse = total_mse / total_mask if total_mask > 0 else 0.0
            mse_list.append(mse)
    mean_mse = sum(mse_list) / len(mse_list) if mse_list else 0.0
    logging.info(f"Test MSE: {mean_mse:.6f}")
    print(f"Test MSE: {mean_mse:.6f}")
    writer.add_scalar(f'{gan_type}/Test_MSE', mean_mse)
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    for gan_type in ['gan']:
        train_gan(args, gan_type)
