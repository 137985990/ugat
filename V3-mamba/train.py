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
from model import TGATUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train T-GAT-UNet on time series data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save logs and models")
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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Logging
    log_file = os.path.join(args.output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.output_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Dataset
    dataset = create_dataset_from_config(args.config)
    total = len(dataset)
    train_len = int(0.6 * total)
    val_len = int(0.2 * total)
    test_len = total - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model, optimizer, scheduler
    in_channels = dataset[0].shape[0]
    out_channels = in_channels
    model = TGATUNet(in_channels, hidden_channels=64, out_channels=out_channels)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    criterion = MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(model, train_loader, optimizer, criterion, device)
        val_loss = eval_loop(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            logging.info(f"Saved best model at epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            logging.info("Early stopping triggered.")
            break

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss = eval_loop(model, test_loader, criterion, device)
    logging.info(f"Test Loss: {test_loss:.6f}")
    writer.add_scalar('Loss/test', test_loss)

    writer.close()

if __name__ == '__main__':
    main()
