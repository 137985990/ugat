# src/train.py

"""
Updated train.py

Supports:
- TGATUNet with GSL integration
- Optional attention-to-GSL feedback loss
- Logging structure loss
"""

import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

# 修正clip_grad_norm_和SummaryWriter的导入
from torch.nn.utils.clip_grad import clip_grad_norm_
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # 如果没有tensorboardX或torch的SummaryWriter则为None

from data import create_dataset_from_config
from model import TGATUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train T-GAT-UNet on time series data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save logs and models")
    parser.add_argument("--use_feedback", action="store_true", help="Use GAT attention to guide GSL")
    args = parser.parse_args()
    return args


def mask_channel(x, mask_ratio=0.2):
    batch, C, T = x.size()
    mask_idx = torch.randint(0, C, (batch,), device=x.device)
    x_masked = x.clone()
    batch_indices = torch.arange(batch, device=x.device)
    x_masked[batch_indices, mask_idx, :] = 0
    return x_masked, mask_idx


def train_loop(model, dataloader, optimizer, criterion, device, use_feedback=False):
    model.train()
    total_loss = 0.0
    total_struct_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        masked, mask_idx = mask_channel(batch)
        cur_batch_size, C, T = batch.size()  # 兼容最后一个batch

        optimizer.zero_grad()
        # 向量化处理，提升速度
        # [B, C, T] -> [B, T, C]
        masked_t = masked.permute(0, 2, 1)  # [B, T, C]
        outputs = []
        attn_list_all = []
        adj_all = []
        if use_feedback:
            for i in range(cur_batch_size):
                out, attn_list, adj = model(masked_t[i], return_attention=True)
                outputs.append(out)
                attn_list_all.append(attn_list)
                adj_all.append(adj)
        else:
            for i in range(cur_batch_size):
                out = model(masked_t[i])
                outputs.append(out)

        # 堆叠输出 [B, C, T]
        outs = torch.stack(outputs, dim=0)
        # 取预测和目标
        batch_indices = torch.arange(cur_batch_size, device=batch.device)
        pred = outs[batch_indices, mask_idx, :]
        target = batch[batch_indices, mask_idx, :]
        loss = criterion(pred, target)

        struct_loss = 0.0
        if use_feedback:
            for i in range(cur_batch_size):
                attn_list = attn_list_all[i]
                adj = adj_all[i]
                edge_index = attn_list[-1][0]
                attn_weights = attn_list[-1][1]
                if attn_weights.dim() == 2:
                    attn_weights_mean = attn_weights.mean(dim=1)
                elif attn_weights.dim() == 3:
                    attn_weights_mean = attn_weights.mean(dim=(1,2))
                else:
                    attn_weights_mean = attn_weights.squeeze()
                adj_edges = adj[edge_index[0], edge_index[1]].squeeze()
                min_len = min(adj_edges.shape[0], attn_weights_mean.shape[0])
                struct_loss_sample = MSELoss()(adj_edges[:min_len].detach(), attn_weights_mean[:min_len].detach())
                struct_loss += struct_loss_sample.item()
            struct_loss = struct_loss / cur_batch_size
            loss = loss + struct_loss
            total_struct_loss += struct_loss * cur_batch_size

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * cur_batch_size

    return total_loss / len(dataloader.dataset), total_struct_loss / len(dataloader.dataset) if use_feedback else 0.0


def eval_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

    log_file = os.path.join(args.output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    writer = SummaryWriter(log_dir=args.output_dir) if SummaryWriter is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = create_dataset_from_config(args.config)
    total = len(dataset)
    train_len = int(0.6 * total)
    val_len = int(0.2 * total)
    test_len = total - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    in_channels = dataset[0].shape[0]
    out_channels = in_channels
    model = TGATUNet(in_channels, hidden_channels=64, out_channels=out_channels,
                     shared_graph=True, use_skip=True, use_attention_feedback=args.use_feedback)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    criterion = MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10


    for epoch in range(1, args.epochs + 1):
        train_loss, struct_loss = train_loop(model, train_loader, optimizer, criterion, device, use_feedback=args.use_feedback)
        val_loss = eval_loop(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, struct_loss={struct_loss:.6f}")
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            if args.use_feedback:
                writer.add_scalar('Loss/structure', struct_loss, epoch)

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


    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss = eval_loop(model, test_loader, criterion, device)
    logging.info(f"Test Loss: {test_loss:.6f}")
    if writer is not None:
        writer.add_scalar('Loss/test', test_loss)
        writer.close()


if __name__ == '__main__':
    main()
