# src/train.py

"""
Updated train.py for multi-layer GSL feedback.
- Compatible with TGATUNet using MultiLayerGraphLearner
- Feedback loss computed per-layer using final GATEncoder layer's attention
"""
import os
import argparse
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

from torch.utils.tensorboard import SummaryWriter

from data import create_dataset_from_config
from model import TGATUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train T-GAT-UNet with GSL")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save logs and models")
    parser.add_argument("--use_feedback", action="store_true", help="Enable attention-to-GSL structure loss")
    return parser.parse_args()


# In src/train.py

def mask_channel(x, mask_ratio=0.2):
    """
    Masks a random channel for each sample in a batch. Vectorized version.
    
    Args:
        x (torch.Tensor): Input batch of shape [batch, channels, timesteps].
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The masked batch and the indices of masked channels.
    """
    batch, C, T = x.size()
    
    # Randomly select one channel index per sample in the batch
    mask_idx = torch.randint(0, C, (batch,), device=x.device)
    
    x_masked = x.clone()
    
    # Use advanced indexing to mask the channels for all samples at once
    # Create an array [0, 1, 2, ...] to index the batch dimension
    batch_indices = torch.arange(batch, device=x.device)
    x_masked[batch_indices, mask_idx, :] = 0
    
    return x_masked, mask_idx


def train_loop(model, dataloader, optimizer, criterion, device, use_feedback=False):
    model.train()
    total_loss, total_struct_loss = 0.0, 0.0

    for batch in dataloader:
        batch = batch.to(device)
        masked, mask_idx = mask_channel(batch)
        batch_size = batch.size(0)

        loss, struct_loss = 0.0, 0.0
        optimizer.zero_grad()

        for i in range(batch_size):
            window = masked[i].t()
            if use_feedback:
                out, attn_list, adjs = model(window, return_attention=True)
                last_attn = attn_list[-1]  # last GATEncoder layer attention
                if isinstance(last_attn, tuple):
                    edge_index, attn_weights = last_attn
                    if attn_weights.dim() > 1:
                        attn_mean = attn_weights.mean(dim=1)
                    else:
                        attn_mean = attn_weights
                    adj_last = adjs[len(attn_list) - 1]  # match to last encoder layer
                    adj_values = adj_last[edge_index[0], edge_index[1]]
                    min_len = min(attn_mean.shape[0], adj_values.shape[0])
                    struct_loss += MSELoss()(adj_values[:min_len], attn_mean[:min_len].detach())
            else:
                out = model(window)

            target = batch[i, mask_idx[i], :]
            pred = out[mask_idx[i], :]
            loss += criterion(pred, target)

        loss = loss / batch_size
        if use_feedback:
            struct_loss = struct_loss / batch_size
            loss += struct_loss
            total_struct_loss += struct_loss.item() * batch_size

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_size

    return total_loss / len(dataloader.dataset), total_struct_loss / len(dataloader.dataset) if use_feedback else 0.0


def eval_loop(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            masked, mask_idx = mask_channel(batch)
            batch_size = batch.size(0)
            loss = 0.0
            for i in range(batch_size):
                window = masked[i].t()
                out = model(window)
                target = batch[i, mask_idx[i], :]
                pred = out[mask_idx[i], :]
                loss += criterion(pred, target)
            total_loss += (loss / batch_size).item() * batch_size
    return total_loss / len(dataloader.dataset)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    writer = SummaryWriter(log_dir=args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = TGATUNet(in_channels, hidden_channels=64, out_channels=in_channels,
                     use_skip=True, use_attention_feedback=args.use_feedback)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = MSELoss()

    best_val_loss, epochs_no_improve = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        train_loss, struct_loss = train_loop(model, train_loader, optimizer, criterion, device, args.use_feedback)
        val_loss = eval_loop(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logging.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, struct_loss={struct_loss:.6f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        if args.use_feedback:
            writer.add_scalar('Loss/structure', struct_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 10:
            logging.info("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss = eval_loop(model, test_loader, criterion, device)
    logging.info(f"Test Loss: {test_loss:.6f}")
    writer.add_scalar('Loss/test', test_loss)
    writer.close()


if __name__ == '__main__':
    main()
