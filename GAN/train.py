# src/train.py

"""
train.py

Module to train the GAN model with GAT-based generator and dual discriminators.
Implements adversarial training for time series completion.
"""
import os
import argparse
import logging
from datetime import datetime

import torch
from torch import nn
try:
    from torch.nn.utils.clip_grad import clip_grad_norm_
except ImportError:
    from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("Warning: TensorBoard not available")
        SummaryWriter = None

from data import create_dataset_from_config
# GAN imports
from model import GANGenerator, GANDiscriminator, ClassifierDiscriminator

class ListDataset(Dataset):
    """简单的数据集包装器，将list转换为Dataset兼容的对象"""
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


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
        clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    """
    训练完整的GAN模型，包括：
    1. GAT-based Generator (生成器)
    2. Primary GAN Discriminator (主判别器)
    3. Classifier Discriminator (分类判别器)
    """
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"train_gan_{gan_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    try:
        writer = SummaryWriter(log_dir=args.output_dir) if SummaryWriter else None
    except:
        writer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 数据加载
    train_set, val_set, test_set = create_dataset_from_config(args.config)
    
    # 合并训练和验证集（简单的list合并）
    train_data = train_set + val_set
    test_data = test_set
    
    # 使用包装器创建Dataset兼容对象
    train_dataset = ListDataset(train_data)
    test_dataset = ListDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 获取数据维度
    sample_data = train_data[0]
    if hasattr(sample_data, 'shape'):
        in_channels = sample_data.shape[0]
        seq_len = sample_data.shape[1]
    else:
        # 如果是tensor，应该有shape属性
        in_channels = sample_data.size(0) if hasattr(sample_data, 'size') else 7
        seq_len = sample_data.size(1) if hasattr(sample_data, 'size') else 320
    
    # 初始化模型
    G = GANGenerator(in_channels, in_channels, seq_len).to(device)
    D1 = GANDiscriminator(in_channels, seq_len).to(device)  # 主判别器
    D2 = ClassifierDiscriminator(in_channels, seq_len, num_classes=3).to(device)  # 分类判别器
    
    # 优化器
    g_optimizer = Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d1_optimizer = Adam(D1.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d2_optimizer = Adam(D2.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # 损失函数
    mse_loss = MSELoss()
    bce_loss = BCEWithLogitsLoss()
    ce_loss = CrossEntropyLoss()
    
    # 训练配置
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    patience = args.patience if args.patience is not None else cfg.get('patience', 10)
    best_g_loss = float('inf')
    patience_counter = 0
    
    # 标签
    real_label = 1.0
    fake_label = 0.0
    
    logging.info("Starting GAN training with dual discriminators...")
    
    for epoch in range(1, args.epochs + 1):
        G.train()
        D1.train()
        D2.train()
        
        g_loss_epoch = 0.0
        d1_loss_epoch = 0.0
        d2_loss_epoch = 0.0
        recon_loss_epoch = 0.0
        
        for batch_idx, real_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # =================== 训练判别器 D1 (主判别器) ===================
            d1_optimizer.zero_grad()
            
            # 真实数据
            real_output1 = D1(real_data)
            real_labels = torch.full((batch_size, 1), real_label, dtype=torch.float, device=device)
            d1_real_loss = bce_loss(real_output1, real_labels)
            
            # 生成假数据
            masked_data, mask_idx = mask_channel(real_data, mask_ratio=0.3)
            fake_data = G(masked_data).detach()
            fake_output1 = D1(fake_data)
            fake_labels = torch.full((batch_size, 1), fake_label, dtype=torch.float, device=device)
            d1_fake_loss = bce_loss(fake_output1, fake_labels)
            
            d1_loss = (d1_real_loss + d1_fake_loss) / 2
            d1_loss.backward()
            d1_optimizer.step()
            
            # =================== 训练判别器 D2 (分类判别器) ===================
            d2_optimizer.zero_grad()
            
            # 真实数据 - 判别 + 分类
            d2_real_disc, d2_real_class = D2(real_data)
            d2_real_disc_loss = bce_loss(d2_real_disc, real_labels)
            
            # 假设我们有类别标签（这里随机生成，实际应该从数据中获取）
            real_class_labels = torch.randint(0, 3, (batch_size,), device=device)
            d2_real_class_loss = ce_loss(d2_real_class, real_class_labels)
            
            # 假数据
            fake_data = G(masked_data).detach()
            d2_fake_disc, d2_fake_class = D2(fake_data)
            d2_fake_disc_loss = bce_loss(d2_fake_disc, fake_labels)
            
            d2_loss = d2_real_disc_loss + d2_fake_disc_loss + d2_real_class_loss
            d2_loss.backward()
            d2_optimizer.step()
            
            # =================== 训练生成器 G ===================
            g_optimizer.zero_grad()
            
            # 生成假数据
            fake_data = G(masked_data)
            
            # 对抗损失 - 欺骗D1
            fake_output1 = D1(fake_data)
            g_adv_loss1 = bce_loss(fake_output1, real_labels)
            
            # 对抗损失 - 欺骗D2
            d2_fake_disc, d2_fake_class = D2(fake_data)
            g_adv_loss2 = bce_loss(d2_fake_disc, real_labels)
            
            # 重建损失 - 只在mask的部分计算
            recon_loss = 0.0
            total_masked = 0
            for i in range(batch_size):
                for idx in mask_idx[i]:
                    recon_loss += mse_loss(fake_data[i, idx, :], real_data[i, idx, :])
                    total_masked += 1
            
            if total_masked > 0:
                recon_loss = recon_loss / total_masked
            
            # 总生成器损失
            g_loss = g_adv_loss1 + g_adv_loss2 + 10.0 * recon_loss  # 重建损失权重
            
            g_loss.backward()
            # 梯度裁剪
            clip_grad_norm_(G.parameters(), max_norm=1.0)
            g_optimizer.step()
            
            # 记录损失
            g_loss_epoch += g_loss.item()
            d1_loss_epoch += d1_loss.item()
            d2_loss_epoch += d2_loss.item()
            recon_loss_epoch += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
        
        # 计算平均损失
        num_batches = len(train_loader)
        g_loss_epoch /= num_batches
        d1_loss_epoch /= num_batches
        d2_loss_epoch /= num_batches
        recon_loss_epoch /= num_batches
        
        # 记录日志
        logging.info(f"Epoch {epoch}: G_loss={g_loss_epoch:.6f}, D1_loss={d1_loss_epoch:.6f}, "
                    f"D2_loss={d2_loss_epoch:.6f}, Recon_loss={recon_loss_epoch:.6f}")
        
        if writer:
            writer.add_scalar(f'{gan_type}/G_loss', g_loss_epoch, epoch)
            writer.add_scalar(f'{gan_type}/D1_loss', d1_loss_epoch, epoch)
            writer.add_scalar(f'{gan_type}/D2_loss', d2_loss_epoch, epoch)
            writer.add_scalar(f'{gan_type}/Recon_loss', recon_loss_epoch, epoch)
        
        # 保存最佳模型
        if recon_loss_epoch < best_g_loss:
            best_g_loss = recon_loss_epoch
            torch.save({
                'generator': G.state_dict(),
                'discriminator1': D1.state_dict(),
                'discriminator2': D2.state_dict(),
                'epoch': epoch,
                'g_loss': g_loss_epoch,
                'recon_loss': recon_loss_epoch
            }, os.path.join(args.output_dir, f'best_gan_{gan_type}.pth'))
            logging.info(f"Saved best model at epoch {epoch}")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"No improvement in reconstruction loss. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch} due to no improvement for {patience} epochs.")
            break

    # ====== 测试集评估 ======
    G.eval()
    test_mse_list = []
    
    with torch.no_grad():
        for real_data in tqdm(test_loader, desc="Testing"):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Mask测试数据
            masked_data, mask_idx = mask_channel(real_data, mask_ratio=0.3)
            fake_data = G(masked_data)
            
            # 计算MSE（仅在masked区域）
            batch_mse = 0.0
            total_masked = 0
            for i in range(batch_size):
                for idx in mask_idx[i]:
                    batch_mse += mse_loss(fake_data[i, idx, :], real_data[i, idx, :]).item()
                    total_masked += 1
            
            if total_masked > 0:
                batch_mse /= total_masked
                test_mse_list.append(batch_mse)
    
    mean_test_mse = sum(test_mse_list) / len(test_mse_list) if test_mse_list else 0.0
    logging.info(f"Test MSE: {mean_test_mse:.6f}")
    print(f"Test MSE: {mean_test_mse:.6f}")
    if writer:
        writer.add_scalar(f'{gan_type}/Test_MSE', mean_test_mse)
        writer.close()

if __name__ == '__main__':
    args = parse_args()
    for gan_type in ['gan']:
        train_gan(args, gan_type)
