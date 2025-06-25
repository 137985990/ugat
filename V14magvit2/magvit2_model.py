# magvit2_model.py - Magvit2 VAE实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """Vector Quantization module for Magvit2"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
          # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, height, width, embedding_dim] or [batch_size, seq_len, embedding_dim]
        Returns:
            quantized: quantized tensor
            vq_loss: vector quantization loss
            encoding_indices: indices of closest embeddings
        """
        # Calculate distances to embedding vectors
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(inputs.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices.reshape(inputs.shape[:-1])


class ResidualBlock(nn.Module):
    """Residual block for encoder/decoder"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.activation = nn.SiLU()
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        return self.activation(out + residual)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Compute attention
        q = q.permute(0, 2, 1)  # [b, h, c]
        k = k.permute(0, 2, 1)  # [b, h, c]
        v = v.permute(0, 2, 1)  # [b, h, c]
        
        attn = torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1)  # [b, c, h]
        
        out = self.proj_out(out)
        return out + residual


class Magvit2Encoder(nn.Module):
    """Magvit2 Encoder"""
    
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int, num_res_blocks: int = 2):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels * 2),
            AttentionBlock(hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, hidden_channels * 4),
        ])
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels * 4, hidden_channels * 4),
            AttentionBlock(hidden_channels * 4),
            ResidualBlock(hidden_channels * 4, hidden_channels * 4),
        ])
        
        # Output projection
        self.conv_out = nn.Conv1d(hidden_channels * 4, latent_dim, kernel_size=3, padding=1)
        self.norm_out = nn.GroupNorm(1, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, in_channels, seq_len]
        Returns:
            latent: [batch_size, latent_dim, seq_len//4]
        """
        # Initial conv
        h = self.conv_in(x)
        
        # Downsampling
        for i, block in enumerate(self.down_blocks):
            h = block(h)
            if i in [0, 2]:  # Downsample after certain blocks
                h = F.avg_pool1d(h, kernel_size=2)
        
        # Middle blocks
        for block in self.mid_blocks:
            h = block(h)
        
        # Output
        h = self.conv_out(h)
        h = self.norm_out(h)
        
        return h


class Magvit2Decoder(nn.Module):
    """Magvit2 Decoder"""
    
    def __init__(self, latent_dim: int, hidden_channels: int, out_channels: int, num_res_blocks: int = 2):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv1d(latent_dim, hidden_channels * 4, kernel_size=3, padding=1)
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels * 4, hidden_channels * 4),
            AttentionBlock(hidden_channels * 4),
            ResidualBlock(hidden_channels * 4, hidden_channels * 4),
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels * 4, hidden_channels * 2),
            AttentionBlock(hidden_channels * 2),
            ResidualBlock(hidden_channels * 2, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
        ])
        
        # Output projection
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, latent_dim, seq_len//4]
        Returns:
            output: [batch_size, out_channels, seq_len]
        """
        # Initial conv
        h = self.conv_in(x)
        
        # Middle blocks
        for block in self.mid_blocks:
            h = block(h)
        
        # Upsampling
        for i, block in enumerate(self.up_blocks):
            if i in [1, 3]:  # Upsample before certain blocks
                h = F.interpolate(h, scale_factor=2, mode='linear', align_corners=False)
            h = block(h)
        
        # Output
        output = self.conv_out(h)
        
        return output


class Magvit2VAE(nn.Module):
    """
    Magvit2 VAE for multimodal time series data
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int = 128,
                 latent_dim: int = 64,
                 num_embeddings: int = 1024,
                 num_classes: int = 2,
                 commitment_cost: float = 0.25):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = Magvit2Encoder(in_channels, hidden_channels, latent_dim)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        # Decoder
        self.decoder = Magvit2Decoder(latent_dim, hidden_channels, in_channels)
        
        # Classifier (operates on quantized latent features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, num_classes)
        )
        
        print(f"[Magvit2VAE] Initialized with in_channels={in_channels}, "
              f"hidden_channels={hidden_channels}, latent_dim={latent_dim}, "
              f"num_embeddings={num_embeddings}, num_classes={num_classes}")
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x, return_loss=True):
        """
        Args:
            x: [batch_size, in_channels, seq_len] or [seq_len, in_channels] for single sample
        Returns:
            If return_loss=True: (reconstruction, classification_logits, vq_loss)
            If return_loss=False: (reconstruction, classification_logits)
        """
        # Handle single sample input
        if x.dim() == 2:
            x = x.t().unsqueeze(0)  # [seq_len, in_channels] -> [1, in_channels, seq_len]
            single_sample = True
        else:
            single_sample = False
        
        # Encode
        z_e = self.encoder(x)
        
        # Vector quantization
        z_q, vq_loss, _ = self.vq(z_e.permute(0, 2, 1))  # VQ expects [batch, seq, dim]
        z_q = z_q.permute(0, 2, 1)  # Back to [batch, dim, seq]
        
        # Decode
        x_recon = self.decoder(z_q)
        
        # Classification
        logits = self.classifier(z_q)
          # Handle single sample output
        if single_sample:
            x_recon = x_recon.squeeze(0).t()  # [1, in_channels, seq_len] -> [seq_len, in_channels]
            logits = logits.squeeze(0)  # [1, num_classes] -> [num_classes]
        
        if return_loss:
            return x_recon, logits, vq_loss
        else:
            return x_recon, logits
    
    def forward_batch(self, windows_batch):
        """
        批量前向传播接口，兼容原始训练脚本
        Args:
            windows_batch: [batch_size, seq_len, in_channels]
        Returns:
            batch_out: [batch_size, in_channels, seq_len]
            batch_logits: [batch_size, num_classes]
        """
        # Convert to [batch_size, in_channels, seq_len]
        x = windows_batch.permute(0, 2, 1)
        
        # Forward pass
        result = self.forward(x, return_loss=True)
        if len(result) == 3:
            x_recon, logits, _ = result
        else:
            x_recon, logits = result
        
        # Convert output back to expected format
        batch_out = x_recon  # Already [batch_size, in_channels, seq_len]
        batch_logits = logits
        
        return batch_out, batch_logits
    
    def get_vq_loss(self, x):
        """Get vector quantization loss separately"""
        if x.dim() == 2:
            x = x.t().unsqueeze(0)
        
        z_e = self.encoder(x)
        _, vq_loss, _ = self.vq(z_e.permute(0, 2, 1))
        return vq_loss
    
    def get_reconstruction_loss(self, x, x_recon, reduction='mean'):
        """Calculate reconstruction loss"""
        if reduction == 'mean':
            return F.mse_loss(x_recon, x)
        elif reduction == 'sum':
            return F.mse_loss(x_recon, x, reduction='sum')
        else:
            return F.mse_loss(x_recon, x, reduction='none')
