# train_staged.py - 分阶段训练脚本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
import os
from typing import Dict, Tuple, Optional

from data import MultiModalDataset, collate_fn
from model import TGATUNet
from enhanced_validation_integration import EnhancedValidationManager

class StagedTrainer:
    """分阶段训练管理器"""
    
    def __init__(self, config_path: str):
        """
        初始化分阶段训练器
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_model()
        self.setup_data()
        self.setup_validation()
        
        # 训练阶段配置
        self.stage_config = {
            'reconstruction_first': {
                'epochs': self.config.get('reconstruction_epochs', 50),
                'lr': self.config.get('reconstruction_lr', 0.001),
                'stage': 'reconstruction'
            },
            'classification_second': {
                'epochs': self.config.get('classification_epochs', 30), 
                'lr': self.config.get('classification_lr', 0.0005),
                'stage': 'classification'
            },
            'joint_training': {
                'epochs': self.config.get('joint_epochs', 20),
                'lr': self.config.get('joint_lr', 0.0002),
                'stage': 'both'
            }
        }
        
    def setup_logging(self):
        """设置日志"""
        log_dir = self.config.get('log_dir', 'Logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'staged_training.log')),
                logging.StreamHandler()
            ]
        )
        
    def setup_model(self):
        """设置模型"""
        self.model = TGATUNet(
            in_channels=self.config['in_channels'],
            hidden_channels=self.config['hidden_channels'],
            out_channels=self.config['out_channels'],
            encoder_layers=self.config.get('encoder_layers', 3),
            decoder_layers=self.config.get('decoder_layers', 3),
            heads=self.config.get('attention_heads', 4),
            time_k=self.config.get('time_k', 1),
            trans_nhead=self.config.get('attention_heads', 4),
            trans_layers=self.config.get('transformer_layers', 1),
            num_classes=self.config.get('num_classes', 2)
        ).to(self.device)
        
        logging.info(f"模型已加载到设备: {self.device}")
        
    def setup_data(self):
        """设置数据"""
        # 创建数据集
        self.train_dataset = MultiModalDataset(
            config=self.config,
            split='train'
        )
        
        self.val_dataset = MultiModalDataset(
            config=self.config,
            split='val'
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', True)
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', True)
        )
        
        logging.info(f"数据加载完成 - 训练集: {len(self.train_dataset)}, 验证集: {len(self.val_dataset)}")
        
    def setup_validation(self):
        """设置验证管理器"""
        self.val_manager = EnhancedValidationManager(
            patience=self.config.get('patience', 15),
            min_delta=self.config.get('min_delta', 1e-6),
            save_dir=os.path.join(self.config.get('log_dir', 'Logs'), 'staged_validation')
        )
        
    def setup_optimizer_for_stage(self, stage: str) -> optim.Optimizer:
        """为特定阶段设置优化器"""
        stage_cfg = self.stage_config[stage]
        lr = stage_cfg['lr']
        
        # 设置梯度计算
        self.model.set_stage_gradients(stage_cfg['stage'])
        
        # 只为需要梯度的参数创建优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        logging.info(f"为阶段 '{stage}' 设置优化器 - 学习率: {lr}, 可训练参数: {len(trainable_params)}")
        return optimizer
        
    def compute_loss(self, batch_recon_out: torch.Tensor, batch_cls_out: torch.Tensor,
                     targets: torch.Tensor, labels: torch.Tensor, stage: str) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}
        total_loss = 0.0
        
        # 重建损失
        if batch_recon_out is not None and stage in ['reconstruction_first', 'joint_training']:
            recon_loss = nn.MSELoss()(batch_recon_out, targets)
            losses['recon_loss'] = recon_loss
            total_loss += recon_loss * self.config.get('recon_weight', 1.0)
            
        # 分类损失
        if batch_cls_out is not None and stage in ['classification_second', 'joint_training']:
            cls_loss = nn.CrossEntropyLoss()(batch_cls_out, labels)
            losses['cls_loss'] = cls_loss
            total_loss += cls_loss * self.config.get('cls_weight', 1.0)
            
        losses['total_loss'] = total_loss
        return losses
        
    def train_stage(self, stage_name: str) -> Dict:
        """训练特定阶段"""
        stage_cfg = self.stage_config[stage_name]
        stage = stage_cfg['stage']
        epochs = stage_cfg['epochs']
        
        logging.info(f"\n=== 开始训练阶段: {stage_name} ===")
        logging.info(f"阶段配置: {stage_cfg}")
        
        # 设置优化器
        optimizer = self.setup_optimizer_for_stage(stage_name)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        best_loss = float('inf')
        stage_metrics = []
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_losses = {'recon_loss': 0.0, 'cls_loss': 0.0, 'total_loss': 0.0}
            train_samples = 0
            
            for batch_idx, batch_data in enumerate(self.train_loader):
                if len(batch_data) == 4:
                    batch, labels, _, is_real_mask = batch_data
                else:
                    batch, labels, _, is_real_mask, _ = batch_data
                    
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                targets = batch.permute(0, 2, 1)  # [B, C, T]
                
                # 分阶段前向传播
                batch_recon_out, batch_cls_out = self.model.forward_batch_staged(batch, stage=stage)
                
                # 计算损失
                losses = self.compute_loss(batch_recon_out, batch_cls_out, targets, labels, stage_name)
                
                # 反向传播
                optimizer.zero_grad()
                losses['total_loss'].backward()
                
                # 梯度裁剪
                if self.config.get('gradient_clipping', False):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                optimizer.step()
                
                # 累积损失
                batch_size = batch.size(0)
                train_samples += batch_size
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        train_losses[key] += value.item() * batch_size
                        
            # 计算平均损失
            for key in train_losses:
                train_losses[key] /= train_samples
                
            # 验证 (每5个epoch验证一次)
            if epoch % 5 == 0 or epoch == epochs - 1:
                val_metrics = self.validate_stage(stage)
                val_loss = val_metrics['val_loss']
                
                # 学习率调度
                scheduler.step(val_loss)
                
                # 记录最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(f'best_{stage_name}.pth', epoch, optimizer, val_metrics)
                    
                logging.info(
                    f"[{stage_name}] Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_losses['total_loss']:.6f}, "
                    f"val_loss={val_loss:.6f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )
                
                stage_metrics.append({
                    'epoch': epoch,
                    'train_metrics': train_losses,
                    'val_metrics': val_metrics
                })
                
        logging.info(f"=== 阶段 {stage_name} 训练完成，最佳验证损失: {best_loss:.6f} ===\n")
        return {'best_loss': best_loss, 'metrics_history': stage_metrics}
        
    def validate_stage(self, stage: str) -> Dict:
        """验证特定阶段"""
        self.model.eval()
        val_losses = {'recon_loss': 0.0, 'cls_loss': 0.0, 'total_loss': 0.0}
        val_samples = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                if len(batch_data) == 4:
                    batch, labels, _, is_real_mask = batch_data
                else:
                    batch, labels, _, is_real_mask, _ = batch_data
                    
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                targets = batch.permute(0, 2, 1)  # [B, C, T]
                
                # 分阶段前向传播
                batch_recon_out, batch_cls_out = self.model.forward_batch_staged(batch, stage=stage)
                
                # 计算损失
                losses = self.compute_loss(batch_recon_out, batch_cls_out, targets, labels, 'joint_training')
                
                # 累积损失
                batch_size = batch.size(0)
                val_samples += batch_size
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        val_losses[key] += value.item() * batch_size
                        
        # 计算平均损失
        for key in val_losses:
            val_losses[key] /= val_samples
            
        return {
            'val_loss': val_losses['total_loss'],
            'val_recon_loss': val_losses['recon_loss'],
            'val_cls_loss': val_losses['cls_loss'],
            'val_samples': val_samples
        }
        
    def save_checkpoint(self, filename: str, epoch: int, optimizer: optim.Optimizer, metrics: Dict):
        """保存检查点"""
        ckpt_dir = self.config.get('ckpt_dir', 'Checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(ckpt_dir, filename))
        logging.info(f"检查点已保存: {filename}")
        
    def run_staged_training(self):
        """运行完整的分阶段训练"""
        logging.info("开始分阶段训练流程")
        
        all_results = {}
        
        # 阶段1: 重建训练
        logging.info("🔄 阶段1: 专注重建任务训练")
        stage1_results = self.train_stage('reconstruction_first')
        all_results['reconstruction_first'] = stage1_results
        
        # 阶段2: 分类训练 (冻结重建分支)
        logging.info("🎯 阶段2: 专注分类任务训练")
        self.model.freeze_reconstruction_branch()
        stage2_results = self.train_stage('classification_second')
        all_results['classification_second'] = stage2_results
        
        # 阶段3: 联合训练
        logging.info("🔗 阶段3: 重建+分类联合训练")
        self.model.unfreeze_all()
        stage3_results = self.train_stage('joint_training')
        all_results['joint_training'] = stage3_results
        
        logging.info("✅ 分阶段训练完成")
        return all_results

def main():
    """主函数"""
    # 加载配置
    config_path = 'config.yaml'
    
    # 创建分阶段训练器
    trainer = StagedTrainer(config_path)
    
    # 运行训练
    results = trainer.run_staged_training()
    
    # 保存训练结果
    import json
    with open('staged_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info("训练结果已保存到 staged_training_results.json")

if __name__ == "__main__":
    main()
