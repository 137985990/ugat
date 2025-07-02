# src/model.py

"""
model.py

Module defining the Temporal Graph Attention U-Net (T-GAT-UNet) architecture,
with Graph-based encoder (GAT), Transformer bottleneck, and Graph-based decoder (GAT).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from graph import build_graph


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer: in_channels -> hidden_channels via multi-head
        self.layers.append(
            GATConv(in_channels, hidden_channels // heads, heads=heads)
        )
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads)
            )
        self.act = nn.ReLU()

    def forward(self, x, edge_index, return_attention=False):
        # x: [N_nodes, in_channels]
        attentions = []
        for gat in self.layers:
            if return_attention:
                x, attn = gat(x, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                x = gat(x, edge_index)
            x = self.act(x)
        if return_attention:
            return x, attentions  # [N_nodes, hidden_channels], list of attention weights
        return x  # [N_nodes, hidden_channels]


class TransformerBottleneck(nn.Module):
    def __init__(self, hidden_channels, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch_size, N_nodes, hidden_channels]
        return self.transformer(x)


class GraphDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # First decoder GAT
        self.layers.append(
            GATConv(hidden_channels, hidden_channels // heads, heads=heads)
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads)
            )
        # Last layer maps to out_channels
        self.layers.append(
            GATConv(hidden_channels, out_channels, heads=1)
        )
        self.act = nn.ReLU()

    def forward(self, x, edge_index, return_attention=False):
        attentions = []
        for gat in self.layers[:-1]:
            if return_attention:
                x, attn = gat(x, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                x = gat(x, edge_index)
            x = self.act(x)
        if return_attention:
            x, attn = self.layers[-1](x, edge_index, return_attention_weights=True)
            attentions.append(attn)
            return x, attentions  # [N_nodes, out_channels], list of attention weights
        x = self.layers[-1](x, edge_index)
        return x  # [N_nodes, out_channels]


class TGATUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 encoder_layers=3, decoder_layers=3,
                 heads=4, time_k=1,
                 trans_nhead=4, trans_layers=2, trans_dim_feedforward=512,
                 num_classes=2,
                 use_discriminator=False):
        super().__init__()
        print(f"[DEBUG] TGATUNet in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}")
        self.time_k = time_k
        # Modules
        self.encoder = GraphEncoder(in_channels, hidden_channels, num_layers=encoder_layers, heads=heads)
        self.bottleneck = TransformerBottleneck(hidden_channels,
                                                nhead=trans_nhead,
                                                num_layers=trans_layers,
                                                dim_feedforward=trans_dim_feedforward)
        self.decoder = GraphDecoder(hidden_channels, out_channels,
                                    num_layers=decoder_layers, heads=heads)
        
        # UNet Skip Connection: 将编码器特征连接到解码器
        self.skip_proj = nn.Linear(hidden_channels + hidden_channels, hidden_channels)
        
        # 分类头：全局池化后MLP（主分类器）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # 验证生成分类器：专门用于验证生成数据质量和分类性能
        self.generation_validator = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels // 2, num_classes)
        )
        # 判别器分支（可选，后续可扩展为独立判别器）
        self.use_discriminator = use_discriminator
        if use_discriminator:
            # 简单判别器：输入为解码后特征，输出为一标量（可扩展为更复杂结构）
            self.discriminator = nn.Sequential(
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                nn.Linear(out_channels // 2, 1),
                nn.Sigmoid()            )
        else:
            self.discriminator = None

    def forward(self, window, return_attention=False, phase="encode", use_generation_validator=False):
        """
        UNet范式前向传播，单阶段encode/decode为同一前向过程
        phase: "encode" or "decode"
        use_generation_validator: 是否使用验证生成分类器
        - encode: 返回 (out, logits)
        - decode: 返回 (out, disc_pred)
        - 当use_generation_validator=True时，额外返回生成验证分类结果
        """
        device = window.device
        data = build_graph(window, time_k=self.time_k)
        x = data.x.to(device) if data.x is not None else None
        edge_index = data.edge_index.to(device) if data.edge_index is not None else None
        
        # UNet Encode: GAT下采样
        if return_attention:
            h_encoded, encoder_attn = self.encoder(x, edge_index, return_attention=True)  # [T, hidden], list
        else:
            h_encoded = self.encoder(x, edge_index)
        
        # UNet Bottleneck: Transformer瓶颈
        h_trans = h_encoded.unsqueeze(0)  # [1, T, hidden]
        h_bottleneck = self.bottleneck(h_trans)  # [1, T, hidden]
        h_bottleneck = h_bottleneck.squeeze(0)  # [T, hidden]
        
        # UNet Skip Connection: 融合编码器特征和瓶颈特征
        h_skip = torch.cat([h_encoded, h_bottleneck], dim=-1)  # [T, hidden*2]
        h_skip = self.skip_proj(h_skip)  # [T, hidden]
        
        # 分类分支：全局平均池化
        h_cls = h_skip.mean(dim=0)  # [hidden]
        logits = self.classifier(h_cls)  # [num_classes]
        
        # UNet Decode: GAT上采样
        if return_attention:
            out, decoder_attn = self.decoder(h_skip, edge_index, return_attention=True)  # [T, out_channels], list
            # Return output, attention maps, and logits
            if use_generation_validator:
                # 验证生成分类器：基于解码输出进行分类
                h_gen = out.mean(dim=0)  # [out_channels] - 全局平均池化
                gen_logits = self.generation_validator(h_gen)  # [num_classes]
                return out.t(), encoder_attn, decoder_attn, logits, gen_logits
            return out.t(), encoder_attn, decoder_attn, logits
        out = self.decoder(h_skip, edge_index)  # [T, out_channels]
        out_t = out.t()  # [out_channels, T]
        
        # 验证生成分类器处理
        gen_logits = None
        if use_generation_validator:
            # 验证生成分类器：基于解码输出进行分类
            h_gen = out.mean(dim=0)  # [out_channels] - 全局平均池化
            gen_logits = self.generation_validator(h_gen)  # [num_classes]
        
        if phase == "encode":
            # encode 阶段：返回 (out, logits) 或 (out, logits, gen_logits)
            if use_generation_validator:
                return out_t, logits, gen_logits
            return out_t, logits
        elif phase == "decode":
            # decode 阶段：返回 (out, disc_pred) 或 (out, disc_pred, gen_logits)
            if self.use_discriminator and self.discriminator is not None:
                # 判别器输入：对每个时间点的补全结果做池化（可自定义）
                # 这里简单取均值池化
                pooled = out_t.mean(dim=1)  # [out_channels]
                disc_pred = self.discriminator(pooled)  # [1]
            else:
                disc_pred = None
            if use_generation_validator:
                return out_t, disc_pred, gen_logits
            return out_t, disc_pred
        else:
            # 默认兼容旧接口
            if use_generation_validator:
                return out_t, logits, gen_logits
            return out_t, logits

    def forward_batch(self, windows_batch, use_generation_validator=False):
        """
        批量前向传播 - 充分利用16GB显存
        Args:
            windows_batch: [batch_size, T, C] 批量窗口数据
            use_generation_validator: 是否使用验证生成分类器
        Returns:
            batch_out: [batch_size, C, T] 批量重建输出
            batch_logits: [batch_size, num_classes] 批量分类输出
            batch_gen_logits: [batch_size, num_classes] 批量验证生成分类输出 (当use_generation_validator=True时)
        """
        batch_size, T, C = windows_batch.size()
        device = windows_batch.device
        
        # 批量处理所有样本的图构建
        batch_outputs = []
        batch_logits = []
        batch_gen_logits = []
          # 可以进一步优化：尝试向量化图构建
        for i in range(batch_size):
            window = windows_batch[i]  # [T, C]
            result = self.forward(window, use_generation_validator=use_generation_validator)
            
            if use_generation_validator:
                if len(result) == 3:
                    out, logits, gen_logits = result
                elif len(result) == 5:
                    out, _, _, logits, gen_logits = result
                else:
                    out, logits, gen_logits = result[0], result[-2], result[-1]
                batch_gen_logits.append(gen_logits)
            else:
                if len(result) == 2:
                    out, logits = result
                elif len(result) == 4:
                    out, _, _, logits = result
                else:
                    out, logits = result[0], result[-1]  # 取第一个和最后一个
            
            batch_outputs.append(out)
            batch_logits.append(logits)
        
        # 堆叠结果
        batch_out = torch.stack(batch_outputs, dim=0)  # [batch_size, C, T]
        batch_logits = torch.stack(batch_logits, dim=0)  # [batch_size, num_classes]
        
        if use_generation_validator:
            batch_gen_logits = torch.stack(batch_gen_logits, dim=0)  # [batch_size, num_classes]
            return batch_out, batch_logits, batch_gen_logits
        
        return batch_out, batch_logits
    
    def forward_batch_parallel(self, windows_batch):
        """
        并行批量前向传播 - 最大化显存利用
        使用torch.jit.script或其他并行化技术
        """
        batch_size, T, C = windows_batch.size()
        
        # 尝试使用编译优化的批量处理
        try:
            # 使用torch.compile进行批量优化
            @torch.compile
            def compiled_batch_forward(windows):
                return self.forward_batch(windows)
            
            return compiled_batch_forward(windows_batch)
        except:
            # 回退到标准批量处理
            return self.forward_batch(windows_batch)
    
    def forward_staged(self, window, stage="reconstruction", return_attention=False, use_generation_validator=False):
        """
        分阶段前向传播
        Args:
            window: 输入窗口数据
            stage: "reconstruction" | "classification" | "generation_validation" | "both"
            return_attention: 是否返回注意力权重
            use_generation_validator: 是否使用验证生成分类器
        Returns:
            根据stage返回不同内容：
            - "reconstruction": (重建输出, None, None)
            - "classification": (None, 分类输出, None) 
            - "generation_validation": (None, None, 验证生成分类输出)
            - "both": (重建输出, 分类输出, 验证生成分类输出 if use_generation_validator else None)
        """
        device = window.device
        data = build_graph(window, time_k=self.time_k)
        x = data.x.to(device) if data.x is not None else None
        edge_index = data.edge_index.to(device) if data.edge_index is not None else None
        
        # Encode: 总是需要执行
        if return_attention:
            h_encoded, encoder_attn = self.encoder(x, edge_index, return_attention=True)
        else:
            h_encoded = self.encoder(x, edge_index)
            encoder_attn = None
        
        # Transformer bottleneck: 总是需要执行
        h_trans = h_encoded.unsqueeze(0)  # [1, T, hidden]
        h_bottleneck = self.bottleneck(h_trans)  # [1, T, hidden]
        h_bottleneck = h_bottleneck.squeeze(0)  # [T, hidden]
        
        # UNet Skip Connection
        h_skip = torch.cat([h_encoded, h_bottleneck], dim=-1)  # [T, hidden*2]
        h_skip = self.skip_proj(h_skip)  # [T, hidden]
        
        # 根据阶段决定执行哪些分支
        reconstruction_out = None
        classification_out = None
        generation_validation_out = None
        decoder_attn = None
        
        if stage in ["reconstruction", "both"]:
            # 执行重建分支
            if return_attention:
                out, decoder_attn = self.decoder(h_skip, edge_index, return_attention=True)
            else:
                out = self.decoder(h_skip, edge_index)
            reconstruction_out = out.t()  # [out_channels, T]
            
            # 如果需要验证生成分类器，基于重建输出计算
            if (stage == "both" and use_generation_validator) or stage == "generation_validation":
                h_gen = out.mean(dim=0)  # [out_channels] - 全局平均池化
                generation_validation_out = self.generation_validator(h_gen)  # [num_classes]
        
        if stage in ["classification", "both"]:
            # 执行分类分支
            h_cls = h_skip.mean(dim=0)  # [hidden] - 全局平均池化
            classification_out = self.classifier(h_cls)  # [num_classes]
        
        if stage == "generation_validation" and reconstruction_out is None:
            # 单独的验证生成分类，需要先计算重建
            if return_attention:
                out, decoder_attn = self.decoder(h_skip, edge_index, return_attention=True)
            else:
                out = self.decoder(h_skip, edge_index)
            h_gen = out.mean(dim=0)  # [out_channels] - 全局平均池化
            generation_validation_out = self.generation_validator(h_gen)  # [num_classes]
        
        if return_attention:
            return reconstruction_out, classification_out, generation_validation_out, encoder_attn, decoder_attn
        else:
            return reconstruction_out, classification_out, generation_validation_out

    def forward_batch_staged(self, windows_batch, stage="reconstruction", use_generation_validator=False):
        """
        分阶段批量前向传播
        Args:
            windows_batch: [batch_size, T, C] 批量窗口数据
            stage: "reconstruction" | "classification" | "generation_validation" | "both"
            use_generation_validator: 是否使用验证生成分类器
        Returns:
            根据stage返回 (batch_recon_out, batch_cls_out, batch_gen_val_out)
            不需要的输出为None
        """
        batch_size, T, C = windows_batch.size()
        
        batch_recon_outputs = []
        batch_cls_outputs = []
        batch_gen_val_outputs = []
        
        for i in range(batch_size):
            window = windows_batch[i]  # [T, C]
            result = self.forward_staged(window, stage=stage, use_generation_validator=use_generation_validator)
            
            if len(result) >= 3:
                recon_out, cls_out, gen_val_out = result[0], result[1], result[2]
            else:
                recon_out, cls_out = result[0], result[1] if len(result) > 1 else None
                gen_val_out = None
            
            if recon_out is not None:
                batch_recon_outputs.append(recon_out)
            if cls_out is not None:
                batch_cls_outputs.append(cls_out)
            if gen_val_out is not None:
                batch_gen_val_outputs.append(gen_val_out)
        
        # 堆叠结果
        batch_recon_out = torch.stack(batch_recon_outputs, dim=0) if batch_recon_outputs else None
        batch_cls_out = torch.stack(batch_cls_outputs, dim=0) if batch_cls_outputs else None
        batch_gen_val_out = torch.stack(batch_gen_val_outputs, dim=0) if batch_gen_val_outputs else None
        
        return batch_recon_out, batch_cls_out, batch_gen_val_out

    def set_stage_gradients(self, stage="reconstruction"):
        """
        设置特定阶段的梯度计算
        Args:
            stage: "reconstruction" | "classification" | "generation_validation" | "both"
        """
        # 首先关闭所有梯度
        for param in self.parameters():
            param.requires_grad = False
        
        # 编码器和Transformer总是需要梯度（因为是共享的）
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.bottleneck.parameters():
            param.requires_grad = True
        for param in self.skip_proj.parameters():
            param.requires_grad = True
        
        if stage in ["reconstruction", "both"]:
            # 开启解码器梯度
            for param in self.decoder.parameters():
                param.requires_grad = True
        
        if stage in ["classification", "both"]:
            # 开启分类器梯度
            for param in self.classifier.parameters():
                param.requires_grad = True
        
        if stage in ["generation_validation", "both"]:
            # 开启验证生成分类器梯度（需要解码器输出）
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.generation_validator.parameters():
                param.requires_grad = True

    def freeze_reconstruction_branch(self):
        """冻结重建分支（编码器+解码器）"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        # 保持Transformer和分类器可训练
        for param in self.bottleneck.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def freeze_classification_branch(self):
        """冻结分类分支"""
        for param in self.classifier.parameters():
            param.requires_grad = False
        # 保持重建相关组件可训练
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.bottleneck.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def freeze_generation_validator(self):
        """冻结验证生成分类器"""
        for param in self.generation_validator.parameters():
            param.requires_grad = False

    def freeze_main_classifier(self):
        """冻结主分类器"""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

    def get_generation_validator_params(self):
        """获取验证生成分类器的参数，用于单独优化"""
        return list(self.generation_validator.parameters())
