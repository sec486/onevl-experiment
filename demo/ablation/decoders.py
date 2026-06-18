"""
OneVL 消融实验解码器架构
========================
两个辅助解码器在训练时提供梯度信号:
  - VisualDecoderV2: 从 ViT + 隐式状态预测未来帧 token
  - LangDecoder: 从语言隐式状态重建 CoT 文本

推理时两个解码器均丢弃 → 零开销。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualDecoderV2(nn.Module):
    """
    OneVL 公式3: Z_v = [W_v(V), W_v(H_v)]
    
    从以下输入预测离散视觉 token (Emu3 codebook):
      - ViT patch 嵌入（基座模型 ViT 的视觉特征）
      - 隐式 token 的 hidden state（序列中学习到的表征）
    
    架构: 线性投影 → 可学习 queries → Transformer 解码器 → CE 分类头
    """
    def __init__(self, vit_dim=1024, hidden_dim=2048, codebook_size=32768,
                 num_queries=64, num_layers=2, num_heads=8, inner_dim=512):
        super().__init__()
        self.proj_vit = nn.Linear(vit_dim, inner_dim)
        self.proj_lat = nn.Linear(hidden_dim, inner_dim)
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, inner_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=inner_dim, nhead=num_heads,
            dim_feedforward=inner_dim * 2, batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(inner_dim, codebook_size)

    def forward(self, vit_emb, latent_h=None, gt_ids=None):
        """
        Args:
            vit_emb: (B, N_vit, vit_dim) — ViT patch embeddings
            latent_h: (B, C_v, hidden_dim) — latent hidden states (None for Preliminary)
            gt_ids: (B, num_queries) — target discrete token IDs
        Returns:
            loss (scalar) if gt_ids provided, else logits (B, num_queries, codebook_size)
        """
        B = vit_emb.shape[0]
        if latent_h is not None:
            z_v = torch.cat([self.proj_vit(vit_emb), self.proj_lat(latent_h)], dim=1)
        else:
            z_v = self.proj_vit(vit_emb)

        queries = self.queries.expand(B, -1, -1)
        decoded = self.decoder(queries, z_v)
        logits = self.head(decoded)

        if gt_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), gt_ids.view(-1))
            return loss
        return logits


class LangDecoder(nn.Module):
    """
    语言辅助解码器
    
    从语言隐式 token 的 hidden state 重建 Chain-of-Thought 文本。
    强制模型在隐式 token 中编码有意义的推理信息。
    
    架构: 线性投影 → Transformer 解码器 → 词表分类头
    """
    def __init__(self, d_model=2048, n_heads=8, vocab_size=151669, inner_dim=512):
        super().__init__()
        self.proj_in = nn.Linear(d_model, inner_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=inner_dim, nhead=n_heads,
            dim_feedforward=inner_dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.head = nn.Linear(inner_dim, vocab_size)

    def forward(self, latent_h, target_ids):
        """
        Args:
            latent_h: (B, C_l, d_model) — language latent hidden states
            target_ids: (B, L) — target token IDs for CoT text
        Returns:
            CE loss (scalar)
        """
        B, L = target_ids.shape
        latent_proj = self.proj_in(latent_h)
        inner_dim = latent_proj.shape[-1]
        causal_mask = torch.triu(
            torch.ones(L, L, device=target_ids.device), diagonal=1
        ).bool()
        tgt = torch.zeros(B, L, inner_dim, device=latent_h.device, dtype=latent_h.dtype)
        decoded = self.decoder(tgt, latent_proj, tgt_mask=causal_mask)
        logits = self.head(decoded)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100
        )
