"""
Auxiliary Language Decoder for OneVL × Cosmos-Reason2 Experiment (Phase 1B)

A lightweight 2-layer transformer decoder that reconstructs CoT reasoning text
from the latent token hidden states. This provides direct supervisory signal
to the latent tokens during training.

Architecture:
- Input: hidden states at latent token positions (2 tokens × 2048d)
- Cross-attention decoder (2 layers, 8 heads, d=2048)
- Output: CoT text token logits (teacher-forced during training)
- Discarded at inference (zero overhead)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LanguageDecoderHead(nn.Module):
    """
    Lightweight transformer decoder that reconstructs CoT text from latent states.
    
    During training:
        latent_hidden_states (B, num_latent, d_model) → CoT text logits (B, seq_len, vocab_size)
    
    At inference:
        Not used (discarded). Latent tokens feed directly to trajectory output.
    """
    
    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 8,
        n_layers: int = 2,
        vocab_size: int = 151936,  # Qwen3-VL vocab size
        max_cot_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_cot_length = max_cot_length
        
        # Positional embedding for decoder input (CoT tokens)
        self.pos_embedding = nn.Embedding(max_cot_length, d_model)
        
        # Token embedding for decoder input (shared with base model or separate)
        # Using separate lightweight embedding to avoid messing with base model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer decoder layers with cross-attention to latent states
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable for small decoders)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection to vocab
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Layer norm before output
        self.ln_f = nn.LayerNorm(d_model)
        
    def forward(
        self,
        latent_hidden_states: torch.Tensor,  # (B, num_latent, d_model) — from base model
        target_ids: torch.Tensor,            # (B, seq_len) — teacher-forced CoT token IDs
        target_mask: Optional[torch.Tensor] = None,  # (B, seq_len) — attention mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (teacher-forced training).
        
        Args:
            latent_hidden_states: Hidden states extracted at latent token positions
            target_ids: Ground-truth CoT token IDs (shifted right for teacher forcing)
            target_mask: 1 for valid tokens, 0 for padding
            
        Returns:
            logits: (B, seq_len, vocab_size)
            loss: scalar CrossEntropyLoss
        """
        B, seq_len = target_ids.shape
        device = target_ids.device
        
        # Embed target tokens + positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        tgt = self.token_embedding(target_ids) + self.pos_embedding(positions)
        
        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device, dtype=latent_hidden_states.dtype
        )
        
        # Padding mask for target
        tgt_key_padding_mask = None
        if target_mask is not None:
            tgt_key_padding_mask = (target_mask == 0)  # True = masked (PyTorch convention)
        
        # Decode: cross-attend to latent_hidden_states
        # memory = latent_hidden_states (B, num_latent, d_model)
        # tgt = embedded CoT tokens (B, seq_len, d_model)
        decoded = self.decoder(
            tgt=tgt,
            memory=latent_hidden_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        
        # Project to vocab
        decoded = self.ln_f(decoded)
        logits = self.output_proj(decoded)  # (B, seq_len, vocab_size)
        
        # Compute loss (shift by 1 for next-token prediction)
        # Input: target_ids[:-1], Label: target_ids[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        # Flatten for cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,  # Ignore padding
            reduction="mean",
        )
        
        return logits, loss
    
    def get_parameter_count(self) -> dict:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "token_embedding": self.token_embedding.weight.numel(),
            "pos_embedding": self.pos_embedding.weight.numel(),
            "decoder_layers": sum(
                p.numel() for n, p in self.named_parameters() 
                if "decoder" in n
            ),
            "output_proj": self.output_proj.weight.numel(),
        }


def extract_latent_hidden_states(
    model_outputs,
    hidden_states: torch.Tensor,
    latent_token_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Extract hidden states at latent token positions from the last transformer layer.
    
    Args:
        model_outputs: outputs from model forward (not used if hidden_states provided)
        hidden_states: (B, seq_len, d_model) — last layer hidden states
        latent_token_positions: (B, num_latent) — indices of latent tokens in sequence
        
    Returns:
        latent_states: (B, num_latent, d_model)
    """
    B, num_latent = latent_token_positions.shape
    d_model = hidden_states.shape[-1]
    
    # Gather hidden states at latent positions
    # Expand positions for gather: (B, num_latent, 1) → (B, num_latent, d_model)
    positions_expanded = latent_token_positions.unsqueeze(-1).expand(B, num_latent, d_model)
    latent_states = hidden_states.gather(1, positions_expanded)
    
    return latent_states


def find_latent_token_positions(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Find positions of latent tokens in the input sequence.
    
    Looks for tokens between <|start-latent|> and <|end-latent|> markers.
    
    Args:
        input_ids: (B, seq_len)
        tokenizer: the model's tokenizer
        
    Returns:
        positions: (B, num_latent) — padded with -1 if not found
    """
    # Get special token IDs
    start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    
    # Also check visual latent tokens
    start_latent_vis_id = tokenizer.convert_tokens_to_ids("<|start-latent-vis|>")
    end_latent_vis_id = tokenizer.convert_tokens_to_ids("<|end-latent-vis|>")
    latent_vis_id = tokenizer.convert_tokens_to_ids("<|latent-vis|>")
    
    B, seq_len = input_ids.shape
    all_positions = []
    
    for b in range(B):
        positions = []
        ids = input_ids[b].tolist()
        
        # Find language latent positions
        if latent_id is not None and latent_id in ids:
            for i, tid in enumerate(ids):
                if tid == latent_id:
                    positions.append(i)
        
        # Find visual latent positions
        if latent_vis_id is not None and latent_vis_id in ids:
            for i, tid in enumerate(ids):
                if tid == latent_vis_id:
                    positions.append(i)
        
        all_positions.append(positions)
    
    # Pad to max length
    max_latent = max(len(p) for p in all_positions) if all_positions else 0
    if max_latent == 0:
        max_latent = 1  # Avoid empty tensor
    
    padded = torch.full((B, max_latent), -1, dtype=torch.long)
    for b, positions in enumerate(all_positions):
        for i, pos in enumerate(positions):
            padded[b, i] = pos
    
    return padded


if __name__ == "__main__":
    # Quick test
    print("=== Language Decoder Head Test ===")
    
    decoder = LanguageDecoderHead(d_model=2048, vocab_size=151936)
    params = decoder.get_parameter_count()
    print(f"Parameter count: {params['total']:,} total")
    print(f"  Token embedding: {params['token_embedding']:,}")
    print(f"  Pos embedding: {params['pos_embedding']:,}")
    print(f"  Decoder layers: {params['decoder_layers']:,}")
    print(f"  Output proj: {params['output_proj']:,}")
    
    # Simulate forward pass
    B, num_latent, d_model = 2, 6, 2048  # 4 visual + 2 language = 6 latent
    seq_len = 50  # CoT length
    
    latent_states = torch.randn(B, num_latent, d_model)
    target_ids = torch.randint(0, 1000, (B, seq_len))
    
    logits, loss = decoder(latent_states, target_ids)
    print(f"\n  Input: latent_states {latent_states.shape}, target_ids {target_ids.shape}")
    print(f"  Output: logits {logits.shape}, loss {loss.item():.4f}")
    print(f"  Memory: ~{sum(p.numel() * 2 for p in decoder.parameters()) / 1e9:.2f} GB (BF16)")
    print("\n✅ Decoder test passed!")
