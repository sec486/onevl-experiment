#!/usr/bin/env python3
"""
W7 v3: Train c-full with REAL trajectory GT (BF-032 fix).
Fixed: Visual decoder now correctly accesses ViT encoder output.

Key fixes from v2:
- Visual encoder at model.base_model.model.model.visual (not model.base_model.model.visual)
- ViT output is BaseModelOutputWithDeepstackFeatures, access via .last_hidden_state
- Output shape: (num_patches, 1024) where num_patches=784 for 448x448 image
"""
import json, torch, time, os, sys, gc
import numpy as np
from PIL import Image

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from language_decoder import LanguageDecoderHead
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
OUTPUT_DIR = '/opt/onevl-experiment/output/w7_cfull_real_traj_v3'
LAMBDA_LANG = 0.5
LAMBDA_VIS = 0.1
NUM_EPOCHS = 5
MAX_SAMPLES = 200
GRAD_ACCUM = 4
LR = 2e-5

print('=' * 60)
print('  W7 v3: c-full with REAL Trajectory GT + Visual Decoder FIX')
print(f'  Lambda: lang={LAMBDA_LANG}, vis={LAMBDA_VIS}')
print(f'  Samples: {MAX_SAMPLES}, Epochs: {NUM_EPOCHS}')
print('=' * 60)

# Load data
print("\n=== Loading data ===")
samples = []
with open(DATA_PATH) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))
            if len(samples) >= MAX_SAMPLES:
                break
print(f"Loaded {len(samples)} samples")

# Load model
print("\n=== Loading model ===")
device = 'cuda'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, attn_implementation="eager",
).to(device)

# Apply LoRA
print("\n=== Applying LoRA (r=64 + embed_tokens) ===")
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["embed_tokens"],
    task_type=TaskType.CAUSAL_LM, bias="none",
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

# Access visual encoder (correct path after LoRA wrapping)
vis_encoder = model.base_model.model.model.visual
print(f"Visual encoder: {type(vis_encoder).__name__}")

hidden_size = 2048
vocab_size = len(tokenizer)
vit_dim = 1024  # ViT output dimension for this model

# Language decoder
print(f"\n=== Initializing decoders ===")
lang_decoder = LanguageDecoderHead(
    d_model=hidden_size, n_heads=8, n_layers=2,
    vocab_size=vocab_size, max_cot_length=128,
).to(device).to(torch.bfloat16)
print(f"Language decoder: {sum(p.numel() for p in lang_decoder.parameters()):,} params")

# Visual decoder
class VisualDecoder(nn.Module):
    """Predicts future frame ViT patch features from visual latent states."""
    def __init__(self, d_model=2048, vit_dim=1024, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 2, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.proj = nn.Linear(d_model, vit_dim)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, vis_latent, gt_patches):
        """
        vis_latent: (B, 4, d_model) visual latent hidden states
        gt_patches: (B, num_patches, vit_dim) ViT output patches
        Returns: MSE loss
        """
        B = vis_latent.shape[0]
        q = self.queries.expand(B, -1, -1).to(vis_latent.dtype)
        decoded = self.decoder(q, vis_latent)
        decoded = self.ln(decoded)
        pred = self.proj(decoded)  # (B, num_queries, vit_dim)
        
        # Pool GT patches to match num_queries
        # gt_patches: (B, N, vit_dim) → (B, num_queries, vit_dim)
        gt_pooled = F.adaptive_avg_pool1d(
            gt_patches.transpose(1, 2), self.num_queries
        ).transpose(1, 2)
        
        return F.mse_loss(pred, gt_pooled.to(pred.dtype))

vis_decoder = VisualDecoder(d_model=hidden_size, vit_dim=vit_dim, num_queries=32).to(device).to(torch.bfloat16)
print(f"Visual decoder: {sum(p.numel() for p in vis_decoder.parameters()):,} params")

# Optimizer
all_params = (
    [p for p in model.parameters() if p.requires_grad] +
    list(lang_decoder.parameters()) +
    list(vis_decoder.parameters())
)
optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)

# Default CoT target for language decoder
default_cot = "The vehicle should maintain safe trajectory based on the current driving conditions and road geometry."
default_cot_ids = tokenizer(default_cot, return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids']

print(f"\n=== Starting Training ===")
print(f"Steps/epoch: {MAX_SAMPLES // GRAD_ACCUM}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.train()
lang_decoder.train()
vis_decoder.train()

total_steps = 0
start_time = time.time()
best_loss = float('inf')
vis_active_count = 0

for epoch in range(NUM_EPOCHS):
    epoch_traj, epoch_lang, epoch_vis = [], [], []
    optimizer.zero_grad()
    
    for idx, sample in enumerate(samples):
        messages = sample['messages']
        
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            img_path = sample.get('images', [None])[0]
            image = None
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB').resize((448, 448))
            
            if image:
                inputs = processor(text=[text], images=[image], return_tensors="pt",
                                   padding=True, truncation=True, max_length=512).to(device)
            else:
                inputs = processor(text=[text], return_tensors="pt",
                                   padding=True, truncation=True, max_length=512).to(device)
        except Exception as e:
            if idx < 2:
                print(f"  Prep error #{idx}: {e}")
            continue
        
        try:
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            
            # 1. Trajectory loss
            logits = outputs.logits
            labels = inputs['input_ids'].clone()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            traj_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 2. Language decoder loss
            hidden = outputs.hidden_states[-1]  # (1, seq_len, 2048)
            seq_len = hidden.shape[1]
            lang_latent = hidden[:, max(0, seq_len-4):max(2, seq_len-2), :]
            cot_ids = default_cot_ids.to(device)
            _, lang_loss = lang_decoder(lang_latent, cot_ids)
            
            # 3. Visual decoder loss — FIXED
            vis_loss = torch.tensor(0.0, device=device)
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                vis_latent = hidden[:, max(0, seq_len-8):max(4, seq_len-4), :]
                
                with torch.no_grad():
                    pv = inputs['pixel_values']
                    grid_thw = inputs.get('image_grid_thw', None)
                    vit_output = vis_encoder(pv, grid_thw=grid_thw)
                    gt_patches = vit_output.last_hidden_state  # (num_patches, vit_dim)
                    if gt_patches.dim() == 2:
                        gt_patches = gt_patches.unsqueeze(0)  # (1, num_patches, vit_dim)
                
                vis_loss = vis_decoder(vis_latent, gt_patches)
                vis_active_count += 1
            
            # Combined loss
            total_loss = traj_loss + LAMBDA_LANG * lang_loss + LAMBDA_VIS * vis_loss
            (total_loss / GRAD_ACCUM).backward()
            
            epoch_traj.append(traj_loss.item())
            epoch_lang.append(lang_loss.item())
            epoch_vis.append(vis_loss.item())
            
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at sample {idx}, skipping")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue
        except Exception as e:
            if idx < 5 or (idx % 50 == 0):
                print(f"  Error #{idx}: {type(e).__name__}: {str(e)[:100]}")
            continue
        
        # Optimizer step
        if (idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_steps += 1
            
            if total_steps % 10 == 0:
                t_avg = np.mean(epoch_traj[-GRAD_ACCUM*10:])
                l_avg = np.mean(epoch_lang[-GRAD_ACCUM*10:])
                v_avg = np.mean(epoch_vis[-GRAD_ACCUM*10:])
                elapsed = time.time() - start_time
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step={total_steps} traj={t_avg:.3f} lang={l_avg:.3f} vis={v_avg:.4f} mem={mem:.1f}GB t={elapsed:.0f}s vis_active={vis_active_count}")
    
    # Epoch summary
    if epoch_traj:
        avg_t = np.mean(epoch_traj)
        avg_l = np.mean(epoch_lang)
        avg_v = np.mean(epoch_vis)
        total_avg = avg_t + LAMBDA_LANG * avg_l + LAMBDA_VIS * avg_v
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}: total={total_avg:.4f} traj={avg_t:.4f} lang={avg_l:.4f} vis={avg_v:.4f} vis_active={vis_active_count}")
        
        if total_avg < best_loss:
            best_loss = total_avg
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'best'))
            torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'lang_decoder.pt'))
            torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'vis_decoder.pt'))
            print(f"  Saved best (loss={best_loss:.4f})")

# Final save
print(f"\n=== Saving ===")
model.save_pretrained(OUTPUT_DIR)
torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))
torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'vis_decoder.pt'))

config = {
    'fix': 'BF-032 — real trajectories + working visual decoder',
    'lambda_lang': LAMBDA_LANG, 'lambda_vis': LAMBDA_VIS,
    'epochs': NUM_EPOCHS, 'samples': len(samples),
    'lr': LR, 'lora_r': 64, 'total_steps': total_steps,
    'best_loss': best_loss, 'vis_active_samples': vis_active_count,
    'vit_dim': vit_dim, 'num_patches_per_image': 784,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

elapsed = time.time() - start_time
print(f"\nDone! {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Steps: {total_steps}, best_loss: {best_loss:.4f}")
print(f"Visual decoder active on {vis_active_count}/{len(samples)*NUM_EPOCHS} samples")
