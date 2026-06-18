#!/usr/bin/env python3
"""
W8 方案 C: Three-Stage Training with Emu3 Discrete Visual Tokens (v5)
=====================================================================
Based on the WORKING c-lang script pattern (w8_train_clang_1000.py).
Key: resize images to 448x448, use apply_chat_template with raw messages.

Stage 0: Trajectory warmup (main model only)
Stage 1: Decoder warmup (decoders only, DETACHED)
Stage 2: Joint training (all, NO DETACH)
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# Config
# ============================================================
MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
VIS_TOKENS_PATH = '/opt/onevl-experiment/data/emu3_visual_tokens_1000.pt'
OUTPUT_DIR = '/opt/onevl-experiment/output/w8c_three_stage'
LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log_v5.txt')

MAX_SAMPLES = 1000
GRAD_ACCUM = 8
LR = 2e-5
CODEBOOK_SIZE = 32768
NUM_VIS_QUERIES = 64  # Small to fit 22GB
LAMBDA_LANG = 0.5
LAMBDA_VIS = 0.5

STAGE_0_EPOCHS = 5
STAGE_1_EPOCHS = 3
STAGE_2_EPOCHS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ============================================================
# Visual Decoder (small, fp16-safe)
# ============================================================
class VisualDecoderCE(nn.Module):
    def __init__(self, hidden_dim=2048, codebook_size=32768, num_queries=64, num_layers=2, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.codebook_size = codebook_size
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, codebook_size)

    def forward(self, vis_latent, gt_token_ids=None):
        B = vis_latent.shape[0]
        queries = self.queries.expand(B, -1, -1)
        decoded = self.decoder(queries, vis_latent)
        logits = self.output_proj(decoded)
        if gt_token_ids is not None:
            loss = F.cross_entropy(logits.view(-1, self.codebook_size), gt_token_ids.view(-1))
            return loss
        return logits

# ============================================================
# Language Decoder (1 layer, lightweight)
# ============================================================
class LangDecoderHead(nn.Module):
    def __init__(self, d_model=2048, n_heads=8, vocab_size=151936, max_len=64):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, latent, target_ids):
        B, L = target_ids.shape
        d = latent.shape[-1]
        causal_mask = torch.triu(torch.ones(L, L, device=target_ids.device), diagonal=1).bool()
        tgt = torch.zeros(B, L, d, device=latent.device, dtype=latent.dtype)
        decoded = self.decoder(tgt, latent, tgt_mask=causal_mask)
        logits = self.output_proj(decoded)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)
        return loss

# ============================================================
# Helpers
# ============================================================
def subsample_tokens(full_tokens, target=64):
    """Subsample (96,170) → (target,) via uniform grid."""
    flat = full_tokens.flatten()
    indices = torch.linspace(0, len(flat)-1, target).long()
    return flat[indices]

def process_sample(sample, processor, device):
    """Process one sample using the WORKING pattern from c-lang script."""
    messages = sample['messages']
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if text is None:
        return None
    
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
    return inputs

# ============================================================
# Main
# ============================================================
def main():
    log("=" * 60)
    log("  W8 方案 C: Three-Stage Training v5 (Emu3 CE, working pattern)")
    log("=" * 60)
    
    device = 'cuda'
    
    # Load visual tokens
    log("\n=== Loading Emu3 visual tokens ===")
    vis_tokens_dict = torch.load(VIS_TOKENS_PATH, weights_only=False)
    log(f"  {len(vis_tokens_dict)} scene encodings")
    
    # Load data
    log("\n=== Loading data ===")
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= MAX_SAMPLES:
                    break
    log(f"  {len(samples)} samples")
    
    # Match samples to visual tokens
    for s in samples:
        s['_scene'] = s.get('scene', '')
        s['_has_vis'] = s['_scene'] in vis_tokens_dict
    
    n_vis = sum(1 for s in samples if s['_has_vis'])
    log(f"  Matched visual tokens: {n_vis}/{len(samples)}")
    
    # Load model (same as working c-lang script)
    log("\n=== Loading Cosmos-Reason2-2B ===")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to(device)
    log(f"  Loaded in {time.time()-t0:.1f}s")
    
    # LoRA (same as c-lang)
    log("\n=== LoRA ===")
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    log(f"  {trainable/1e6:.1f}M trainable / {total_p/1e6:.1f}M total ({100*trainable/total_p:.1f}%)")
    
    hidden_dim = 2048
    vocab_size = len(tokenizer)
    
    # Decoders (small, bfloat16)
    log("\n=== Decoders ===")
    vis_decoder = VisualDecoderCE(
        hidden_dim=hidden_dim, codebook_size=CODEBOOK_SIZE,
        num_queries=NUM_VIS_QUERIES, num_layers=2, num_heads=8
    ).to(device).to(torch.bfloat16)
    
    lang_decoder = LangDecoderHead(
        d_model=hidden_dim, n_heads=8, vocab_size=vocab_size, max_len=64
    ).to(device).to(torch.bfloat16)
    
    log(f"  Visual decoder: {sum(p.numel() for p in vis_decoder.parameters())/1e6:.1f}M (CE on {CODEBOOK_SIZE} codebook, {NUM_VIS_QUERIES} queries)")
    log(f"  Language decoder: {sum(p.numel() for p in lang_decoder.parameters())/1e6:.1f}M")
    
    # CoT target (fixed for now — same as c-lang script)
    default_cot = "The vehicle should maintain safe trajectory based on the current driving conditions and road geometry."
    default_cot_ids = tokenizer(default_cot, return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids'].to(device)
    
    # ============================================================
    # STAGE 0: Trajectory Warmup
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 0: Trajectory Warmup ({STAGE_0_EPOCHS} epochs)")
    log(f"{'='*60}")
    
    vis_decoder.eval()
    lang_decoder.eval()
    for p in vis_decoder.parameters(): p.requires_grad = False
    for p in lang_decoder.parameters(): p.requires_grad = False
    
    optimizer_s0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    
    total_steps = 0
    t_stage = time.time()
    
    for epoch in range(STAGE_0_EPOCHS):
        epoch_losses = []
        optimizer_s0.zero_grad()
        errors = 0
        
        for idx, sample in enumerate(samples):
            try:
                inputs = process_sample(sample, processor, device)
                if inputs is None:
                    errors += 1
                    continue
                
                outputs = model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                labels = inputs['input_ids'].clone()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                traj_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                (traj_loss / GRAD_ACCUM).backward()
                epoch_losses.append(traj_loss.item())
                
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer_s0.step()
                    optimizer_s0.zero_grad()
                    total_steps += 1
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer_s0.zero_grad()
                errors += 1
                continue
            except Exception as e:
                errors += 1
                if errors <= 3 and epoch == 0:
                    log(f"  S0 err {idx}: {type(e).__name__}: {str(e)[:100]}")
                continue
        
        avg = np.mean(epoch_losses) if epoch_losses else 0
        log(f"  S0 Epoch {epoch+1}/{STAGE_0_EPOCHS}: traj={avg:.4f}, samples={len(epoch_losses)}, errors={errors}, steps={total_steps}")
        torch.cuda.empty_cache()
    
    s0_time = time.time() - t_stage
    log(f"  Stage 0 done: {s0_time/60:.1f} min")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 's0_model.pt'))
    
    # ============================================================
    # STAGE 1: Decoder Warmup (DETACHED)
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 1: Decoder Warmup ({STAGE_1_EPOCHS} epochs, DETACHED)")
    log(f"{'='*60}")
    
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    vis_decoder.train()
    lang_decoder.train()
    for p in vis_decoder.parameters(): p.requires_grad = True
    for p in lang_decoder.parameters(): p.requires_grad = True
    
    optimizer_s1 = torch.optim.AdamW(
        list(vis_decoder.parameters()) + list(lang_decoder.parameters()),
        lr=LR * 2
    )
    
    t_stage = time.time()
    
    for epoch in range(STAGE_1_EPOCHS):
        epoch_vis, epoch_lang = [], []
        optimizer_s1.zero_grad()
        errors = 0
        
        for idx, sample in enumerate(samples):
            try:
                inputs = process_sample(sample, processor, device)
                if inputs is None:
                    errors += 1
                    continue
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]  # (1, seq, 2048) DETACHED
                
                seq_len = hidden.shape[1]
                vis_latent = hidden[:, max(0, seq_len-8):max(4, seq_len-4), :]
                lang_latent = hidden[:, max(0, seq_len-4):max(2, seq_len-2), :]
                
                # Visual decoder (only if scene has tokens)
                vis_loss = torch.tensor(0.0, device=device)
                if sample['_has_vis']:
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES)
                    vis_gt = vis_gt.unsqueeze(0).to(device)
                    vis_loss = vis_decoder(vis_latent, vis_gt)
                
                # Language decoder
                lang_loss = lang_decoder(lang_latent, default_cot_ids)
                
                total = (LAMBDA_VIS * vis_loss + LAMBDA_LANG * lang_loss) / GRAD_ACCUM
                total.backward()
                
                if vis_loss.item() > 0:
                    epoch_vis.append(vis_loss.item())
                epoch_lang.append(lang_loss.item())
                
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(list(vis_decoder.parameters()) + list(lang_decoder.parameters()), 1.0)
                    optimizer_s1.step()
                    optimizer_s1.zero_grad()
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer_s1.zero_grad()
                errors += 1
                continue
            except Exception as e:
                errors += 1
                if errors <= 3 and epoch == 0:
                    log(f"  S1 err {idx}: {type(e).__name__}: {str(e)[:100]}")
                continue
        
        avg_v = np.mean(epoch_vis) if epoch_vis else 0
        avg_l = np.mean(epoch_lang) if epoch_lang else 0
        log(f"  S1 Epoch {epoch+1}/{STAGE_1_EPOCHS}: vis={avg_v:.2f}, lang={avg_l:.4f}, vis_samples={len(epoch_vis)}, lang_samples={len(epoch_lang)}, errors={errors}")
        torch.cuda.empty_cache()
    
    s1_time = time.time() - t_stage
    log(f"  Stage 1 done: {s1_time/60:.1f} min")
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 's1_vis_decoder.pt'))
    torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 's1_lang_decoder.pt'))
    
    # ============================================================
    # STAGE 2: Joint (NO DETACH)
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 2: Joint Training ({STAGE_2_EPOCHS} epochs, NO DETACH)")
    log(f"{'='*60}")
    
    model.train()
    for name, p in model.named_parameters():
        if 'lora' in name or 'embed_tokens' in name:
            p.requires_grad = True
    vis_decoder.train()
    lang_decoder.train()
    
    all_params = (
        [p for p in model.parameters() if p.requires_grad] +
        list(vis_decoder.parameters()) +
        list(lang_decoder.parameters())
    )
    optimizer_s2 = torch.optim.AdamW(all_params, lr=LR * 0.5)
    
    t_stage = time.time()
    
    for epoch in range(STAGE_2_EPOCHS):
        epoch_traj, epoch_vis, epoch_lang = [], [], []
        optimizer_s2.zero_grad()
        errors = 0
        
        for idx, sample in enumerate(samples):
            try:
                inputs = process_sample(sample, processor, device)
                if inputs is None:
                    errors += 1
                    continue
                
                # Forward WITH gradient (NO DETACH!)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Trajectory loss
                logits = outputs.logits
                labels = inputs['input_ids'].clone()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                traj_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Hidden states for decoders (NOT detached in Stage 2!)
                hidden = outputs.hidden_states[-1]
                seq_len = hidden.shape[1]
                vis_latent = hidden[:, max(0, seq_len-8):max(4, seq_len-4), :]
                lang_latent = hidden[:, max(0, seq_len-4):max(2, seq_len-2), :]
                
                # Visual decoder
                vis_loss = torch.tensor(0.0, device=device)
                if sample['_has_vis']:
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES)
                    vis_gt = vis_gt.unsqueeze(0).to(device)
                    vis_loss = vis_decoder(vis_latent, vis_gt)
                
                # Language decoder
                lang_loss = lang_decoder(lang_latent, default_cot_ids)
                
                # Total (all CE, naturally balanced!)
                total = (traj_loss + LAMBDA_VIS * vis_loss + LAMBDA_LANG * lang_loss) / GRAD_ACCUM
                total.backward()
                
                epoch_traj.append(traj_loss.item())
                if vis_loss.item() > 0:
                    epoch_vis.append(vis_loss.item())
                epoch_lang.append(lang_loss.item())
                
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer_s2.step()
                    optimizer_s2.zero_grad()
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer_s2.zero_grad()
                errors += 1
                continue
            except Exception as e:
                errors += 1
                if errors <= 3 and epoch == 0:
                    log(f"  S2 err {idx}: {type(e).__name__}: {str(e)[:100]}")
                continue
        
        avg_t = np.mean(epoch_traj) if epoch_traj else 0
        avg_v = np.mean(epoch_vis) if epoch_vis else 0
        avg_l = np.mean(epoch_lang) if epoch_lang else 0
        log(f"  S2 Epoch {epoch+1}/{STAGE_2_EPOCHS}: traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}, "
            f"samples={len(epoch_traj)}, vis_active={len(epoch_vis)}, errors={errors}")
        torch.cuda.empty_cache()
        
        # Save checkpoint each epoch
        with open(os.path.join(OUTPUT_DIR, 'partial_results.json'), 'w') as f:
            json.dump({'stage': 2, 'epoch': epoch+1, 'traj': avg_t, 'vis': avg_v, 'lang': avg_l,
                      'samples': len(epoch_traj), 'vis_active': len(epoch_vis)}, f, indent=2)
    
    s2_time = time.time() - t_stage
    log(f"  Stage 2 done: {s2_time/60:.1f} min")
    
    # Save final
    model.save_pretrained(os.path.join(OUTPUT_DIR, 'final'))
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'final_vis_decoder.pt'))
    torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'final_lang_decoder.pt'))
    
    # ============================================================
    # Summary
    # ============================================================
    total_time = s0_time + s1_time + s2_time
    log(f"\n{'='*60}")
    log(f"  COMPLETE")
    log(f"{'='*60}")
    log(f"  S0: {s0_time/60:.1f} min | S1: {s1_time/60:.1f} min | S2: {s2_time/60:.1f} min | Total: {total_time/60:.1f} min")
    log(f"  Final: traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}")
    log(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    log(f"  Key Q: vis_loss initial ~{np.log(CODEBOOK_SIZE):.1f} → final {avg_v:.2f} (drop = {100*(np.log(CODEBOOK_SIZE)-avg_v)/np.log(CODEBOOK_SIZE):.1f}%)")
    
    with open(os.path.join(OUTPUT_DIR, 'TRAINING_COMPLETE.txt'), 'w') as f:
        f.write(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {total_time/60:.1f} min\n")
        f.write(f"traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}\n")

if __name__ == "__main__":
    main()
