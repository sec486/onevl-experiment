#!/usr/bin/env python3
"""
W8 方案 B: Three-Stage Training with CE Loss on Discrete Visual Tokens.

Stage 0: Trajectory warmup (5 epochs, no decoders)
Stage 1: Decoder warmup (3 epochs, latent states DETACHED from main model)
Stage 2: Joint training (5 epochs, NO DETACH — gradients flow back to main model)

CE loss on k-means codebook (k=8192) tokens. Expected initial loss ≈ log(8192) ≈ 9.0.
"""
import json, torch, time, os, sys, signal
import numpy as np
from PIL import Image

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from language_decoder import LanguageDecoderHead
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/output/w8_codebook/train_with_visual_tokens.jsonl'
CODEBOOK_PATH = '/opt/onevl-experiment/output/w8_codebook/codebook.pt'
OUTPUT_DIR = '/opt/onevl-experiment/output/w8_three_stage'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
MAX_SAMPLES = 1000
GRAD_ACCUM = 8
LR = 2e-5
LAMBDA_LANG = 0.5
LAMBDA_VIS = 0.5  # CE loss is same scale as traj — natural balance!

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

emergency_state = {'stage': 0, 'epoch': 0, 'step': 0}
def save_emergency(signum, frame):
    with open(os.path.join(OUTPUT_DIR, 'emergency_state.json'), 'w') as f:
        json.dump(emergency_state, f)
signal.signal(signal.SIGTERM, save_emergency)
signal.signal(signal.SIGINT, save_emergency)

print('=' * 60)
print('  W8 方案 B: Three-Stage CE Loss Training')
print('  Stage 0: trajectory warmup (5 ep)')
print('  Stage 1: decoder warmup + detach (3 ep)')
print('  Stage 2: joint NO detach (5 ep)')
print('=' * 60)

# Load data with visual tokens
samples = []
with open(DATA_PATH) as f:
    for line in f:
        if line.strip():
            s = json.loads(line)
            if s.get('future_visual_tokens') is not None:
                samples.append(s)
            if len(samples) >= MAX_SAMPLES:
                break
print(f"Loaded {len(samples)} samples (all have future visual tokens)")

# Load codebook
cb_data = torch.load(CODEBOOK_PATH)
CODEBOOK_SIZE = cb_data['k']
PATCH_DIM = cb_data['dim']
NUM_PATCHES = cb_data['num_patches_per_frame']
print(f"Codebook: k={CODEBOOK_SIZE}, dim={PATCH_DIM}, patches/frame={NUM_PATCHES}")

# Load model
device = 'cuda'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
).to(device)

lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                         modules_to_save=["embed_tokens"], task_type=TaskType.CAUSAL_LM, bias="none")
model = get_peft_model(model, lora_config)
print(f"Main model trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

hidden_size = 2048
vocab_size = len(tokenizer)

# Language decoder
lang_decoder = LanguageDecoderHead(d_model=hidden_size, n_heads=8, n_layers=2,
                                    vocab_size=vocab_size, max_cot_length=128).to(device).to(torch.bfloat16)

# Visual decoder with CE loss (cross-attention + classification head)
class VisualDecoderCE(nn.Module):
    """Predicts discrete visual token IDs from visual latent states via cross-attention."""
    def __init__(self, d_model=2048, codebook_size=8192, num_patches=784, num_queries=64):
        super().__init__()
        self.num_queries = num_queries
        self.codebook_size = codebook_size
        self.num_patches = num_patches
        # Learnable queries (predict groups of patches)
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        # Cross-attention decoder
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*2, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=3)
        self.ln = nn.LayerNorm(d_model)
        # Classification head: project to codebook size
        self.head = nn.Linear(d_model, codebook_size)
    
    def forward(self, vis_latent, gt_token_ids=None):
        """
        vis_latent: (B, 4, d_model) visual latent hidden states
        gt_token_ids: (B, num_patches) discrete GT token IDs for CE loss
        Returns: loss (scalar) if gt_token_ids provided, else logits
        """
        B = vis_latent.shape[0]
        q = self.queries.expand(B, -1, -1).to(vis_latent.dtype)
        decoded = self.decoder(q, vis_latent)  # (B, num_queries, d_model)
        decoded = self.ln(decoded)
        logits = self.head(decoded)  # (B, num_queries, codebook_size)
        
        if gt_token_ids is not None:
            # Pool GT tokens to match num_queries (average groups)
            # gt_token_ids: (B, num_patches) → need to reduce to (B, num_queries)
            # Take every (num_patches // num_queries) token as representative
            stride = self.num_patches // self.num_queries
            gt_subsampled = gt_token_ids[:, ::stride][:, :self.num_queries]  # (B, num_queries)
            loss = F.cross_entropy(logits.view(-1, self.codebook_size), gt_subsampled.view(-1))
            return loss
        return logits

vis_decoder = VisualDecoderCE(d_model=hidden_size, codebook_size=CODEBOOK_SIZE, 
                               num_patches=NUM_PATCHES, num_queries=64).to(device).to(torch.bfloat16)
print(f"Lang decoder: {sum(p.numel() for p in lang_decoder.parameters()):,}")
print(f"Vis decoder (CE): {sum(p.numel() for p in vis_decoder.parameters()):,}")

# Default CoT target
default_cot_ids = tokenizer("The vehicle maintains safe trajectory based on driving conditions.",
                            return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids']

# Fix message format for processor (BF-035 lesson)
def fix_msg(sample):
    msgs = json.loads(json.dumps(sample['messages']))
    for msg in msgs:
        if isinstance(msg.get('content'), list):
            for item in msg['content']:
                if item.get('type') == 'image' and 'image' in item:
                    item['path'] = item.pop('image')
    return msgs

def prepare_inputs(sample):
    """Prepare model inputs from a sample."""
    messages = fix_msg(sample)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    img_path = sample.get('images', [None])[0]
    image = Image.open(img_path).convert('RGB').resize((448, 448)) if img_path and os.path.exists(img_path) else None
    if image:
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    return inputs

# ==================== TRAINING ====================
total_steps = 0
start_time = time.time()

def run_stage(stage_num, num_epochs, train_main, train_decoders, detach_latent, use_vis):
    """Run one training stage."""
    global total_steps
    
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}: epochs={num_epochs}, main={'ON' if train_main else 'OFF'}, "
          f"decoders={'ON' if train_decoders else 'OFF'}, detach={'YES' if detach_latent else 'NO'}, vis={'ON' if use_vis else 'OFF'}")
    print(f"{'='*60}")
    
    # Set trainable components
    if train_main:
        model.train()
        main_params = [p for p in model.parameters() if p.requires_grad]
    else:
        model.eval()
        main_params = []
    
    if train_decoders:
        lang_decoder.train(); vis_decoder.train()
        decoder_params = list(lang_decoder.parameters()) + list(vis_decoder.parameters())
    else:
        lang_decoder.eval(); vis_decoder.eval()
        decoder_params = []
    
    all_params = main_params + decoder_params
    if not all_params:
        print("  No trainable params — skipping stage")
        return
    
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)
    best_loss = float('inf')
    errors = 0
    
    for epoch in range(num_epochs):
        ep_traj, ep_lang, ep_vis = [], [], []
        optimizer.zero_grad()
        
        for idx, sample in enumerate(samples):
            try:
                inputs = prepare_inputs(sample)
            except Exception as e:
                errors += 1
                continue
            
            try:
                if train_main:
                    outputs = model(**inputs, output_hidden_states=True)
                else:
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                
                hidden = outputs.hidden_states[-1]
                seq_len = hidden.shape[1]
                
                # Trajectory loss (only if training main model)
                traj_loss = torch.tensor(0.0, device=device)
                if train_main:
                    logits = outputs.logits
                    labels = inputs['input_ids'].clone()
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    traj_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Latent states
                if detach_latent:
                    lang_latent = hidden[:, max(0, seq_len-4):max(2, seq_len-2), :].detach()
                    vis_latent = hidden[:, max(0, seq_len-8):max(4, seq_len-4), :].detach()
                else:
                    lang_latent = hidden[:, max(0, seq_len-4):max(2, seq_len-2), :]
                    vis_latent = hidden[:, max(0, seq_len-8):max(4, seq_len-4), :]
                
                # Language decoder loss
                lang_loss = torch.tensor(0.0, device=device)
                if train_decoders:
                    _, lang_loss = lang_decoder(lang_latent, default_cot_ids.to(device))
                
                # Visual decoder loss (CE on discrete tokens!)
                vis_loss = torch.tensor(0.0, device=device)
                if use_vis and train_decoders:
                    gt_tokens = torch.tensor([sample['future_visual_tokens']], dtype=torch.long, device=device)
                    vis_loss = vis_decoder(vis_latent, gt_tokens)
                
                # Total loss
                total_loss = traj_loss + LAMBDA_LANG * lang_loss + LAMBDA_VIS * vis_loss
                if total_loss.requires_grad:
                    (total_loss / GRAD_ACCUM).backward()
                
                ep_traj.append(traj_loss.item())
                ep_lang.append(lang_loss.item())
                ep_vis.append(vis_loss.item())
                
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); optimizer.zero_grad(); errors += 1; continue
            except Exception as e:
                errors += 1
                if errors <= 5: print(f"  Err: {type(e).__name__}: {str(e)[:60]}")
                continue
            
            if (idx + 1) % GRAD_ACCUM == 0 and all_params:
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step(); optimizer.zero_grad()
                total_steps += 1
                emergency_state.update({'stage': stage_num, 'epoch': epoch, 'step': total_steps})
                
                if total_steps % 25 == 0:
                    print(f"  S{stage_num} step={total_steps} traj={np.mean(ep_traj[-200:]):.3f} lang={np.mean(ep_lang[-200:]):.3f} vis={np.mean(ep_vis[-200:]):.2f} err={errors} t={time.time()-start_time:.0f}s")
        
        # Epoch summary
        if ep_traj:
            avg_t, avg_l, avg_v = np.mean(ep_traj), np.mean(ep_lang), np.mean(ep_vis) if ep_vis else 0
            total_avg = avg_t + LAMBDA_LANG * avg_l + LAMBDA_VIS * avg_v
            print(f"\n  S{stage_num} Epoch {epoch+1}/{num_epochs}: traj={avg_t:.3f} lang={avg_l:.3f} vis={avg_v:.2f} total={total_avg:.3f} err={errors}")
            
            # Checkpoint
            model.save_pretrained(os.path.join(CHECKPOINT_DIR, f'stage{stage_num}-epoch{epoch+1}'))
            if total_avg < best_loss:
                best_loss = total_avg
                model.save_pretrained(os.path.join(OUTPUT_DIR, 'best'))
                torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'lang_decoder.pt'))
                torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'vis_decoder.pt'))
            
            with open(os.path.join(OUTPUT_DIR, 'partial_results.json'), 'w') as f:
                json.dump({'stage': stage_num, 'epoch': epoch+1, 'traj': avg_t, 'lang': avg_l, 'vis': avg_v, 'best': best_loss, 'errors': errors, 'total_steps': total_steps}, f, indent=2)
    
    return best_loss

# ===== Execute Three Stages =====
# Stage 0: Trajectory warmup (main model only, no decoders)
s0_loss = run_stage(stage_num=0, num_epochs=5, train_main=True, train_decoders=False, detach_latent=True, use_vis=False)

# Stage 1: Decoder warmup (decoders only, main model frozen, latent DETACHED)
s1_loss = run_stage(stage_num=1, num_epochs=3, train_main=False, train_decoders=True, detach_latent=True, use_vis=True)

# Stage 2: Joint training (everything, NO DETACH!)
s2_loss = run_stage(stage_num=2, num_epochs=5, train_main=True, train_decoders=True, detach_latent=False, use_vis=True)

# Final save
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"  THREE-STAGE TRAINING COMPLETE")
print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Total steps: {total_steps}")
print(f"  Stage losses: S0={s0_loss:.4f}, S1={s1_loss:.4f}, S2={s2_loss:.4f}")
print(f"{'='*60}")

model.save_pretrained(OUTPUT_DIR)
torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))
torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'vis_decoder.pt'))

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump({'method': 'three_stage_CE_codebook', 'codebook_k': CODEBOOK_SIZE, 'stages': '0+1+2',
               'total_steps': total_steps, 'time_sec': elapsed,
               's0_loss': s0_loss, 's1_loss': s1_loss, 's2_loss': s2_loss}, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'TRAINING_COMPLETE.txt'), 'w') as f:
    f.write(f"Done {time.strftime('%Y-%m-%d %H:%M:%S')}\nS0={s0_loss:.4f} S1={s1_loss:.4f} S2={s2_loss:.4f}\nsteps={total_steps}\n")
print("TRAINING_COMPLETE.")
