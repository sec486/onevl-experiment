#!/usr/bin/env python3
"""
W8 c-full PRAGMATIC: Apply 3 of 4 BF-034 fixes (skip latent token insertion).

Fixes applied:
  1. Loss normalization (vis_loss scaled to traj_loss magnitude)
  3. vis_latent.detach() (visual gradients don't corrupt LoRA)
  4. Future frame (t+1.0s) as GT target

NOT applied (BF-035 blocks it):
  2. Real latent token positions (still uses tail positions)

This gives us 3/4 fixes and ACTUALLY TRAINS (0 errors, like the original v4).
If ADE improves vs broken c-full (2.81m) → fixes 1+3+4 help.
If ADE matches c-lang (1.17m) → latent positions don't matter much.
"""
import json, torch, time, os, sys, signal, re
import numpy as np
from PIL import Image

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from language_decoder import LanguageDecoderHead
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
OUTPUT_DIR = '/opt/onevl-experiment/output/w8_cfull_pragmatic'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LAMBDA_LANG = 0.5
NUM_EPOCHS = 5
MAX_SAMPLES = 1000
GRAD_ACCUM = 8
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

emergency_state = {'step': 0, 'epoch': 0}
def save_emergency(signum, frame):
    with open(os.path.join(OUTPUT_DIR, 'emergency_state.json'), 'w') as f:
        json.dump(emergency_state, f)
signal.signal(signal.SIGTERM, save_emergency)
signal.signal(signal.SIGINT, save_emergency)

print('=' * 60)
print('  W8 c-full PRAGMATIC (fixes 1+3+4, skip latent tokens)')
print('  FIX 1: loss normalization')
print('  FIX 3: vis_latent.detach()')
print('  FIX 4: future frame t+1.0s')
print(f'  Samples: {MAX_SAMPLES}, Epochs: {NUM_EPOCHS}')
print('=' * 60)

samples = []
with open(DATA_PATH) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))
            if len(samples) >= MAX_SAMPLES:
                break
print(f"Loaded {len(samples)} samples")

# Future frames (FIX 4)
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'
TRAINVAL_ALT = '/opt/onevl-experiment/OneVL_training/navsim_v1.1_all/dataset/sensor_blobs/trainval'
img_base = TRAINVAL if os.path.exists(TRAINVAL) else TRAINVAL_ALT
future_frames = {}
for idx, s in enumerate(samples):
    scene, frame_idx = s.get('scene', ''), s.get('frame_idx', 0)
    cam_dir = os.path.join(img_base, scene, 'CAM_F0')
    if os.path.exists(cam_dir):
        frames = sorted(os.listdir(cam_dir))
        if frame_idx + 2 < len(frames):
            future_frames[idx] = os.path.join(cam_dir, frames[frame_idx + 2])
print(f"Future frames: {len(future_frames)}/{len(samples)}")

device = 'cuda'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
).to(device)

lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                         modules_to_save=["embed_tokens"], task_type=TaskType.CAUSAL_LM, bias="none")
model = get_peft_model(model, lora_config)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

vis_encoder = model.base_model.model.model.visual
hidden_size, vocab_size, vit_dim = 2048, len(tokenizer), 1024

lang_decoder = LanguageDecoderHead(d_model=hidden_size, n_heads=8, n_layers=2,
                                    vocab_size=vocab_size, max_cot_length=128).to(device).to(torch.bfloat16)

class VisualDecoder(nn.Module):
    def __init__(self, d_model=2048, vit_dim=1024, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model*2, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.proj = nn.Linear(d_model, vit_dim)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, vis_latent, gt_patches):
        B = vis_latent.shape[0]
        q = self.queries.expand(B,-1,-1).to(vis_latent.dtype)
        decoded = self.decoder(q, vis_latent)
        pred = self.proj(self.ln(decoded))
        gt_pooled = F.adaptive_avg_pool1d(gt_patches.transpose(1,2), self.num_queries).transpose(1,2)
        return F.mse_loss(pred, gt_pooled.to(pred.dtype))

vis_decoder = VisualDecoder(d_model=hidden_size, vit_dim=vit_dim).to(device).to(torch.bfloat16)
print(f"Decoders: lang={sum(p.numel() for p in lang_decoder.parameters()):,}, vis={sum(p.numel() for p in vis_decoder.parameters()):,}")

# FIX 3: Separate optimizer for visual decoder (doesn't touch LoRA)
main_params = [p for p in model.parameters() if p.requires_grad] + list(lang_decoder.parameters())
vis_params = list(vis_decoder.parameters())
opt_main = torch.optim.AdamW(main_params, lr=LR, weight_decay=0.01)
opt_vis = torch.optim.AdamW(vis_params, lr=LR*5, weight_decay=0.01)

default_cot_ids = tokenizer("The vehicle maintains safe trajectory based on driving conditions.",
                            return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids']

print(f"\n=== Training ===")
model.train(); lang_decoder.train(); vis_decoder.train()
total_steps = 0; start_time = time.time(); best_loss = float('inf'); vis_active = 0; errors = 0

for epoch in range(NUM_EPOCHS):
    ep_traj, ep_lang, ep_vis = [], [], []
    opt_main.zero_grad(); opt_vis.zero_grad()

    for idx, sample in enumerate(samples):
        try:
            messages = sample['messages']
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            img_path = sample.get('images', [None])[0]
            image = Image.open(img_path).convert('RGB').resize((448, 448)) if img_path and os.path.exists(img_path) else None
            if image:
                inputs = processor(text=[text], images=[image], return_tensors="pt",
                                   padding=True, truncation=True, max_length=512).to(device)
            else:
                inputs = processor(text=[text], return_tensors="pt",
                                   padding=True, truncation=True, max_length=512).to(device)
        except Exception as e:
            errors += 1
            if errors <= 3: print(f"  Prep err: {e}")
            continue

        try:
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            labels = inputs['input_ids'].clone()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            traj_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Use tail positions for latent states (same as original v4)
            hidden = outputs.hidden_states[-1]
            seq_len = hidden.shape[1]
            lang_latent = hidden[:, max(0, seq_len-4):max(2, seq_len-2), :]
            _, lang_loss = lang_decoder(lang_latent, default_cot_ids.to(device))

            # FIX 3: DETACH + FIX 4: FUTURE FRAME
            vis_loss = torch.tensor(0.0, device=device)
            if idx in future_frames and 'pixel_values' in inputs:
                vis_latent = hidden[:, max(0, seq_len-8):max(4, seq_len-4), :].detach()  # FIX 3
                future_img = Image.open(future_frames[idx]).convert('RGB').resize((448, 448))
                fut_inputs = processor(images=[future_img], return_tensors='pt').to(device)
                with torch.no_grad():
                    fut_vit = vis_encoder(fut_inputs['pixel_values'], grid_thw=fut_inputs.get('image_grid_thw'))
                    gt_patches = fut_vit.last_hidden_state
                    if gt_patches.dim() == 2: gt_patches = gt_patches.unsqueeze(0)
                vis_loss = vis_decoder(vis_latent, gt_patches)
                vis_active += 1

            # FIX 1: Main loss only for shared params. Visual normalized separately.
            main_loss = traj_loss + LAMBDA_LANG * lang_loss
            (main_loss / GRAD_ACCUM).backward()

            if vis_loss.item() > 0:
                vis_scale = max(0.001, traj_loss.item() / (vis_loss.item() + 1e-8))
                (vis_loss * vis_scale * 0.5 / GRAD_ACCUM).backward()

            ep_traj.append(traj_loss.item()); ep_lang.append(lang_loss.item()); ep_vis.append(vis_loss.item())

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); opt_main.zero_grad(); opt_vis.zero_grad(); errors += 1; continue
        except Exception as e:
            errors += 1
            if errors <= 5: print(f"  Err: {type(e).__name__}: {str(e)[:60]}")
            continue

        if (idx+1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            torch.nn.utils.clip_grad_norm_(vis_params, 1.0)
            opt_main.step(); opt_vis.step(); opt_main.zero_grad(); opt_vis.zero_grad()
            total_steps += 1; emergency_state.update({'step': total_steps, 'epoch': epoch})
            if total_steps % 25 == 0:
                print(f"  step={total_steps} traj={np.mean(ep_traj[-200:]):.3f} lang={np.mean(ep_lang[-200:]):.3f} vis={np.mean(ep_vis[-200:]):.1f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB t={time.time()-start_time:.0f}s err={errors} vis={vis_active}")

    if ep_traj:
        avg_t, avg_l, avg_v = np.mean(ep_traj), np.mean(ep_lang), np.mean(ep_vis) if ep_vis else 0
        total_avg = avg_t + LAMBDA_LANG * avg_l
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}: traj={avg_t:.3f} lang={avg_l:.4f} vis={avg_v:.1f} err={errors} vis={vis_active} t={time.time()-start_time:.0f}s")
        model.save_pretrained(os.path.join(CHECKPOINT_DIR, f'epoch-{epoch+1}'))
        if total_avg < best_loss:
            best_loss = total_avg
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'best'))
            torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'lang_decoder.pt'))
            torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'vis_decoder.pt'))
            print(f"  Saved best ({best_loss:.4f})")
        with open(os.path.join(OUTPUT_DIR, 'partial_results.json'), 'w') as f:
            json.dump({'epoch': epoch+1, 'traj': avg_t, 'lang': avg_l, 'vis': avg_v, 'best': best_loss, 'errors': errors, 'vis_active': vis_active}, f, indent=2)

elapsed = time.time() - start_time
print(f"\n=== Done! {elapsed:.0f}s ({elapsed/60:.1f}min) steps={total_steps} best={best_loss:.4f} err={errors} vis={vis_active} ===")
model.save_pretrained(OUTPUT_DIR)
torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))
torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'vis_decoder.pt'))
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump({'version': 'pragmatic', 'fixes': ['loss_norm','vis_detach','future_frame'], 'skipped': ['latent_tokens_BF035'], 'best_loss': best_loss, 'errors': errors, 'vis_active': vis_active, 'time': elapsed}, f, indent=2)
with open(os.path.join(OUTPUT_DIR, 'TRAINING_COMPLETE.txt'), 'w') as f:
    f.write(f"Done {time.strftime('%Y-%m-%d %H:%M:%S')}\nbest={best_loss:.4f} err={errors} vis={vis_active}\n")
print("TRAINING_COMPLETE.")
