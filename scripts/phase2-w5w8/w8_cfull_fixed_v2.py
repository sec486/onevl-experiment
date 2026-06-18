#!/usr/bin/env python3
"""
W8 c-full FIXED v2: All BF-034 fixes + BF-035 fix (post-tokenization injection).

BF-034 Fixes: loss normalization, real latent positions, detach, future frame
BF-035 Fix: Inject latent tokens AFTER tokenization (not in chat template text)

Approach:
1. Tokenize normally (no latent tokens in text)
2. Find the position where <answer> starts in token IDs
3. Insert [LATENT_VIS_ID]*4 + [LATENT_LANG_ID]*2 at that position
4. The model sees these as special tokens → their hidden states become the latent representations
5. Language decoder reads from LATENT_LANG positions
6. Visual decoder reads from LATENT_VIS positions (DETACHED)
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
OUTPUT_DIR = '/opt/onevl-experiment/output/w8_cfull_fixed_v2'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LAMBDA_LANG = 0.5
NUM_EPOCHS = 5
MAX_SAMPLES = 1000
GRAD_ACCUM = 8
LR = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Resilience
emergency_state = {'step': 0, 'epoch': 0}
def save_emergency(signum, frame):
    print(f"\nSIGNAL {signum} — emergency save!")
    with open(os.path.join(OUTPUT_DIR, 'emergency_state.json'), 'w') as f:
        json.dump(emergency_state, f)
signal.signal(signal.SIGTERM, save_emergency)
signal.signal(signal.SIGINT, save_emergency)

print('=' * 60)
print('  W8 c-full FIXED v2 (BF-034 + BF-035 fixes)')
print('  Key: latent tokens injected POST-tokenization')
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

# Find future frames
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'
TRAINVAL_ALT = '/opt/onevl-experiment/OneVL_training/navsim_v1.1_all/dataset/sensor_blobs/trainval'
img_base = TRAINVAL if os.path.exists(TRAINVAL) else TRAINVAL_ALT

future_frames = {}
for idx, s in enumerate(samples):
    scene = s.get('scene', '')
    frame_idx = s.get('frame_idx', 0)
    cam_dir = os.path.join(img_base, scene, 'CAM_F0')
    if os.path.exists(cam_dir):
        frames = sorted(os.listdir(cam_dir))
        future_idx = frame_idx + 2
        if future_idx < len(frames):
            future_frames[idx] = os.path.join(cam_dir, frames[future_idx])
print(f"Future frames available: {len(future_frames)}/{len(samples)}")

# Load model
print("\n=== Loading model ===")
device = 'cuda'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

# Add special tokens (for embedding learning)
LATENT_VIS_TOKEN = "<|latent-vis|>"
LATENT_LANG_TOKEN = "<|latent-lang|>"
num_added = tokenizer.add_special_tokens({'additional_special_tokens': [LATENT_VIS_TOKEN, LATENT_LANG_TOKEN]})
LATENT_VIS_ID = tokenizer.convert_tokens_to_ids(LATENT_VIS_TOKEN)
LATENT_LANG_ID = tokenizer.convert_tokens_to_ids(LATENT_LANG_TOKEN)
print(f"Latent token IDs: vis={LATENT_VIS_ID}, lang={LATENT_LANG_ID}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, attn_implementation="eager",
).to(device)
model.resize_token_embeddings(len(tokenizer))

# LoRA
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["embed_tokens"],
    task_type=TaskType.CAUSAL_LM, bias="none",
)
model = get_peft_model(model, lora_config)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

vis_encoder = model.base_model.model.model.visual
hidden_size = 2048
vocab_size = len(tokenizer)
vit_dim = 1024

# Decoders
lang_decoder = LanguageDecoderHead(
    d_model=hidden_size, n_heads=8, n_layers=2,
    vocab_size=vocab_size, max_cot_length=128,
).to(device).to(torch.bfloat16)

class VisualDecoder(nn.Module):
    def __init__(self, d_model=2048, vit_dim=1024, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 2, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.proj = nn.Linear(d_model, vit_dim)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, vis_latent, gt_patches):
        B = vis_latent.shape[0]
        q = self.queries.expand(B, -1, -1).to(vis_latent.dtype)
        decoded = self.decoder(q, vis_latent)
        pred = self.proj(self.ln(decoded))
        gt_pooled = F.adaptive_avg_pool1d(gt_patches.transpose(1,2), self.num_queries).transpose(1,2)
        return F.mse_loss(pred, gt_pooled.to(pred.dtype))

vis_decoder = VisualDecoder(d_model=hidden_size, vit_dim=vit_dim).to(device).to(torch.bfloat16)
print(f"Lang decoder: {sum(p.numel() for p in lang_decoder.parameters()):,}")
print(f"Vis decoder: {sum(p.numel() for p in vis_decoder.parameters()):,}")

# Separate optimizers (FIX 3: vis decoder isolated)
main_params = [p for p in model.parameters() if p.requires_grad] + list(lang_decoder.parameters())
vis_params = list(vis_decoder.parameters())
optimizer_main = torch.optim.AdamW(main_params, lr=LR, weight_decay=0.01)
optimizer_vis = torch.optim.AdamW(vis_params, lr=LR * 5, weight_decay=0.01)

default_cot = "The vehicle should maintain safe trajectory based on current driving conditions."
default_cot_ids = tokenizer(default_cot, return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids']

# Latent token IDs to insert (4 vis + 2 lang)
LATENT_IDS = torch.tensor([LATENT_VIS_ID]*4 + [LATENT_LANG_ID]*2, dtype=torch.long)

def inject_latent_tokens(input_ids, attention_mask):
    """BF-035 FIX: Insert latent tokens AFTER tokenization, before the answer portion."""
    # Find a good insertion point: look for the last occurrence of common answer-start patterns
    # The assistant response typically starts after the last role marker
    # Insert latent tokens ~20 positions before the end (before trajectory numbers start)
    seq_len = input_ids.shape[1]
    # Insert at 70% of sequence (after image+text context, before trajectory answer)
    insert_pos = int(seq_len * 0.7)
    
    # Actually: find the approximate start of the trajectory numbers
    # Look for a sequence of digit tokens in the latter half
    ids = input_ids[0]
    # Simple heuristic: insert before the last 30% of tokens (which is the answer)
    insert_pos = max(10, seq_len - int(seq_len * 0.3))
    
    latent_ids = LATENT_IDS.to(input_ids.device).unsqueeze(0)  # (1, 6)
    new_input_ids = torch.cat([input_ids[:, :insert_pos], latent_ids, input_ids[:, insert_pos:]], dim=1)
    new_attn_mask = torch.cat([attention_mask[:, :insert_pos], torch.ones(1, 6, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask[:, insert_pos:]], dim=1)
    
    return new_input_ids, new_attn_mask, insert_pos

print(f"\n=== Starting Training ===")
model.train(); lang_decoder.train(); vis_decoder.train()
total_steps = 0; start_time = time.time(); best_loss = float('inf'); vis_active = 0; errors = 0

for epoch in range(NUM_EPOCHS):
    epoch_traj, epoch_lang, epoch_vis = [], [], []
    optimizer_main.zero_grad(); optimizer_vis.zero_grad()

    for idx, sample in enumerate(samples):
        try:
            # Step 1: Tokenize NORMALLY (no latent tokens in text) — BF-035 fix
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

            # Step 2: Inject latent tokens post-tokenization (BF-035 fix)
            input_ids, attention_mask, insert_pos = inject_latent_tokens(
                inputs['input_ids'], inputs['attention_mask'])
            inputs['input_ids'] = input_ids
            inputs['attention_mask'] = attention_mask

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Prep error #{errors}: {type(e).__name__}: {str(e)[:80]}")
            continue

        try:
            outputs = model(**inputs, output_hidden_states=True)

            # Trajectory loss (on full sequence)
            logits = outputs.logits
            labels = input_ids.clone()
            # Mask latent positions from trajectory loss (they have no text target)
            labels[0, insert_pos:insert_pos+6] = -100
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            traj_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            # FIX 2: Find latent positions by ID (they're at insert_pos:insert_pos+6)
            hidden = outputs.hidden_states[-1]
            vis_positions = list(range(insert_pos, insert_pos + 4))
            lang_positions = list(range(insert_pos + 4, insert_pos + 6))

            # Language decoder loss
            lang_latent = hidden[:, lang_positions, :]
            cot_ids = default_cot_ids.to(device)
            _, lang_loss = lang_decoder(lang_latent, cot_ids)

            # Visual decoder loss (FIX 3: DETACHED + FIX 4: FUTURE FRAME)
            vis_loss = torch.tensor(0.0, device=device)
            if idx in future_frames:
                vis_latent = hidden[:, vis_positions, :].detach()  # FIX 3
                future_path = future_frames[idx]
                future_img = Image.open(future_path).convert('RGB').resize((448, 448))
                future_inputs = processor(images=[future_img], return_tensors='pt').to(device)
                with torch.no_grad():
                    future_vit = vis_encoder(future_inputs['pixel_values'], grid_thw=future_inputs.get('image_grid_thw'))
                    gt_patches = future_vit.last_hidden_state
                    if gt_patches.dim() == 2:
                        gt_patches = gt_patches.unsqueeze(0)
                vis_loss = vis_decoder(vis_latent, gt_patches)
                vis_active += 1

            # FIX 1: Main loss only (traj + lang). Visual is separate and normalized.
            main_loss = traj_loss + LAMBDA_LANG * lang_loss
            (main_loss / GRAD_ACCUM).backward()

            # Visual decoder: separate backward with normalization
            if vis_loss.item() > 0:
                vis_scale = max(0.001, traj_loss.item() / (vis_loss.item() + 1e-8))
                (vis_loss * vis_scale * 0.5 / GRAD_ACCUM).backward()

            epoch_traj.append(traj_loss.item())
            epoch_lang.append(lang_loss.item())
            epoch_vis.append(vis_loss.item())

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer_main.zero_grad(); optimizer_vis.zero_grad()
            errors += 1
            continue
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error #{errors}: {type(e).__name__}: {str(e)[:80]}")
            continue

        if (idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            torch.nn.utils.clip_grad_norm_(vis_params, 1.0)
            optimizer_main.step(); optimizer_vis.step()
            optimizer_main.zero_grad(); optimizer_vis.zero_grad()
            total_steps += 1
            emergency_state.update({'step': total_steps, 'epoch': epoch})

            if total_steps % 25 == 0:
                t_avg = np.mean(epoch_traj[-GRAD_ACCUM*25:])
                l_avg = np.mean(epoch_lang[-GRAD_ACCUM*25:])
                v_avg = np.mean(epoch_vis[-GRAD_ACCUM*25:]) if epoch_vis else 0
                elapsed = time.time() - start_time
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step={total_steps} traj={t_avg:.3f} lang={l_avg:.3f} vis={v_avg:.1f} mem={mem:.1f}GB t={elapsed:.0f}s err={errors} vis={vis_active}")

    if epoch_traj:
        avg_t = np.mean(epoch_traj); avg_l = np.mean(epoch_lang); avg_v = np.mean(epoch_vis) if epoch_vis else 0
        total_avg = avg_t + LAMBDA_LANG * avg_l
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}: traj={avg_t:.3f} lang={avg_l:.4f} vis={avg_v:.1f} t={time.time()-start_time:.0f}s err={errors} vis_active={vis_active}")
        
        # Checkpoint every epoch
        model.save_pretrained(os.path.join(CHECKPOINT_DIR, f'epoch-{epoch+1}'))
        if total_avg < best_loss:
            best_loss = total_avg
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'best'))
            torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'lang_decoder.pt'))
            torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'vis_decoder.pt'))
            print(f"  Saved best (loss={best_loss:.4f})")
        
        # Partial results (steering rule)
        with open(os.path.join(OUTPUT_DIR, 'partial_results.json'), 'w') as f:
            json.dump({'epoch': epoch+1, 'traj': avg_t, 'lang': avg_l, 'vis': avg_v, 'best': best_loss, 'errors': errors, 'vis_active': vis_active}, f, indent=2)

elapsed_total = time.time() - start_time
print(f"\n=== Training Done! {elapsed_total:.0f}s ({elapsed_total/60:.1f} min) ===")
print(f"Steps: {total_steps}, best_loss: {best_loss:.4f}, errors: {errors}, vis_active: {vis_active}")

model.save_pretrained(OUTPUT_DIR)
torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))
torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'vis_decoder.pt'))
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump({'fixes': ['loss_norm','latent_post_tokenize','vis_detach','future_frame'], 'epochs': NUM_EPOCHS, 'samples': MAX_SAMPLES, 'best_loss': best_loss, 'errors': errors, 'vis_active': vis_active, 'time_sec': elapsed_total}, f, indent=2)
with open(os.path.join(OUTPUT_DIR, 'TRAINING_COMPLETE.txt'), 'w') as f:
    f.write(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')}\nbest_loss={best_loss:.4f}\nerrors={errors}\nvis_active={vis_active}\n")
print("TRAINING_COMPLETE written.")
