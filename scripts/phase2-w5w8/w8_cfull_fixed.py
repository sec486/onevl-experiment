#!/usr/bin/env python3
"""
W8 §11 RETRY: c-full with ALL 4 BF-034 fixes applied.
Fair A/B comparison against c-lang (ADE 1.17m).

BF-034 Fixes:
1. Loss normalization (vis_loss scaled to same magnitude as traj_loss)
2. Actual latent token positions (insert <|latent-vis|> tokens, find by ID)
3. Detach vis_latent (prevent visual gradients from corrupting LoRA)
4. Future frame target (t+1.0s = frame_idx+2, not current frame)

Architecture:
  Input: [image tokens] [text tokens] [<|latent-vis|>x4] [<|latent|>x2] <answer>[trajectory]</answer>
  Language decoder: reads hidden states at <|latent|> positions → reconstruct CoT
  Visual decoder: reads hidden states at <|latent-vis|> positions (DETACHED) → predict future frame patches
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
OUTPUT_DIR = '/opt/onevl-experiment/output/w8_cfull_fixed'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LAMBDA_LANG = 0.5
NUM_EPOCHS = 5
MAX_SAMPLES = 1000
GRAD_ACCUM = 8
LR = 2e-5

# === RESILIENCE BOILERPLATE (per steering rule) ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

emergency_state = {'step': 0, 'epoch': 0}
def save_emergency(signum, frame):
    print(f"\nSIGNAL {signum} — emergency save!")
    with open(os.path.join(OUTPUT_DIR, 'emergency_state.json'), 'w') as f:
        json.dump(emergency_state, f)
signal.signal(signal.SIGTERM, save_emergency)
signal.signal(signal.SIGINT, save_emergency)
# === END BOILERPLATE ===

print('=' * 60)
print('  W8 RETRY: c-full with BF-034 fixes')
print('  Fix 1: Loss normalization')
print('  Fix 2: Latent token markers in prompt')
print('  Fix 3: vis_latent.detach()')
print('  Fix 4: Future frame (t+1.0s) as GT')
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

# FIX 4: Identify samples that have a future frame available
# Each sample has scene + frame_idx; future = frame_idx + 2 (t+1.0s at 2Hz)
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'
TRAINVAL_ALT = '/opt/onevl-experiment/OneVL_training/navsim_v1.1_all/dataset/sensor_blobs/trainval'
img_base = TRAINVAL if os.path.exists(TRAINVAL) else TRAINVAL_ALT

future_frames = {}  # sample_idx → future frame path
for idx, s in enumerate(samples):
    scene = s.get('scene', '')
    frame_idx = s.get('frame_idx', 0)
    cam_dir = os.path.join(img_base, scene, 'CAM_F0')
    if os.path.exists(cam_dir):
        frames = sorted(os.listdir(cam_dir))
        future_idx = frame_idx + 2  # t+1.0s at 2Hz
        if future_idx < len(frames):
            future_frames[idx] = os.path.join(cam_dir, frames[future_idx])

print(f"Samples with future frame available: {len(future_frames)}/{len(samples)}")

# Load model
print("\n=== Loading model ===")
device = 'cuda'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

# FIX 2: Add latent token markers to tokenizer
LATENT_VIS_TOKEN = "<|latent-vis|>"
LATENT_LANG_TOKEN = "<|latent-lang|>"
special_tokens = [LATENT_VIS_TOKEN, LATENT_LANG_TOKEN]
num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
print(f"Added {num_added} special tokens: {special_tokens}")
LATENT_VIS_ID = tokenizer.convert_tokens_to_ids(LATENT_VIS_TOKEN)
LATENT_LANG_ID = tokenizer.convert_tokens_to_ids(LATENT_LANG_TOKEN)
print(f"  {LATENT_VIS_TOKEN} → ID {LATENT_VIS_ID}")
print(f"  {LATENT_LANG_TOKEN} → ID {LATENT_LANG_ID}")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, attn_implementation="eager",
).to(device)
model.resize_token_embeddings(len(tokenizer))  # Resize for new tokens

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
print(f"Trainable: {trainable:,} ({100*trainable/sum(p.numel() for p in model.parameters()):.2f}%)")

# Visual encoder (correct path after LoRA)
vis_encoder = model.base_model.model.model.visual
print(f"Visual encoder: {type(vis_encoder).__name__}")

hidden_size = 2048
vocab_size = len(tokenizer)
vit_dim = 1024

# Language decoder
print(f"\n=== Initializing decoders ===")
lang_decoder = LanguageDecoderHead(
    d_model=hidden_size, n_heads=8, n_layers=2,
    vocab_size=vocab_size, max_cot_length=128,
).to(device).to(torch.bfloat16)
print(f"Language decoder: {sum(p.numel() for p in lang_decoder.parameters()):,} params")

# Visual decoder (separate optimizer — FIX 3 complement)
class VisualDecoder(nn.Module):
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
        B = vis_latent.shape[0]
        q = self.queries.expand(B, -1, -1).to(vis_latent.dtype)
        decoded = self.decoder(q, vis_latent)
        decoded = self.ln(decoded)
        pred = self.proj(decoded)
        gt_pooled = F.adaptive_avg_pool1d(
            gt_patches.transpose(1, 2), self.num_queries
        ).transpose(1, 2)
        return F.mse_loss(pred, gt_pooled.to(pred.dtype))

vis_decoder = VisualDecoder(d_model=hidden_size, vit_dim=vit_dim, num_queries=32).to(device).to(torch.bfloat16)
print(f"Visual decoder: {sum(p.numel() for p in vis_decoder.parameters()):,} params")

# Separate optimizers (FIX 3: visual decoder doesn't touch LoRA)
main_params = [p for p in model.parameters() if p.requires_grad] + list(lang_decoder.parameters())
vis_params = list(vis_decoder.parameters())
optimizer_main = torch.optim.AdamW(main_params, lr=LR, weight_decay=0.01)
optimizer_vis = torch.optim.AdamW(vis_params, lr=LR * 5, weight_decay=0.01)  # Higher LR for vis decoder

# CoT target
default_cot = "The vehicle should maintain safe trajectory based on the current driving conditions and road geometry."
default_cot_ids = tokenizer(default_cot, return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids']

# FIX 2: Template that inserts latent tokens
LATENT_INSERT = f" {LATENT_VIS_TOKEN}{LATENT_VIS_TOKEN}{LATENT_VIS_TOKEN}{LATENT_VIS_TOKEN}{LATENT_LANG_TOKEN}{LATENT_LANG_TOKEN}"

print(f"\n=== Starting Training ===")
print(f"Steps/epoch: {MAX_SAMPLES // GRAD_ACCUM}")

model.train()
lang_decoder.train()
vis_decoder.train()

total_steps = 0
start_time = time.time()
best_loss = float('inf')
vis_active = 0
errors = 0

for epoch in range(NUM_EPOCHS):
    epoch_traj, epoch_lang, epoch_vis = [], [], []
    optimizer_main.zero_grad()
    optimizer_vis.zero_grad()

    for idx, sample in enumerate(samples):
        messages = sample['messages']

        try:
            # FIX 2: Insert latent tokens before <answer>
            modified_messages = json.loads(json.dumps(messages))
            assistant_content = modified_messages[1]['content']
            # Insert latent tokens before <answer>
            if '<answer>' in assistant_content:
                assistant_content = assistant_content.replace('<answer>', f'{LATENT_INSERT} <answer>')
            else:
                assistant_content = f'{LATENT_INSERT} {assistant_content}'
            modified_messages[1]['content'] = assistant_content

            text = processor.apply_chat_template(modified_messages, tokenize=False, add_generation_prompt=False)
            img_path = sample.get('images', [None])[0]
            image = None
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB').resize((448, 448))
            if image:
                inputs = processor(text=[text], images=[image], return_tensors="pt",
                                   padding=True, truncation=True, max_length=600).to(device)
            else:
                inputs = processor(text=[text], return_tensors="pt",
                                   padding=True, truncation=True, max_length=600).to(device)
        except Exception as e:
            errors += 1
            continue

        try:
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

            # FIX 2: Find actual latent token positions
            input_ids = inputs['input_ids'][0]
            vis_positions = (input_ids == LATENT_VIS_ID).nonzero(as_tuple=True)[0]
            lang_positions = (input_ids == LATENT_LANG_ID).nonzero(as_tuple=True)[0]

            hidden = outputs.hidden_states[-1]  # (1, seq_len, 2048)

            # 2. Language decoder loss (at actual latent-lang positions)
            lang_loss = torch.tensor(0.0, device=device)
            if len(lang_positions) >= 2:
                lang_latent = hidden[:, lang_positions, :]
                cot_ids = default_cot_ids.to(device)
                _, lang_loss = lang_decoder(lang_latent, cot_ids)

            # 3. Visual decoder loss (FIX 3: DETACHED + FIX 4: FUTURE FRAME)
            vis_loss = torch.tensor(0.0, device=device)
            if len(vis_positions) >= 4 and idx in future_frames:
                # FIX 3: Detach visual latent states
                vis_latent = hidden[:, vis_positions, :].detach()

                # FIX 4: Load FUTURE frame (t+1.0s) as GT
                future_path = future_frames[idx]
                future_img = Image.open(future_path).convert('RGB').resize((448, 448))
                future_inputs = processor(images=[future_img], return_tensors='pt').to(device)

                with torch.no_grad():
                    future_vit = vis_encoder(
                        future_inputs['pixel_values'],
                        grid_thw=future_inputs.get('image_grid_thw')
                    )
                    gt_patches = future_vit.last_hidden_state
                    if gt_patches.dim() == 2:
                        gt_patches = gt_patches.unsqueeze(0)

                vis_loss = vis_decoder(vis_latent, gt_patches)
                vis_active += 1

            # FIX 1: Normalize visual loss to same scale as trajectory loss
            if vis_loss.item() > 0:
                # Scale vis_loss to be approximately same magnitude as traj_loss
                vis_scale = traj_loss.item() / (vis_loss.item() + 1e-8)
                vis_loss_scaled = vis_loss * vis_scale * 0.5  # 0.5 = lambda_vis
            else:
                vis_loss_scaled = vis_loss

            # Main loss (trajectory + language)
            main_loss = traj_loss + LAMBDA_LANG * lang_loss
            (main_loss / GRAD_ACCUM).backward()

            # Visual decoder loss (separate, only updates vis_decoder params)
            if vis_loss.item() > 0:
                (vis_loss_scaled / GRAD_ACCUM).backward()

            epoch_traj.append(traj_loss.item())
            epoch_lang.append(lang_loss.item())
            epoch_vis.append(vis_loss.item())

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer_main.zero_grad()
            optimizer_vis.zero_grad()
            errors += 1
            continue
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error #{errors}: {type(e).__name__}: {str(e)[:100]}")
            continue

        if (idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            torch.nn.utils.clip_grad_norm_(vis_params, 1.0)
            optimizer_main.step()
            optimizer_vis.step()
            optimizer_main.zero_grad()
            optimizer_vis.zero_grad()
            total_steps += 1
            emergency_state['step'] = total_steps
            emergency_state['epoch'] = epoch

            if total_steps % 25 == 0:
                t_avg = np.mean(epoch_traj[-GRAD_ACCUM*25:])
                l_avg = np.mean(epoch_lang[-GRAD_ACCUM*25:])
                v_avg = np.mean(epoch_vis[-GRAD_ACCUM*25:]) if epoch_vis else 0
                elapsed = time.time() - start_time
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step={total_steps} traj={t_avg:.3f} lang={l_avg:.3f} vis={v_avg:.1f} mem={mem:.1f}GB t={elapsed:.0f}s vis_active={vis_active}")

    # Epoch summary + checkpoint (per steering rule)
    if epoch_traj:
        avg_t = np.mean(epoch_traj)
        avg_l = np.mean(epoch_lang)
        avg_v = np.mean(epoch_vis) if epoch_vis else 0
        total_avg = avg_t + LAMBDA_LANG * avg_l
        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}: traj={avg_t:.3f} lang={avg_l:.4f} vis={avg_v:.1f} t={elapsed:.0f}s errors={errors} vis_active={vis_active}")

        # Save checkpoint every epoch (steering rule)
        model.save_pretrained(os.path.join(CHECKPOINT_DIR, f'epoch-{epoch+1}'))
        torch.save(vis_decoder.state_dict(), os.path.join(CHECKPOINT_DIR, f'epoch-{epoch+1}', 'vis_decoder.pt'))

        if total_avg < best_loss:
            best_loss = total_avg
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'best'))
            torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'lang_decoder.pt'))
            torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'vis_decoder.pt'))
            print(f"  Saved best (loss={best_loss:.4f})")

        # Write partial results to disk immediately (steering rule)
        partial = {
            'epoch': epoch+1, 'traj_loss': avg_t, 'lang_loss': avg_l, 'vis_loss': avg_v,
            'best_loss': best_loss, 'steps': total_steps, 'errors': errors,
            'vis_active': vis_active, 'elapsed_sec': elapsed,
        }
        with open(os.path.join(OUTPUT_DIR, 'partial_results.json'), 'w') as f:
            json.dump(partial, f, indent=2)

# Final save
elapsed_total = time.time() - start_time
print(f"\n=== Training Done! {elapsed_total:.0f}s ({elapsed_total/60:.1f} min) ===")
print(f"Steps: {total_steps}, best_loss: {best_loss:.4f}, errors: {errors}, vis_active: {vis_active}")

model.save_pretrained(OUTPUT_DIR)
torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))
torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'vis_decoder.pt'))

config = {
    'experiment': 'W8 c-full with BF-034 fixes (4 fixes applied)',
    'fixes': ['loss_normalization', 'real_latent_tokens', 'vis_detach', 'future_frame_target'],
    'lambda_lang': LAMBDA_LANG, 'epochs': NUM_EPOCHS, 'samples': MAX_SAMPLES,
    'lr': LR, 'lora_r': 64, 'total_steps': total_steps,
    'best_loss': best_loss, 'vis_active': vis_active, 'errors': errors,
    'training_time_sec': elapsed_total,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Write completion marker for ops agent
with open(os.path.join(OUTPUT_DIR, 'TRAINING_COMPLETE.txt'), 'w') as f:
    f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Best loss: {best_loss:.4f}\n")
    f.write(f"Steps: {total_steps}, Errors: {errors}, Vis active: {vis_active}\n")
    f.write(f"Time: {elapsed_total:.0f}s\n")
    f.write(f"Next: run w8_eval_fixed.py for ADE/FDE evaluation\n")

print("\nTRAINING_COMPLETE marker written. Ops agent can now trigger eval.")
