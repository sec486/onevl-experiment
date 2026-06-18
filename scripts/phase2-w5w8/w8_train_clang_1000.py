#!/usr/bin/env python3
"""
W8 §11: Fair A/B — Train c-lang-only on the SAME 1000 real NAVSIM samples.
No visual decoder, only language decoder. Same LoRA r=64, same data, same eval.

This isolates the visual decoder's contribution:
  - c-full v4 (both decoders): ADE=2.81m on held-out (W7 result)
  - c-lang (this run): ADE=? on same held-out
  - If c-lang ADE < 2.81 → visual decoder HURTS
  - If c-lang ADE > 2.81 → visual decoder HELPS
  - If c-lang ADE ≈ 2.81 → visual decoder is neutral
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from language_decoder import LanguageDecoderHead
import torch.nn as nn
import torch.nn.functional as F
import re

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
OUTPUT_DIR = '/opt/onevl-experiment/output/w8_clang_only_1000'
LAMBDA_LANG = 0.5
NUM_EPOCHS = 5
MAX_SAMPLES = 1000
GRAD_ACCUM = 8
LR = 2e-5

print('=' * 60)
print('  W8 §11: c-lang-only — Fair A/B Comparison')
print(f'  Lambda_lang={LAMBDA_LANG}, NO visual decoder')
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

# Apply LoRA (same config as c-full for fair comparison)
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

hidden_size = 2048
vocab_size = len(tokenizer)

# Language decoder ONLY (no visual decoder)
print(f"\n=== Language decoder only (no visual) ===")
lang_decoder = LanguageDecoderHead(
    d_model=hidden_size, n_heads=8, n_layers=2,
    vocab_size=vocab_size, max_cot_length=128,
).to(device).to(torch.bfloat16)
print(f"Language decoder: {sum(p.numel() for p in lang_decoder.parameters()):,} params")
print("Visual decoder: DISABLED (this is the A/B control)")

# Optimizer
all_params = (
    [p for p in model.parameters() if p.requires_grad] +
    list(lang_decoder.parameters())
)
optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=0.01)

# Default CoT target
default_cot = "The vehicle should maintain safe trajectory based on the current driving conditions and road geometry."
default_cot_ids = tokenizer(default_cot, return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids']

print(f"\n=== Starting Training ===")
print(f"Steps/epoch: {MAX_SAMPLES // GRAD_ACCUM}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.train()
lang_decoder.train()

total_steps = 0
start_time = time.time()
best_loss = float('inf')
errors = 0

for epoch in range(NUM_EPOCHS):
    epoch_traj, epoch_lang = [], []
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
            
            # 2. Language decoder loss ONLY
            hidden = outputs.hidden_states[-1]
            seq_len = hidden.shape[1]
            lang_latent = hidden[:, max(0, seq_len-4):max(2, seq_len-2), :]
            cot_ids = default_cot_ids.to(device)
            _, lang_loss = lang_decoder(lang_latent, cot_ids)
            
            # NO visual decoder loss
            total_loss = traj_loss + LAMBDA_LANG * lang_loss
            (total_loss / GRAD_ACCUM).backward()
            
            epoch_traj.append(traj_loss.item())
            epoch_lang.append(lang_loss.item())
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            errors += 1
            continue
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {type(e).__name__}: {str(e)[:80]}")
            continue
        
        if (idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_steps += 1
            
            if total_steps % 25 == 0:
                t_avg = np.mean(epoch_traj[-GRAD_ACCUM*25:])
                l_avg = np.mean(epoch_lang[-GRAD_ACCUM*25:])
                elapsed = time.time() - start_time
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  step={total_steps} traj={t_avg:.3f} lang={l_avg:.3f} mem={mem:.1f}GB t={elapsed:.0f}s")
    
    if epoch_traj:
        avg_t = np.mean(epoch_traj)
        avg_l = np.mean(epoch_lang)
        total_avg = avg_t + LAMBDA_LANG * avg_l
        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS}: total={total_avg:.4f} traj={avg_t:.3f} lang={avg_l:.4f} t={elapsed:.0f}s errors={errors}")
        
        if total_avg < best_loss:
            best_loss = total_avg
            model.save_pretrained(os.path.join(OUTPUT_DIR, 'best'))
            torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'best', 'lang_decoder.pt'))
            print(f"  Saved best (loss={best_loss:.4f})")

# Save final
print(f"\n=== Saving ===")
model.save_pretrained(OUTPUT_DIR)
torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))

config = {
    'experiment': 'W8 §11 Fair A/B — c-lang only (no visual decoder)',
    'lambda_lang': LAMBDA_LANG, 'lambda_vis': 0,
    'epochs': NUM_EPOCHS, 'samples': len(samples),
    'lr': LR, 'lora_r': 64, 'grad_accum': GRAD_ACCUM,
    'total_steps': total_steps, 'best_loss': best_loss, 'errors': errors,
    'comparison_target': 'c-full v4 ADE=2.81m (same data, held-out 1000-1049)',
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

elapsed_train = time.time() - start_time
print(f"\nTraining done! {elapsed_train:.0f}s ({elapsed_train/60:.1f} min), {total_steps} steps, best_loss={best_loss:.4f}")

# ==================== EVALUATION ====================
print("\n" + "=" * 60)
print("  EVALUATION: c-lang vs c-full on held-out")
print("=" * 60)

def parse_traj(text):
    wps = []
    for m in re.finditer(r'\[([\d.\-]+),\s*([\d.\-]+)', text):
        x, y = float(m.group(1)), float(m.group(2))
        if abs(x) < 200 and abs(y) < 200:
            wps.append([x, y])
    return np.array(wps[:8]) if len(wps) >= 8 else None

# Load held-out samples
eval_samples = []
with open(DATA_PATH) as f:
    for i, line in enumerate(f):
        if 1000 <= i < 1050:
            eval_samples.append(json.loads(line))
print(f"\nEval samples: {len(eval_samples)} (held-out 1000-1049)")

# Evaluate c-lang (just trained)
# Use PeftModel.from_pretrained for correct loading (BF-033 lesson!)
del model
torch.cuda.empty_cache()

print("\n--- Loading c-lang adapter (PeftModel.from_pretrained) ---")
base = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
).to(device)
model = PeftModel.from_pretrained(base, os.path.join(OUTPUT_DIR, 'best'))
model.eval()

ades, fdes = [], []
for idx, sample in enumerate(eval_samples):
    user_msg = sample['messages'][0]
    forced = json.loads(json.dumps(user_msg))
    if isinstance(forced['content'], list):
        for item in forced['content']:
            if item.get('type') == 'text':
                item['text'] += ' Respond with ONLY: <answer>[x1,y1,h1], [x2,y2,h2], ..., [x8,y8,h8]</answer>'
                break
    
    text = processor.apply_chat_template([forced], tokenize=False, add_generation_prompt=True)
    img_path = sample.get('images', [None])[0]
    image = Image.open(img_path).convert('RGB').resize((448, 448)) if img_path and os.path.exists(img_path) else None
    if image:
        inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    else:
        inputs = processor(text=[text], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    pred = parse_traj(generated)
    gt = parse_traj(sample['messages'][1]['content'])
    
    if pred is not None and gt is not None:
        errs = np.sqrt(np.sum((pred - gt)**2, axis=1))
        ades.append(float(np.mean(errs)))
        fdes.append(float(errs[-1]))
    
    if (idx+1) % 10 == 0 and ades:
        print(f"  {idx+1}/50: valid={len(ades)}, ADE={np.mean(ades):.2f}m")

print(f"\n{'='*60}")
print(f"  RESULTS: Fair A/B Comparison")
print(f"{'='*60}")
print(f"\n{'Model':<40} {'ADE (m)':<10} {'FDE (m)':<10} {'Valid':<8}")
print("-" * 70)

clang_ade = np.mean(ades) if ades else None
clang_fde = np.mean(fdes) if fdes else None
cfull_ade = 2.81  # W7 result

if clang_ade:
    print(f"{'c-lang (this run, no vis decoder)':<40} {clang_ade:<10.2f} {clang_fde:<10.2f} {len(ades)}/50")
print(f"{'c-full v4 (W7, with vis decoder)':<40} {cfull_ade:<10.2f} {'5.29':<10} {'50/50':<8}")
print(f"{'Phase 1 ref (different data)':<40} {'2.10':<10} {'4.37':<10} {'50/50':<8}")

print(f"\n--- THE ANSWER ---")
if clang_ade:
    delta = cfull_ade - clang_ade
    if delta < -0.1:
        print(f"  c-full ({cfull_ade:.2f}) < c-lang ({clang_ade:.2f}) by {abs(delta):.2f}m")
        print(f"  >> VISUAL DECODER HELPS! (improves ADE by {abs(delta):.2f}m)")
    elif delta > 0.1:
        print(f"  c-full ({cfull_ade:.2f}) > c-lang ({clang_ade:.2f}) by {delta:.2f}m")
        print(f"  >> Visual decoder HURTS (or is noise). Language-only is better.")
    else:
        print(f"  c-full ({cfull_ade:.2f}) ≈ c-lang ({clang_ade:.2f}) (Δ={delta:.2f}m)")
        print(f"  >> Visual decoder is NEUTRAL. No incremental benefit.")

# Save
results = {
    'clang_ade': clang_ade, 'clang_fde': clang_fde, 'clang_valid': len(ades),
    'cfull_ade': cfull_ade, 'cfull_fde': 5.29,
    'delta': cfull_ade - clang_ade if clang_ade else None,
    'conclusion': 'helps' if (clang_ade and cfull_ade < clang_ade - 0.1) else 'hurts' if (clang_ade and cfull_ade > clang_ade + 0.1) else 'neutral',
    'training_time_sec': elapsed_train,
}
with open('/opt/onevl-experiment/output/w8_fair_ab_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to /opt/onevl-experiment/output/w8_fair_ab_results.json")

total_elapsed = time.time() - start_time
print(f"Total time (train + eval): {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
