#!/usr/bin/env python3
"""
W8 Evaluation: Compare fixed c-full vs c-lang (1.17m) on held-out samples.
SEPARATE script from training (per steering rule — never combine in one long command).
"""
import json, torch, time, os, sys, re
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
CFULL_FIXED = '/opt/onevl-experiment/output/w8_cfull_fixed_v2/best'
CLANG_ADAPTER = '/opt/onevl-experiment/output/w8_clang_only_1000/best'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
RESULTS_PATH = '/opt/onevl-experiment/output/w8_fixed_eval_results.json'

device = 'cuda'

def parse_traj(text):
    wps = []
    for m in re.finditer(r'\[([\d.\-]+),\s*([\d.\-]+)', text):
        x, y = float(m.group(1)), float(m.group(2))
        if abs(x) < 200 and abs(y) < 200:
            wps.append([x, y])
    return np.array(wps[:8]) if len(wps) >= 8 else None

def evaluate_adapter(adapter_path, processor, tokenizer, eval_samples, name):
    print(f"\n--- Evaluating: {name} ---")
    print(f"  Adapter: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        print(f"  ERROR: Adapter not found!")
        return None
    
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
    ).to(device)
    
    # Resize embeddings if needed (for fixed version with new tokens)
    if 'fixed' in adapter_path:
        LATENT_VIS_TOKEN = "<|latent-vis|>"
        LATENT_LANG_TOKEN = "<|latent-lang|>"
        tokenizer.add_special_tokens({'additional_special_tokens': [LATENT_VIS_TOKEN, LATENT_LANG_TOKEN]})
        base.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base, adapter_path)
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
    
    del model, base
    torch.cuda.empty_cache()
    
    if ades:
        result = {'ade': float(np.mean(ades)), 'fde': float(np.mean(fdes)), 'valid': len(ades), 'total': 50}
        print(f"  RESULT: ADE={result['ade']:.2f}m, FDE={result['fde']:.2f}m, valid={result['valid']}/50")
        return result
    else:
        print("  ERROR: No valid predictions!")
        return None

# Load eval samples
print("=" * 60)
print("  W8 EVALUATION: Fixed c-full vs c-lang")
print("=" * 60)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

eval_samples = []
with open(DATA_PATH) as f:
    for i, line in enumerate(f):
        if 1000 <= i < 1050:
            eval_samples.append(json.loads(line))
print(f"Eval samples: {len(eval_samples)} (held-out 1000-1049)")

# Evaluate both
results = {}
results['cfull_fixed'] = evaluate_adapter(CFULL_FIXED, processor, tokenizer, eval_samples, "c-full (BF-034 fixed)")
results['clang'] = evaluate_adapter(CLANG_ADAPTER, processor, tokenizer, eval_samples, "c-lang (baseline)")

# Summary
print(f"\n{'='*60}")
print(f"  FINAL COMPARISON")
print(f"{'='*60}")
print(f"\n{'Model':<45} {'ADE (m)':<10} {'FDE (m)':<10} {'Valid':<8}")
print("-" * 75)

if results['cfull_fixed']:
    print(f"{'c-full FIXED (BF-034 all 4 fixes)':<45} {results['cfull_fixed']['ade']:<10.2f} {results['cfull_fixed']['fde']:<10.2f} {results['cfull_fixed']['valid']}/50")
if results['clang']:
    print(f"{'c-lang (language decoder only)':<45} {results['clang']['ade']:<10.2f} {results['clang']['fde']:<10.2f} {results['clang']['valid']}/50")
print(f"{'[ref] c-lang W8 (from earlier run)':<45} {'1.17':<10} {'2.88':<10} {'50/50':<8}")

if results['cfull_fixed'] and results['clang']:
    delta = results['cfull_fixed']['ade'] - results['clang']['ade']
    print(f"\nΔ (c-full - c-lang) = {delta:+.2f}m")
    if delta < -0.1:
        print(">> WITH FIXES: Visual decoder NOW HELPS!")
    elif delta > 0.1:
        print(">> Even with fixes, visual decoder still hurts. Concept limitation.")
    else:
        print(">> Visual decoder is neutral (no significant difference).")

# Save results
results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
results['fixes_applied'] = ['loss_normalization', 'real_latent_tokens', 'vis_detach', 'future_frame_target']
with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")

# Write status for ops agent
with open('/opt/onevl-experiment/output/W8_EVAL_COMPLETE.txt', 'w') as f:
    f.write(f"Evaluation completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if results['cfull_fixed']:
        f.write(f"c-full FIXED ADE: {results['cfull_fixed']['ade']:.2f}m\n")
    if results['clang']:
        f.write(f"c-lang ADE: {results['clang']['ade']:.2f}m\n")
    if results['cfull_fixed'] and results['clang']:
        f.write(f"Delta: {results['cfull_fixed']['ade'] - results['clang']['ade']:+.2f}m\n")
        f.write(f"Conclusion: {'visual_helps' if delta < -0.1 else 'visual_hurts' if delta > 0.1 else 'neutral'}\n")
