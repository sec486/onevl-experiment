#!/usr/bin/env python3
"""
W7 Evaluation: Compare c-full (real traj, v4) vs Phase 1 c-lang (ADE 2.10).
This is the critical test: does adding a visual decoder improve trajectory prediction?

Evaluation on held-out samples (use samples 1000-1100 from the dataset, which
were NOT in training).
"""
import json, torch, time, os, sys, re
import numpy as np
from PIL import Image

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
CFULL_ADAPTER = '/opt/onevl-experiment/output/w7_cfull_1000samples/best'
CLANG_ADAPTER = '/opt/onevl-experiment/output/latent_cot_with_decoder_v5'  # Phase 1 best
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
NUM_EVAL = 50  # Evaluation samples

print('=' * 60)
print('  W7 ADE/FDE Evaluation')
print('  c-full (1000 samples, visual decoder) vs baselines')
print('=' * 60)

# Load held-out eval data (samples beyond training set)
print("\n=== Loading eval data ===")
all_samples = []
with open(DATA_PATH) as f:
    for line in f:
        if line.strip():
            all_samples.append(json.loads(line))

# Use samples 1000-1050 as held-out test (training used 0-999)
eval_samples = all_samples[1000:1000 + NUM_EVAL]
if len(eval_samples) < NUM_EVAL:
    # If not enough held-out, use every 20th sample from training as test
    print(f"  Only {len(eval_samples)} held-out samples. Using every 20th from training instead.")
    eval_samples = all_samples[::20][:NUM_EVAL]

print(f"Eval samples: {len(eval_samples)}")

# Extract GT trajectories
def parse_trajectory(answer_text):
    """Parse trajectory from answer format: <answer>[x,y,h], [x,y,h], ...</answer>"""
    match = re.search(r'<answer>(.*?)</answer>', answer_text, re.DOTALL)
    if not match:
        return None
    traj_text = match.group(1).strip()
    
    # Parse [x,y,h] groups
    waypoints = []
    for wp_match in re.finditer(r'\[([^]]+)\]', traj_text):
        parts = wp_match.group(1).split(',')
        if len(parts) >= 2:
            x, y = float(parts[0]), float(parts[1])
            waypoints.append([x, y])
    
    return np.array(waypoints) if len(waypoints) == 8 else None

gt_trajectories = []
for s in eval_samples:
    answer = s['messages'][1]['content']
    traj = parse_trajectory(answer)
    gt_trajectories.append(traj)

valid_count = sum(1 for t in gt_trajectories if t is not None)
print(f"Valid GT trajectories: {valid_count}/{len(eval_samples)}")

# Load model + processor
print("\n=== Loading model ===")
device = 'cuda'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

def load_base_model():
    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device)

def evaluate_model(model, eval_samples, gt_trajectories, model_name):
    """Run inference on eval samples and compute ADE/FDE."""
    print(f"\n--- Evaluating: {model_name} ---")
    model.eval()
    
    ades, fdes = [], []
    valid_preds = 0
    
    for idx, (sample, gt) in enumerate(zip(eval_samples, gt_trajectories)):
        if gt is None:
            continue
        
        # Prepare input (user message only, no assistant)
        user_msg = sample['messages'][0]
        messages_input = [user_msg]
        
        try:
            text = processor.apply_chat_template(
                messages_input, tokenize=False, add_generation_prompt=True
            )
            
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
            
            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=1.0,
                )
            
            # Decode
            generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse predicted trajectory
            pred_traj = parse_trajectory(f"<answer>{generated}</answer>")
            if pred_traj is None:
                # Try without answer tags
                pred_traj = parse_trajectory(generated)
            
            if pred_traj is None:
                continue
            
            # Compute ADE/FDE
            errors = np.sqrt(np.sum((pred_traj - gt[:len(pred_traj)]) ** 2, axis=1))
            ade = np.mean(errors)
            fde = errors[-1]
            
            ades.append(ade)
            fdes.append(fde)
            valid_preds += 1
            
        except Exception as e:
            if idx < 3:
                print(f"  Error sample {idx}: {e}")
            continue
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(eval_samples)}, valid={valid_preds}, ADE so far={np.mean(ades):.2f}m")
    
    results = {
        'model': model_name,
        'ade': float(np.mean(ades)) if ades else None,
        'fde': float(np.mean(fdes)) if fdes else None,
        'ade_std': float(np.std(ades)) if ades else None,
        'fde_std': float(np.std(fdes)) if fdes else None,
        'valid_preds': valid_preds,
        'total_samples': len(eval_samples),
    }
    
    if ades:
        print(f"  Result: ADE={results['ade']:.2f}m (±{results['ade_std']:.2f}), FDE={results['fde']:.2f}m, valid={valid_preds}/{len(eval_samples)}")
    else:
        print(f"  NO valid predictions!")
    
    return results

# Evaluate c-full v4 (1000 samples)
print("\n=== Model 1: c-full v4 (1000 samples, visual decoder) ===")
model = load_base_model()
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["embed_tokens"],
    task_type=TaskType.CAUSAL_LM, bias="none",
)
model = get_peft_model(model, lora_config)

# Load adapter weights
if os.path.exists(CFULL_ADAPTER):
    from safetensors.torch import load_file
    adapter_path = os.path.join(CFULL_ADAPTER, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        state_dict = load_file(adapter_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded adapter from {CFULL_ADAPTER}")
    else:
        print(f"  WARNING: No adapter_model.safetensors in {CFULL_ADAPTER}")
else:
    print(f"  WARNING: Adapter path not found: {CFULL_ADAPTER}")

cfull_results = evaluate_model(model, eval_samples, gt_trajectories, "c-full v4 (1000 samples)")

# Free memory
del model
torch.cuda.empty_cache()

# Evaluate base model (no fine-tuning) as lower bound
print("\n=== Model 2: Base model (no fine-tuning) ===")
model = load_base_model()
base_results = evaluate_model(model, eval_samples, gt_trajectories, "Base model (no ft)")

del model
torch.cuda.empty_cache()

# Evaluate c-lang (Phase 1 best) if adapter exists
clang_results = None
if os.path.exists(CLANG_ADAPTER):
    print(f"\n=== Model 3: c-lang (Phase 1 best, ADE 2.10) ===")
    model = load_base_model()
    model = get_peft_model(model, lora_config)
    adapter_path = os.path.join(CLANG_ADAPTER, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        state_dict = load_file(adapter_path)
        model.load_state_dict(state_dict, strict=False)
        clang_results = evaluate_model(model, eval_samples, gt_trajectories, "c-lang (Phase 1)")
    del model
    torch.cuda.empty_cache()
else:
    print(f"\n  Phase 1 adapter not found at {CLANG_ADAPTER}")
    print(f"  Using Phase 1 reported ADE=2.10 as reference")

# Summary
print("\n" + "=" * 60)
print("  EVALUATION SUMMARY")
print("=" * 60)
print(f"\n{'Model':<35} {'ADE (m)':<12} {'FDE (m)':<12} {'Valid%':<10}")
print("-" * 70)

results_all = [cfull_results, base_results]
if clang_results:
    results_all.append(clang_results)

for r in results_all:
    if r and r['ade'] is not None:
        pct = f"{100*r['valid_preds']/r['total_samples']:.0f}%"
        print(f"{r['model']:<35} {r['ade']:<12.2f} {r['fde']:<12.2f} {pct:<10}")

# Reference from Phase 1
print(f"\n{'[Phase 1 ref] c-lang':<35} {'2.10':<12} {'4.37':<12} {'100%':<10}")
print(f"{'[Phase 1 ref] answer-only r=64':<35} {'7.37':<12} {'14.56':<12} {'100%':<10}")

print("\n--- Key Question ---")
print("Is c-full ADE < c-lang ADE (2.10)?")
if cfull_results and cfull_results['ade'] is not None:
    if cfull_results['ade'] < 2.10:
        print(f"  YES! c-full ADE={cfull_results['ade']:.2f} < 2.10 → Visual decoder helps!")
    else:
        print(f"  NO. c-full ADE={cfull_results['ade']:.2f} >= 2.10 → Visual decoder doesn't help (at this config)")
        print("  Possible reasons: visual loss barely converging, different data distribution, need more epochs")

# Save results
results_file = '/opt/onevl-experiment/output/w7_eval_results.json'
with open(results_file, 'w') as f:
    json.dump({
        'cfull_v4': cfull_results,
        'base': base_results,
        'clang_phase1': clang_results,
        'eval_samples': NUM_EVAL,
        'data_source': DATA_PATH,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, f, indent=2)
print(f"\nResults saved to {results_file}")
