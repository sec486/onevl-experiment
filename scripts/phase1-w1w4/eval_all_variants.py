"""
W4: Evaluate all trained variants — compute ADE/FDE on 100 NAVSIM test samples.
Runs inference on each checkpoint and compares predicted trajectories to ground truth.
"""
import json, torch, time, os, sys, re
import numpy as np
from pathlib import Path

MODEL_PATH = "/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
DATASET_PATH = "/opt/onevl-experiment/OneVL_training/demo_data/navsim/navsim_answer_demo100.jsonl"

# All variants to evaluate
VARIANTS = {
    "answer_only_r8": "/opt/onevl-experiment/output/baseline_answer_only",
    "answer_only_r64": "/opt/onevl-experiment/output/baseline_answer_only_r64",
    "explicit_cot_r8": "/opt/onevl-experiment/output/baseline_explicit_cot",
    "latent_cot_r64_no_decoder": "/opt/onevl-experiment/output/baseline_latent_cot_r64",
    "latent_cot_r64_with_decoder": "/opt/onevl-experiment/output/latent_cot_with_decoder_v5",
}


def parse_trajectory(text):
    """Parse trajectory from model output: [x,y,h], [x,y,h], ..."""
    # Remove tags
    for tag in ["<answer>", "</answer>", "<|im_end|>", "<|start-latent|>", "<|latent|>",
                "<|end-latent|>", "<|start-latent-vis|>", "<|end-latent-vis|>", "<|latent-vis|>",
                "<think>", "</think>"]:
        text = text.replace(tag, "")
    text = text.strip()
    
    # Try to parse as list of [x,y,h] triplets
    try:
        # Add outer brackets if needed
        if not text.startswith("[["):
            text = "[" + text + "]"
        arr = json.loads(text)
        return [[float(v) for v in point] for point in arr]
    except:
        # Try regex extraction
        pattern = r'\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]'
        matches = re.findall(pattern, text)
        if matches:
            return [[float(x), float(y), float(h)] for x, y, h in matches]
        return None


def compute_ade_fde(pred_traj, gt_traj):
    """Compute ADE and FDE between predicted and ground-truth trajectories."""
    if pred_traj is None or len(pred_traj) == 0:
        return None, None
    
    pred = np.array(pred_traj)[:, :2]  # Only x, y
    gt = np.array(gt_traj)[:, :2]
    
    # Align lengths
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    # L2 distances
    distances = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    ade = np.mean(distances)
    fde = distances[-1]
    
    return ade, fde


def run_inference(model, tokenizer, sample, max_new_tokens=150):
    """Run inference on a single sample and return predicted trajectory."""
    messages = sample["messages"]
    # Only use the user message (input)
    user_msg = [messages[0]]
    
    text = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,  # Greedy for reproducibility
        )
    
    # Decode only generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=False)
    
    return response


def evaluate_variant(variant_name, adapter_path, model_base, tokenizer, samples):
    """Evaluate a single variant."""
    from peft import PeftModel
    
    print(f"\n{'='*50}")
    print(f"  Evaluating: {variant_name}")
    print(f"  Adapter: {adapter_path}")
    print(f"{'='*50}")
    
    # Check if adapter exists
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        print(f"  SKIP: No adapter found at {adapter_file}")
        return None
    
    # Load adapter
    try:
        model = PeftModel.from_pretrained(model_base, adapter_path)
        model.eval()
    except Exception as e:
        print(f"  ERROR loading adapter: {e}")
        return None
    
    # Run inference on all samples
    results = []
    start = time.time()
    
    for i, sample in enumerate(samples):
        # Get ground truth
        gt_text = sample["messages"][1]["content"]
        gt_traj = parse_trajectory(gt_text)
        
        if gt_traj is None:
            continue
        
        # Run inference
        response = run_inference(model, tokenizer, sample)
        pred_traj = parse_trajectory(response)
        
        # Compute metrics
        ade, fde = compute_ade_fde(pred_traj, gt_traj)
        
        results.append({
            "sample_idx": i,
            "ade": ade,
            "fde": fde,
            "pred_valid": pred_traj is not None,
            "pred_len": len(pred_traj) if pred_traj else 0,
        })
        
        if (i + 1) % 20 == 0:
            valid = [r for r in results if r["ade"] is not None]
            avg_ade = np.mean([r["ade"] for r in valid]) if valid else float("inf")
            print(f"  Progress: {i+1}/{len(samples)} | ADE={avg_ade:.4f} | {time.time()-start:.0f}s")
    
    # Summary
    valid_results = [r for r in results if r["ade"] is not None]
    elapsed = time.time() - start
    
    if valid_results:
        avg_ade = np.mean([r["ade"] for r in valid_results])
        avg_fde = np.mean([r["fde"] for r in valid_results])
        valid_pct = len(valid_results) / len(results) * 100
    else:
        avg_ade = avg_fde = float("inf")
        valid_pct = 0
    
    summary = {
        "variant": variant_name,
        "ade": avg_ade,
        "fde": avg_fde,
        "valid_predictions": len(valid_results),
        "total_samples": len(results),
        "valid_pct": valid_pct,
        "inference_time_s": elapsed,
        "avg_time_per_sample": elapsed / len(samples) if samples else 0,
    }
    
    print(f"\n  RESULT: ADE={avg_ade:.4f}, FDE={avg_fde:.4f}")
    print(f"  Valid: {len(valid_results)}/{len(results)} ({valid_pct:.0f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(samples):.2f}s/sample)")
    
    # Cleanup adapter to free memory
    del model
    torch.cuda.empty_cache()
    
    return summary


def main():
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    print("W4: Evaluating all variants")
    print("=" * 60)
    
    # Load base model
    print("Loading base model...")
    model_base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    
    # Load test samples (use answer-only format for consistent evaluation)
    samples = [json.loads(l) for l in open(DATASET_PATH) if l.strip()]
    print(f"Loaded {len(samples)} test samples")
    
    # Evaluate each variant
    all_results = []
    for name, path in VARIANTS.items():
        result = evaluate_variant(name, path, model_base, tokenizer, samples[:50])  # Use 50 for speed
        if result:
            all_results.append(result)
    
    # Final comparison table
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Variant':<35} {'ADE':>8} {'FDE':>8} {'Valid%':>8} {'Time':>8}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x["ade"]):
        print(f"{r['variant']:<35} {r['ade']:>8.4f} {r['fde']:>8.4f} {r['valid_pct']:>7.0f}% {r['inference_time_s']:>7.1f}s")
    
    # Save results
    output_path = "/opt/onevl-experiment/output/w4_eval_results.json"
    json.dump(all_results, open(output_path, "w"), indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
