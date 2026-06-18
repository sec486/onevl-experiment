"""
CoT Annotation Generator using Cosmos-Reason2-2B (local GPU inference)
Generates chain-of-thought reasoning from actual driving images.
This is self-distillation: the model generates its own training CoT.

Usage:
    python3 cot_generator_cosmos.py --input <answer_jsonl> --output <cot_jsonl> [--limit N] [--test]
"""
import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path


MODEL_PATH = "/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
BASE_DIR = "/opt/onevl-experiment/OneVL_training"

COT_PROMPT = """You are an expert autonomous driving system. Look at this front camera image and the driving context below. Generate a brief chain-of-thought reasoning (2-4 sentences) explaining what the ego vehicle should do and why.

Your reasoning MUST contain:
1. Perception: What you observe in the scene (road, objects, signals)
2. Causal reasoning: Why a specific action is needed
3. Decision: The high-level driving action

Context:
- Command: {command}
- Current velocity: {velocity}
- Historical trajectory: {history}

Generate ONLY the reasoning text (2-4 sentences), nothing else."""


def load_model():
    """Load Cosmos-Reason2-2B for inference."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print("Loading Cosmos-Reason2-2B...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"Model loaded. Device: {next(model.parameters()).device}")
    return model, processor


def parse_scene_info(sample):
    """Extract scene info from a NAVSIM answer-only sample."""
    messages = sample.get("messages", [])
    user_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    command = "UNKNOWN"
    velocity = "UNKNOWN"
    history = "UNKNOWN"

    if "Command:" in user_msg:
        parts = user_msg.split("Command:")
        if len(parts) > 1:
            cmd_part = parts[1].split(".")[0].strip()
            command = cmd_part

    if "Velocity:" in user_msg:
        parts = user_msg.split("Velocity:")
        if len(parts) > 1:
            vel_part = parts[1].split(".")[0].strip()
            velocity = vel_part

    if "Historical" in user_msg or "history" in user_msg.lower():
        for line in user_msg.split("\n"):
            if "histor" in line.lower() or "past" in line.lower():
                history = line.strip()[:200]
                break

    return {
        "command": command,
        "velocity": velocity,
        "history": history,
    }


def get_image_path(sample):
    """Get the first image path from the sample."""
    images = sample.get("images", [])
    if not images:
        return None
    # Image paths are relative to OneVL_training/
    img_rel = images[0]
    img_full = os.path.join(BASE_DIR, img_rel)
    if os.path.exists(img_full):
        return img_full
    return None


def generate_cot_cosmos(model, processor, image_path, scene_info):
    """Generate CoT using Cosmos-Reason2-2B with the actual image."""
    from qwen_vl_utils import process_vision_info

    prompt_text = COT_PROMPT.format(**scene_info)

    # Build messages in Qwen3-VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Process with the processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )

    # Decode only the generated part
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0].strip()

    return output_text


def generate_cot_text_only(model, processor, scene_info):
    """Fallback: generate CoT without image (text-only prompt)."""
    prompt_text = COT_PROMPT.format(**scene_info)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0].strip()

    return output_text


def convert_to_cot_format(original_sample, cot_text):
    """Convert an answer-only sample to explicit CoT format."""
    cot_sample = original_sample.copy()
    cot_sample["think_steps"] = cot_text

    messages = cot_sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            original_answer = msg["content"]
            msg["content"] = f"<think>{cot_text}</think>\n{original_answer}"
            break

    return cot_sample


def main():
    parser = argparse.ArgumentParser(description="Generate CoT using Cosmos-Reason2-2B (local GPU)")
    parser.add_argument("--input", required=True, help="Input answer-only JSONL file")
    parser.add_argument("--output", required=True, help="Output CoT JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--test", action="store_true", help="Test mode: generate 5 samples and print")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--no-image", action="store_true", help="Text-only mode (no images)")
    args = parser.parse_args()

    # Load input data
    print(f"Loading input: {args.input}")
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")

    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    # Resume support
    existing_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"Resuming from sample {existing_count}")
        samples = samples[existing_count:]

    if not samples:
        print("No samples to process. Done.")
        return

    # Load model
    model, processor = load_model()

    # Check first sample for image availability
    first_img = get_image_path(samples[0])
    if first_img and not args.no_image:
        print(f"Image mode: using actual driving images")
        print(f"  First image: {first_img}")
        use_images = True
    else:
        print(f"Text-only mode: no images available or --no-image flag set")
        use_images = False

    # Process samples
    mode = "a" if args.resume else "w"
    output_file = open(args.output, mode) if not args.test else None

    success = 0
    errors = 0
    img_missing = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        try:
            scene_info = parse_scene_info(sample)

            if use_images:
                img_path = get_image_path(sample)
                if img_path:
                    cot_text = generate_cot_cosmos(model, processor, img_path, scene_info)
                else:
                    img_missing += 1
                    cot_text = generate_cot_text_only(model, processor, scene_info)
            else:
                cot_text = generate_cot_text_only(model, processor, scene_info)

            cot_sample = convert_to_cot_format(sample, cot_text)

            if args.test:
                print(f"\n{'='*60}")
                print(f"Sample {i+1}:")
                print(f"  Command: {scene_info['command']}")
                print(f"  Image: {get_image_path(sample) or 'N/A'}")
                print(f"  Generated CoT: {cot_text}")
                print(f"{'='*60}")
                if i >= 4:
                    break
            else:
                output_file.write(json.dumps(cot_sample, ensure_ascii=False) + "\n")
                output_file.flush()

            success += 1

            # Progress
            if (i + 1) % 5 == 0 or args.test:
                elapsed = time.time() - start_time
                rate = success / elapsed if elapsed > 0 else 0
                eta = (len(samples) - i - 1) / rate if rate > 0 else 0
                print(f"  Progress: {existing_count + i + 1}/{existing_count + len(samples)} "
                      f"({success} ok, {errors} err, {img_missing} no-img, "
                      f"{rate:.1f} samples/s, ETA {eta/60:.1f} min)")

        except Exception as e:
            errors += 1
            print(f"  ERROR sample {i}: {e}")
            if errors > 20:
                print("Too many errors, stopping.")
                break
            # Clear GPU cache on error
            torch.cuda.empty_cache()

    if output_file:
        output_file.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done. {success} generated, {errors} errors, {img_missing} missing images")
    print(f"Elapsed: {elapsed:.1f}s ({success/elapsed:.2f} samples/s)")
    print(f"Output: {args.output}")
    print(f"GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB peak")


if __name__ == "__main__":
    main()
