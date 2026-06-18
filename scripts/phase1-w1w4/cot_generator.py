"""
CoT Annotation Generator for OneVL × Cosmos-Reason2 Experiment
Uses Bedrock Claude to generate chain-of-thought reasoning for NAVSIM driving scenes.
Following OneVL Appendix 8.3 prompt structure.

Usage:
    python3 cot_generator.py --input <answer_jsonl> --output <cot_jsonl> [--limit N] [--test]
"""
import argparse
import boto3
import json
import os
import sys
import time
from pathlib import Path


# OneVL Appendix 8.3 style prompt for CoT generation
COT_SYSTEM_PROMPT = """You are an expert autonomous driving annotator. Given a driving scene description, generate a brief chain-of-thought reasoning that explains what the ego vehicle should do and why.

Your reasoning MUST contain exactly 3 elements:
1. **Perception**: Key observations about the scene (road layout, nearby vehicles, pedestrians, traffic signals, weather)
2. **Causal reasoning**: Why a specific action is needed (cause → effect logic)
3. **Decision**: The high-level driving action (meta-action: follow lane, change lane, yield, stop, accelerate, decelerate, turn)

Requirements:
- Keep reasoning concise: 2-4 sentences total
- Focus on safety-critical elements only
- Use present tense ("The road curves left", not "The road curved left")
- Do NOT include trajectory coordinates in the reasoning
- Do NOT use bullet points or numbered lists — write as flowing text

Example output:
"The ego vehicle is on a straight highway with moderate traffic. A slower vehicle ahead in the same lane is maintaining consistent speed with no gap in adjacent lanes. The ego should maintain current lane and gradually decelerate to match the lead vehicle's speed while monitoring for an opening to pass."
"""

COT_USER_TEMPLATE = """Generate chain-of-thought reasoning for this driving scene:

Command: {command}
Current velocity (m/s): {velocity}
Historical trajectory (past 2s, [x,y,heading]): {history}

The ego vehicle's planned future trajectory spans 4 seconds with 8 waypoints at 0.5s intervals.

Generate ONLY the reasoning text (2-4 sentences), nothing else."""


def parse_scene_info(sample):
    """Extract scene info from a NAVSIM answer-only sample."""
    messages = sample.get("messages", [])
    user_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    # Parse command, velocity, history from the user message
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
        # Try to extract trajectory info
        for line in user_msg.split("\n"):
            if "histor" in line.lower() or "past" in line.lower():
                history = line.strip()[:200]
                break

    return {
        "command": command,
        "velocity": velocity,
        "history": history,
    }


def generate_cot(bedrock_client, scene_info, model_id="us.anthropic.claude-sonnet-4-6"):
    """Generate CoT annotation using Bedrock Claude."""
    user_content = COT_USER_TEMPLATE.format(**scene_info)

    body = json.dumps({
        "messages": [
            {"role": "user", "content": user_content}
        ],
        "system": COT_SYSTEM_PROMPT,
        "max_tokens": 300,
        "temperature": 0.3,
        "anthropic_version": "bedrock-2023-05-31"
    })

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


def convert_to_cot_format(original_sample, cot_text):
    """Convert an answer-only sample to explicit CoT format."""
    cot_sample = original_sample.copy()

    # Add think_steps field
    cot_sample["think_steps"] = cot_text

    # Modify assistant message to include CoT before answer
    messages = cot_sample.get("messages", [])
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            original_answer = msg["content"]
            # Wrap with <think> tags like OneVL format
            msg["content"] = f"<think>{cot_text}</think>\n{original_answer}"
            break

    return cot_sample


def main():
    parser = argparse.ArgumentParser(description="Generate CoT annotations for NAVSIM data")
    parser.add_argument("--input", required=True, help="Input answer-only JSONL file")
    parser.add_argument("--output", required=True, help="Output CoT JSONL file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--test", action="store_true", help="Test mode: generate 5 samples and print")
    parser.add_argument("--model", default="us.anthropic.claude-sonnet-4-6", help="Bedrock model ID")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
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

    # Init Bedrock client
    bedrock = boto3.client("bedrock-runtime", region_name=args.region)

    # Process samples
    mode = "a" if args.resume else "w"
    output_file = open(args.output, mode) if not args.test else None
    
    success = 0
    errors = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        try:
            scene_info = parse_scene_info(sample)
            cot_text = generate_cot(bedrock, scene_info, model_id=args.model)
            cot_sample = convert_to_cot_format(sample, cot_text)

            if args.test:
                print(f"\n{'='*60}")
                print(f"Sample {i+1}:")
                print(f"  Command: {scene_info['command']}")
                print(f"  Generated CoT: {cot_text}")
                print(f"{'='*60}")
                if i >= 4:  # Test mode: only 5 samples
                    break
            else:
                output_file.write(json.dumps(cot_sample, ensure_ascii=False) + "\n")
                output_file.flush()

            success += 1

            # Rate limiting (Bedrock throttling)
            if not args.test:
                time.sleep(0.5)

            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = success / elapsed * 3600
                print(f"  Progress: {existing_count + i + 1}/{existing_count + len(samples)} "
                      f"({success} ok, {errors} err, {rate:.0f}/hr)")

        except Exception as e:
            errors += 1
            print(f"  ERROR sample {i}: {e}")
            if errors > 10 and errors > success:
                print("Too many errors, stopping.")
                break
            time.sleep(2)  # Back off on error

    if output_file:
        output_file.close()

    elapsed = time.time() - start_time
    print(f"\nDone. {success} generated, {errors} errors, {elapsed:.1f}s elapsed")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
