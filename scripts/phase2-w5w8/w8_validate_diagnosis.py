#!/usr/bin/env python3
"""
W8 Validation: Confirm Quick's diagnosis about missing image placeholders.
Tests the full pipeline on a SINGLE sample to pinpoint the exact failure.
"""
import json, torch, traceback
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_FIXED = '/opt/onevl-experiment/data/navsim_real_traj_1000_fixed.jsonl'
DATA_ORIG = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'

print('=' * 60)
print('  W8 Validation — Confirming Quick Diagnosis')
print('=' * 60)

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load a sample (use fixed data if exists, else orig)
import os
data_path = DATA_FIXED if os.path.exists(DATA_FIXED) else DATA_ORIG
print(f"\nUsing data: {data_path}")
with open(data_path) as f:
    sample = json.loads(f.readline())

print(f"\nSample message format:")
print(json.dumps(sample['messages'][0]['content'], indent=2)[:400])

messages = sample['messages']

# ========== TEST 1: apply_chat_template (tokenize=False) ==========
print("\n" + "="*60)
print("TEST 1: apply_chat_template(tokenize=False)")
print("="*60)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(f"Text type: {type(text)}, length: {len(text) if text else 'None'}")
print(f"Contains <|vision_start|>: {'<|vision_start|>' in text if text else 'N/A'}")
print(f"Contains <|image_pad|>: {'<|image_pad|>' in text if text else 'N/A'}")
print(f"Contains <|vision_end|>: {'<|vision_end|>' in text if text else 'N/A'}")
print(f"\nFirst 400 chars:\n{text[:400] if text else 'None'}")

# ========== TEST 2: processor() two-step ==========
print("\n" + "="*60)
print("TEST 2: processor(text, images) two-step")
print("="*60)
img_path = sample['images'][0]
print(f"Image path: {img_path}")
print(f"Image exists: {os.path.exists(img_path)}")
try:
    image = Image.open(img_path).convert('RGB').resize((448, 448))
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True, max_length=512)
    print(f"Processor output keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} = {v}")
    print(f"pixel_values is None: {inputs.get('pixel_values') is None}")
except Exception as e:
    print("FAILED:")
    traceback.print_exc()

# ========== TEST 3: apply_chat_template (tokenize=True) — Quick's Solution 1 ==========
print("\n" + "="*60)
print("TEST 3: apply_chat_template(tokenize=True, return_dict=True) — Quick Solution 1")
print("="*60)
try:
    inputs2 = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt", max_length=512, truncation=True,
    )
    print(f"Output keys: {list(inputs2.keys())}")
    for k, v in inputs2.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
    print(f"pixel_values is None: {inputs2.get('pixel_values') is None}")
    print(">>> Solution 1 WORKS" if inputs2.get('pixel_values') is not None else ">>> Solution 1 has no pixel_values")
except Exception as e:
    print("FAILED:")
    traceback.print_exc()

# ========== TEST 4: Forward pass with LoRA ==========
print("\n" + "="*60)
print("TEST 4: Model forward pass (with LoRA + output_hidden_states)")
print("="*60)
device = 'cuda'
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
).to(device)
lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                         modules_to_save=["embed_tokens"], task_type=TaskType.CAUSAL_LM, bias="none")
model = get_peft_model(model, lora_config)

# Try with Solution 1 inputs (tokenize=True)
try:
    inputs2 = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs2.items()}
    outputs = model(**inputs2, output_hidden_states=True)
    print(f"Forward pass OK!")
    print(f"  hidden_states: {type(outputs.hidden_states)}, len={len(outputs.hidden_states) if outputs.hidden_states else 'None'}")
    print(f"  logits: {outputs.logits.shape}")
    print(">>> SOLUTION 1 + FORWARD PASS WORKS!")
except Exception as e:
    print("Forward pass FAILED with Solution 1 inputs:")
    traceback.print_exc()

# ========== TEST 5: Future frame processing ==========
print("\n" + "="*60)
print("TEST 5: Future frame — processor.image_processor (Quick fix)")
print("="*60)
try:
    fut_img = Image.open(img_path).convert('RGB').resize((448, 448))
    # Quick's recommended approach
    fut_inputs = processor.image_processor(images=[fut_img], return_tensors='pt')
    print(f"image_processor output keys: {list(fut_inputs.keys())}")
    for k, v in fut_inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}")
    print(">>> image_processor WORKS for future frame")
except Exception as e:
    print("FAILED:")
    traceback.print_exc()

print("\n" + "="*60)
print("  VALIDATION COMPLETE")
print("="*60)
