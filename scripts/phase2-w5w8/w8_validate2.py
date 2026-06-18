#!/usr/bin/env python3
"""
W8 Validation 2: Confirm the TRUNCATION hypothesis.
Test multiple samples through the EXACT training path (resize 448 + two-step processor)
and the forward pass, to find which samples fail and why.
"""
import json, torch, traceback, os
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA = '/opt/onevl-experiment/data/navsim_real_traj_1000_fixed.jsonl'
device = 'cuda'

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
).to(device)
lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                         modules_to_save=["embed_tokens"], task_type=TaskType.CAUSAL_LM, bias="none")
model = get_peft_model(model, lora_config)

samples = []
with open(DATA) as f:
    for line in f:
        samples.append(json.loads(line))
        if len(samples) >= 20:
            break

print("="*60)
print("Testing 20 samples through EXACT training path")
print("="*60)

# Test A: WITH truncation max_length=512 (current training config)
print("\n--- Config A: max_length=512, truncation=True (current) ---")
ok_a, fail_a = 0, 0
for idx, sample in enumerate(samples):
    try:
        messages = sample['messages']
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        img_path = sample['images'][0]
        image = Image.open(img_path).convert('RGB').resize((448, 448))
        inputs = processor(text=[text], images=[image], return_tensors="pt",
                           padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        ok_a += 1
    except Exception as e:
        fail_a += 1
        if fail_a <= 2:
            print(f"  Sample {idx} FAILED: {type(e).__name__}: {str(e)[:120]}")
print(f"  Config A: {ok_a} OK, {fail_a} FAILED")

# Test B: NO truncation, larger max_length
print("\n--- Config B: max_length=2048, truncation=False ---")
ok_b, fail_b = 0, 0
for idx, sample in enumerate(samples):
    try:
        messages = sample['messages']
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        img_path = sample['images'][0]
        image = Image.open(img_path).convert('RGB').resize((448, 448))
        inputs = processor(text=[text], images=[image], return_tensors="pt",
                           padding=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        ok_b += 1
    except Exception as e:
        fail_b += 1
        if fail_b <= 2:
            print(f"  Sample {idx} FAILED: {type(e).__name__}: {str(e)[:120]}")
print(f"  Config B: {ok_b} OK, {fail_b} FAILED")

# Test C: Check token counts to understand truncation
print("\n--- Token count analysis ---")
for idx in [0, 1, 2]:
    sample = samples[idx]
    messages = sample['messages']
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    img_path = sample['images'][0]
    image = Image.open(img_path).convert('RGB').resize((448, 448))
    # Without truncation
    full = processor(text=[text], images=[image], return_tensors="pt", padding=False)
    full_len = full['input_ids'].shape[1]
    img_tokens = (full['input_ids'][0] == tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum().item()
    print(f"  Sample {idx}: full_len={full_len}, image_pad_tokens={img_tokens}, exceeds_512={full_len > 512}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if fail_a > ok_a and ok_b > fail_b:
    print(">>> CONFIRMED: Truncation at max_length=512 cuts image tokens → failure")
    print(">>> FIX: Disable truncation OR increase max_length to fit image tokens")
elif ok_a > fail_a:
    print(">>> Config A works in isolation — failure may be intermittent/sample-specific")
