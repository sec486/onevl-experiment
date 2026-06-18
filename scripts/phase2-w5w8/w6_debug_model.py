#!/usr/bin/env python3
"""W6: Debug model structure to find ViT patch extraction path."""
import torch, os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from qwen_vl_utils import process_vision_info

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'

model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda')
processor = AutoProcessor.from_pretrained(MODEL_PATH)

lora_config = LoraConfig(r=64, lora_alpha=256, target_modules=['q_proj','k_proj','v_proj','o_proj'], modules_to_save=['embed_tokens'], lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)

print('=== Model structure ===')
for name, child in model.base_model.model.named_children():
    print(f'  {name}: {type(child).__name__}')

print('\n=== Visual modules ===')
visual_modules = [(n, type(m).__name__) for n, m in model.named_modules() if 'visual' in n.lower()][:10]
for n, t in visual_modules:
    print(f'  {n}: {t}')

# Process an image
scene = os.listdir(TRAINVAL)[0]
cam_dir = os.path.join(TRAINVAL, scene, 'CAM_F0')
img_path = os.path.join(cam_dir, os.listdir(cam_dir)[0])
img = Image.open(img_path).convert('RGB')
print(f'\nImage: {img_path}, size={img.size}')

messages = [{'role': 'user', 'content': [{'type': 'image', 'image': 'file://' + img_path}, {'type': 'text', 'text': 'describe'}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors='pt').to('cuda')

print(f'\nInput keys: {list(inputs.keys())}')
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: {v.shape}')

# Forward with hidden states
model.eval()
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
    print(f'\nHidden states: {len(out.hidden_states)} layers')
    print(f'Last hidden: {out.hidden_states[-1].shape}')
    
    # Find image token positions
    input_ids = inputs['input_ids'][0]
    # Qwen3-VL uses special token for image patches
    # Check common IDs: 151655 (image_pad), 151652 (vision_start), 151653 (vision_end)
    for check_id in [151655, 151652, 151653, 151654, 151656]:
        count = (input_ids == check_id).sum().item()
        if count > 0:
            positions = (input_ids == check_id).nonzero(as_tuple=True)[0]
            print(f'  Token ID {check_id}: {count} occurrences, range [{positions[0].item()}, {positions[-1].item()}]')
    
    # The image patch positions in hidden states = our GT for visual decoder
    # Find the large block of repeated IDs (those are the patches)
    id_counts = {}
    for tid in input_ids.tolist():
        id_counts[tid] = id_counts.get(tid, 0) + 1
    # Find ID with most occurrences (likely image patches)
    most_common = sorted(id_counts.items(), key=lambda x: -x[1])[:5]
    print(f'\n  Most common token IDs: {most_common}')
    
    patch_id = most_common[0][0]
    patch_positions = (input_ids == patch_id).nonzero(as_tuple=True)[0]
    print(f'  Patch token ID: {patch_id}, count: {len(patch_positions)}')
    if len(patch_positions) > 10:
        patch_hidden = out.hidden_states[-1][0, patch_positions]
        print(f'  Patch hidden states shape: {patch_hidden.shape}')
        print(f'  THIS is the GT for visual decoder (num_patches x hidden_dim)')

print(f'\nGPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
