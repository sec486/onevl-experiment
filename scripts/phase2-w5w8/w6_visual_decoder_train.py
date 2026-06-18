#!/usr/bin/env python3
"""
W6.1: Visual Decoder Training
- Processes real images through ViT to get patch representations
- Extracts future frame patches as GT
- Trains visual decoder (from 4 visual latent states -> predict future patches)
- Combined loss: L_trajectory + lambda1*L_language + lambda2*L_visual
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '.')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from language_decoder import LanguageDecoderHead
from qwen_vl_utils import process_vision_info
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'
OUTPUT_DIR = '/opt/onevl-experiment/output/visual_decoder_v1'

IMAGE_TOKEN_ID = 151655
SUBSAMPLE_FACTOR = 10  # Predict every 10th patch (2040 -> 204 targets)

class VisualDecoderHead(nn.Module):
    """Predicts subsampled future frame patches from visual latent hidden states."""
    def __init__(self, d_model=2048, n_heads=8, n_layers=2, num_target_patches=204):
        super().__init__()
        self.num_target_patches = num_target_patches
        # Learnable query tokens for cross-attention
        self.query_embed = nn.Parameter(torch.randn(num_target_patches, d_model) * 0.02)
        # Transformer decoder: queries attend to visual latent states
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, visual_latent_hidden, gt_patches=None):
        """
        visual_latent_hidden: (B, num_visual_latent, d_model) - from base model
        gt_patches: (B, num_target_patches, d_model) - from future frame ViT
        Returns: predicted_patches, loss
        """
        B = visual_latent_hidden.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        
        # Decode: queries cross-attend to visual latent states
        predicted = self.decoder(tgt=queries, memory=visual_latent_hidden)
        predicted = self.ln(predicted)
        
        loss = torch.tensor(0.0, device=visual_latent_hidden.device)
        if gt_patches is not None:
            loss = F.mse_loss(predicted, gt_patches)
        
        return predicted, loss


def extract_image_patches(model, processor, img_path):
    """Extract ViT patch hidden states from an image."""
    img = Image.open(img_path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': 'file://' + img_path},
        {'type': 'text', 'text': 'x'}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors='pt').to('cuda')
    
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    
    # Extract patch positions
    input_ids = inputs['input_ids'][0]
    patch_positions = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    patch_hidden = out.hidden_states[-1][0, patch_positions]
    
    # Subsample
    indices = torch.arange(0, len(patch_positions), SUBSAMPLE_FACTOR)
    subsampled = patch_hidden[indices]
    
    return subsampled


def main():
    print('=' * 60)
    print('  W6.1: Visual Decoder Training')
    print('=' * 60)
    
    # Load model
    print('\nLoading model...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda'
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    
    # Apply LoRA
    lora_config = LoraConfig(r=64, lora_alpha=256,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],
        modules_to_save=['embed_tokens'], lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create decoders
    lang_decoder = LanguageDecoderHead(
        d_model=2048, n_heads=8, n_layers=2,
        vocab_size=len(tokenizer), max_cot_length=128, dropout=0.1
    ).to(dtype=torch.bfloat16, device='cuda')
    
    vis_decoder = VisualDecoderHead(
        d_model=2048, n_heads=8, n_layers=2, num_target_patches=204
    ).to(dtype=torch.bfloat16, device='cuda')
    
    print(f'Language decoder: {sum(p.numel() for p in lang_decoder.parameters()):,} params')
    print(f'Visual decoder: {sum(p.numel() for p in vis_decoder.parameters()):,} params')
    
    # Step 1: Pre-extract future frame patches (expensive, do once)
    print('\n=== Pre-extracting future frame patches ===')
    scenes = sorted(os.listdir(TRAINVAL))[:15]  # 15 scenes for speed
    
    training_data = []
    model.eval()
    
    for scene_idx, scene in enumerate(scenes):
        cam_dir = os.path.join(TRAINVAL, scene, 'CAM_F0')
        if not os.path.exists(cam_dir):
            continue
        frames = sorted(os.listdir(cam_dir))
        
        for idx in range(min(len(frames) - 2, 4)):  # 4 pairs per scene, need future frame
            current_path = os.path.join(cam_dir, frames[idx])
            future_path = os.path.join(cam_dir, frames[idx + 2])  # t+1.0s
            
            try:
                future_patches = extract_image_patches(model, processor, future_path)
                training_data.append({
                    'scene': scene,
                    'current': current_path,
                    'future_patches': future_patches,  # (num_patches, 2048)
                    'think_steps': 'The vehicle is driving forward on a clear road.',
                })
            except Exception as e:
                print(f'  Error on {scene} frame {idx}: {e}')
                continue
        
        if (scene_idx + 1) % 5 == 0:
            print(f'  Extracted {len(training_data)} pairs from {scene_idx+1} scenes')
        
        if len(training_data) >= 50:
            break
    
    print(f'\nTotal training pairs: {len(training_data)}')
    if not training_data:
        print('ERROR: No training data extracted.')
        return
    
    num_patches = training_data[0]['future_patches'].shape[0]
    print(f'Patches per sample: {num_patches}')
    
    # Update visual decoder if patch count differs
    if num_patches != 204:
        print(f'Adjusting visual decoder for {num_patches} patches')
        vis_decoder = VisualDecoderHead(
            d_model=2048, n_heads=8, n_layers=2, num_target_patches=num_patches
        ).to(dtype=torch.bfloat16, device='cuda')
    
    # Step 2: Train with dual decoders
    print('\n=== Training with language + visual decoders ===')
    
    LAMBDA1 = 0.5  # language decoder weight
    LAMBDA2 = 0.1  # visual decoder weight (start small)
    EPOCHS = 3
    GRAD_ACCUM = 4
    LR = 2e-5
    
    all_params = (
        list(model.parameters()) + 
        list(lang_decoder.parameters()) + 
        list(vis_decoder.parameters())
    )
    trainable_params = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.requires_grad], 'lr': LR},
        {'params': lang_decoder.parameters(), 'lr': LR * 2},
        {'params': vis_decoder.parameters(), 'lr': LR * 2},
    ], weight_decay=0.01)
    
    model.train()
    lang_decoder.train()
    vis_decoder.train()
    model.gradient_checkpointing_enable()
    
    print(f'Config: lambda1={LAMBDA1}, lambda2={LAMBDA2}, epochs={EPOCHS}')
    start_time = time.time()
    step = 0
    
    for epoch in range(EPOCHS):
        et, el, ev, n = 0, 0, 0, 0
        
        for i, entry in enumerate(training_data):
            # Prepare text input (latent format)
            text_input = '<|im_start|>user\nCommand: MOVE FORWARD.<|im_end|>\n<|im_start|>assistant\n<|start-latent-vis|><|latent-vis|><|latent-vis|><|latent-vis|><|latent-vis|><|end-latent-vis|><|start-latent|><|latent|><|latent|><|end-latent|><answer>[1.0, 0.0, 0.0]</answer><|im_end|>'
            
            enc = tokenizer(text_input, return_tensors='pt', truncation=True, max_length=512)
            ids = enc['input_ids'].to('cuda')
            mask = enc['attention_mask'].to('cuda')
            labels = ids.clone()
            
            # Forward
            out = model(input_ids=ids, attention_mask=mask, labels=labels, output_hidden_states=True)
            traj_loss = out.loss
            
            # Find latent positions (same method as Phase 1B)
            text_no_latent = '<|im_start|>user\nCommand: MOVE FORWARD.<|im_end|>\n<|im_start|>assistant\n<answer>[1.0, 0.0, 0.0]</answer><|im_end|>'
            tok_no_latent = tokenizer.encode(text_no_latent, truncation=True, max_length=512)
            tok_full = tokenizer.encode(text_input, truncation=True, max_length=512)
            num_latent = len(tok_full) - len(tok_no_latent)
            
            start_pos = 0
            for j in range(min(len(tok_full), len(tok_no_latent))):
                if tok_full[j] != tok_no_latent[j]:
                    start_pos = j
                    break
            
            latent_pos = list(range(start_pos, start_pos + num_latent))
            
            # Extract latent hidden states
            hid = out.hidden_states[-1]
            hid_len = hid.shape[1]
            valid_pos = [p for p in latent_pos if p < hid_len]
            
            lang_loss = torch.tensor(0.0, device='cuda')
            vis_loss = torch.tensor(0.0, device='cuda')
            
            if valid_pos and len(valid_pos) >= 6:
                pt = torch.tensor(valid_pos, device='cuda').unsqueeze(0)
                latent_h = hid.gather(1, pt.unsqueeze(-1).expand(1, len(valid_pos), 2048))
                
                # Split: first 4 = visual, last 2 = language (matching OneVL format)
                # In our token format: vis×4 then lang×2
                vis_latent = latent_h[:, :4, :]  # (1, 4, 2048)
                lang_latent = latent_h[:, 4:, :]  # (1, 2+, 2048)
                
                # Language decoder loss
                cot = entry.get('think_steps', '')
                if cot and lang_latent.shape[1] >= 2:
                    ce = tokenizer(cot, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
                    cot_ids = ce['input_ids'].to('cuda').clamp(max=len(tokenizer)-1)
                    _, lang_loss = lang_decoder(lang_latent, cot_ids, ce['attention_mask'].to('cuda'))
                
                # Visual decoder loss
                gt_patches = entry['future_patches'].unsqueeze(0).to('cuda')
                _, vis_loss = vis_decoder(vis_latent, gt_patches)
            
            # Combined loss
            total = traj_loss + LAMBDA1 * lang_loss + LAMBDA2 * vis_loss
            (total / GRAD_ACCUM).backward()
            
            et += traj_loss.item()
            el += lang_loss.item()
            ev += vis_loss.item()
            n += 1
            
            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                if step % 5 == 0:
                    print(f'  Step {step} | traj={et/n:.3f} | lang={el/n:.3f} | vis={ev/n:.4f} | {time.time()-start_time:.0f}s')
        
        print(f'  Epoch {epoch+1}: traj={et/n:.4f}, lang={el/n:.4f}, vis={ev/n:.4f} ({time.time()-start_time:.0f}s)')
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'vis_decoder.pt'))
    
    elapsed = time.time() - start_time
    result = {
        'traj_loss': et/n, 'lang_loss': el/n, 'vis_loss': ev/n,
        'time': elapsed, 'samples': len(training_data),
        'lambda1': LAMBDA1, 'lambda2': LAMBDA2,
        'num_patches': num_patches, 'subsample_factor': SUBSAMPLE_FACTOR,
    }
    json.dump(result, open(os.path.join(OUTPUT_DIR, 'result.json'), 'w'), indent=2)
    
    print(f'\n{"="*60}')
    print(f'  RESULT')
    print(f'  Trajectory: {et/n:.4f}')
    print(f'  Language decoder: {el/n:.4f}')
    print(f'  Visual decoder: {ev/n:.4f}')
    print(f'  Time: {elapsed:.0f}s')
    print(f'  GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
