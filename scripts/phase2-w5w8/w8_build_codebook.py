#!/usr/bin/env python3
"""
W8 方案 B Step 1-3: Build visual codebook from Cosmos-Reason2 ViT features.

1. Extract ViT patch features from training frames (1000 scenes)
2. k-means clustering → codebook (k=8192)
3. Encode all future frames → discrete token IDs
4. Save codebook.pt + visual_tokens.json for training

Expected time: ~30-40 min on GPU (ViT extraction) + 5 min CPU (k-means)
"""
import json, torch, time, os, sys, signal
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
OUTPUT_DIR = '/opt/onevl-experiment/output/w8_codebook'
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'
TRAINVAL_ALT = '/opt/onevl-experiment/OneVL_training/navsim_v1.1_all/dataset/sensor_blobs/trainval'

K = 8192  # codebook size
MAX_SAMPLES = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)
img_base = TRAINVAL if os.path.exists(TRAINVAL) else TRAINVAL_ALT

print('=' * 60)
print(f'  W8 方案 B: Build Codebook (k={K})')
print(f'  Step 1: Extract ViT features')
print(f'  Step 2: k-means → codebook')
print(f'  Step 3: Encode future frames → token IDs')
print('=' * 60)

# Load model (only need ViT encoder + processor.image_processor)
device = 'cuda'
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
).to(device)
vis_encoder = model.model.visual  # Qwen3VLVisionModel (without LoRA: model.model.visual)
vis_encoder.eval()

# Load samples
samples = []
with open(DATA_PATH) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))
            if len(samples) >= MAX_SAMPLES:
                break
print(f"Loaded {len(samples)} samples")

# ===== STEP 1: Extract ViT patch features =====
print(f"\n=== Step 1: Extracting ViT features from {len(samples)} frames ===")
start = time.time()
all_patches = []  # Will hold (num_patches, dim) per frame
frame_info = []   # Track which sample/frame each belongs to

for idx, sample in enumerate(samples):
    scene = sample.get('scene', '')
    frame_idx = sample.get('frame_idx', 0)
    cam_dir = os.path.join(img_base, scene, 'CAM_F0')
    
    if not os.path.exists(cam_dir):
        continue
    
    frames = sorted(os.listdir(cam_dir))
    
    # Current frame
    img_path = os.path.join(cam_dir, frames[frame_idx]) if frame_idx < len(frames) else None
    if not img_path or not os.path.exists(img_path):
        continue
    
    try:
        image = Image.open(img_path).convert('RGB').resize((448, 448))
        # Use image_processor (BF-035 lesson!)
        img_inputs = processor.image_processor(images=[image], return_tensors='pt')
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        
        with torch.no_grad():
            vit_out = vis_encoder(img_inputs['pixel_values'], grid_thw=img_inputs['image_grid_thw'])
            patches = vit_out.last_hidden_state.cpu().float()  # (num_patches, dim)
            all_patches.append(patches)
        
        frame_info.append({'idx': idx, 'scene': scene, 'frame_idx': frame_idx})
    except Exception as e:
        if idx < 5:
            print(f"  Error frame {idx}: {e}")
        continue
    
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(samples)} frames extracted, patches so far: {sum(p.shape[0] for p in all_patches)}")

elapsed_step1 = time.time() - start
total_patches = sum(p.shape[0] for p in all_patches)
patch_dim = all_patches[0].shape[1] if all_patches else 0
num_patches_per_frame = all_patches[0].shape[0] if all_patches else 0
print(f"  Done: {len(all_patches)} frames, {total_patches} total patches, dim={patch_dim}, per_frame={num_patches_per_frame}")
print(f"  Time: {elapsed_step1:.0f}s ({elapsed_step1/60:.1f} min)")

# ===== STEP 2: k-means clustering =====
print(f"\n=== Step 2: k-means clustering (k={K}) ===")
start = time.time()

# Stack all patches for clustering
all_patches_flat = torch.cat(all_patches, dim=0).numpy()  # (total_patches, dim)
print(f"  Clustering {all_patches_flat.shape[0]} patches × {all_patches_flat.shape[1]}d into {K} clusters...")

kmeans = MiniBatchKMeans(n_clusters=K, batch_size=min(10000, all_patches_flat.shape[0]), 
                          n_init=3, max_iter=100, verbose=0)
kmeans.fit(all_patches_flat)
codebook = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # (K, dim)

elapsed_step2 = time.time() - start
print(f"  Done: codebook shape = {codebook.shape}")
print(f"  Time: {elapsed_step2:.0f}s")

# Save codebook
codebook_path = os.path.join(OUTPUT_DIR, 'codebook.pt')
torch.save({'codebook': codebook, 'k': K, 'dim': patch_dim, 'num_patches_per_frame': num_patches_per_frame}, codebook_path)
print(f"  Saved: {codebook_path}")

# ===== STEP 3: Encode future frames → discrete token IDs =====
print(f"\n=== Step 3: Encoding future frames → discrete token IDs ===")
start = time.time()

codebook_gpu = codebook.to(device)
encoded_samples = []

for idx, sample in enumerate(samples):
    scene = sample.get('scene', '')
    frame_idx = sample.get('frame_idx', 0)
    cam_dir = os.path.join(img_base, scene, 'CAM_F0')
    
    if not os.path.exists(cam_dir):
        encoded_samples.append(None)
        continue
    
    frames = sorted(os.listdir(cam_dir))
    future_idx = frame_idx + 2  # t+1.0s at 2Hz
    
    if future_idx >= len(frames):
        encoded_samples.append(None)
        continue
    
    try:
        future_path = os.path.join(cam_dir, frames[future_idx])
        future_img = Image.open(future_path).convert('RGB').resize((448, 448))
        fut_inputs = processor.image_processor(images=[future_img], return_tensors='pt')
        fut_inputs = {k: v.to(device) for k, v in fut_inputs.items()}
        
        with torch.no_grad():
            fut_vit = vis_encoder(fut_inputs['pixel_values'], grid_thw=fut_inputs['image_grid_thw'])
            fut_patches = fut_vit.last_hidden_state.float()  # (num_patches, dim)
        
        # Find nearest codebook entry for each patch
        distances = torch.cdist(fut_patches.unsqueeze(0), codebook_gpu.unsqueeze(0)).squeeze(0)  # (num_patches, K)
        token_ids = distances.argmin(dim=-1).cpu().tolist()  # (num_patches,)
        
        encoded_samples.append(token_ids)
    except Exception as e:
        encoded_samples.append(None)
        if idx < 5:
            print(f"  Error encoding {idx}: {e}")
    
    if (idx + 1) % 100 == 0:
        valid = sum(1 for x in encoded_samples if x is not None)
        print(f"  {idx+1}/{len(samples)}: {valid} encoded")

valid_count = sum(1 for x in encoded_samples if x is not None)
elapsed_step3 = time.time() - start
print(f"  Done: {valid_count}/{len(samples)} future frames encoded")
print(f"  Token IDs per frame: {len(encoded_samples[0]) if encoded_samples[0] else 'N/A'}")
print(f"  Time: {elapsed_step3:.0f}s ({elapsed_step3/60:.1f} min)")

# Save augmented training data with visual token IDs
augmented_path = os.path.join(OUTPUT_DIR, 'train_with_visual_tokens.jsonl')
with open(augmented_path, 'w') as f:
    for sample, tokens in zip(samples, encoded_samples):
        sample_out = dict(sample)
        sample_out['future_visual_tokens'] = tokens  # None if no future frame
        f.write(json.dumps(sample_out) + '\n')

print(f"  Saved: {augmented_path}")

# Summary
total_time = time.time() - (start - elapsed_step2 - elapsed_step1)
print(f"\n{'='*60}")
print(f"  CODEBOOK BUILD COMPLETE")
print(f"  Codebook: {codebook_path} ({K} clusters × {patch_dim}d)")
print(f"  Data: {augmented_path} ({valid_count} with visual tokens)")
print(f"  Patches per frame: {num_patches_per_frame}")
print(f"  Total time: {elapsed_step1 + elapsed_step2 + elapsed_step3:.0f}s")
print(f"  CE loss initial (random): log({K}) ≈ {np.log(K):.1f}")
print(f"{'='*60}")

with open(os.path.join(OUTPUT_DIR, 'CODEBOOK_COMPLETE.txt'), 'w') as f:
    f.write(f"Done {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"k={K} dim={patch_dim} patches_per_frame={num_patches_per_frame}\n")
    f.write(f"valid_samples={valid_count}/{len(samples)}\n")
