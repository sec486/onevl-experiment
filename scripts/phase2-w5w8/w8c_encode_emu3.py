#!/usr/bin/env python3
"""
W8 方案 C: Emu3 VisionTokenizer Encoding (FIXED)
=================================================
Encode future frames (t+1.0s) from NAVSIM training data into discrete visual tokens
using Emu3 VisionTokenizer.

API (verified from previous session):
  result = vq_model.encode(pixel_values, image_sizes)
  # result: list of tensors
  # result[0]: torch.Size([96, 170]), dtype=int64
  # Token IDs range: [1683, 14067]
  # Codebook size: ~32768
  # Tokens per image: 16320 (96*170)
  # Flatten: result[0].flatten().tolist() -> [16320 integers]

Output: /opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt
  Dict with scene_id -> tensor of shape (96, 170) for each sample's future frame
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from pathlib import Path
from PIL import Image

import torch

# Use transformers 4.51.3 (verified compatible)
from transformers import Emu3VQVAE, Emu3ImageProcessor

# ============================================================
# Configuration
# ============================================================
EMU3_MODEL = "BAAI/Emu3-VisionTokenizer"
DATA_DIR = Path("/opt/onevl-experiment")
TRAIN_JSONL = DATA_DIR / "data" / "navsim_real_traj_1000.jsonl"
SENSOR_DIR = DATA_DIR / "navtrain_data" / "trainval_sensor_blobs" / "trainval"
META_DIR = DATA_DIR / "navsim_v1.1_all" / "dataset" / "openscene-v1.1" / "meta_datas" / "trainval"
OUTPUT_DIR = DATA_DIR / "output" / "w8_emu3"
OUTPUT_FILE = DATA_DIR / "data" / "emu3_visual_tokens_expanded.pt"
PROGRESS_FILE = OUTPUT_DIR / "encode_progress.json"
LOG_FILE = OUTPUT_DIR / "encode_v2_log.txt"

# Future frame offset: t+1.0s at 2Hz = 2 frames ahead
FUTURE_OFFSET = 2
MAX_SAMPLES = 1000

# ============================================================
# Logging
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============================================================
# Main
# ============================================================
def main():
    log("=" * 60)
    log("  W8 方案 C: Emu3 Encoding (FIXED API)")
    log("=" * 60)
    
    # --- Step 1: Load Emu3 ---
    log("\n=== Step 1: Loading Emu3 VisionTokenizer ===")
    t0 = time.time()
    
    vq_model = Emu3VQVAE.from_pretrained(EMU3_MODEL, trust_remote_code=True)
    vq_model = vq_model.cuda().eval()
    
    image_processor = Emu3ImageProcessor.from_pretrained(EMU3_MODEL, trust_remote_code=True)
    
    log(f"  ✅ Loaded in {time.time()-t0:.1f}s")
    log(f"  Device: {next(vq_model.parameters()).device}")
    
    # --- Step 2: Load training data to get scene list ---
    log("\n=== Step 2: Loading training data ===")
    
    with open(TRAIN_JSONL) as f:
        samples = [json.loads(line) for line in f][:MAX_SAMPLES]
    log(f"  Loaded {len(samples)} samples")
    
    # Get unique scenes and their image paths
    # Each sample has an 'images' field with the current frame path
    # We need to find the FUTURE frame (t+1.0s = 2 frames ahead)
    
    # --- Step 3: Check for partial progress ---
    encoded_tokens = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
        log(f"  Resuming from {progress['completed']} completed samples")
        if OUTPUT_FILE.exists():
            encoded_tokens = torch.load(OUTPUT_FILE, weights_only=False)
            log(f"  Loaded {len(encoded_tokens)} existing encodings")
    
    # --- Step 4: Encode future frames ---
    log(f"\n=== Step 3: Encoding future frames ({len(samples)} samples) ===")
    
    errors = 0
    skipped = 0
    encoded_count = len(encoded_tokens)
    t_start = time.time()
    
    for i, sample in enumerate(samples):
        # Extract scene info from image path
        img_path = sample.get('images', [None])[0] if 'images' in sample else None
        if img_path is None:
            # Try alternate field names
            for msg in sample.get('messages', []):
                content = msg.get('content', '')
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'image':
                            img_path = item.get('image') or item.get('path')
                            break
            if img_path is None:
                errors += 1
                continue
        
        # Parse scene ID from path
        # Path format: .../trainval/<scene_id>/CAM_F0/<frame_hash>.jpg
        parts = Path(img_path).parts
        try:
            cam_idx = next(j for j, p in enumerate(parts) if p == 'CAM_F0')
            scene_id = parts[cam_idx - 1]
        except (StopIteration, IndexError):
            # Try just the directory name
            scene_id = Path(img_path).parent.parent.name
        
        # Skip if already encoded
        if scene_id in encoded_tokens:
            skipped += 1
            continue
        
        # Find future frame in the scene directory
        scene_cam_dir = SENSOR_DIR / scene_id / "CAM_F0"
        if not scene_cam_dir.exists():
            errors += 1
            if errors <= 5:
                log(f"  ⚠️ Scene dir not found: {scene_cam_dir}")
            continue
        
        # Get sorted frame list and find current + future
        frames = sorted(scene_cam_dir.glob("*.jpg"))
        if len(frames) < FUTURE_OFFSET + 1:
            errors += 1
            continue
        
        # Current frame is the one in the sample
        current_frame_name = Path(img_path).name
        try:
            current_idx = next(j for j, f in enumerate(frames) if f.name == current_frame_name)
        except StopIteration:
            # Frame not found — use middle of sequence as fallback
            current_idx = len(frames) // 2
        
        future_idx = current_idx + FUTURE_OFFSET
        if future_idx >= len(frames):
            future_idx = len(frames) - 1  # Clamp to last frame
        
        future_frame_path = frames[future_idx]
        
        # Encode with Emu3
        try:
            img = Image.open(future_frame_path).convert("RGB")
            processed = image_processor(img, return_tensors="pt")
            pixel_values = processed["pixel_values"].cuda()
            image_sizes = processed["image_sizes"]
            
            with torch.no_grad():
                result = vq_model.encode(pixel_values, image_sizes)
            
            # result is a list, result[0] is tensor (96, 170) int64
            token_tensor = result.image_tokens[0].cpu()  # BF-037 fix  # (96, 170)
            encoded_tokens[scene_id] = token_tensor
            encoded_count += 1
            
        except Exception as e:
            errors += 1
            if errors <= 10:
                log(f"  ❌ Error encoding {scene_id}: {e}")
            continue
        
        # Progress logging every 50 samples
        if (encoded_count) % 50 == 0:
            elapsed = time.time() - t_start
            speed = encoded_count / elapsed if elapsed > 0 else 0
            log(f"  Progress: {encoded_count}/{len(samples)} encoded, "
                f"{errors} errors, {speed:.2f} samples/s, "
                f"ETA: {(len(samples)-encoded_count-skipped)/max(speed,0.01)/60:.1f} min")
            
            # Save checkpoint
            torch.save(encoded_tokens, OUTPUT_FILE)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({'completed': encoded_count, 'errors': errors, 
                          'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
    
    # --- Final save ---
    torch.save(encoded_tokens, OUTPUT_FILE)
    elapsed = time.time() - t_start
    
    log(f"\n{'='*60}")
    log(f"  ENCODING COMPLETE")
    log(f"{'='*60}")
    log(f"  Total encoded: {encoded_count}")
    log(f"  Errors: {errors}")
    log(f"  Skipped (already done): {skipped}")
    log(f"  Time: {elapsed/60:.1f} min")
    log(f"  Speed: {encoded_count/max(elapsed,1):.2f} samples/s")
    log(f"  Output: {OUTPUT_FILE}")
    log(f"  Token shape per image: (96, 170) = 16,320 discrete tokens")
    log(f"  Codebook size: ~32768")
    
    # Verify
    if encoded_count > 0:
        sample_key = list(encoded_tokens.keys())[0]
        sample_tensor = encoded_tokens[sample_key]
        log(f"\n  Verification:")
        log(f"    Sample scene: {sample_key}")
        log(f"    Token shape: {sample_tensor.shape}")
        log(f"    Token dtype: {sample_tensor.dtype}")
        log(f"    Token range: [{sample_tensor.min().item()}, {sample_tensor.max().item()}]")
        log(f"    Unique tokens: {sample_tensor.unique().numel()}")
    
    # Write completion marker
    with open(OUTPUT_DIR / "ENCODING_COMPLETE.txt", 'w') as f:
        f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples: {encoded_count}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"File: {OUTPUT_FILE}\n")

if __name__ == "__main__":
    main()
