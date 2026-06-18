#!/opt/onevl-env/bin/python3
# -*- coding: utf-8 -*-
"""
W9 Re-encode: Fix BF-037 by re-running Emu3 encoding with validation.
Previous run produced tokens in range [-5, 13] — clearly wrong.
Expected range: [0, 32768) for Emu3 32K codebook.

This script:
1. Loads Emu3 VisionTokenizer 
2. Validates it produces correct token range on a test image
3. Re-encodes ALL 250 scenes with per-scene validation
4. Saves with explicit int64 dtype assertion
5. Uploads result log to S3

Run: /opt/onevl-env/bin/python3 /opt/onevl-experiment/scripts/w9_reencode_emu3.py
"""
import torch, time, os, sys
import numpy as np
from PIL import Image
from pathlib import Path

SENSOR_DIR = Path('/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval')
OUTPUT_VIS_TOKENS = '/opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt'
ORIGINAL_VIS_TOKENS = '/opt/onevl-experiment/data/emu3_visual_tokens_1000.pt'
OUTPUT_DIR = '/opt/onevl-experiment/output/w9_step2'
LOG_FILE = os.path.join(OUTPUT_DIR, 'reencode_log.txt')
TARGET_SCENES = 250
FUTURE_OFFSET = 2  # frames ahead for "future frame"
CODEBOOK_SIZE = 32768

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    log("=" * 60)
    log("  W9 Re-encode: Emu3 Visual Tokens (BF-037 Fix)")
    log("=" * 60)

    # Step 0: Check if original 35-scene data is valid (as reference)
    log("\n--- Step 0: Validate original data (35 scenes) ---")
    if os.path.exists(ORIGINAL_VIS_TOKENS):
        orig = torch.load(ORIGINAL_VIS_TOKENS, weights_only=False)
        orig_min = min(t.min().item() for t in orig.values())
        orig_max = max(t.max().item() for t in orig.values())
        orig_dtypes = set(str(t.dtype) for t in orig.values())
        log(f"  Original: {len(orig)} scenes, range=[{orig_min}, {orig_max}], dtypes={orig_dtypes}")
        if orig_min >= 0 and orig_max < CODEBOOK_SIZE:
            log(f"  Original data VALID ✅ (will use as reference)")
        else:
            log(f"  Original data also corrupted!")
    else:
        log(f"  Original file not found (expected at {ORIGINAL_VIS_TOKENS})")
        orig = {}

    # Step 1: Load Emu3
    log("\n--- Step 1: Loading Emu3 VisionTokenizer ---")
    t0 = time.time()
    from transformers import Emu3VQVAE, Emu3ImageProcessor
    
    vq_model = Emu3VQVAE.from_pretrained(
        "BAAI/Emu3-VisionTokenizer", 
        trust_remote_code=True
    ).cuda().eval()
    image_processor = Emu3ImageProcessor.from_pretrained(
        "BAAI/Emu3-VisionTokenizer",
        trust_remote_code=True
    )
    log(f"  Loaded in {time.time()-t0:.1f}s")
    log(f"  Model device: {next(vq_model.parameters()).device}")
    log(f"  Codebook size: {vq_model.config.codebook_size if hasattr(vq_model.config, 'codebook_size') else 'unknown'}")

    # Step 2: Validation encode on test image
    log("\n--- Step 2: Test encode on synthetic image ---")
    test_img = Image.new('RGB', (640, 480), color=(128, 64, 32))
    processed = image_processor(test_img, return_tensors="pt")
    pv = processed["pixel_values"].cuda()
    sizes = processed["image_sizes"]
    log(f"  pixel_values shape: {pv.shape}, dtype: {pv.dtype}")
    log(f"  image_sizes: {sizes}")
    
    with torch.no_grad():
        result = vq_model.encode(pv, sizes)
    
    test_tokens = result[0].cpu()
    log(f"  Output shape: {test_tokens.shape}")
    log(f"  Output dtype: {test_tokens.dtype}")
    log(f"  Output range: [{test_tokens.min().item()}, {test_tokens.max().item()}]")
    
    if test_tokens.min().item() < 0 or test_tokens.max().item() >= CODEBOOK_SIZE:
        log(f"  ❌ CRITICAL: Test encode produced invalid range!")
        log(f"  The Emu3 model is NOT working correctly.")
        log(f"  Possible causes: wrong model version, truncated download, GPU issue")
        # Check if this is the known [-5, 13] pattern
        if test_tokens.max().item() < 100:
            log(f"  Pattern matches BF-037: model returning embedding indices, not codebook IDs")
            log(f"  Likely cause: Emu3 quantizer not initialized properly")
        sys.exit(1)
    else:
        log(f"  ✅ Test encode produces valid tokens in [0, {CODEBOOK_SIZE})")

    # Step 3: Test with a REAL NAVSIM image
    log("\n--- Step 3: Test encode on real NAVSIM image ---")
    all_scenes = sorted([d.name for d in SENSOR_DIR.iterdir() if d.is_dir()])
    test_scene = all_scenes[0]
    test_frames = sorted((SENSOR_DIR / test_scene / 'CAM_F0').glob('*.jpg'))
    if test_frames:
        real_img = Image.open(test_frames[0]).convert("RGB")
        log(f"  Image: {test_frames[0].name}, size={real_img.size}")
        processed = image_processor(real_img, return_tensors="pt")
        pv = processed["pixel_values"].cuda()
        sizes = processed["image_sizes"]
        
        with torch.no_grad():
            result = vq_model.encode(pv, sizes)
        
        real_tokens = result[0].cpu()
        log(f"  Shape: {real_tokens.shape}, dtype: {real_tokens.dtype}")
        log(f"  Range: [{real_tokens.min().item()}, {real_tokens.max().item()}]")
        
        if real_tokens.min().item() >= 0 and real_tokens.max().item() < CODEBOOK_SIZE:
            log(f"  ✅ Real image encoding valid")
        else:
            log(f"  ❌ Real image encoding INVALID")
            sys.exit(1)
    
    # Step 4: Encode all 250 scenes
    log(f"\n--- Step 4: Encoding {TARGET_SCENES} scenes ---")
    valid_scenes = []
    for scene in all_scenes:
        cam_dir = SENSOR_DIR / scene / 'CAM_F0'
        if cam_dir.exists():
            frames = sorted(cam_dir.glob('*.jpg'))
            if len(frames) >= 20:
                valid_scenes.append(scene)
    
    selected = valid_scenes[:TARGET_SCENES]
    log(f"  Selected {len(selected)} scenes for encoding")
    
    vis_tokens = {}
    errors = 0
    invalid_range = 0
    t_start = time.time()
    
    for si, scene in enumerate(selected):
        try:
            cam_dir = SENSOR_DIR / scene / 'CAM_F0'
            frames = sorted(cam_dir.glob('*.jpg'))
            
            # Use future frame (middle + offset)
            future_idx = min(len(frames) - 1, len(frames) // 2 + FUTURE_OFFSET)
            future_path = frames[future_idx]
            
            img = Image.open(future_path).convert("RGB")
            processed = image_processor(img, return_tensors="pt")
            pv = processed["pixel_values"].cuda()
            sizes = processed["image_sizes"]
            
            with torch.no_grad():
                result = vq_model.encode(pv, sizes)
            
            tokens = result[0].cpu()
            
            # VALIDATE every single encoding
            assert tokens.dtype == torch.int64, f"Scene {scene}: dtype={tokens.dtype}, expected int64"
            assert tokens.min().item() >= 0, f"Scene {scene}: min={tokens.min().item()} < 0"
            assert tokens.max().item() < CODEBOOK_SIZE, f"Scene {scene}: max={tokens.max().item()} >= {CODEBOOK_SIZE}"
            
            vis_tokens[scene] = tokens
            
        except AssertionError as e:
            invalid_range += 1
            log(f"  ❌ INVALID: {e}")
            if invalid_range >= 3:
                log(f"  Too many invalid encodings. Emu3 model is broken. Stopping.")
                sys.exit(1)
        except Exception as e:
            errors += 1
            if errors <= 5:
                log(f"  Error {scene}: {type(e).__name__}: {str(e)[:80]}")
        
        # Progress + checkpoint
        if (si + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (si + 1) / elapsed
            log(f"  Progress: {si+1}/{len(selected)} ({rate:.1f} scenes/s), "
                f"valid={len(vis_tokens)}, errors={errors}")
            torch.save(vis_tokens, OUTPUT_VIS_TOKENS)
            torch.cuda.empty_cache()
    
    # Final save
    torch.save(vis_tokens, OUTPUT_VIS_TOKENS)
    elapsed = time.time() - t_start
    
    # Final validation
    log(f"\n--- Step 5: Final validation ---")
    all_min = min(t.min().item() for t in vis_tokens.values())
    all_max = max(t.max().item() for t in vis_tokens.values())
    all_dtypes = set(str(t.dtype) for t in vis_tokens.values())
    
    log(f"  Scenes encoded: {len(vis_tokens)}/{TARGET_SCENES}")
    log(f"  Token range: [{all_min}, {all_max}]")
    log(f"  Dtypes: {all_dtypes}")
    log(f"  Errors: {errors}, Invalid range: {invalid_range}")
    log(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if all_min >= 0 and all_max < CODEBOOK_SIZE and all_dtypes == {'torch.int64'}:
        log(f"\n  ✅ ALL TOKENS VALID. Ready for training.")
    else:
        log(f"\n  ❌ VALIDATION FAILED. Do not train.")
        sys.exit(1)
    
    log(f"\n{'='*60}")
    log(f"  RE-ENCODING COMPLETE")
    log(f"  {len(vis_tokens)} scenes, range [{all_min}, {all_max}], int64")
    log(f"{'='*60}")

if __name__ == '__main__':
    main()
