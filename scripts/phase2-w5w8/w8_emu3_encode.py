#!/usr/bin/env python3
"""
W8 方案 C Step 1-3: Encode future frames using Emu3 VisionTokenizer.

Uses BAAI/Emu3-VisionTokenizer to produce discrete visual token IDs
(the same tokenizer OneVL uses in their paper).

Steps:
1. Download & load Emu3 VisionTokenizer
2. Verify on 1 image (check output format, codebook size)
3. Encode 1000 future frames → discrete token IDs
4. Save to JSONL for training
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image

DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
OUTPUT_DIR = '/opt/onevl-experiment/output/w8_emu3'
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'
TRAINVAL_ALT = '/opt/onevl-experiment/OneVL_training/navsim_v1.1_all/dataset/sensor_blobs/trainval'

os.makedirs(OUTPUT_DIR, exist_ok=True)
img_base = TRAINVAL if os.path.exists(TRAINVAL) else TRAINVAL_ALT
device = 'cuda'

print('=' * 60)
print('  W8 方案 C: Emu3 VisionTokenizer Encoding')
print('=' * 60)

# Step 1: Load Emu3 VisionTokenizer
print("\n=== Step 1: Loading Emu3 VisionTokenizer ===")
start = time.time()

try:
    from transformers import Emu3VQVAE, Emu3ImageProcessor
    
    print("  Loading Emu3VQVAE...")
    vq_model = Emu3VQVAE.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True).to(device).eval()
    print("  Loading Emu3ImageProcessor...")
    image_processor = Emu3ImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
    MODEL_ID = "BAAI/Emu3-VisionTokenizer (Emu3VQVAE + Emu3ImageProcessor)"
    print(f"  ✅ Loaded: {MODEL_ID}")

except Exception as e:
    print(f"  Error: {e}")
    print("  Trying Emu3.5-VisionTokenizer...")
    try:
        from transformers import Emu3VQVAE, Emu3ImageProcessor
        vq_model = Emu3VQVAE.from_pretrained("BAAI/Emu3.5-VisionTokenizer", trust_remote_code=True).to(device).eval()
        image_processor = Emu3ImageProcessor.from_pretrained("BAAI/Emu3.5-VisionTokenizer", trust_remote_code=True)
        MODEL_ID = "BAAI/Emu3.5-VisionTokenizer"
        print(f"  ✅ Loaded: {MODEL_ID}")
    except Exception as e2:
        print(f"  Also failed: {e2}")
        import transformers
        print(f"  transformers version: {transformers.__version__}")
        emu_classes = [x for x in dir(transformers) if 'emu' in x.lower() or 'Emu' in x]
        print(f"  Available Emu classes: {emu_classes}")
        sys.exit(1)

load_time = time.time() - start
print(f"  Load time: {load_time:.0f}s")
print(f"  Model device: {next(vq_model.parameters()).device}")
print(f"  Model dtype: {next(vq_model.parameters()).dtype}")

# Step 2: Verify on 1 image
print("\n=== Step 2: Verification (1 image) ===")
samples = []
with open(DATA_PATH) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))
            if len(samples) >= 1000:
                break

# Find first sample with a future frame
test_sample = None
for s in samples:
    scene = s.get('scene', '')
    frame_idx = s.get('frame_idx', 0)
    cam_dir = os.path.join(img_base, scene, 'CAM_F0')
    if os.path.exists(cam_dir):
        frames = sorted(os.listdir(cam_dir))
        if frame_idx + 2 < len(frames):
            test_img_path = os.path.join(cam_dir, frames[frame_idx + 2])
            test_sample = s
            break

if test_sample is None:
    print("  ERROR: No test sample found!")
    sys.exit(1)

print(f"  Test image: {test_img_path}")
test_img = Image.open(test_img_path).convert('RGB')
print(f"  Image size: {test_img.size}")

# Encode
with torch.no_grad():
    processed = image_processor(images=test_img, return_tensors='pt').to(device)
    print(f"  Processed keys: {list(processed.keys())}")
    for k, v in processed.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape}")
    
    # Encode to discrete tokens
    # Emu3VQVAE uses .encode() which returns codes
    if hasattr(vq_model, 'encode'):
        encoded = vq_model.encode(processed['pixel_values'], processed.get('image_sizes'))
    else:
        encoded = vq_model(processed['pixel_values'])
    
    # Extract token IDs from encoded output
    if isinstance(encoded, torch.Tensor):
        token_ids = encoded
    elif hasattr(encoded, 'codes'):
        token_ids = encoded.codes
    elif hasattr(encoded, 'indices'):
        token_ids = encoded.indices
    elif isinstance(encoded, tuple):
        # Try each element
        for elem in encoded:
            if isinstance(elem, torch.Tensor) and elem.dtype in [torch.long, torch.int32, torch.int64]:
                token_ids = elem
                break
        else:
            token_ids = encoded[0]
    else:
        print(f"  Encoded type: {type(encoded)}")
        print(f"  Encoded attrs: {[a for a in dir(encoded) if not a.startswith('_')]}")
        token_ids = encoded

    if hasattr(token_ids, 'shape'):
        print(f"  Token IDs shape: {token_ids.shape}")
        print(f"  Token IDs dtype: {token_ids.dtype}")
        print(f"  Token IDs range: [{token_ids.min().item()}, {token_ids.max().item()}]")
        codebook_size = token_ids.max().item() + 1
        num_tokens = token_ids.numel()
        print(f"  Codebook size (inferred): {codebook_size}")
        print(f"  Tokens per image: {num_tokens}")
        print(f"  CE loss initial (random): log({codebook_size}) ≈ {np.log(codebook_size):.1f}")
    else:
        print(f"  Token IDs type: {type(token_ids)}")
        print(f"  Cannot determine shape — need to debug API")
        sys.exit(1)

# Step 3: Encode all future frames
print(f"\n=== Step 3: Encoding {len(samples)} future frames ===")
start = time.time()
encoded_samples = []
errors = 0

for idx, sample in enumerate(samples):
    scene = sample.get('scene', '')
    frame_idx = sample.get('frame_idx', 0)
    cam_dir = os.path.join(img_base, scene, 'CAM_F0')
    
    if not os.path.exists(cam_dir):
        encoded_samples.append(None)
        continue
    
    frames = sorted(os.listdir(cam_dir))
    future_idx = frame_idx + 2
    
    if future_idx >= len(frames):
        encoded_samples.append(None)
        continue
    
    try:
        future_path = os.path.join(cam_dir, frames[future_idx])
        future_img = Image.open(future_path).convert('RGB')
        
        with torch.no_grad():
            processed = image_processor(images=future_img, return_tensors='pt').to(device)
            if hasattr(vq_model, 'encode'):
                encoded = vq_model.encode(processed['pixel_values'], processed.get('image_sizes'))
            elif hasattr(vq_model, 'quantize'):
                encoded = vq_model.quantize(processed['pixel_values'])
            else:
                encoded = vq_model(processed['pixel_values'])
            
            if hasattr(encoded, 'codes'):
                ids = encoded.codes.cpu().flatten().tolist()
            elif hasattr(encoded, 'indices'):
                ids = encoded.indices.cpu().flatten().tolist()
            elif isinstance(encoded, tuple):
                ids = encoded[1].cpu().flatten().tolist() if len(encoded) > 1 else encoded[0].cpu().flatten().tolist()
            else:
                ids = token_ids.cpu().flatten().tolist()
            
            encoded_samples.append(ids)
    except Exception as e:
        errors += 1
        encoded_samples.append(None)
        if errors <= 3:
            print(f"  Error {idx}: {e}")
    
    if (idx + 1) % 100 == 0:
        valid = sum(1 for x in encoded_samples if x is not None)
        print(f"  {idx+1}/{len(samples)}: {valid} encoded, {errors} errors")

valid_count = sum(1 for x in encoded_samples if x is not None)
elapsed = time.time() - start
print(f"  Done: {valid_count}/{len(samples)} encoded, {errors} errors, {elapsed:.0f}s")

# Step 4: Save
print(f"\n=== Step 4: Saving ===")
# Fix message format + add emu3 visual tokens
output_path = os.path.join(OUTPUT_DIR, 'train_with_emu3_tokens.jsonl')
with open(output_path, 'w') as f:
    for sample, tokens in zip(samples, encoded_samples):
        out = dict(sample)
        # Fix image→path format
        for msg in out['messages']:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') == 'image' and 'image' in item:
                        item['path'] = item.pop('image')
        out['emu3_visual_tokens'] = tokens
        f.write(json.dumps(out) + '\n')

# Save metadata
meta = {
    'tokenizer': MODEL_ID,
    'codebook_size': codebook_size,
    'tokens_per_image': num_tokens,
    'valid_samples': valid_count,
    'total_samples': len(samples),
    'ce_loss_initial': float(np.log(codebook_size)),
}
with open(os.path.join(OUTPUT_DIR, 'emu3_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n{'='*60}")
print(f"  EMU3 ENCODING COMPLETE")
print(f"  Tokenizer: {MODEL_ID}")
print(f"  Codebook: {codebook_size}")
print(f"  Tokens/image: {num_tokens}")
print(f"  Valid: {valid_count}/{len(samples)}")
print(f"  CE loss initial: log({codebook_size}) ≈ {np.log(codebook_size):.1f}")
print(f"  Output: {output_path}")
print(f"{'='*60}")

with open(os.path.join(OUTPUT_DIR, 'ENCODING_COMPLETE.txt'), 'w') as f:
    f.write(f"Done {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"codebook={codebook_size} tokens/img={num_tokens} valid={valid_count}\n")
