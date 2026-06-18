# -*- coding: utf-8 -*-
#!/opt/onevl-env/bin/python3
"""
W9 Step 2: Expand data to 200+ unique scenes.
1. Build training JSONL from ALL available scenes (not just the original 35)
2. Encode future frames with Emu3 for new scenes
3. Output: expanded JSONL + expanded visual tokens .pt

Input: NAVSIM sensor_blobs (1192 scenes), metadata pkls
Output: 
  - /opt/onevl-experiment/data/navsim_real_traj_expanded.jsonl (200+ unique scenes)
  - /opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt (200+ scene encodings)
"""
import json, torch, time, os, sys, pickle, re
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation

# Config
SENSOR_DIR = Path('/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval')
META_DIR = Path('/opt/onevl-experiment/navsim_v1.1_all/dataset/openscene-v1.1/meta_datas/trainval')
OUTPUT_JSONL = '/opt/onevl-experiment/data/navsim_real_traj_expanded.jsonl'
OUTPUT_VIS_TOKENS = '/opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt'
LOG_FILE = '/opt/onevl-experiment/output/w9_correct/expand_data_log.txt'

TARGET_SCENES = 250  # aim for 250 unique scenes
SAMPLES_PER_SCENE = 5  # 5 different frames per scene -> 1250 total samples
FUTURE_OFFSET = 2  # t+1.0s at 2Hz

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def extract_trajectory(meta_frames, start_idx, num_waypoints=8):
    """Extract ego-relative trajectory from NAVSIM metadata frames."""
    if start_idx + num_waypoints >= len(meta_frames):
        return None
    
    # Current position and rotation
    curr_pos = np.array(meta_frames[start_idx]['ego2global_translation'])
    curr_rot_q = meta_frames[start_idx].get('ego2global_rotation', [1, 0, 0, 0])
    
    # Convert quaternion to rotation matrix for ego-relative coords
    try:
        R_inv = Rotation.from_quat([curr_rot_q[1], curr_rot_q[2], curr_rot_q[3], curr_rot_q[0]]).inv()
    except:
        R_inv = Rotation.identity()
    
    waypoints = []
    for i in range(1, num_waypoints + 1):
        future_pos = np.array(meta_frames[start_idx + i]['ego2global_translation'])
        # Convert to ego-relative
        rel_pos = R_inv.apply(future_pos - curr_pos)
        # heading change
        try:
            future_rot_q = meta_frames[start_idx + i].get('ego2global_rotation', [1, 0, 0, 0])
            future_R = Rotation.from_quat([future_rot_q[1], future_rot_q[2], future_rot_q[3], future_rot_q[0]])
            heading = (future_R * R_inv.inv()).as_euler('xyz')[2]  # yaw
        except:
            heading = 0.0
        waypoints.append([round(rel_pos[0], 4), round(rel_pos[1], 4), round(heading, 4)])
    
    return waypoints

def main():
    log("=" * 60)
    log("  W9 Step 2: Expand Data to 200+ Unique Scenes")
    log("=" * 60)
    
    # --- Step 1: Find all available scenes ---
    log("\n=== Step 1: Finding scenes ===")
    all_scenes = sorted([d.name for d in SENSOR_DIR.iterdir() if d.is_dir()])
    log(f"  Total scenes in sensor_blobs: {len(all_scenes)}")
    
    # Filter to scenes that have CAM_F0 images AND metadata
    valid_scenes = []
    for scene in all_scenes:
        cam_dir = SENSOR_DIR / scene / 'CAM_F0'
        meta_file = META_DIR / f"{scene}.pkl"
        if cam_dir.exists() and meta_file.exists():
            frames = sorted(cam_dir.glob('*.jpg'))
            if len(frames) >= 20:  # need enough frames for trajectory + future
                valid_scenes.append(scene)
    
    log(f"  Valid scenes (have CAM_F0 + metadata + >=20 frames): {len(valid_scenes)}")
    
    # Take up to TARGET_SCENES
    selected_scenes = valid_scenes[:TARGET_SCENES]
    log(f"  Selected: {len(selected_scenes)} scenes")
    
    # --- Step 2: Extract trajectories and build JSONL ---
    log("\n=== Step 2: Extracting trajectories ===")
    samples = []
    errors = 0
    
    for si, scene in enumerate(selected_scenes):
        try:
            # Load metadata
            meta_file = META_DIR / f"{scene}.pkl"
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            
            # Get sorted frame files
            cam_dir = SENSOR_DIR / scene / 'CAM_F0'
            frames = sorted(cam_dir.glob('*.jpg'))
            
            # Sample SAMPLES_PER_SCENE frames from this scene
            n_frames = len(meta) if isinstance(meta, list) else len(frames)
            # Pick evenly spaced start indices (leaving room for trajectory + future)
            max_start = n_frames - 10 - FUTURE_OFFSET  # need 8 waypoints + future frame
            if max_start < 1:
                errors += 1
                continue
            
            indices = np.linspace(0, max_start - 1, SAMPLES_PER_SCENE).astype(int)
            
            for idx in indices:
                # Extract trajectory
                if isinstance(meta, list):
                    traj = extract_trajectory(meta, idx)
                else:
                    continue
                
                if traj is None:
                    continue
                
                # Format trajectory as text
                traj_text = '<answer>' + ', '.join([f'[{w[0]}, {w[1]}, {w[2]}]' for w in traj]) + '</answer>'
                
                # Get image path
                frame_file = frames[idx] if idx < len(frames) else frames[0]
                img_path = str(frame_file)
                
                sample = {
                    'messages': [
                        {'role': 'user', 'content': [
                            {'type': 'image', 'image': img_path},
                            {'type': 'text', 'text': 'Predict the ego vehicle trajectory for the next 4 seconds.'}
                        ]},
                        {'role': 'assistant', 'content': traj_text}
                    ],
                    'images': [img_path],
                    'scene': scene,
                    'frame_idx': int(idx),
                    'source': 'navsim_expanded'
                }
                samples.append(sample)
                
        except Exception as e:
            errors += 1
            if errors <= 5:
                log(f"  Error scene {scene}: {type(e).__name__}: {str(e)[:80]}")
            continue
        
        if (si + 1) % 50 == 0:
            log(f"  Progress: {si+1}/{len(selected_scenes)} scenes, {len(samples)} samples")
    
    unique_scenes_in_data = len(set(s['scene'] for s in samples))
    log(f"  Total samples: {len(samples)}")
    log(f"  Unique scenes: {unique_scenes_in_data}")
    log(f"  Errors: {errors}")
    
    # Save JSONL
    with open(OUTPUT_JSONL, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    log(f"  Saved: {OUTPUT_JSONL}")
    
    # --- Step 3: Encode future frames with Emu3 ---
    log("\n=== Step 3: Emu3 encoding (future frames) ===")
    
    from transformers import Emu3VQVAE, Emu3ImageProcessor
    
    vq_model = Emu3VQVAE.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
    vq_model = vq_model.cuda().eval()
    image_processor = Emu3ImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
    log(f"  Emu3 loaded")
    
    # Load existing tokens (if any)
    vis_tokens = {}
    if os.path.exists(OUTPUT_VIS_TOKENS):
        vis_tokens = torch.load(OUTPUT_VIS_TOKENS, weights_only=False)
        log(f"  Loaded {len(vis_tokens)} existing encodings")
    
    encoded = 0
    encode_errors = 0
    
    for si, scene in enumerate(selected_scenes):
        if scene in vis_tokens:
            continue  # already encoded
        
        try:
            cam_dir = SENSOR_DIR / scene / 'CAM_F0'
            frames = sorted(cam_dir.glob('*.jpg'))
            
            # Use a frame from the middle as the "future frame" for this scene
            future_idx = min(len(frames) - 1, len(frames) // 2 + FUTURE_OFFSET)
            future_path = frames[future_idx]
            
            img = Image.open(future_path).convert("RGB")
            processed = image_processor(img, return_tensors="pt")
            pv = processed["pixel_values"].cuda()
            sizes = processed["image_sizes"]
            
            with torch.no_grad():
                result = vq_model.encode(pv, sizes)
            
            vis_tokens[scene] = result[0].cpu()  # (96, 170) int64
            encoded += 1
            
        except Exception as e:
            encode_errors += 1
            if encode_errors <= 5:
                log(f"  Encode error {scene}: {type(e).__name__}: {str(e)[:60]}")
            continue
        
        # Save checkpoint every 50 scenes
        if (encoded) % 50 == 0:
            torch.save(vis_tokens, OUTPUT_VIS_TOKENS)
            log(f"  Checkpoint: {encoded} encoded, {len(vis_tokens)} total")
    
    # Final save
    torch.save(vis_tokens, OUTPUT_VIS_TOKENS)
    
    log(f"\n{'='*60}")
    log(f"  DATA EXPANSION COMPLETE")
    log(f"{'='*60}")
    log(f"  Training JSONL: {len(samples)} samples, {unique_scenes_in_data} unique scenes")
    log(f"  Visual tokens: {len(vis_tokens)} scenes encoded")
    log(f"  New encodings: {encoded}, errors: {encode_errors}")
    log(f"  Output files:")
    log(f"    {OUTPUT_JSONL}")
    log(f"    {OUTPUT_VIS_TOKENS}")
    
    with open('/opt/onevl-experiment/output/w9_correct/DATA_EXPANSION_COMPLETE.txt', 'w') as f:
        f.write(f"Samples: {len(samples)}\n")
        f.write(f"Unique scenes: {unique_scenes_in_data}\n")
        f.write(f"Visual tokens: {len(vis_tokens)}\n")

if __name__ == "__main__":
    main()
