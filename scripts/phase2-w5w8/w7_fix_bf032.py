#!/usr/bin/env python3
"""
W7: Fix BF-032 — Extract REAL trajectory GT from NAVSIM metadata .pkl files,
generate correct training data, then train c-full with both decoders.

BF-032 Root Cause: Previous c-full training used synthetic/random trajectories
instead of real ego-vehicle trajectory ground truth.

Fix: Parse NAVSIM pkl metadata → extract ego2global_translation per frame →
convert to ego-relative waypoints → use as training GT.

Pkl Structure (per scene):
  - List of N frame dicts (N=398-774 typically)
  - Each frame: {ego2global_translation: [x,y,z], ego2global_rotation: [w,x,y,z],
    driving_command: [one-hot 4], ego_dynamic_state: [vx,vy,ax,ay], ...}
  - Frame rate: 2Hz (0.5s between frames)
  - Trajectory GT: 8 future waypoints at 0.5s intervals = 4s horizon
"""
import json, os, sys, pickle, time
import numpy as np

# Paths
METADATA = '/opt/onevl-experiment/navsim_v1.1_all/dataset/openscene-v1.1/meta_datas/trainval'
TRAINVAL = '/opt/onevl-experiment/navtrain_data/trainval_sensor_blobs/trainval'
# Fallback: check symlinked path
TRAINVAL_ALT = '/opt/onevl-experiment/OneVL_training/navsim_v1.1_all/dataset/sensor_blobs/trainval'
OUTPUT_JSONL = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'

# Driving command mapping (one-hot to text)
COMMAND_MAP = {
    (1,0,0,0): "FORWARD",
    (0,1,0,0): "LEFT",
    (0,0,1,0): "RIGHT",
    (0,0,0,1): "STOP",
}

def quaternion_to_yaw(q):
    """Convert quaternion [w, x, y, z] to yaw angle (rotation around z-axis)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def global_to_ego(ref_trans, ref_rot, target_trans):
    """Convert global position to ego-relative coordinates.
    
    Args:
        ref_trans: [x, y, z] global position of ego at reference time
        ref_rot: [w, x, y, z] quaternion of ego at reference time
        target_trans: [x, y, z] global position to convert
    
    Returns:
        [x_ego, y_ego] in ego frame (x=forward, y=left)
    """
    # Get yaw of reference frame
    yaw = quaternion_to_yaw(ref_rot)
    
    # Translation relative to reference
    dx = target_trans[0] - ref_trans[0]
    dy = target_trans[1] - ref_trans[1]
    
    # Rotate into ego frame (inverse rotation by -yaw)
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    x_ego = cos_yaw * dx - sin_yaw * dy
    y_ego = sin_yaw * dx + cos_yaw * dy
    
    return x_ego, y_ego

def extract_trajectory(frames, start_idx, num_waypoints=8):
    """Extract trajectory (8 future waypoints) relative to frame at start_idx.
    
    Returns:
        list of [x, y, heading] tuples, or None if not enough future frames
    """
    if start_idx + num_waypoints >= len(frames):
        return None
    
    ref_frame = frames[start_idx]
    ref_trans = ref_frame['ego2global_translation']
    ref_rot = ref_frame['ego2global_rotation']
    ref_yaw = quaternion_to_yaw(ref_rot)
    
    trajectory = []
    for i in range(1, num_waypoints + 1):
        future_frame = frames[start_idx + i]
        future_trans = future_frame['ego2global_translation']
        future_rot = future_frame['ego2global_rotation']
        
        # Convert to ego-relative position
        x_ego, y_ego = global_to_ego(ref_trans, ref_rot, future_trans)
        
        # Heading = relative yaw change
        future_yaw = quaternion_to_yaw(future_rot)
        heading = future_yaw - ref_yaw
        # Normalize to [-pi, pi]
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        
        trajectory.append([round(float(x_ego), 4), round(float(y_ego), 4), round(float(heading), 4)])
    
    return trajectory

def format_trajectory(traj):
    """Format trajectory as OneVL answer string: [x1,y1,h1], [x2,y2,h2], ..."""
    parts = [f"[{wp[0]:.4f},{wp[1]:.4f},{wp[2]:.4f}]" for wp in traj]
    return ", ".join(parts)

def get_driving_command_text(cmd):
    """Convert one-hot driving command to text."""
    cmd_tuple = tuple(int(c) for c in cmd)
    return COMMAND_MAP.get(cmd_tuple, "FORWARD")

def main():
    print("=" * 60)
    print("  W7 Fix BF-032: Extract Real Trajectory GT from NAVSIM")
    print("=" * 60)
    
    # Determine image path
    if os.path.exists(TRAINVAL):
        img_base = TRAINVAL
    elif os.path.exists(TRAINVAL_ALT):
        img_base = TRAINVAL_ALT
    else:
        print(f"ERROR: Neither {TRAINVAL} nor {TRAINVAL_ALT} exists!")
        sys.exit(1)
    print(f"Image base: {img_base}")
    
    # Find matching scenes (have both images and metadata)
    metadata_files = sorted([f for f in os.listdir(METADATA) if f.endswith('.pkl')])
    scenes_with_meta = {f.replace('.pkl', '') for f in metadata_files}
    scenes_with_images = set(os.listdir(img_base))
    valid_scenes = sorted(scenes_with_meta & scenes_with_images)
    
    print(f"Metadata files: {len(metadata_files)}")
    print(f"Scenes with images: {len(scenes_with_images)}")
    print(f"Valid scenes (both): {len(valid_scenes)}")
    
    if len(valid_scenes) == 0:
        print("ERROR: No valid scenes found! Checking paths...")
        print(f"  First 5 metadata: {metadata_files[:5]}")
        print(f"  First 5 images: {sorted(scenes_with_images)[:5]}")
        sys.exit(1)
    
    # Extract trajectories from all valid scenes
    all_samples = []
    skipped_short = 0
    skipped_no_cam = 0
    
    for scene_idx, scene in enumerate(valid_scenes):
        pkl_path = os.path.join(METADATA, scene + '.pkl')
        cam_dir = os.path.join(img_base, scene, 'CAM_F0')
        
        if not os.path.exists(cam_dir):
            skipped_no_cam += 1
            continue
        
        cam_frames = sorted(os.listdir(cam_dir))
        if len(cam_frames) < 3:
            skipped_short += 1
            continue
        
        # Load metadata
        try:
            with open(pkl_path, 'rb') as f:
                meta = pickle.load(f)
        except Exception as e:
            print(f"  Error loading {scene}: {e}")
            continue
        
        if not isinstance(meta, list) or len(meta) < 10:
            continue
        
        # Sample training frames from this scene
        # Use every 4th frame as a training sample (gives ~2s spacing between samples)
        # Each frame needs 8 future frames (4s) of trajectory
        num_frames = len(meta)
        
        for frame_idx in range(0, num_frames - 8, 4):
            # Extract trajectory
            traj = extract_trajectory(meta, frame_idx, num_waypoints=8)
            if traj is None:
                continue
            
            # Find corresponding camera image
            # Frame indices in pkl map 1:1 to CAM_F0 images (both at 2Hz)
            if frame_idx >= len(cam_frames):
                continue
            
            img_path = os.path.join(cam_dir, cam_frames[frame_idx])
            if not os.path.exists(img_path):
                continue
            
            # Get driving command
            frame_data = meta[frame_idx]
            cmd = get_driving_command_text(frame_data.get('driving_command', [1,0,0,0]))
            
            # Get ego dynamic state for context
            eds = frame_data.get('ego_dynamic_state', [0, 0, 0, 0])
            vx = eds[0] if len(eds) > 0 else 0
            vy = eds[1] if len(eds) > 1 else 0
            speed = np.sqrt(vx**2 + vy**2)
            
            # Build training sample (OneVL answer-only format)
            traj_str = format_trajectory(traj)
            
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": f"You are driving an autonomous vehicle. Navigation command: {cmd}. Current speed: {speed:.2f} m/s. Predict the ego-vehicle trajectory for the next 4 seconds as 8 waypoints in ego frame [x,y,heading]."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": f"<answer>{traj_str}</answer>"
                    }
                ],
                "images": [img_path],
                "scene": scene,
                "frame_idx": frame_idx,
                "source": "navsim_real_gt"
            }
            
            all_samples.append(sample)
        
        if (scene_idx + 1) % 50 == 0:
            print(f"  Processed {scene_idx + 1}/{len(valid_scenes)} scenes, {len(all_samples)} samples so far")
    
    print(f"\n=== Extraction Summary ===")
    print(f"Valid scenes processed: {len(valid_scenes)}")
    print(f"Skipped (no CAM_F0): {skipped_no_cam}")
    print(f"Skipped (too short): {skipped_short}")
    print(f"Total training samples: {len(all_samples)}")
    
    if len(all_samples) == 0:
        print("ERROR: No samples extracted!")
        sys.exit(1)
    
    # Verify trajectory quality (sanity checks)
    print(f"\n=== Trajectory Quality Check ===")
    displacements = []
    for s in all_samples[:100]:
        # Parse trajectory from answer
        answer = s['messages'][1]['content']
        # Last waypoint displacement
        traj_text = answer.replace('<answer>', '').replace('</answer>', '')
        parts = traj_text.split('], [')
        if len(parts) == 8:
            last = parts[-1].replace('[', '').replace(']', '').split(',')
            dx = float(last[0])
            dy = float(last[1])
            displacements.append(np.sqrt(dx**2 + dy**2))
    
    if displacements:
        print(f"  FDE distribution (first 100 samples):")
        print(f"    Mean: {np.mean(displacements):.2f} m")
        print(f"    Std:  {np.std(displacements):.2f} m")
        print(f"    Min:  {np.min(displacements):.2f} m")
        print(f"    Max:  {np.max(displacements):.2f} m")
        
        # Sanity: if mean FDE > 100m in 4s, something is wrong
        if np.mean(displacements) > 100:
            print("  WARNING: Mean FDE > 100m — possible coordinate system issue!")
        elif np.mean(displacements) < 0.01:
            print("  WARNING: Mean FDE < 0.01m — vehicle appears stationary!")
        else:
            print("  OK: Trajectory magnitudes look reasonable for 4s horizon")
    
    # Print example
    print(f"\n=== Example Sample ===")
    ex = all_samples[0]
    print(f"  Scene: {ex['scene']}")
    print(f"  Frame: {ex['frame_idx']}")
    print(f"  Image: {ex['images'][0]}")
    print(f"  Command: extracted from prompt")
    print(f"  Answer: {ex['messages'][1]['content'][:200]}...")
    
    # Save
    with open(OUTPUT_JSONL, 'w') as f:
        for s in all_samples:
            f.write(json.dumps(s) + '\n')
    
    print(f"\nSaved {len(all_samples)} samples to {OUTPUT_JSONL}")
    
    # Also create latent format version (vis4+text2)
    latent_output = OUTPUT_JSONL.replace('.jsonl', '_latent.jsonl')
    with open(latent_output, 'w') as f:
        for s in all_samples:
            latent_sample = s.copy()
            # Add latent token format to the prompt
            user_msg = latent_sample['messages'][0].copy()
            user_content = user_msg['content'] if isinstance(user_msg['content'], list) else [{"type": "text", "text": user_msg['content']}]
            
            # Insert latent token instruction
            for item in user_content:
                if item.get('type') == 'text':
                    item['text'] += " Think using latent tokens: <|start-latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|latent|><|end-latent|>"
                    break
            
            latent_sample['messages'][0]['content'] = user_content
            f.write(json.dumps(latent_sample) + '\n')
    
    print(f"Saved {len(all_samples)} latent-format samples to {latent_output}")
    
    return len(all_samples)

if __name__ == '__main__':
    num_samples = main()
    print(f"\n{'='*60}")
    print(f"  BF-032 Fix: {num_samples} samples with REAL trajectory GT")
    print(f"  Next: Run training with w7_train_cfull.py")
    print(f"{'='*60}")
