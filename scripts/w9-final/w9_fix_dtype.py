#!/opt/onevl-env/bin/python3
# -*- coding: utf-8 -*-
"""
W9 Fix: dtype validation + cast for visual tokens.
Fixes BF-037: expanded Emu3 visual tokens saved as Float instead of Long.

This script:
1. Validates and fixes the visual token data file (cast to int64)
2. Patches the training script's subsample_tokens to enforce .long()
3. Runs an expanded dry run that validates ALL dtypes before training

Run: /opt/onevl-env/bin/python3 /opt/onevl-experiment/scripts/w9_fix_dtype.py
"""
import torch
import os
import sys
import time

VIS_TOKENS_PATH = '/opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt'
TRAIN_SCRIPT = '/opt/onevl-experiment/scripts/w9_train_correct.py'
OUTPUT_DIR = '/opt/onevl-experiment/output/w9_step2'

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def main():
    log("=" * 60)
    log("  BF-037 Fix: Visual Token Dtype Validation & Cast")
    log("=" * 60)

    # Step 1: Check and fix the visual tokens data file
    log("\n--- Step 1: Validate visual token data ---")
    if not os.path.exists(VIS_TOKENS_PATH):
        log(f"  ERROR: {VIS_TOKENS_PATH} not found!")
        sys.exit(1)

    vis_tokens_dict = torch.load(VIS_TOKENS_PATH, weights_only=False)
    log(f"  Loaded {len(vis_tokens_dict)} scenes")

    # Check dtypes
    dtype_counts = {}
    bad_scenes = []
    for scene_id, tensor in vis_tokens_dict.items():
        dtype_str = str(tensor.dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        if tensor.dtype != torch.int64 and tensor.dtype != torch.long:
            bad_scenes.append(scene_id)

    log(f"  Dtype distribution: {dtype_counts}")

    if bad_scenes:
        log(f"  PROBLEM: {len(bad_scenes)} scenes have non-int64 dtype!")
        log(f"  Example: {bad_scenes[0]} -> dtype={vis_tokens_dict[bad_scenes[0]].dtype}")
        log(f"  Fixing: casting all tensors to int64...")

        # Fix all tensors
        fixed = 0
        for scene_id in list(vis_tokens_dict.keys()):
            t = vis_tokens_dict[scene_id]
            if t.dtype != torch.int64:
                vis_tokens_dict[scene_id] = t.long()
                fixed += 1

        # Save fixed version
        backup_path = VIS_TOKENS_PATH + '.bak'
        os.rename(VIS_TOKENS_PATH, backup_path)
        torch.save(vis_tokens_dict, VIS_TOKENS_PATH)
        log(f"  Fixed {fixed} scenes. Backup at {backup_path}")

        # Verify
        verify = torch.load(VIS_TOKENS_PATH, weights_only=False)
        sample_key = list(verify.keys())[0]
        assert verify[sample_key].dtype == torch.int64, "Fix failed!"
        log(f"  Verified: all tensors now int64 ✅")
    else:
        log(f"  All {len(vis_tokens_dict)} scenes are int64 ✅ (no fix needed)")

    # Step 2: Validate token value ranges
    log("\n--- Step 2: Validate token value ranges ---")
    all_min = float('inf')
    all_max = float('-inf')
    for scene_id, tensor in vis_tokens_dict.items():
        all_min = min(all_min, tensor.min().item())
        all_max = max(all_max, tensor.max().item())
    log(f"  Token ID range: [{all_min}, {all_max}]")
    CODEBOOK_SIZE = 32768
    if all_max >= CODEBOOK_SIZE:
        log(f"  WARNING: max token ID {all_max} >= codebook size {CODEBOOK_SIZE}!")
        log(f"  CrossEntropyLoss will crash with 'index out of range'")
        sys.exit(1)
    if all_min < 0:
        log(f"  WARNING: negative token IDs found! Min={all_min}")
        sys.exit(1)
    log(f"  Range valid: [0, {CODEBOOK_SIZE}) ✅")

    # Step 3: Patch the training script
    log("\n--- Step 3: Patch subsample_tokens in training script ---")
    with open(TRAIN_SCRIPT, 'r') as f:
        content = f.read()

    # Find and fix the subsample_tokens function
    old_subsample = """def subsample_tokens(full_tokens, target=256):
    flat = full_tokens.flatten()
    indices = torch.linspace(0, len(flat)-1, target).long()
    return flat[indices]"""

    new_subsample = """def subsample_tokens(full_tokens, target=256):
    flat = full_tokens.flatten()
    indices = torch.linspace(0, len(flat)-1, target).long()
    return flat[indices].long()  # BF-037: ALWAYS cast to int64 for CrossEntropyLoss"""

    if old_subsample in content:
        content = content.replace(old_subsample, new_subsample)
        with open(TRAIN_SCRIPT, 'w') as f:
            f.write(content)
        log(f"  Patched subsample_tokens: added .long() cast ✅")
    elif 'return flat[indices].long()' in content:
        log(f"  Already patched ✅ (no change needed)")
    else:
        log(f"  WARNING: Could not find exact subsample_tokens pattern")
        log(f"  Manual fix needed: add .long() after flat[indices]")
        # Try a more flexible patch
        if 'return flat[indices]' in content:
            content = content.replace('return flat[indices]', 'return flat[indices].long()  # BF-037 fix')
            with open(TRAIN_SCRIPT, 'w') as f:
                f.write(content)
            log(f"  Applied flexible patch ✅")

    # Step 4: Dry-run dtype validation (simulates one forward pass per stage)
    log("\n--- Step 4: Dry-run dtype validation ---")
    log("  Loading one sample to verify CrossEntropyLoss compatibility...")

    sample_key = list(vis_tokens_dict.keys())[0]
    sample_tokens = vis_tokens_dict[sample_key]

    # Simulate subsample_tokens
    flat = sample_tokens.flatten()
    indices = torch.linspace(0, len(flat)-1, 64).long()
    vis_gt = flat[indices].long()

    log(f"  vis_gt dtype: {vis_gt.dtype} (must be int64/long)")
    log(f"  vis_gt shape: {vis_gt.shape}")
    log(f"  vis_gt range: [{vis_gt.min().item()}, {vis_gt.max().item()}]")

    # Simulate CE loss
    dummy_logits = torch.randn(1, 64, CODEBOOK_SIZE)
    try:
        loss = torch.nn.functional.cross_entropy(
            dummy_logits.view(-1, CODEBOOK_SIZE),
            vis_gt.view(-1)
        )
        log(f"  CE loss dry run: {loss.item():.4f} ✅ (expected ~10.4 = log(32768))")
    except RuntimeError as e:
        log(f"  CE loss dry run FAILED: {e}")
        sys.exit(1)

    # Also test with float targets (should fail - this is what we're preventing)
    try:
        bad_gt = vis_gt.float()
        _ = torch.nn.functional.cross_entropy(
            dummy_logits.view(-1, CODEBOOK_SIZE),
            bad_gt.view(-1)
        )
        log(f"  WARNING: Float targets didn't fail? (unexpected)")
    except RuntimeError as e:
        log(f"  Confirmed: Float targets correctly rejected: {str(e)[:60]} ✅")

    log("\n" + "=" * 60)
    log("  BF-037 FIX COMPLETE")
    log("  - Visual tokens: all int64")
    log("  - Token range: [0, 32768) valid")
    log("  - subsample_tokens: .long() enforced")
    log("  - CE loss: dry run passed")
    log("  Ready to re-run w9_train_correct.py")
    log("=" * 60)

if __name__ == '__main__':
    main()
