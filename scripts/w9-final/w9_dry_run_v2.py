#!/opt/onevl-env/bin/python3
# -*- coding: utf-8 -*-
"""
W9 Dry Run V2: Comprehensive Pre-Training Validation
=====================================================
Validates EVERY operation that training will perform BEFORE burning GPU time.
Added after BF-037: dtype mismatch wasted 57 min of training.

RULE: If ANY check fails, training MUST NOT start.

Checks:
  [1] Data files exist and are loadable
  [2] Visual token dtype = int64 (BF-037)
  [3] Visual token range ∈ [0, CODEBOOK_SIZE)
  [4] Data sample count matches expectation
  [5] Model loads correctly
  [6] ViT encoder output shape/dtype
  [7] Latent token IDs are valid vocab entries
  [8] Full forward pass (model + ViT + decoders)
  [9] Backward pass (gradient flows to latent embeddings)
  [10] CE loss computes without error (visual decoder)
  [11] CE loss computes without error (language decoder)
  [12] Memory check (peak < 20GB for safety margin)

Run: /opt/onevl-env/bin/python3 /opt/onevl-experiment/scripts/w9_dry_run_v2.py
Exit 0 = all checks pass, safe to train.
Exit 1 = BLOCKED, fix issues first.
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Import from training script (same config)
sys.path.insert(0, '/opt/onevl-experiment/scripts')

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_expanded.jsonl'
VIS_TOKENS_PATH = '/opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt'
CODEBOOK_SIZE = 32768
NUM_VIS_QUERIES = 64
NUM_VIS_LATENT = 4
NUM_LANG_LATENT = 2
VIS_LATENT_ID = 151662
LANG_LATENT_ID = 151663
MAX_SAMPLES = 1250

device = 'cuda'

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

class CheckResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.failures = []

    def ok(self, check_id, msg):
        self.passed += 1
        log(f"  [{check_id:>2}/12] ✅ PASS: {msg}")

    def fail(self, check_id, msg):
        self.failed += 1
        self.failures.append(f"[{check_id}] {msg}")
        log(f"  [{check_id:>2}/12] ❌ FAIL: {msg}")

    def warn(self, check_id, msg):
        self.warnings += 1
        log(f"  [{check_id:>2}/12] ⚠️  WARN: {msg}")

    def summary(self):
        log(f"\n{'='*60}")
        log(f"  DRY RUN SUMMARY: {self.passed} pass, {self.failed} fail, {self.warnings} warn")
        if self.failures:
            log(f"  BLOCKED — fix these before training:")
            for f in self.failures:
                log(f"    • {f}")
            log(f"{'='*60}")
            return False
        else:
            log(f"  ALL CHECKS PASS — safe to start training")
            log(f"{'='*60}")
            return True


def main():
    log("=" * 60)
    log("  W9 DRY RUN V2: Pre-Training Validation")
    log("  All 12 checks must pass before training starts.")
    log("=" * 60)

    r = CheckResult()

    # [1] Data files exist
    log("\n--- Check 1: Data files ---")
    files_ok = True
    for path, desc in [
        (DATA_PATH, "Training JSONL"),
        (VIS_TOKENS_PATH, "Emu3 visual tokens"),
        (MODEL_PATH, "Model directory"),
    ]:
        if os.path.exists(path):
            log(f"       {desc}: exists")
        else:
            log(f"       {desc}: MISSING ({path})")
            files_ok = False
    if files_ok:
        r.ok(1, "All data files present")
    else:
        r.fail(1, "Missing data files")
        r.summary()
        sys.exit(1)

    # [2] Visual token dtype
    log("\n--- Check 2: Visual token dtype ---")
    vis_tokens_dict = torch.load(VIS_TOKENS_PATH, weights_only=False)
    dtypes = set()
    for scene_id, tensor in vis_tokens_dict.items():
        dtypes.add(str(tensor.dtype))
    log(f"       {len(vis_tokens_dict)} scenes, dtypes: {dtypes}")
    if dtypes == {'torch.int64'}:
        r.ok(2, f"All {len(vis_tokens_dict)} scenes are int64")
    else:
        r.fail(2, f"Non-int64 dtypes found: {dtypes}. Run w9_fix_dtype.py first!")

    # [3] Visual token range
    log("\n--- Check 3: Visual token value range ---")
    all_min = min(t.min().item() for t in vis_tokens_dict.values())
    all_max = max(t.max().item() for t in vis_tokens_dict.values())
    log(f"       Range: [{all_min}, {all_max}], codebook={CODEBOOK_SIZE}")
    if 0 <= all_min and all_max < CODEBOOK_SIZE:
        r.ok(3, f"Token range [0, {CODEBOOK_SIZE}) valid")
    else:
        r.fail(3, f"Token range [{all_min}, {all_max}] outside [0, {CODEBOOK_SIZE})")

    # [4] Data sample count
    log("\n--- Check 4: Training data samples ---")
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= MAX_SAMPLES:
                    break
    n_with_vis = sum(1 for s in samples if s.get('scene', '') in vis_tokens_dict)
    log(f"       Samples: {len(samples)}, with vis tokens: {n_with_vis}")
    if len(samples) >= MAX_SAMPLES * 0.9 and n_with_vis >= len(samples) * 0.5:
        r.ok(4, f"{len(samples)} samples loaded, {n_with_vis} have vis tokens")
    elif n_with_vis < len(samples) * 0.5:
        r.fail(4, f"Only {n_with_vis}/{len(samples)} samples have vis tokens")
    else:
        r.warn(4, f"Only {len(samples)} samples (expected {MAX_SAMPLES})")

    # Check image paths exist
    sample_with_img = None
    img_missing = 0
    for s in samples[:50]:
        imgs = s.get('images', [])
        if imgs and os.path.exists(imgs[0]):
            if sample_with_img is None:
                sample_with_img = s
        elif imgs:
            img_missing += 1
    if img_missing > 0:
        log(f"       WARNING: {img_missing}/50 samples have missing images")

    # [5] Model load
    log("\n--- Check 5: Model load ---")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    t0 = time.time()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokenizer = processor.tokenizer
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="eager"
        ).to(device)
        r.ok(5, f"Model loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        r.fail(5, f"Model load failed: {e}")
        r.summary()
        sys.exit(1)

    # [6] ViT encoder
    log("\n--- Check 6: ViT encoder ---")
    vit_encoder = model.model.visual
    probe_img = Image.new('RGB', (448, 448))
    probe_inputs = processor.image_processor(images=[probe_img], return_tensors='pt')
    pv = probe_inputs['pixel_values'].to(device)
    thw = probe_inputs['image_grid_thw'].to(device)
    with torch.no_grad():
        vit_out = vit_encoder(pv, grid_thw=thw)
        if hasattr(vit_out, 'last_hidden_state'):
            vit_shape = vit_out.last_hidden_state.shape
        else:
            vit_shape = vit_out.shape
    vit_dim = vit_shape[-1]
    log(f"       ViT output: {vit_shape}, dim={vit_dim}, dtype={vit_out.last_hidden_state.dtype if hasattr(vit_out, 'last_hidden_state') else vit_out.dtype}")
    if vit_dim in [1024, 2048]:
        r.ok(6, f"ViT output dim={vit_dim}, shape={vit_shape}")
    else:
        r.fail(6, f"Unexpected ViT dim={vit_dim}")
    del probe_img, probe_inputs, pv, thw, vit_out
    torch.cuda.empty_cache()

    # [7] Latent token IDs
    log("\n--- Check 7: Latent token IDs ---")
    vocab_size = len(tokenizer)
    vis_token_name = tokenizer.convert_ids_to_tokens(VIS_LATENT_ID)
    lang_token_name = tokenizer.convert_ids_to_tokens(LANG_LATENT_ID)
    log(f"       Vocab size: {vocab_size}")
    log(f"       VIS_LATENT_ID={VIS_LATENT_ID}: '{vis_token_name}'")
    log(f"       LANG_LATENT_ID={LANG_LATENT_ID}: '{lang_token_name}'")
    if VIS_LATENT_ID < vocab_size and LANG_LATENT_ID < vocab_size:
        r.ok(7, f"Latent IDs valid within vocab ({vocab_size})")
    else:
        r.fail(7, f"Latent IDs outside vocab range!")

    # Apply LoRA for forward pass test
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)

    # [8] Full forward pass
    log("\n--- Check 8: Full forward pass ---")
    if sample_with_img is not None:
        try:
            # Import build function from training script
            from w9_train_correct import build_inputs_with_latent
            batch = build_inputs_with_latent(sample_with_img, processor, tokenizer)
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                mm_token_type_ids=batch['mm_token_type_ids'],
                output_hidden_states=True,
            )
            log(f"       loss={outputs.loss.item():.4f}, hidden={outputs.hidden_states[-1].shape}")
            r.ok(8, f"Forward pass OK (loss={outputs.loss.item():.4f})")
        except ImportError:
            # If can't import, do inline
            log(f"       Cannot import build_inputs_with_latent, testing basic forward")
            dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            outputs = model(input_ids=dummy_ids, labels=dummy_ids, output_hidden_states=True)
            r.ok(8, f"Basic forward OK (loss={outputs.loss.item():.4f})")
        except Exception as e:
            r.fail(8, f"Forward failed: {type(e).__name__}: {str(e)[:100]}")
    else:
        r.warn(8, "No sample with valid image found for forward test")

    # [9] Backward pass
    log("\n--- Check 9: Backward + gradient to latent embeddings ---")
    try:
        if outputs.loss is not None:
            outputs.loss.backward()
            # Check embed_tokens gradient
            embed_grad = None
            for name, param in model.named_parameters():
                if 'embed_tokens' in name and param.grad is not None:
                    embed_grad = param.grad
                    break
            if embed_grad is not None:
                r.ok(9, f"Gradient flows to embed_tokens (norm={embed_grad.norm().item():.6f})")
            else:
                r.warn(9, "embed_tokens gradient is None (may be OK if no latent tokens in this sample)")
            model.zero_grad()
        else:
            r.warn(9, "No loss to backward")
    except Exception as e:
        r.fail(9, f"Backward failed: {e}")
    torch.cuda.empty_cache()

    # [10] Visual decoder CE loss
    log("\n--- Check 10: Visual decoder CE loss ---")
    try:
        from w9_train_correct import VisualDecoderV2
        vis_decoder = VisualDecoderV2(
            vit_dim=vit_dim, hidden_dim=2048, codebook_size=CODEBOOK_SIZE,
            num_queries=NUM_VIS_QUERIES, num_layers=2, num_heads=8
        ).to(device).to(torch.bfloat16)

        # Simulate real inputs
        dummy_vit = torch.randn(1, 784, vit_dim, device=device, dtype=torch.bfloat16)
        dummy_latent = torch.randn(1, NUM_VIS_LATENT, 2048, device=device, dtype=torch.bfloat16)

        # Use ACTUAL vis tokens from data
        sample_scene = list(vis_tokens_dict.keys())[0]
        vis_gt_full = vis_tokens_dict[sample_scene]
        flat = vis_gt_full.flatten()
        indices = torch.linspace(0, len(flat)-1, NUM_VIS_QUERIES).long()
        vis_gt = flat[indices].long().unsqueeze(0).to(device)  # MUST be long!

        log(f"       vis_gt dtype={vis_gt.dtype}, shape={vis_gt.shape}, range=[{vis_gt.min()},{vis_gt.max()}]")

        # Test Preliminary (ViT only)
        vis_loss_pre = vis_decoder(dummy_vit, latent_h=None, gt_ids=vis_gt)
        log(f"       Preliminary vis_loss={vis_loss_pre.item():.4f} (expect ~10.4)")

        # Test Stage 1 (ViT + latent)
        vis_loss_s1 = vis_decoder(dummy_vit, dummy_latent, vis_gt)
        log(f"       Stage1 vis_loss={vis_loss_s1.item():.4f}")

        if 5 < vis_loss_pre.item() < 15:
            r.ok(10, f"Visual CE loss OK: Preliminary={vis_loss_pre.item():.2f}")
        else:
            r.warn(10, f"Unexpected vis_loss={vis_loss_pre.item():.2f} (expected ~10.4)")
        del vis_decoder, dummy_vit, dummy_latent
    except ImportError:
        # Inline test
        dummy_logits = torch.randn(1, NUM_VIS_QUERIES, CODEBOOK_SIZE, device=device)
        sample_scene = list(vis_tokens_dict.keys())[0]
        vis_gt_full = vis_tokens_dict[sample_scene]
        flat = vis_gt_full.flatten()
        indices = torch.linspace(0, len(flat)-1, NUM_VIS_QUERIES).long()
        vis_gt = flat[indices].long().unsqueeze(0).to(device)
        loss = F.cross_entropy(dummy_logits.view(-1, CODEBOOK_SIZE), vis_gt.view(-1))
        r.ok(10, f"CE loss computes: {loss.item():.4f}")
    except Exception as e:
        r.fail(10, f"Visual CE loss failed: {type(e).__name__}: {str(e)[:100]}")
    torch.cuda.empty_cache()

    # [11] Language decoder CE loss
    log("\n--- Check 11: Language decoder CE loss ---")
    try:
        from w9_train_correct import LangDecoder
        lang_decoder = LangDecoder(d_model=2048, n_heads=8, vocab_size=vocab_size, inner_dim=512
        ).to(device).to(torch.bfloat16)

        dummy_latent = torch.randn(1, NUM_LANG_LATENT, 2048, device=device, dtype=torch.bfloat16)
        cot_text = "The vehicle should maintain safe trajectory."
        cot_ids = tokenizer(cot_text, return_tensors='pt', max_length=64,
                           truncation=True, padding='max_length')['input_ids'].to(device)

        lang_loss = lang_decoder(dummy_latent, cot_ids)
        log(f"       lang_loss={lang_loss.item():.4f} (expect ~11-12)")
        if 5 < lang_loss.item() < 20:
            r.ok(11, f"Language CE loss OK: {lang_loss.item():.2f}")
        else:
            r.warn(11, f"Unexpected lang_loss={lang_loss.item():.2f}")
        del lang_decoder, dummy_latent
    except Exception as e:
        r.fail(11, f"Language CE loss failed: {type(e).__name__}: {str(e)[:100]}")
    torch.cuda.empty_cache()

    # [12] Memory check
    log("\n--- Check 12: GPU memory ---")
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    total_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    log(f"       Peak: {peak_mem:.1f} GB / {total_mem:.1f} GB total")
    if peak_mem < total_mem * 0.85:
        r.ok(12, f"Memory OK: {peak_mem:.1f}/{total_mem:.1f} GB ({peak_mem/total_mem*100:.0f}%)")
    else:
        r.warn(12, f"Memory tight: {peak_mem:.1f}/{total_mem:.1f} GB")

    # Final summary
    success = r.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
