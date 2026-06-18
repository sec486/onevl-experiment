#!/opt/onevl-env/bin/python3
"""
Variant 4: OneVL w/o Language Decoder (Visual Decoder Only)
============================================================
Latent tokens + visual decoder. No language decoder.
4-stage: Preliminary → Stage 0 (traj) → Stage 1 (vis dec, detach) → Stage 2 (joint)
OneVL equivalent: ablation row "w/o lang. dec."
"""
import sys, os, json, torch, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from decoders import VisualDecoderV2
from data_utils import (load_data, build_inputs_with_latent, subsample_tokens,
                        save_checkpoint, log, evaluate_model)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

VARIANT = "v4_vis_only"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, VARIANT)
LOG_FILE = None

def main():
    global LOG_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "train_log.txt")

    log(f"{'='*60}", LOG_FILE)
    log(f"  Variant 4: OneVL w/o Lang Decoder (vis only)", LOG_FILE)
    log(f"  4-stage: Preliminary → S0 → S1 → S2", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    train_samples, eval_samples, vis_tokens_dict = load_data()
    log(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}, Vis scenes: {len(vis_tokens_dict)}", LOG_FILE)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to('cuda')

    vit_encoder = model.model.visual
    hidden_dim = 2048

    # Probe ViT dim
    _p = processor.image_processor(images=[Image.new('RGB', (448, 448))], return_tensors='pt')
    with torch.no_grad():
        _o = vit_encoder(_p['pixel_values'].to('cuda'), grid_thw=_p['image_grid_thw'].to('cuda'))
        vit_dim = _o.last_hidden_state.shape[-1] if hasattr(_o, 'last_hidden_state') else _o.shape[-1]
    del _p, _o; torch.cuda.empty_cache()

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)

    vis_decoder = VisualDecoderV2(
        vit_dim=vit_dim, hidden_dim=hidden_dim, codebook_size=CODEBOOK_SIZE,
        num_queries=NUM_VIS_QUERIES, num_layers=2, num_heads=8
    ).to('cuda').to(torch.bfloat16)

    # ========== Preliminary: Visual Decoder Pretrain ==========
    log(f"\n{'='*60}", LOG_FILE)
    log(f"  PRELIMINARY: Visual Decoder Pretrain ({PRELIMINARY_EPOCHS} ep)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    vis_decoder.train()
    opt_pre = torch.optim.AdamW(vis_decoder.parameters(), lr=LR * 3)
    t0 = time.time()

    for epoch in range(PRELIMINARY_EPOCHS):
        losses, errors = [], 0
        opt_pre.zero_grad()
        for idx, sample in enumerate(train_samples):
            if not sample['_has_vis']: continue
            try:
                img_path = sample.get('images', [None])[0]
                image = Image.open(img_path).convert('RGB').resize((448, 448))
                img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                pv = img_inputs['pixel_values'].to('cuda')
                grid_thw = img_inputs['image_grid_thw'].to('cuda')
                with torch.no_grad():
                    vit_out = vit_encoder(pv, grid_thw=grid_thw)
                    vit_emb = vit_out.last_hidden_state.unsqueeze(0) if vit_out.last_hidden_state.dim() == 2 else vit_out.last_hidden_state
                vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).to('cuda')
                vis_loss = vis_decoder(vit_emb, latent_h=None, gt_ids=vis_gt)
                (vis_loss / GRAD_ACCUM).backward()
                losses.append(vis_loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(vis_decoder.parameters(), 1.0)
                    opt_pre.step(); opt_pre.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_pre.zero_grad(); errors += 1
            except: errors += 1
        log(f"  PRE Epoch {epoch+1}: vis={np.mean(losses) if losses else 0:.2f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(None, vis_decoder, None, OUTPUT_BASE, VARIANT, "preliminary")
    log(f"  Preliminary done: {(time.time()-t0)/60:.1f} min", LOG_FILE)

    # ========== Stage 0: Trajectory Warmup ==========
    log(f"\n  STAGE 0: Traj Warmup ({STAGE_0_EPOCHS} ep)", LOG_FILE)
    vis_decoder.eval()
    for p in vis_decoder.parameters(): p.requires_grad = False
    model.train()
    opt_s0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    t1 = time.time()
    for epoch in range(STAGE_0_EPOCHS):
        losses, errors = [], 0
        opt_s0.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'], mm_token_type_ids=batch['mm_token_type_ids'])
                loss = outputs.loss
                if loss is None or torch.isnan(loss): continue
                (loss / GRAD_ACCUM).backward()
                losses.append(loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_s0.step(); opt_s0.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s0.zero_grad(); errors += 1
            except: errors += 1
        log(f"  S0 Epoch {epoch+1}: traj={np.mean(losses) if losses else 0:.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(model, None, None, OUTPUT_BASE, VARIANT, "stage0")
    log(f"  Stage 0 done: {(time.time()-t1)/60:.1f} min", LOG_FILE)

    # ========== Stage 1: Vis Decoder Warmup (DETACH) ==========
    log(f"\n  STAGE 1: Vis Decoder Warmup ({STAGE_1_EPOCHS} ep, DETACH)", LOG_FILE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    vis_decoder.train()
    for p in vis_decoder.parameters(): p.requires_grad = True
    opt_s1 = torch.optim.AdamW(vis_decoder.parameters(), lr=LR * 2)

    t2 = time.time()
    for epoch in range(STAGE_1_EPOCHS):
        vis_losses, errors = [], 0
        opt_s1.zero_grad()
        for idx, sample in enumerate(train_samples):
            if not sample['_has_vis']: continue
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                with torch.no_grad():
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'],
                        mm_token_type_ids=batch['mm_token_type_ids'], output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    vit_out = vit_encoder(img_inputs['pixel_values'].to('cuda'), grid_thw=img_inputs['image_grid_thw'].to('cuda'))
                    vit_emb = vit_out.last_hidden_state.unsqueeze(0) if vit_out.last_hidden_state.dim() == 2 else vit_out.last_hidden_state
                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).to('cuda')
                vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)
                (vis_loss / GRAD_ACCUM).backward()
                vis_losses.append(vis_loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(vis_decoder.parameters(), 1.0)
                    opt_s1.step(); opt_s1.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s1.zero_grad(); errors += 1
            except: errors += 1
        log(f"  S1 Epoch {epoch+1}: vis={np.mean(vis_losses) if vis_losses else 0:.2f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(None, vis_decoder, None, OUTPUT_BASE, VARIANT, "stage1")
    log(f"  Stage 1 done: {(time.time()-t2)/60:.1f} min", LOG_FILE)

    # ========== Stage 2: Joint (NO DETACH) ==========
    log(f"\n  STAGE 2: Joint ({STAGE_2_EPOCHS} ep, NO DETACH)", LOG_FILE)
    model.train()
    vis_decoder.train()
    for p in model.parameters(): p.requires_grad = True
    for p in vis_decoder.parameters(): p.requires_grad = True
    all_params = [p for p in model.parameters() if p.requires_grad] + list(vis_decoder.parameters())
    opt_s2 = torch.optim.AdamW(all_params, lr=LR * 0.5)

    t3 = time.time()
    for epoch in range(STAGE_2_EPOCHS):
        traj_losses, vis_losses, errors = [], [], 0
        opt_s2.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'], mm_token_type_ids=batch['mm_token_type_ids'],
                    output_hidden_states=True)
                traj_loss = outputs.loss
                if traj_loss is None or torch.isnan(traj_loss): continue
                hidden = outputs.hidden_states[-1]

                vis_loss = torch.tensor(0.0, device='cuda')
                if sample['_has_vis']:
                    vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    with torch.no_grad():
                        vit_out = vit_encoder(img_inputs['pixel_values'].to('cuda'), grid_thw=img_inputs['image_grid_thw'].to('cuda'))
                        vit_emb = vit_out.last_hidden_state.unsqueeze(0) if vit_out.last_hidden_state.dim() == 2 else vit_out.last_hidden_state
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).to('cuda')
                    vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)

                total = (traj_loss + LAMBDA_VIS * vis_loss) / GRAD_ACCUM
                total.backward()
                traj_losses.append(traj_loss.item())
                if vis_loss.item() > 0: vis_losses.append(vis_loss.item())

                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt_s2.step(); opt_s2.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s2.zero_grad(); errors += 1
            except: errors += 1
        log(f"  S2 Epoch {epoch+1}: traj={np.mean(traj_losses) if traj_losses else 0:.4f}, "
            f"vis={np.mean(vis_losses) if vis_losses else 0:.2f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(model, vis_decoder, None, OUTPUT_BASE, VARIANT, "stage2_final")
    log(f"  Stage 2 done: {(time.time()-t3)/60:.1f} min", LOG_FILE)

    # Evaluate
    log("\n=== Evaluation ===", LOG_FILE)
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V4: Vis Only (no lang dec)", include_latent=True, log_file=LOG_FILE)
    result['train_time_min'] = (time.time() - t0) / 60
    result['final_traj_loss'] = np.mean(traj_losses) if traj_losses else 0
    result['final_vis_loss'] = np.mean(vis_losses) if vis_losses else 0
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
