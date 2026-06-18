#!/opt/onevl-env/bin/python3
"""
Variant 4 (Memory-Fixed): OneVL w/o Language Decoder (Visual Decoder Only)
==========================================================================
Same as v4_vis_only.py but with aggressive memory management for Stage 2.
Fix: process in smaller effective batch, explicit del + empty_cache.
"""
import sys, os, json, torch, time, gc
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
    # Clear old log
    open(LOG_FILE, 'w').close()

    log(f"{'='*60}", LOG_FILE)
    log(f"  Variant 4: OneVL w/o Lang Decoder (vis only) [MEM-FIXED]", LOG_FILE)
    log(f"  4-stage: Preliminary -> S0 -> S1 -> S2", LOG_FILE)
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

    # ========== Preliminary ==========
    log(f"\n  PRELIMINARY ({PRELIMINARY_EPOCHS} ep)", LOG_FILE)
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
                del vit_emb, vis_gt, vis_loss, pv, grid_thw, img_inputs
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

    # ========== Stage 0 ==========
    log(f"\n  STAGE 0 ({STAGE_0_EPOCHS} ep)", LOG_FILE)
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
                del outputs, loss, batch
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_s0.step(); opt_s0.zero_grad()
                if (idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s0.zero_grad(); errors += 1
            except: errors += 1
        log(f"  S0 Epoch {epoch+1}: traj={np.mean(losses) if losses else 0:.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(model, None, None, OUTPUT_BASE, VARIANT, "stage0")
    log(f"  Stage 0 done: {(time.time()-t1)/60:.1f} min", LOG_FILE)

    # ========== Stage 1 (DETACH) ==========
    log(f"\n  STAGE 1 ({STAGE_1_EPOCHS} ep, DETACH)", LOG_FILE)
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
                del outputs, hidden, vit_emb, vis_latent_h, vis_gt, vis_loss, batch
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(vis_decoder.parameters(), 1.0)
                    opt_s1.step(); opt_s1.zero_grad()
                if (idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s1.zero_grad(); errors += 1
            except: errors += 1
        log(f"  S1 Epoch {epoch+1}: vis={np.mean(vis_losses) if vis_losses else 0:.2f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(None, vis_decoder, None, OUTPUT_BASE, VARIANT, "stage1")
    log(f"  Stage 1 done: {(time.time()-t2)/60:.1f} min", LOG_FILE)

    # ========== Stage 2 (NO DETACH) — MEMORY-FIXED ==========
    log(f"\n  STAGE 2 ({STAGE_2_EPOCHS} ep, NO DETACH, mem-optimized)", LOG_FILE)
    model.gradient_checkpointing_enable()
    model.train()
    vis_decoder.train()
    for p in model.parameters(): p.requires_grad = True
    for p in vis_decoder.parameters(): p.requires_grad = True
    all_params = [p for p in model.parameters() if p.requires_grad] + list(vis_decoder.parameters())
    # Gradient checkpointing for Stage 2 memory
    model.gradient_checkpointing_enable()
    opt_s2 = torch.optim.AdamW(all_params, lr=LR * 0.5)

    t3 = time.time()
    for epoch in range(STAGE_2_EPOCHS):
        traj_losses, vis_losses, errors = [], [], 0
        opt_s2.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                
                # Model forward (with grad)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'], mm_token_type_ids=batch['mm_token_type_ids'],
                    output_hidden_states=True)
                traj_loss = outputs.loss
                if traj_loss is None or torch.isnan(traj_loss):
                    del outputs, batch
                    continue
                hidden = outputs.hidden_states[-1]

                # Vis decoder (ViT frozen, decoder with grad)
                vis_loss = torch.tensor(0.0, device='cuda')
                if sample['_has_vis']:
                    vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    with torch.no_grad():
                        vit_out = vit_encoder(img_inputs['pixel_values'].to('cuda'),
                                            grid_thw=img_inputs['image_grid_thw'].to('cuda'))
                        vit_emb = vit_out.last_hidden_state.unsqueeze(0) if vit_out.last_hidden_state.dim() == 2 else vit_out.last_hidden_state
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).to('cuda')
                    vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)
                    del vit_emb, img_inputs, vit_out

                total = (traj_loss + LAMBDA_VIS * vis_loss) / GRAD_ACCUM
                total.backward()
                
                traj_losses.append(traj_loss.item())
                if vis_loss.item() > 0: vis_losses.append(vis_loss.item())
                
                # Aggressive cleanup
                del outputs, hidden, traj_loss, vis_loss, total, batch
                gc.collect()

                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt_s2.step(); opt_s2.zero_grad()
                
                # Empty cache frequently
                if (idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                opt_s2.zero_grad()
                errors += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log(f"    S2 err: {type(e).__name__}: {str(e)[:80]}", LOG_FILE)

        avg_traj = np.mean(traj_losses) if traj_losses else 0
        avg_vis = np.mean(vis_losses) if vis_losses else 0
        log(f"  S2 Epoch {epoch+1}: traj={avg_traj:.4f}, vis={avg_vis:.2f}, "
            f"samples={len(traj_losses)}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()
        gc.collect()

    save_checkpoint(model, vis_decoder, None, OUTPUT_BASE, VARIANT, "stage2_final")
    log(f"  Stage 2 done: {(time.time()-t3)/60:.1f} min", LOG_FILE)
    log(f"  Total train time: {(time.time()-t0)/60:.1f} min", LOG_FILE)

    # Evaluate
    log("\n=== Evaluation ===", LOG_FILE)
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V4: Vis Only (no lang dec)", include_latent=True, log_file=LOG_FILE)
    result['train_time_min'] = (time.time() - t0) / 60
    result['final_traj_loss'] = avg_traj
    result['final_vis_loss'] = avg_vis
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
