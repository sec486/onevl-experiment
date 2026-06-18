#!/usr/bin/env python3
"""
变体 4: OneVL 无语言解码器（仅视觉解码器）
============================================
4 阶段训练，仅使用视觉辅助解码器:
  - 预训练: 视觉解码器预训练（仅 ViT）
  - Stage 0: 轨迹 warmup
  - Stage 1: 视觉解码器 warmup（梯度截断）
  - Stage 2: 联合训练（24GB GPU 需要 gradient checkpointing）

预期结果: ADE ~8.6m, 训练 ~108 分钟 (L4)。
注意: 24GB GPU 上 Stage 2 需要启用 model.gradient_checkpointing_enable()。
"""
import sys, os, json, torch, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from decoders import VisualDecoderV2
from data_utils import (load_data, build_inputs_with_latent, save_checkpoint,
                        log, evaluate_model, subsample_tokens)
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
    log(f"  Variant 4: Visual Decoder Only (4-stage)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    train_samples, eval_samples, vis_tokens_dict = load_data()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to('cuda')

    vit_encoder = model.model.visual
    hidden_dim = 2048

    # Probe ViT dim
    _probe = processor.image_processor(images=[Image.new('RGB', (448, 448))], return_tensors='pt')
    with torch.no_grad():
        _out = vit_encoder(_probe['pixel_values'].cuda(), grid_thw=_probe['image_grid_thw'].cuda())
        vit_dim = (_out.last_hidden_state if hasattr(_out, 'last_hidden_state') else _out).shape[-1]
    del _probe, _out; torch.cuda.empty_cache()

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)

    vis_decoder = VisualDecoderV2(
        vit_dim=vit_dim, hidden_dim=hidden_dim, codebook_size=CODEBOOK_SIZE,
        num_queries=NUM_VIS_QUERIES
    ).to('cuda').to(torch.bfloat16)

    t0 = time.time()

    # PRELIMINARY: Visual Decoder Pretrain
    log(f"\n  PRELIMINARY: Vis Decoder Pretrain ({PRELIMINARY_EPOCHS} epochs)", LOG_FILE)
    vis_decoder.train()
    opt_pre = torch.optim.AdamW(vis_decoder.parameters(), lr=LR * 3)
    for epoch in range(PRELIMINARY_EPOCHS):
        losses, errors = [], 0
        opt_pre.zero_grad()
        for idx, sample in enumerate(train_samples):
            if not sample['_has_vis']: continue
            try:
                img_path = sample.get('images', [None])[0]
                image = Image.open(img_path).convert('RGB').resize((448, 448))
                img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                with torch.no_grad():
                    vit_out = vit_encoder(img_inputs['pixel_values'].cuda(),
                                          grid_thw=img_inputs['image_grid_thw'].cuda())
                    vit_emb = (vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out)
                    if vit_emb.dim() == 2: vit_emb = vit_emb.unsqueeze(0)
                vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).cuda()
                vis_loss = vis_decoder(vit_emb, latent_h=None, gt_ids=vis_gt)
                (vis_loss / GRAD_ACCUM).backward()
                losses.append(vis_loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(vis_decoder.parameters(), 1.0)
                    opt_pre.step(); opt_pre.zero_grad()
            except: errors += 1; continue
        log(f"    Epoch {epoch+1}: vis={np.mean(losses):.2f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    # STAGE 0: Trajectory Warmup
    log(f"\n  STAGE 0: Trajectory Warmup ({STAGE_0_EPOCHS} epochs)", LOG_FILE)
    vis_decoder.eval()
    for p in vis_decoder.parameters(): p.requires_grad = False
    model.train()
    opt_s0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
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
        log(f"    Epoch {epoch+1}: traj={np.mean(losses):.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    # STAGE 1: Vis Decoder Warmup (Detached)
    log(f"\n  STAGE 1: Vis Decoder Warmup ({STAGE_1_EPOCHS} epochs, DETACHED)", LOG_FILE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    vis_decoder.train()
    for p in vis_decoder.parameters(): p.requires_grad = True
    opt_s1 = torch.optim.AdamW(vis_decoder.parameters(), lr=LR * 2)
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
                    vit_out = vit_encoder(img_inputs['pixel_values'].cuda(), grid_thw=img_inputs['image_grid_thw'].cuda())
                    vit_emb = (vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out)
                    if vit_emb.dim() == 2: vit_emb = vit_emb.unsqueeze(0)
                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).cuda()
                vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)
                (vis_loss / GRAD_ACCUM).backward()
                vis_losses.append(vis_loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(vis_decoder.parameters(), 1.0)
                    opt_s1.step(); opt_s1.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s1.zero_grad(); errors += 1
            except: errors += 1
        log(f"    Epoch {epoch+1}: vis={np.mean(vis_losses):.2f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    # STAGE 2: Joint (No Detach) — needs gradient checkpointing on 24GB
    log(f"\n  STAGE 2: Joint ({STAGE_2_EPOCHS} epochs, gradient checkpointing)", LOG_FILE)
    model.train()
    model.gradient_checkpointing_enable()  # Required for 24GB GPU
    for name, p in model.named_parameters():
        if 'lora' in name or 'embed_tokens' in name: p.requires_grad = True
    all_params = [p for p in model.parameters() if p.requires_grad] + list(vis_decoder.parameters())
    opt_s2 = torch.optim.AdamW(all_params, lr=LR * 0.5)

    for epoch in range(STAGE_2_EPOCHS):
        traj_losses, vis_losses, errors = [], [], 0
        opt_s2.zero_grad()
        for idx, sample in enumerate(train_samples):
            if not sample['_has_vis']: continue
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'], mm_token_type_ids=batch['mm_token_type_ids'],
                    output_hidden_states=True)
                traj_loss = outputs.loss
                if traj_loss is None or torch.isnan(traj_loss): continue
                hidden = outputs.hidden_states[-1]
                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                with torch.no_grad():
                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    vit_out = vit_encoder(img_inputs['pixel_values'].cuda(), grid_thw=img_inputs['image_grid_thw'].cuda())
                    vit_emb = (vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out)
                    if vit_emb.dim() == 2: vit_emb = vit_emb.unsqueeze(0)
                vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).cuda()
                vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)
                total = (traj_loss + LAMBDA_VIS * vis_loss) / GRAD_ACCUM
                total.backward()
                traj_losses.append(traj_loss.item())
                vis_losses.append(vis_loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt_s2.step(); opt_s2.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s2.zero_grad(); errors += 1
            except: errors += 1
        log(f"    Epoch {epoch+1}: traj={np.mean(traj_losses):.4f}, vis={np.mean(vis_losses):.2f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log(f"\n  Training done: {elapsed/60:.1f} min", LOG_FILE)
    save_checkpoint(model, vis_decoder, None, OUTPUT_BASE, VARIANT, "final")

    log("\n=== Evaluation ===", LOG_FILE)
    model.gradient_checkpointing_disable()
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V4: Vis Only", include_latent=True, log_file=LOG_FILE)
    result['train_time_min'] = elapsed / 60
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
