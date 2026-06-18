#!/usr/bin/env python3
"""
变体 5: OneVL 无分阶段训练（直接联合训练）
============================================
完整架构（双解码器）但跳过分阶段 warmup，从头直接联合训练。

用于验证分阶段训练流程是否重要。
预期结果: ADE ~7.7m, 训练 ~95 分钟 (L4)。
关键发现: vis_loss=6.32（几乎未收敛）vs V6 的 3.90 — 分阶段训练至关重要。
"""
import sys, os, json, torch, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from decoders import VisualDecoderV2, LangDecoder
from data_utils import (load_data, build_inputs_with_latent, save_checkpoint,
                        log, evaluate_model, subsample_tokens)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

VARIANT = "v5_no_staged"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, VARIANT)
LOG_FILE = None

# Reduced epochs due to no staged warmup (10 total like other variants)
TOTAL_EPOCHS = 10


def main():
    global LOG_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "train_log.txt")

    log(f"{'='*60}", LOG_FILE)
    log(f"  Variant 5: No Staged Training (direct joint, {TOTAL_EPOCHS} epochs)", LOG_FILE)
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

    lang_decoder = LangDecoder(
        d_model=hidden_dim, n_heads=8, vocab_size=len(tokenizer), inner_dim=512
    ).to('cuda').to(torch.bfloat16)

    default_cot_ids = tokenizer(DEFAULT_COT, return_tensors='pt', max_length=64,
                                truncation=True, padding='max_length')['input_ids'].to('cuda')

    # All parameters trained jointly from the start (no staged warmup)
    model.train()
    vis_decoder.train()
    lang_decoder.train()
    all_params = (
        [p for p in model.parameters() if p.requires_grad] +
        list(vis_decoder.parameters()) + list(lang_decoder.parameters())
    )
    opt = torch.optim.AdamW(all_params, lr=LR)

    log(f"\n  Joint training {TOTAL_EPOCHS} epochs (no warmup stages)...", LOG_FILE)
    t0 = time.time()

    for epoch in range(TOTAL_EPOCHS):
        traj_losses, vis_losses, lang_losses, errors = [], [], [], 0
        opt.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                outputs = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],
                    mm_token_type_ids=batch['mm_token_type_ids'],
                    output_hidden_states=True,
                )
                traj_loss = outputs.loss
                if traj_loss is None or torch.isnan(traj_loss): continue
                hidden = outputs.hidden_states[-1]
                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                lang_latent_h = hidden[:, batch['lang_lat_pos'], :]

                # Visual decoder
                vis_loss = torch.tensor(0.0, device='cuda')
                if sample['_has_vis']:
                    with torch.no_grad():
                        img_path = sample.get('images', [None])[0]
                        image = Image.open(img_path).convert('RGB').resize((448, 448))
                        img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                        vit_out = vit_encoder(img_inputs['pixel_values'].cuda(), grid_thw=img_inputs['image_grid_thw'].cuda())
                        vit_emb = (vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out)
                        if vit_emb.dim() == 2: vit_emb = vit_emb.unsqueeze(0)
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).cuda()
                    vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)

                lang_loss = lang_decoder(lang_latent_h, default_cot_ids)
                total = (traj_loss + LAMBDA_VIS * vis_loss + LAMBDA_LANG * lang_loss) / GRAD_ACCUM
                total.backward()

                traj_losses.append(traj_loss.item())
                if vis_loss.item() > 0: vis_losses.append(vis_loss.item())
                lang_losses.append(lang_loss.item())

                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt.step(); opt.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt.zero_grad(); errors += 1
            except: errors += 1

        avg_t = np.mean(traj_losses) if traj_losses else 0
        avg_v = np.mean(vis_losses) if vis_losses else 0
        avg_l = np.mean(lang_losses) if lang_losses else 0
        log(f"    Epoch {epoch+1}: traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log(f"\n  Training done: {elapsed/60:.1f} min", LOG_FILE)
    save_checkpoint(model, vis_decoder, lang_decoder, OUTPUT_BASE, VARIANT, "final")

    log("\n=== Evaluation ===", LOG_FILE)
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V5: No Staged", include_latent=True, log_file=LOG_FILE)
    result['train_time_min'] = elapsed / 60
    result['final_vis_loss'] = avg_v
    result['final_lang_loss'] = avg_l
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
