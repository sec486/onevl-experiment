#!/usr/bin/env python3
"""
变体 3: OneVL 无视觉解码器（仅语言解码器）
============================================
3 阶段训练，仅使用语言辅助解码器:
  - Stage 0: 轨迹 warmup（带隐式 token）
  - Stage 1: 语言解码器 warmup（梯度截断）
  - Stage 2: 联合训练（语言解码器梯度回传）

预期结果: ADE ~8.3m, 训练 ~61 分钟 (L4)。
"""
import sys, os, json, torch, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from decoders import LangDecoder
from data_utils import (load_data, build_inputs_with_latent, save_checkpoint,
                        log, evaluate_model, subsample_tokens)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

VARIANT = "v3_lang_only"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, VARIANT)
LOG_FILE = None


def main():
    global LOG_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "train_log.txt")

    log(f"{'='*60}", LOG_FILE)
    log(f"  Variant 3: Language Decoder Only (3-stage)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    train_samples, eval_samples, vis_tokens_dict = load_data()
    log(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}", LOG_FILE)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to('cuda')

    hidden_dim = 2048
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)

    lang_decoder = LangDecoder(
        d_model=hidden_dim, n_heads=8, vocab_size=len(tokenizer), inner_dim=512
    ).to('cuda').to(torch.bfloat16)

    default_cot_ids = tokenizer(DEFAULT_COT, return_tensors='pt', max_length=64,
                                truncation=True, padding='max_length')['input_ids'].to('cuda')

    t0 = time.time()

    # STAGE 0: Trajectory Warmup
    log(f"\n  STAGE 0: Trajectory Warmup ({STAGE_0_EPOCHS} epochs)", LOG_FILE)
    lang_decoder.eval()
    for p in lang_decoder.parameters(): p.requires_grad = False
    model.train()
    opt_s0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    for epoch in range(STAGE_0_EPOCHS):
        losses, errors = [], 0
        opt_s0.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                outputs = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],
                    mm_token_type_ids=batch['mm_token_type_ids'],
                )
                loss = outputs.loss
                if loss is None or torch.isnan(loss): continue
                (loss / GRAD_ACCUM).backward()
                losses.append(loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_s0.step()
                    opt_s0.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s0.zero_grad(); errors += 1
            except: errors += 1
        log(f"    Epoch {epoch+1}: traj={np.mean(losses):.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    # STAGE 1: Decoder Warmup (Detached)
    log(f"\n  STAGE 1: Lang Decoder Warmup ({STAGE_1_EPOCHS} epochs, DETACHED)", LOG_FILE)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lang_decoder.train()
    for p in lang_decoder.parameters(): p.requires_grad = True
    opt_s1 = torch.optim.AdamW(lang_decoder.parameters(), lr=LR * 2)

    for epoch in range(STAGE_1_EPOCHS):
        lang_losses, errors = [], 0
        opt_s1.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=True)
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'],
                        mm_token_type_ids=batch['mm_token_type_ids'],
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[-1]
                lang_latent_h = hidden[:, batch['lang_lat_pos'], :]
                lang_loss = lang_decoder(lang_latent_h, default_cot_ids)
                (lang_loss / GRAD_ACCUM).backward()
                lang_losses.append(lang_loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(lang_decoder.parameters(), 1.0)
                    opt_s1.step()
                    opt_s1.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s1.zero_grad(); errors += 1
            except: errors += 1
        log(f"    Epoch {epoch+1}: lang={np.mean(lang_losses):.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    # STAGE 2: Joint (No Detach)
    log(f"\n  STAGE 2: Joint Training ({STAGE_2_EPOCHS} epochs, NO DETACH)", LOG_FILE)
    model.train()
    for name, p in model.named_parameters():
        if 'lora' in name or 'embed_tokens' in name: p.requires_grad = True
    all_params = [p for p in model.parameters() if p.requires_grad] + list(lang_decoder.parameters())
    opt_s2 = torch.optim.AdamW(all_params, lr=LR * 0.5)

    for epoch in range(STAGE_2_EPOCHS):
        traj_losses, lang_losses, errors = [], [], 0
        opt_s2.zero_grad()
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
                lang_latent_h = hidden[:, batch['lang_lat_pos'], :]
                lang_loss = lang_decoder(lang_latent_h, default_cot_ids)
                total = (traj_loss + LAMBDA_LANG * lang_loss) / GRAD_ACCUM
                total.backward()
                traj_losses.append(traj_loss.item())
                lang_losses.append(lang_loss.item())
                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt_s2.step()
                    opt_s2.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt_s2.zero_grad(); errors += 1
            except: errors += 1
        log(f"    Epoch {epoch+1}: traj={np.mean(traj_losses):.4f}, lang={np.mean(lang_losses):.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log(f"\n  Training done: {elapsed/60:.1f} min", LOG_FILE)

    save_checkpoint(model, None, lang_decoder, OUTPUT_BASE, VARIANT, "final")

    log("\n=== Evaluation ===", LOG_FILE)
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V3: Lang Only", include_latent=True, log_file=LOG_FILE)
    result['train_time_min'] = elapsed / 60
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
