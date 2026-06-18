#!/opt/onevl-env/bin/python3
"""
Variant 3: OneVL w/o Visual Decoder (Language Decoder Only)
============================================================
Latent tokens + language decoder. No visual decoder.
3-stage training: Stage 0 (traj warmup) → Stage 1 (lang dec, detach) → Stage 2 (joint, no detach)
OneVL equivalent: ablation row "w/o vis. dec."
"""
import sys, os, json, torch, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from decoders import LangDecoder
from data_utils import (load_data, build_inputs_with_latent, subsample_tokens,
                        save_checkpoint, log, evaluate_model)
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
    log(f"  Variant 3: OneVL w/o Visual Decoder (lang only)", LOG_FILE)
    log(f"  3-stage: S0 (traj) → S1 (lang dec, detach) → S2 (joint)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    train_samples, eval_samples, _ = load_data()
    log(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}", LOG_FILE)

    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to('cuda')

    hidden_dim = 2048
    vocab_size = len(tokenizer)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Language decoder only (no visual)
    lang_decoder = LangDecoder(
        d_model=hidden_dim, n_heads=8, vocab_size=vocab_size, inner_dim=512
    ).to('cuda').to(torch.bfloat16)

    default_cot_ids = tokenizer(DEFAULT_COT, return_tensors='pt', max_length=64,
                                truncation=True, padding='max_length')['input_ids'].to('cuda')

    # ========== Stage 0: Trajectory Warmup ==========
    log(f"\n{'='*60}", LOG_FILE)
    log(f"  STAGE 0: Trajectory Warmup ({STAGE_0_EPOCHS} epochs)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    lang_decoder.eval()
    for p in lang_decoder.parameters(): p.requires_grad = False
    model.train()
    opt_s0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    t0 = time.time()
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
        log(f"  S0 Epoch {epoch+1}/{STAGE_0_EPOCHS}: traj={np.mean(losses) if losses else 0:.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(model, None, None, OUTPUT_BASE, VARIANT, "stage0")
    log(f"  Stage 0 done: {(time.time()-t0)/60:.1f} min", LOG_FILE)

    # ========== Stage 1: Language Decoder Warmup (DETACH) ==========
    log(f"\n{'='*60}", LOG_FILE)
    log(f"  STAGE 1: Language Decoder Warmup ({STAGE_1_EPOCHS} epochs, DETACH)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    model.eval()
    for p in model.parameters(): p.requires_grad = False
    lang_decoder.train()
    for p in lang_decoder.parameters(): p.requires_grad = True
    opt_s1 = torch.optim.AdamW(lang_decoder.parameters(), lr=LR * 2)

    t1 = time.time()
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
        log(f"  S1 Epoch {epoch+1}/{STAGE_1_EPOCHS}: lang={np.mean(lang_losses) if lang_losses else 0:.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(None, None, lang_decoder, OUTPUT_BASE, VARIANT, "stage1")
    log(f"  Stage 1 done: {(time.time()-t1)/60:.1f} min", LOG_FILE)

    # ========== Stage 2: Joint (NO DETACH) ==========
    log(f"\n{'='*60}", LOG_FILE)
    log(f"  STAGE 2: Joint Training ({STAGE_2_EPOCHS} epochs, NO DETACH)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    model.train()
    lang_decoder.train()
    for p in model.parameters(): p.requires_grad = True
    for p in lang_decoder.parameters(): p.requires_grad = True
    all_params = [p for p in model.parameters() if p.requires_grad] + list(lang_decoder.parameters())
    opt_s2 = torch.optim.AdamW(all_params, lr=LR * 0.5)

    t2 = time.time()
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
        log(f"  S2 Epoch {epoch+1}/{STAGE_2_EPOCHS}: traj={np.mean(traj_losses) if traj_losses else 0:.4f}, "
            f"lang={np.mean(lang_losses) if lang_losses else 0:.4f}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    save_checkpoint(model, None, lang_decoder, OUTPUT_BASE, VARIANT, "stage2_final")
    log(f"  Stage 2 done: {(time.time()-t2)/60:.1f} min", LOG_FILE)

    # Evaluate
    log("\n=== Evaluation ===", LOG_FILE)
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V3: Lang Only (no vis dec)", include_latent=True, log_file=LOG_FILE)
    result['train_time_min'] = (time.time() - t0) / 60
    result['final_traj_loss'] = np.mean(traj_losses) if traj_losses else 0
    result['final_lang_loss'] = np.mean(lang_losses) if lang_losses else 0
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
