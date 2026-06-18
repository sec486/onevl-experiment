#!/usr/bin/env python3
"""
变体 1: AR Answer（基线）
=========================
标准 SFT 轨迹预测。无隐式 token，无解码器。
对应 OneVL: sft_distributed_qwen3vl_answer_bs64.sh

最简单的基线 — 模型直接从图像+prompt 预测 [x,y,h]。
预期结果: ADE ~7.1m, 训练 ~86 分钟 (L4)。
"""
import sys, os, json, torch, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_utils import load_data, build_inputs_with_latent, save_checkpoint, log, evaluate_model
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

VARIANT = "v1_ar_answer"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, VARIANT)
LOG_FILE = None


def main():
    global LOG_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "train_log.txt")

    log(f"{'='*60}", LOG_FILE)
    log(f"  Variant 1: AR Answer (no latent, no decoder)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    train_samples, eval_samples, _ = load_data()
    log(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}", LOG_FILE)

    # Load model
    log("\nLoading model...", LOG_FILE)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to('cuda')

    # LoRA (same config as all variants for fair comparison)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    # Train: 10 epochs (= Stage 0 + Stage 2 combined for fair time comparison)
    TOTAL_EPOCHS = STAGE_0_EPOCHS + STAGE_2_EPOCHS
    log(f"\nTraining {TOTAL_EPOCHS} epochs (no latent tokens)...", LOG_FILE)

    t0 = time.time()
    for epoch in range(TOTAL_EPOCHS):
        losses, errors = [], 0
        opt.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer, include_latent=False)
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],
                    mm_token_type_ids=batch['mm_token_type_ids'],
                )
                loss = outputs.loss
                if loss is None or torch.isnan(loss):
                    continue
                (loss / GRAD_ACCUM).backward()
                losses.append(loss.item())

                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt.zero_grad()
                errors += 1
            except Exception as e:
                errors += 1

        avg = np.mean(losses) if losses else 0
        log(f"  Epoch {epoch+1}/{TOTAL_EPOCHS}: traj={avg:.4f}, samples={len(losses)}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log(f"\n  Training done: {elapsed/60:.1f} min", LOG_FILE)

    # Save checkpoint
    save_checkpoint(model, None, None, OUTPUT_BASE, VARIANT, "final")

    # Evaluate
    log("\n=== Evaluation ===", LOG_FILE)
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V1: AR Answer", include_latent=False, log_file=LOG_FILE)

    # Save results
    result['train_time_min'] = elapsed / 60
    result['final_traj_loss'] = avg
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)

    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
