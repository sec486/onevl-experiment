#!/opt/onevl-env/bin/python3
"""
Variant 2: AR CoT+Answer
=========================
Standard SFT with explicit CoT reasoning text before trajectory.
No latent tokens, no decoders. Model generates full text CoT then trajectory.
OneVL equivalent: sft_distributed_qwen3vl_cot_64.sh
"""
import sys, os, json, torch, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_utils import load_data, save_checkpoint, log, evaluate_model, build_eval_prompt, parse_trajectory, compute_ade_fde
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

VARIANT = "v2_ar_cot"
OUTPUT_DIR = os.path.join(OUTPUT_BASE, VARIANT)
LOG_FILE = None

def build_cot_input(sample, processor, tokenizer, device='cuda'):
    """Build input with explicit CoT text before trajectory (no latent tokens)."""
    img_path = sample.get('images', [None])[0]
    traj_text = ""
    for msg in sample.get('messages', []):
        if msg.get('role') == 'assistant':
            traj_text = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])

    # Prepend CoT reasoning before trajectory
    cot_prefix = DEFAULT_COT + " "
    full_answer = cot_prefix + traj_text

    image = Image.open(img_path).convert('RGB').resize((448, 448))
    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
    pixel_values = img_inputs['pixel_values'].to(device)
    image_grid_thw = img_inputs['image_grid_thw'].to(device)
    t, h, w = image_grid_thw[0].tolist()
    num_img_tokens = int(t * h * w // 4)

    im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    vis_start = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vis_end = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    img_pad = tokenizer.convert_tokens_to_ids('<|image_pad|>')
    nl = tokenizer.encode('\n', add_special_tokens=False)

    sys_role = tokenizer.encode('system', add_special_tokens=False)
    sys_text = tokenizer.encode('You are a helpful assistant.', add_special_tokens=False)
    usr_role = tokenizer.encode('user', add_special_tokens=False)
    usr_text = tokenizer.encode('Predict the ego vehicle trajectory for the next 4 seconds.',
                                 add_special_tokens=False)
    ast_role = tokenizer.encode('assistant', add_special_tokens=False)
    answer_ids = tokenizer.encode(full_answer, add_special_tokens=False)

    input_ids = []
    labels = []

    sys_part = [im_start] + sys_role + nl + sys_text + [im_end] + nl
    input_ids += sys_part
    labels += [-100] * len(sys_part)

    usr_part = ([im_start] + usr_role + nl +
                [vis_start] + [img_pad] * num_img_tokens + [vis_end] +
                usr_text + [im_end] + nl)
    input_ids += usr_part
    labels += [-100] * len(usr_part)

    ast_prefix = [im_start] + ast_role + nl
    input_ids += ast_prefix
    labels += [-100] * len(ast_prefix)

    # Full answer (CoT + trajectory) is supervised
    input_ids += answer_ids + [im_end]
    labels += answer_ids + [im_end]

    mm_token_type_ids = [0] * len(input_ids)
    img_start_idx = len(sys_part) + len([im_start]) + len(usr_role) + len(nl) + 1
    for i in range(img_start_idx, img_start_idx + num_img_tokens):
        if i < len(mm_token_type_ids):
            mm_token_type_ids[i] = 1

    return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long, device=device),
        'labels': torch.tensor([labels], dtype=torch.long, device=device),
        'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long, device=device),
        'mm_token_type_ids': torch.tensor([mm_token_type_ids], dtype=torch.long, device=device),
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
    }


def main():
    global LOG_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, "train_log.txt")

    log(f"{'='*60}", LOG_FILE)
    log(f"  Variant 2: AR CoT+Answer (explicit reasoning, no latent)", LOG_FILE)
    log(f"{'='*60}", LOG_FILE)

    train_samples, eval_samples, _ = load_data()
    log(f"  Train: {len(train_samples)}, Eval: {len(eval_samples)}", LOG_FILE)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to('cuda')

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    TOTAL_EPOCHS = STAGE_0_EPOCHS + STAGE_2_EPOCHS
    log(f"\nTraining {TOTAL_EPOCHS} epochs (CoT+Answer, no latent)...", LOG_FILE)

    t0 = time.time()
    for epoch in range(TOTAL_EPOCHS):
        losses, errors = [], 0
        opt.zero_grad()
        for idx, sample in enumerate(train_samples):
            try:
                batch = build_cot_input(sample, processor, tokenizer)
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
        log(f"  Epoch {epoch+1}/{TOTAL_EPOCHS}: loss={avg:.4f}, samples={len(losses)}, errors={errors}", LOG_FILE)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    log(f"\n  Training done: {elapsed/60:.1f} min", LOG_FILE)

    save_checkpoint(model, None, None, OUTPUT_BASE, VARIANT, "final")

    # Eval (no latent tokens in prompt)
    log("\n=== Evaluation ===", LOG_FILE)
    model.eval()
    result = evaluate_model(model, processor, tokenizer, eval_samples,
                           "V2: AR CoT+Answer", include_latent=False, log_file=LOG_FILE)

    result['train_time_min'] = elapsed / 60
    result['final_loss'] = avg
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)

    log(f"\n  DONE: ADE={result['ade_mean']:.3f}m, FDE={result['fde_mean']:.3f}m", LOG_FILE)


if __name__ == '__main__':
    main()
