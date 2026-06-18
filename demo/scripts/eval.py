# -*- coding: utf-8 -*-
"""
评估: ADE/FDE 轨迹预测指标
============================
在留出的 NAVSIM 样本上评估训练好的模型:
  - ADE (平均位移误差): 所有路径点的平均 L2 距离
  - FDE (终点位移误差): 最后一个路径点的 L2 距离

用法:
    python eval.py --checkpoint /path/to/checkpoint --latent

参数:
    --checkpoint: LoRA checkpoint 目录路径（包含 adapter_model.safetensors）
    --latent: 生成时在 prompt 中包含隐式 token（用于 OneVL 变体）
    --num-eval: 评估样本数（默认: 50）
"""
import argparse, json, os, re, time, torch
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ============================================================
# 配置
# ============================================================
MODEL_PATH = '/opt/onevl/models/Cosmos-Reason2-2B'
DATA_PATH = '/opt/onevl/data/navsim_real_traj_expanded.jsonl'

# 隐式 token ID（必须与训练时一致）
VIS_LATENT_ID = 151662
LANG_LATENT_ID = 151663
NUM_VIS_LATENT = 4
NUM_LANG_LATENT = 2

DEVICE = 'cuda'


# ============================================================
# 工具函数
# ============================================================
def parse_trajectory(text):
    """从模型输出文本中解析 [x,y,h] 路径点。"""
    pattern = r'\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]'
    matches = re.findall(pattern, text)
    if len(matches) < 4:
        return None
    return [(float(x), float(y), float(h)) for x, y, h in matches[:8]]


def compute_ade_fde(pred, gt):
    """计算 ADE 和 FDE（单位: 米）。"""
    pred = np.array(pred)[:, :2]  # x, y only
    gt = np.array(gt)[:, :2]
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    distances = np.linalg.norm(pred - gt, axis=1)
    ade = float(distances.mean())
    fde = float(distances[-1])
    return ade, fde


def build_eval_prompt(sample, processor, tokenizer, include_latent=True):
    """构建生成用的 prompt（system + user/image + assistant 前缀 + 可选隐式 token）。"""
    img_path = sample.get('images', [None])[0]
    image = Image.open(img_path).convert('RGB').resize((448, 448))
    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
    pixel_values = img_inputs['pixel_values'].to(DEVICE)
    image_grid_thw = img_inputs['image_grid_thw'].to(DEVICE)
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

    input_ids = []
    input_ids += [im_start] + sys_role + nl + sys_text + [im_end] + nl
    input_ids += ([im_start] + usr_role + nl +
                  [vis_start] + [img_pad] * num_img_tokens + [vis_end] +
                  usr_text + [im_end] + nl)
    input_ids += [im_start] + ast_role + nl

    if include_latent:
        input_ids += [VIS_LATENT_ID] * NUM_VIS_LATENT + [LANG_LATENT_ID] * NUM_LANG_LATENT

    mm_token_type_ids = [0] * len(input_ids)
    sys_len = len([im_start]) + len(sys_role) + len(nl) + len(sys_text) + len([im_end]) + len(nl)
    img_start_idx = sys_len + len([im_start]) + len(usr_role) + len(nl) + 1
    for i in range(img_start_idx, img_start_idx + num_img_tokens):
        if i < len(mm_token_type_ids):
            mm_token_type_ids[i] = 1

    return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long, device=DEVICE),
        'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long, device=DEVICE),
        'mm_token_type_ids': torch.tensor([mm_token_type_ids], dtype=torch.long, device=DEVICE),
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='评估轨迹预测模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='LoRA checkpoint 路径')
    parser.add_argument('--latent', action='store_true', help='在 prompt 中包含隐式 token')
    parser.add_argument('--num-eval', type=int, default=50, help='评估样本数')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Evaluating: {args.checkpoint}")
    print(f"  Latent tokens: {args.latent}")
    print(f"{'='*60}")

    # Load data (use last N samples as eval set)
    print("\nLoading data...")
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    eval_samples = samples[-args.num_eval:]
    print(f"  Eval samples: {len(eval_samples)}")

    # Load model + checkpoint
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to(DEVICE)
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()
    print("  Model loaded with LoRA checkpoint")

    # Evaluate
    print("\nRunning evaluation...")
    ades, fdes, valid, errors = [], [], 0, 0
    t0 = time.time()

    for i, sample in enumerate(eval_samples):
        try:
            # Get ground truth
            gt_text = ""
            for msg in sample.get('messages', []):
                if msg.get('role') == 'assistant':
                    gt_text = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])
            gt_traj = parse_trajectory(gt_text)
            if gt_traj is None:
                continue

            img_path = sample.get('images', [None])[0]
            if not img_path or not os.path.exists(img_path):
                continue

            # Generate
            batch = build_eval_prompt(sample, processor, tokenizer, include_latent=args.latent)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],
                    max_new_tokens=150,
                    do_sample=False,
                )
            gen_ids = output_ids[0][batch['input_ids'].shape[1]:]
            generated = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred_traj = parse_trajectory(generated)

            if pred_traj is None:
                errors += 1
                continue

            ade, fde = compute_ade_fde(pred_traj, gt_traj)
            ades.append(ade)
            fdes.append(fde)
            valid += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(eval_samples)}] valid={valid}, "
                      f"ADE={np.mean(ades):.3f}m, FDE={np.mean(fdes):.3f}m")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error: {type(e).__name__}: {str(e)[:80]}")

    elapsed = time.time() - t0

    # Results
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    if ades:
        print(f"  ADE: {np.mean(ades):.3f} ± {np.std(ades):.3f} m")
        print(f"  FDE: {np.mean(fdes):.3f} ± {np.std(fdes):.3f} m")
    print(f"  Valid: {valid}/{len(eval_samples)}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s")

    # Save results
    result = {
        "checkpoint": args.checkpoint,
        "include_latent": args.latent,
        "valid": valid,
        "total": len(eval_samples),
        "errors": errors,
        "ade_mean": float(np.mean(ades)) if ades else -1,
        "ade_std": float(np.std(ades)) if ades else 0,
        "fde_mean": float(np.mean(fdes)) if fdes else -1,
        "fde_std": float(np.std(fdes)) if fdes else 0,
        "time_s": elapsed,
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
