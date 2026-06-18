"""Shared data loading, tokenization, and evaluation utilities."""
import json, re, os, time, torch
import numpy as np
from PIL import Image
from config import *


def log(msg, log_file=None):
    """Print + write to log file."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_data():
    """Load training samples and visual tokens."""
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= MAX_SAMPLES + NUM_EVAL:
                    break

    vis_tokens_dict = torch.load(VIS_TOKENS_PATH, weights_only=False)

    # Split: first MAX_SAMPLES for training, last NUM_EVAL for eval
    train_samples = samples[:MAX_SAMPLES]
    eval_samples = samples[-NUM_EVAL:]

    for s in train_samples + eval_samples:
        s['_scene'] = s.get('scene', '')
        s['_has_vis'] = s['_scene'] in vis_tokens_dict

    return train_samples, eval_samples, vis_tokens_dict


def subsample_tokens(full_tokens, target=64):
    """Subsample visual tokens to target count."""
    flat = full_tokens.flatten()
    indices = torch.linspace(0, len(flat) - 1, target).long()
    return flat[indices].long()


def build_inputs_with_latent(sample, processor, tokenizer, device='cuda',
                              include_latent=True):
    """
    Manual tokenization with optional latent tokens.
    If include_latent=False: standard assistant turn (for AR answer/CoT variants)
    If include_latent=True: inject VIS_LATENT + LANG_LATENT before trajectory
    """
    img_path = sample.get('images', [None])[0]
    traj_text = ""
    cot_text = ""
    for msg in sample.get('messages', []):
        if msg.get('role') == 'assistant':
            content = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])
            traj_text = content

    # Process image
    image = Image.open(img_path).convert('RGB').resize((448, 448))
    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
    pixel_values = img_inputs['pixel_values'].to(device)
    image_grid_thw = img_inputs['image_grid_thw'].to(device)
    t, h, w = image_grid_thw[0].tolist()
    num_img_tokens = int(t * h * w // 4)

    # Token IDs
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
    answer_ids = tokenizer.encode(traj_text, add_special_tokens=False)

    # Build sequence
    input_ids = []
    labels = []

    # System
    sys_part = [im_start] + sys_role + nl + sys_text + [im_end] + nl
    input_ids += sys_part
    labels += [-100] * len(sys_part)

    # User (with image)
    usr_part = ([im_start] + usr_role + nl +
                [vis_start] + [img_pad] * num_img_tokens + [vis_end] +
                usr_text + [im_end] + nl)
    input_ids += usr_part
    labels += [-100] * len(usr_part)

    # Assistant
    ast_prefix = [im_start] + ast_role + nl
    input_ids += ast_prefix
    labels += [-100] * len(ast_prefix)

    # Latent tokens (optional)
    vis_lat_pos = []
    lang_lat_pos = []
    if include_latent:
        latent_start = len(input_ids)
        latent_ids = [VIS_LATENT_ID] * NUM_VIS_LATENT + [LANG_LATENT_ID] * NUM_LANG_LATENT
        input_ids += latent_ids
        labels += [-100] * len(latent_ids)
        vis_lat_pos = list(range(latent_start, latent_start + NUM_VIS_LATENT))
        lang_lat_pos = list(range(latent_start + NUM_VIS_LATENT,
                                   latent_start + NUM_VIS_LATENT + NUM_LANG_LATENT))

    # Trajectory answer
    input_ids += answer_ids + [im_end]
    labels += answer_ids + [im_end]

    # mm_token_type_ids
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
        'vis_lat_pos': vis_lat_pos,
        'lang_lat_pos': lang_lat_pos,
    }


def build_eval_prompt(sample, processor, tokenizer, device='cuda', include_latent=True):
    """Build input_ids for generation (no labels, stop after latent tokens or assistant prefix)."""
    img_path = sample.get('images', [None])[0]
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
        'input_ids': torch.tensor([input_ids], dtype=torch.long, device=device),
        'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long, device=device),
        'mm_token_type_ids': torch.tensor([mm_token_type_ids], dtype=torch.long, device=device),
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
    }


def parse_trajectory(text):
    """Parse [x,y,h] waypoints from model output."""
    pattern = r'\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]'
    matches = re.findall(pattern, text)
    if len(matches) < 4:
        return None
    return [(float(x), float(y), float(h)) for x, y, h in matches[:8]]


def compute_ade_fde(pred, gt):
    """Compute ADE and FDE in meters."""
    pred = np.array(pred)[:, :2]
    gt = np.array(gt)[:, :2]
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    distances = np.linalg.norm(pred - gt, axis=1)
    return float(distances.mean()), float(distances[-1])


def evaluate_model(model, processor, tokenizer, eval_samples, label,
                   include_latent=True, log_file=None):
    """Run greedy eval on samples, return ADE/FDE/valid dict."""
    log(f"Evaluating: {label} ({len(eval_samples)} samples)", log_file)
    ades, fdes, valid, errors = [], [], 0, 0
    t0 = time.time()

    for i, sample in enumerate(eval_samples):
        try:
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

            batch = build_eval_prompt(sample, processor, tokenizer,
                                      include_latent=include_latent)
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
                log(f"  [{i+1}/{len(eval_samples)}] valid={valid}, ADE={np.mean(ades):.3f}m",
                    log_file)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                log(f"  err: {type(e).__name__}: {str(e)[:80]}", log_file)

    result = {
        "label": label,
        "valid": valid,
        "total": len(eval_samples),
        "errors": errors,
        "ade_mean": float(np.mean(ades)) if ades else -1,
        "ade_std": float(np.std(ades)) if ades else 0,
        "fde_mean": float(np.mean(fdes)) if fdes else -1,
        "fde_std": float(np.std(fdes)) if fdes else 0,
        "time_s": time.time() - t0,
    }
    log(f"  → ADE={result['ade_mean']:.3f}±{result['ade_std']:.3f}m, "
        f"FDE={result['fde_mean']:.3f}±{result['fde_std']:.3f}m, "
        f"valid={valid}/{len(eval_samples)}", log_file)
    return result


def save_checkpoint(model, vis_decoder, lang_decoder, output_dir, variant_name, stage_name):
    """Save model + decoders to organized checkpoint folder."""
    ckpt_dir = os.path.join(output_dir, variant_name, f"checkpoint_{stage_name}")
    os.makedirs(ckpt_dir, exist_ok=True)

    if model is not None:
        model.save_pretrained(ckpt_dir)
    if vis_decoder is not None:
        torch.save(vis_decoder.state_dict(), os.path.join(ckpt_dir, "vis_decoder.pt"))
    if lang_decoder is not None:
        torch.save(lang_decoder.state_dict(), os.path.join(ckpt_dir, "lang_decoder.pt"))

    log(f"  Saved checkpoint: {ckpt_dir}")
    return ckpt_dir
