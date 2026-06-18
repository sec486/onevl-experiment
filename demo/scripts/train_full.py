# -*- coding: utf-8 -*-
"""
OneVL × Cosmos-Reason2: 完整 4 阶段训练
========================================
在 Cosmos-Reason2-2B 上实现 OneVL 完整训练方法:
  - 预训练阶段: 视觉解码器独立预训练（仅 ViT 嵌入）
  - Stage 0: 轨迹 warmup（主模型学习基础轨迹预测）
  - Stage 1: 解码器 warmup（解码器学习读取隐式状态，梯度截断）
  - Stage 2: 联合训练（解码器梯度回传至主模型）

用法:
    python train_full.py

环境要求:
    - GPU: 24GB+ (L4/A10G/L40S)
    - transformers==5.4.0, peft==0.19.1, torch==2.6.0+cu124
    - 预编码的视觉 token（先运行 encode_emu3.py）
    - 训练数据: navsim_real_traj_expanded.jsonl (250 场景, 1250 样本)

预期结果: L4 上 ~93 分钟, vis_loss 6.94→3.90, traj_loss 0.624→0.608
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# 配置（根据你的环境修改路径）
# ============================================================
MODEL_PATH = '/opt/onevl/models/Cosmos-Reason2-2B'
DATA_PATH = '/opt/onevl/data/navsim_real_traj_expanded.jsonl'
VIS_TOKENS_PATH = '/opt/onevl/data/emu3_visual_tokens_expanded.pt'
OUTPUT_DIR = '/opt/onevl/output/full_training'

MAX_SAMPLES = 1250
GRAD_ACCUM = 8
LR = 2e-5

# 架构参数
CODEBOOK_SIZE = 32768       # Emu3 VisionTokenizer 词表大小
NUM_VIS_QUERIES = 64        # 下采样后的视觉 token 目标数
NUM_VIS_LATENT = 4          # 序列中的视觉隐式 token 数量
NUM_LANG_LATENT = 2         # 序列中的语言隐式 token 数量
LAMBDA_VIS = 0.5            # 视觉解码器损失权重
LAMBDA_LANG = 0.5           # 语言解码器损失权重

# 隐式 token ID（选用在驾驶场景中不会出现的稀有词表 ID）
VIS_LATENT_ID = 151662      # <|fim_pad|>
LANG_LATENT_ID = 151663     # <|repo_name|>

# 训练调度
PRELIMINARY_EPOCHS = 3
STAGE_0_EPOCHS = 5
STAGE_1_EPOCHS = 3
STAGE_2_EPOCHS = 5

DEVICE = 'cuda'

# ============================================================
# 日志
# ============================================================
LOG_FILE = None

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================
# 视觉解码器 (OneVL 论文 Section 3.4)
# ============================================================
class VisualDecoderV2(nn.Module):
    """
    OneVL 公式3: Z_v = [W_v(V), W_v(H_v)]
    输入: ViT patch 嵌入 + 隐式 token 的 hidden state
    输出: 离散视觉 token 预测（CE loss 对比 Emu3 token）
    """
    def __init__(self, vit_dim=1024, hidden_dim=2048, codebook_size=32768,
                 num_queries=64, num_layers=2, num_heads=8, inner_dim=512):
        super().__init__()
        self.proj_vit = nn.Linear(vit_dim, inner_dim)
        self.proj_lat = nn.Linear(hidden_dim, inner_dim)
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, inner_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=inner_dim, nhead=num_heads,
            dim_feedforward=inner_dim * 2, batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(inner_dim, codebook_size)

    def forward(self, vit_emb, latent_h=None, gt_ids=None):
        B = vit_emb.shape[0]
        if latent_h is not None:
            z_v = torch.cat([self.proj_vit(vit_emb), self.proj_lat(latent_h)], dim=1)
        else:
            z_v = self.proj_vit(vit_emb)
        queries = self.queries.expand(B, -1, -1)
        decoded = self.decoder(queries, z_v)
        logits = self.head(decoded)
        if gt_ids is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), gt_ids.view(-1))
        return logits


# ============================================================
# 语言解码器
# ============================================================
class LangDecoder(nn.Module):
    """从语言隐式 token 的 hidden state 重建 CoT 文本。"""
    def __init__(self, d_model=2048, n_heads=8, vocab_size=151936, inner_dim=512):
        super().__init__()
        self.proj_in = nn.Linear(d_model, inner_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=inner_dim, nhead=n_heads,
            dim_feedforward=inner_dim * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.head = nn.Linear(inner_dim, vocab_size)

    def forward(self, latent_h, target_ids):
        B, L = target_ids.shape
        latent_proj = self.proj_in(latent_h)
        inner_dim = latent_proj.shape[-1]
        causal_mask = torch.triu(torch.ones(L, L, device=target_ids.device), diagonal=1).bool()
        tgt = torch.zeros(B, L, inner_dim, device=latent_h.device, dtype=latent_h.dtype)
        decoded = self.decoder(tgt, latent_proj, tgt_mask=causal_mask)
        logits = self.head(decoded)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)


# ============================================================
# 辅助函数
# ============================================================
def subsample_tokens(full_tokens, target=64):
    """均匀下采样视觉 token 到目标数量。"""
    flat = full_tokens.flatten()
    indices = torch.linspace(0, len(flat) - 1, target).long()
    return flat[indices]


def build_inputs_with_latent(sample, processor, tokenizer):
    """
    构建带有隐式 token 的 tokenized 输入。
    序列: [system][user+image][assistant: vis_latent×4 + lang_latent×2 + trajectory]
    """
    img_path = sample.get('images', [None])[0]
    traj_text = ""
    for msg in sample.get('messages', []):
        if msg.get('role') == 'assistant':
            traj_text = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])

    image = Image.open(img_path).convert('RGB').resize((448, 448))
    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
    pixel_values = img_inputs['pixel_values'].to(DEVICE)
    image_grid_thw = img_inputs['image_grid_thw'].to(DEVICE)
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
    usr_text = tokenizer.encode('Predict the ego vehicle trajectory for the next 4 seconds.', add_special_tokens=False)
    ast_role = tokenizer.encode('assistant', add_special_tokens=False)
    answer_ids = tokenizer.encode(traj_text, add_special_tokens=False)

    input_ids, labels = [], []

    # System turn
    sys_part = [im_start] + sys_role + nl + sys_text + [im_end] + nl
    input_ids += sys_part
    labels += [-100] * len(sys_part)

    # User turn (with image)
    usr_part = ([im_start] + usr_role + nl +
                [vis_start] + [img_pad] * num_img_tokens + [vis_end] +
                usr_text + [im_end] + nl)
    input_ids += usr_part
    labels += [-100] * len(usr_part)

    # Assistant turn with latent tokens
    ast_prefix = [im_start] + ast_role + nl
    input_ids += ast_prefix
    labels += [-100] * len(ast_prefix)

    # Latent tokens (no loss)
    latent_start = len(input_ids)
    latent_ids = [VIS_LATENT_ID] * NUM_VIS_LATENT + [LANG_LATENT_ID] * NUM_LANG_LATENT
    input_ids += latent_ids
    labels += [-100] * len(latent_ids)

    # Trajectory answer (CE loss)
    input_ids += answer_ids + [im_end]
    labels += answer_ids + [im_end]

    vis_lat_pos = list(range(latent_start, latent_start + NUM_VIS_LATENT))
    lang_lat_pos = list(range(latent_start + NUM_VIS_LATENT, latent_start + NUM_VIS_LATENT + NUM_LANG_LATENT))

    # mm_token_type_ids (1 for image tokens, 0 otherwise)
    mm_token_type_ids = [0] * len(input_ids)
    img_start_idx = len(sys_part) + len([im_start]) + len(usr_role) + len(nl) + 1
    for i in range(img_start_idx, img_start_idx + num_img_tokens):
        if i < len(mm_token_type_ids):
            mm_token_type_ids[i] = 1

    return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long, device=DEVICE),
        'labels': torch.tensor([labels], dtype=torch.long, device=DEVICE),
        'attention_mask': torch.ones(1, len(input_ids), dtype=torch.long, device=DEVICE),
        'mm_token_type_ids': torch.tensor([mm_token_type_ids], dtype=torch.long, device=DEVICE),
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'vis_lat_pos': vis_lat_pos,
        'lang_lat_pos': lang_lat_pos,
    }


# ============================================================
# 主训练循环
# ============================================================
def main():
    global LOG_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')

    log("=" * 60)
    log("  OneVL × Cosmos-Reason2: 4-Stage Full Training")
    log("=" * 60)

    # Load visual tokens
    log("\n=== Loading Emu3 visual tokens ===")
    vis_tokens_dict = torch.load(VIS_TOKENS_PATH, weights_only=False)
    log(f"  {len(vis_tokens_dict)} scenes encoded")

    # Load data
    log("\n=== Loading training data ===")
    samples = []
    with open(DATA_PATH) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= MAX_SAMPLES:
                    break
    for s in samples:
        s['_scene'] = s.get('scene', '')
        s['_has_vis'] = s['_scene'] in vis_tokens_dict
    log(f"  {len(samples)} samples, {sum(1 for s in samples if s['_has_vis'])} with vis tokens")

    # Load model
    log("\n=== Loading Cosmos-Reason2-2B ===")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager"
    ).to(DEVICE)
    log(f"  Loaded in {time.time()-t0:.1f}s")

    # ViT encoder reference (before LoRA wrapping changes paths)
    vit_encoder = model.model.visual
    hidden_dim = 2048

    # Probe ViT output dimension
    _probe_img = Image.new('RGB', (448, 448))
    _probe_inputs = processor.image_processor(images=[_probe_img], return_tensors='pt')
    with torch.no_grad():
        _probe_out = vit_encoder(_probe_inputs['pixel_values'].to(DEVICE),
                                  grid_thw=_probe_inputs['image_grid_thw'].to(DEVICE))
        vit_dim = (_probe_out.last_hidden_state if hasattr(_probe_out, 'last_hidden_state') else _probe_out).shape[-1]
    log(f"  ViT output dim: {vit_dim}")
    del _probe_img, _probe_inputs, _probe_out
    torch.cuda.empty_cache()

    # Apply LoRA
    log("\n=== Applying LoRA ===")
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],  # Critical: allows latent token embeddings to learn
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  {trainable/1e6:.1f}M trainable parameters")

    # Initialize decoders
    vis_decoder = VisualDecoderV2(
        vit_dim=vit_dim, hidden_dim=hidden_dim, codebook_size=CODEBOOK_SIZE,
        num_queries=NUM_VIS_QUERIES
    ).to(DEVICE).to(torch.bfloat16)

    lang_decoder = LangDecoder(
        d_model=hidden_dim, n_heads=8, vocab_size=len(tokenizer), inner_dim=512
    ).to(DEVICE).to(torch.bfloat16)

    log(f"  Visual decoder: {sum(p.numel() for p in vis_decoder.parameters())/1e6:.1f}M params")
    log(f"  Language decoder: {sum(p.numel() for p in lang_decoder.parameters())/1e6:.1f}M params")

    # CoT target for language decoder
    default_cot = "The vehicle should maintain safe trajectory based on the current driving conditions."
    default_cot_ids = tokenizer(default_cot, return_tensors='pt', max_length=64,
                                truncation=True, padding='max_length')['input_ids'].to(DEVICE)

    # ==================================================================
    # 预训练阶段: 视觉解码器独立预训练
    # ==================================================================
    log(f"\n{'='*60}")
    log(f"  预训练阶段: 视觉解码器预训练 ({PRELIMINARY_EPOCHS} epochs)")
    log(f"  输入: 仅 ViT 嵌入（无隐式 token, 不走主模型）")
    log(f"{'='*60}")

    vis_decoder.train()
    opt_pre = torch.optim.AdamW(vis_decoder.parameters(), lr=LR * 3)
    t_stage = time.time()

    for epoch in range(PRELIMINARY_EPOCHS):
        losses, errors = [], 0
        opt_pre.zero_grad()
        for idx, sample in enumerate(samples):
            if not sample['_has_vis']:
                continue
            try:
                img_path = sample.get('images', [None])[0]
                image = Image.open(img_path).convert('RGB').resize((448, 448))
                img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                with torch.no_grad():
                    vit_out = vit_encoder(img_inputs['pixel_values'].to(DEVICE),
                                          grid_thw=img_inputs['image_grid_thw'].to(DEVICE))
                    vit_emb = (vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out)
                    if vit_emb.dim() == 2:
                        vit_emb = vit_emb.unsqueeze(0)

                vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).to(DEVICE)
                vis_loss = vis_decoder(vit_emb, latent_h=None, gt_ids=vis_gt)
                (vis_loss / GRAD_ACCUM).backward()
                losses.append(vis_loss.item())

                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(vis_decoder.parameters(), 1.0)
                    opt_pre.step()
                    opt_pre.zero_grad()
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log(f"  PRE err: {type(e).__name__}: {str(e)[:80]}")
                continue

        log(f"  PRE Epoch {epoch+1}/{PRELIMINARY_EPOCHS}: vis_loss={np.mean(losses):.2f}, samples={len(losses)}, errors={errors}")
        torch.cuda.empty_cache()

    pre_time = time.time() - t_stage
    log(f"  Preliminary done: {pre_time/60:.1f} min")
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'preliminary_vis_decoder.pt'))

    # ==================================================================
    # STAGE 0: 轨迹 Warmup
    # ==================================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 0: 轨迹 Warmup ({STAGE_0_EPOCHS} epochs)")
    log(f"  序列包含隐式 token, 但 labels=-100（无解码器损失）")
    log(f"{'='*60}")

    vis_decoder.eval()
    lang_decoder.eval()
    for p in vis_decoder.parameters(): p.requires_grad = False
    for p in lang_decoder.parameters(): p.requires_grad = False
    model.train()

    opt_s0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    t_stage = time.time()

    for epoch in range(STAGE_0_EPOCHS):
        losses, errors = [], 0
        opt_s0.zero_grad()
        for idx, sample in enumerate(samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer)
                outputs = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
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
                    opt_s0.step()
                    opt_s0.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt_s0.zero_grad()
                errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5 and epoch == 0:
                    log(f"  S0 err: {type(e).__name__}: {str(e)[:100]}")
                continue

        log(f"  S0 Epoch {epoch+1}/{STAGE_0_EPOCHS}: traj={np.mean(losses):.4f}, samples={len(losses)}, errors={errors}")
        torch.cuda.empty_cache()

    s0_time = time.time() - t_stage
    log(f"  Stage 0 done: {s0_time/60:.1f} min")

    # ==================================================================
    # STAGE 1: 解码器 Warmup（梯度截断）
    # ==================================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 1: 解码器 Warmup ({STAGE_1_EPOCHS} epochs, 梯度截断)")
    log(f"  解码器学习读取隐式状态; 主模型冻结")
    log(f"{'='*60}")

    model.eval()
    for p in model.parameters(): p.requires_grad = False
    vis_decoder.train()
    lang_decoder.train()
    for p in vis_decoder.parameters(): p.requires_grad = True
    for p in lang_decoder.parameters(): p.requires_grad = True

    opt_s1 = torch.optim.AdamW(
        list(vis_decoder.parameters()) + list(lang_decoder.parameters()), lr=LR * 2
    )
    t_stage = time.time()

    for epoch in range(STAGE_1_EPOCHS):
        vis_losses, lang_losses, errors = [], [], 0
        opt_s1.zero_grad()
        for idx, sample in enumerate(samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer)
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'],
                        mm_token_type_ids=batch['mm_token_type_ids'],
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[-1]

                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    vit_out = vit_encoder(img_inputs['pixel_values'].to(DEVICE),
                                          grid_thw=img_inputs['image_grid_thw'].to(DEVICE))
                    vit_emb = (vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out)
                    if vit_emb.dim() == 2:
                        vit_emb = vit_emb.unsqueeze(0)

                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                lang_latent_h = hidden[:, batch['lang_lat_pos'], :]

                vis_loss = torch.tensor(0.0, device=DEVICE)
                if sample['_has_vis']:
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).to(DEVICE)
                    vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)

                lang_loss = lang_decoder(lang_latent_h, default_cot_ids)
                total = (LAMBDA_VIS * vis_loss + LAMBDA_LANG * lang_loss) / GRAD_ACCUM
                total.backward()

                if vis_loss.item() > 0:
                    vis_losses.append(vis_loss.item())
                lang_losses.append(lang_loss.item())

                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(vis_decoder.parameters()) + list(lang_decoder.parameters()), 1.0)
                    opt_s1.step()
                    opt_s1.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt_s1.zero_grad()
                errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5 and epoch == 0:
                    log(f"  S1 err: {type(e).__name__}: {str(e)[:100]}")
                continue

        avg_v = np.mean(vis_losses) if vis_losses else 0
        avg_l = np.mean(lang_losses) if lang_losses else 0
        log(f"  S1 Epoch {epoch+1}/{STAGE_1_EPOCHS}: vis={avg_v:.2f}, lang={avg_l:.4f}, samples={len(vis_losses)}, errors={errors}")
        torch.cuda.empty_cache()

    s1_time = time.time() - t_stage
    log(f"  Stage 1 done: {s1_time/60:.1f} min")
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 's1_vis_decoder.pt'))
    torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 's1_lang_decoder.pt'))

    # ==================================================================
    # STAGE 2: 联合训练（梯度回传）
    # ==================================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 2: 联合训练 ({STAGE_2_EPOCHS} epochs, 梯度回传)")
    log(f"  解码器梯度回传 → 模型学习更好的隐式表征")
    log(f"{'='*60}")

    model.train()
    for name, p in model.named_parameters():
        if 'lora' in name or 'embed_tokens' in name:
            p.requires_grad = True

    all_params = (
        [p for p in model.parameters() if p.requires_grad] +
        list(vis_decoder.parameters()) + list(lang_decoder.parameters())
    )
    opt_s2 = torch.optim.AdamW(all_params, lr=LR * 0.5)
    t_stage = time.time()

    for epoch in range(STAGE_2_EPOCHS):
        traj_losses, vis_losses, lang_losses, errors = [], [], [], 0
        opt_s2.zero_grad()
        for idx, sample in enumerate(samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer)
                outputs = model(
                    input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    labels=batch['labels'], pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],
                    mm_token_type_ids=batch['mm_token_type_ids'],
                    output_hidden_states=True,
                )
                traj_loss = outputs.loss
                if traj_loss is None or torch.isnan(traj_loss):
                    continue

                hidden = outputs.hidden_states[-1]
                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                lang_latent_h = hidden[:, batch['lang_lat_pos'], :]

                # ViT embeddings (detached: don't train ViT backbone)
                with torch.no_grad():
                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    vit_out = vit_encoder(img_inputs['pixel_values'].to(DEVICE),
                                          grid_thw=img_inputs['image_grid_thw'].to(DEVICE))
                    vit_emb = (vit_out.last_hidden_state if hasattr(vit_out, 'last_hidden_state') else vit_out)
                    if vit_emb.dim() == 2:
                        vit_emb = vit_emb.unsqueeze(0)

                vis_loss = torch.tensor(0.0, device=DEVICE)
                if sample['_has_vis']:
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES).unsqueeze(0).to(DEVICE)
                    vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)

                lang_loss = lang_decoder(lang_latent_h, default_cot_ids)
                total = (traj_loss + LAMBDA_VIS * vis_loss + LAMBDA_LANG * lang_loss) / GRAD_ACCUM
                total.backward()

                traj_losses.append(traj_loss.item())
                if vis_loss.item() > 0:
                    vis_losses.append(vis_loss.item())
                lang_losses.append(lang_loss.item())

                if (idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt_s2.step()
                    opt_s2.zero_grad()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt_s2.zero_grad()
                errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5 and epoch == 0:
                    log(f"  S2 err: {type(e).__name__}: {str(e)[:100]}")
                continue

        avg_t = np.mean(traj_losses) if traj_losses else 0
        avg_v = np.mean(vis_losses) if vis_losses else 0
        avg_l = np.mean(lang_losses) if lang_losses else 0
        log(f"  S2 Epoch {epoch+1}/{STAGE_2_EPOCHS}: traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}, "
            f"samples={len(traj_losses)}, errors={errors}")
        torch.cuda.empty_cache()

    s2_time = time.time() - t_stage
    log(f"  Stage 2 done: {s2_time/60:.1f} min")

    # 保存最终模型
    log("\n=== 保存最终模型 ===")
    final_dir = os.path.join(OUTPUT_DIR, 'final_model')
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    torch.save(vis_decoder.state_dict(), os.path.join(final_dir, 'vis_decoder.pt'))
    torch.save(lang_decoder.state_dict(), os.path.join(final_dir, 'lang_decoder.pt'))
    log(f"  已保存到 {final_dir}")

    # 总结
    total_time = pre_time + s0_time + s1_time + s2_time
    log(f"\n{'='*60}")
    log(f"  训练完成")
    log(f"{'='*60}")
    log(f"  预训练: {pre_time/60:.1f}m | S0: {s0_time/60:.1f}m | S1: {s1_time/60:.1f}m | S2: {s2_time/60:.1f}m")
    log(f"  总时长: {total_time/60:.1f} 分钟")
    log(f"  最终: traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}")
    log(f"  GPU 峰值显存: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
