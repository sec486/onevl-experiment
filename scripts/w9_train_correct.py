# -*- coding: utf-8 -*-
#!/opt/onevl-env/bin/python3
"""
W9: Correct OneVL Visual Decoder Implementation
================================================
Fixes ALL 5 issues from Quick Review:
  #1: Dedicated latent tokens (rare vocab IDs, information bottleneck)
  #2: Visual decoder input = ViT embeddings + latent hidden states
  #3: Preliminary stage (decoder pretrained independently)
  #4: No add_special_tokens (use rare vocab IDs)
  #5: Data diversity (35 scenes for Step 1 validation)

Step 1 Goal: Verify vis_loss starts at ~5-10 (not 0.24)
Success = information bottleneck is working correctly.
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# Config
# ============================================================
MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_expanded.jsonl'
VIS_TOKENS_PATH = '/opt/onevl-experiment/data/emu3_visual_tokens_expanded.pt'
OUTPUT_DIR = '/opt/onevl-experiment/output/w9_step2'
LOG_FILE = None  # Set after mkdir

MAX_SAMPLES = 1250
GRAD_ACCUM = 8
LR = 2e-5
CODEBOOK_SIZE = 32768
NUM_VIS_QUERIES = 64   # Reduced from 256 to fit 22GB
NUM_VIS_LATENT = 4
NUM_LANG_LATENT = 2
LAMBDA_LANG = 0.5
LAMBDA_VIS = 0.5

# Rare vocab IDs for latent tokens (OneVL: don't use add_special_tokens!)
# Vocab size = 151669. Use existing special tokens that won't appear in driving data.
VIS_LATENT_ID = 151662   # <|fim_pad|> -?never used in driving context
LANG_LATENT_ID = 151663  # <|repo_name|> -?never used in driving context

PRELIMINARY_EPOCHS = 3
STAGE_0_EPOCHS = 5
STAGE_1_EPOCHS = 3
STAGE_2_EPOCHS = 5

device = 'cuda'

# ============================================================
# Setup
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, 'train_log.txt')

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ============================================================
# Visual Decoder V2 (OneVL Section 3.4: input = ViT + latent)
# ============================================================
class VisualDecoderV2(nn.Module):
    """
    OneVL Eq.3: Z_v = [W_v(V), W_v(H_v)]
    Memory-optimized: projects to 512d internally to fit 22GB GPU.
    ViT output dim = 1024 (Qwen3-VL-2B), hidden_dim = 2048 (LM).
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
        """
        vit_emb: (B, N_vit, d) -?ViT patch embeddings
        latent_h: (B, 4, d) -?latent token hidden states (None for Preliminary)
        gt_ids: (B, num_queries) -?target token IDs
        """
        B = vit_emb.shape[0]
        if latent_h is not None:
            z_v = torch.cat([self.proj_vit(vit_emb), self.proj_lat(latent_h)], dim=1)
        else:
            # Preliminary: only ViT embeddings (no latent)
            z_v = self.proj_vit(vit_emb)

        queries = self.queries.expand(B, -1, -1)
        decoded = self.decoder(queries, z_v)
        logits = self.head(decoded)

        if gt_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), gt_ids.view(-1))
            return loss
        return logits

# ============================================================
# Language Decoder (same as before, 1 layer)
# ============================================================
class LangDecoder(nn.Module):
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
        d = latent_h.shape[-1]
        latent_proj = self.proj_in(latent_h)  # (B, 2, inner_dim)
        inner_dim = latent_proj.shape[-1]
        causal_mask = torch.triu(torch.ones(L, L, device=target_ids.device), diagonal=1).bool()
        tgt = torch.zeros(B, L, inner_dim, device=latent_h.device, dtype=latent_h.dtype)
        decoded = self.decoder(tgt, latent_proj, tgt_mask=causal_mask)
        logits = self.head(decoded)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)

# ============================================================
# Helpers
# ============================================================
def subsample_tokens(full_tokens, target=256):
    flat = full_tokens.flatten()
    indices = torch.linspace(0, len(flat)-1, target).long()
    return flat[indices]

def build_inputs_with_latent(sample, processor, tokenizer):
    """
    Manual tokenization with latent tokens inserted.
    Sequence: [system][user+image][assistant: latent_vis-4 + latent_lang-2 + trajectory]
    """
    img_path = sample.get('images', [None])[0]
    traj_text = ""
    for msg in sample.get('messages', []):
        if msg.get('role') == 'assistant':
            traj_text = msg['content'] if isinstance(msg['content'], str) else str(msg['content'])

    # Process image
    image = Image.open(img_path).convert('RGB').resize((448, 448))
    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
    pixel_values = img_inputs['pixel_values'].to(device)
    image_grid_thw = img_inputs['image_grid_thw'].to(device)
    t, h, w = image_grid_thw[0].tolist()
    # Qwen3-VL spatial merge: actual image tokens in sequence = t*h*w / merge_size^2
    # merge_size=2 for Qwen3-VL -?784 patches become 196 tokens in the LM sequence
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

    # Assistant with LATENT TOKENS
    ast_prefix = [im_start] + ast_role + nl
    input_ids += ast_prefix
    labels += [-100] * len(ast_prefix)

    # Latent tokens (NO traj loss on these!)
    latent_start = len(input_ids)
    latent_ids = [VIS_LATENT_ID] * NUM_VIS_LATENT + [LANG_LATENT_ID] * NUM_LANG_LATENT
    input_ids += latent_ids
    labels += [-100] * len(latent_ids)

    # Trajectory answer (CE loss here)
    input_ids += answer_ids + [im_end]
    labels += answer_ids + [im_end]

    # Positions
    vis_lat_pos = list(range(latent_start, latent_start + NUM_VIS_LATENT))
    lang_lat_pos = list(range(latent_start + NUM_VIS_LATENT, latent_start + NUM_VIS_LATENT + NUM_LANG_LATENT))

    # mm_token_type_ids: 0 for text, 1 for image tokens
    # Image tokens are the img_pad section in the user turn
    mm_token_type_ids = [0] * len(input_ids)
    # Find image pad start: after usr_role + nl + vis_start
    img_start_idx = len(sys_part) + len([im_start]) + len(usr_role) + len(nl) + 1  # +1 for vis_start
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

# ============================================================
# Main
# ============================================================
def main():
    log("=" * 60)
    log("  W9: Correct OneVL Implementation (Step 1 Validation)")
    log("=" * 60)

    # Load visual tokens
    log("\n=== Loading Emu3 visual tokens ===")
    vis_tokens_dict = torch.load(VIS_TOKENS_PATH, weights_only=False)
    log(f"  {len(vis_tokens_dict)} scenes")

    # Load data
    log("\n=== Loading data ===")
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
    ).to(device)
    log(f"  Loaded in {time.time()-t0:.1f}s")

    # Get ViT encoder reference (for extracting patch embeddings)
    # Qwen3VL architecture: model.model.visual (not model.visual)
    # Note: set this BEFORE LoRA wrapping, as path changes after
    vit_encoder = model.model.visual
    hidden_dim = 2048
    vocab_size = len(tokenizer)

    # Probe ViT output dim with a dummy image
    log("\n=== Probing ViT output dim ===")
    _probe_img = Image.new('RGB', (448, 448))
    _probe_inputs = processor.image_processor(images=[_probe_img], return_tensors='pt')
    _probe_pv = _probe_inputs['pixel_values'].to(device)
    _probe_thw = _probe_inputs['image_grid_thw'].to(device)
    with torch.no_grad():
        _probe_out = vit_encoder(_probe_pv, grid_thw=_probe_thw)
        if hasattr(_probe_out, 'last_hidden_state'):
            _vit_shape = _probe_out.last_hidden_state.shape
        else:
            _vit_shape = _probe_out.shape
    vit_dim = _vit_shape[-1]  # Last dim is feature dim
    log(f"  ViT output shape: {_vit_shape}, dim={vit_dim}")
    del _probe_img, _probe_inputs, _probe_pv, _probe_thw, _probe_out
    torch.cuda.empty_cache()

    # Verify rare token IDs exist
    log(f"\n=== Latent token IDs ===")
    log(f"  VIS_LATENT_ID={VIS_LATENT_ID}: '{tokenizer.convert_ids_to_tokens(VIS_LATENT_ID)}'")
    log(f"  LANG_LATENT_ID={LANG_LATENT_ID}: '{tokenizer.convert_ids_to_tokens(LANG_LATENT_ID)}'")

    # Apply LoRA
    log("\n=== LoRA ===")
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  {trainable/1e6:.1f}M trainable")

    # Decoders
    log("\n=== Decoders ===")
    vis_decoder = VisualDecoderV2(
        vit_dim=vit_dim, hidden_dim=hidden_dim, codebook_size=CODEBOOK_SIZE,
        num_queries=NUM_VIS_QUERIES, num_layers=2, num_heads=8
    ).to(device).to(torch.bfloat16)

    lang_decoder = LangDecoder(
        d_model=hidden_dim, n_heads=8, vocab_size=vocab_size, inner_dim=512
    ).to(device).to(torch.bfloat16)

    log(f"  Visual: {sum(p.numel() for p in vis_decoder.parameters())/1e6:.1f}M (input=ViT+latent, output={NUM_VIS_QUERIES} tokens)")
    log(f"  Language: {sum(p.numel() for p in lang_decoder.parameters())/1e6:.1f}M")

    # CoT target
    default_cot = "The vehicle should maintain safe trajectory based on the current driving conditions."
    default_cot_ids = tokenizer(default_cot, return_tensors='pt', max_length=64,
                                truncation=True, padding='max_length')['input_ids'].to(device)

    # ============================================================
    # PRELIMINARY: Visual Decoder Independent Pretraining
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  PRELIMINARY: Visual Decoder Pretrain ({PRELIMINARY_EPOCHS} epochs)")
    log(f"  Input: ViT embeddings ONLY (no latent, no main model forward)")
    log(f"{'='*60}")

    vis_decoder.train()
    opt_pre = torch.optim.AdamW(vis_decoder.parameters(), lr=LR * 3)

    t_stage = time.time()
    for epoch in range(PRELIMINARY_EPOCHS):
        losses = []
        opt_pre.zero_grad()
        errors = 0
        for idx, sample in enumerate(samples):
            if not sample['_has_vis']:
                continue
            try:
                img_path = sample.get('images', [None])[0]
                image = Image.open(img_path).convert('RGB').resize((448, 448))
                img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                pv = img_inputs['pixel_values'].to(device)
                grid_thw = img_inputs['image_grid_thw'].to(device)

                with torch.no_grad():
                    vit_out = vit_encoder(pv, grid_thw=grid_thw)
                    if hasattr(vit_out, 'last_hidden_state'):
                        vit_emb = vit_out.last_hidden_state.unsqueeze(0) if vit_out.last_hidden_state.dim() == 2 else vit_out.last_hidden_state
                    else:
                        vit_emb = vit_out.unsqueeze(0) if vit_out.dim() == 2 else vit_out

                # GT tokens (subsampled)
                vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES)
                vis_gt = vis_gt.unsqueeze(0).to(device)

                # Forward: ViT only, NO latent (Preliminary stage)
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
                    log(f"  PRE err {idx}: {type(e).__name__}: {str(e)[:100]}")
                continue

        avg = np.mean(losses) if losses else 0
        log(f"  PRE Epoch {epoch+1}/{PRELIMINARY_EPOCHS}: vis_loss={avg:.2f}, samples={len(losses)}, errors={errors}")
        torch.cuda.empty_cache()

    pre_time = time.time() - t_stage
    log(f"  Preliminary done: {pre_time/60:.1f} min")
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'preliminary_vis_decoder.pt'))

    # ============================================================
    # STAGE 0: Trajectory Warmup (with latent tokens in sequence)
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 0: Trajectory Warmup ({STAGE_0_EPOCHS} epochs)")
    log(f"  Sequence includes latent tokens, labels=-100 on them")
    log(f"{'='*60}")

    vis_decoder.eval()
    lang_decoder.eval()
    for p in vis_decoder.parameters(): p.requires_grad = False
    for p in lang_decoder.parameters(): p.requires_grad = False

    opt_s0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()

    t_stage = time.time()
    for epoch in range(STAGE_0_EPOCHS):
        losses = []
        opt_s0.zero_grad()
        errors = 0
        for idx, sample in enumerate(samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer)
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
                    opt_s0.step()
                    opt_s0.zero_grad()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt_s0.zero_grad()
                errors += 1
                continue
            except Exception as e:
                errors += 1
                if errors <= 5 and epoch == 0:
                    log(f"  S0 err {idx}: {type(e).__name__}: {str(e)[:120]}")
                continue

        avg = np.mean(losses) if losses else 0
        log(f"  S0 Epoch {epoch+1}/{STAGE_0_EPOCHS}: traj={avg:.4f}, samples={len(losses)}, errors={errors}")
        torch.cuda.empty_cache()

    s0_time = time.time() - t_stage
    log(f"  Stage 0 done: {s0_time/60:.1f} min")

    # ============================================================
    # STAGE 1: Decoder Warmup (DETACHED, ViT + latent input)
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 1: Decoder Warmup ({STAGE_1_EPOCHS} epochs, DETACHED)")
    log(f"  Decoder input = ViT embeddings + latent hidden states")
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
        vis_losses, lang_losses = [], []
        opt_s1.zero_grad()
        errors = 0
        for idx, sample in enumerate(samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer)

                # Forward model (detached) to get hidden states + ViT embeddings
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        pixel_values=batch['pixel_values'],
                        image_grid_thw=batch['image_grid_thw'],
                    mm_token_type_ids=batch['mm_token_type_ids'],
                        output_hidden_states=True,
                    )
                    hidden = outputs.hidden_states[-1]  # (1, seq, 2048)

                    # Extract ViT embeddings independently
                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    pv = img_inputs['pixel_values'].to(device)
                    grid_thw = img_inputs['image_grid_thw'].to(device)
                    vit_out = vit_encoder(pv, grid_thw=grid_thw)
                    if hasattr(vit_out, 'last_hidden_state'):
                        vit_emb = vit_out.last_hidden_state.unsqueeze(0) if vit_out.last_hidden_state.dim() == 2 else vit_out.last_hidden_state
                    else:
                        vit_emb = vit_out.unsqueeze(0) if vit_out.dim() == 2 else vit_out

                # Extract latent hidden states from correct positions
                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]  # (1, 4, 2048)
                lang_latent_h = hidden[:, batch['lang_lat_pos'], :]  # (1, 2, 2048)

                # Visual decoder: ViT + latent (detached)
                vis_loss = torch.tensor(0.0, device=device)
                if sample['_has_vis']:
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES)
                    vis_gt = vis_gt.unsqueeze(0).to(device)
                    vis_loss = vis_decoder(vit_emb, vis_latent_h, vis_gt)

                # Language decoder
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
                continue
            except Exception as e:
                errors += 1
                if errors <= 5 and epoch == 0:
                    log(f"  S1 err {idx}: {type(e).__name__}: {str(e)[:120]}")
                continue

        avg_v = np.mean(vis_losses) if vis_losses else 0
        avg_l = np.mean(lang_losses) if lang_losses else 0
        log(f"  S1 Epoch {epoch+1}/{STAGE_1_EPOCHS}: vis={avg_v:.2f}, lang={avg_l:.4f}, "
            f"vis_samples={len(vis_losses)}, errors={errors}")
        torch.cuda.empty_cache()

        # KEY CHECK: Is vis_loss ~5-10 (success) or ~0 (memorization)?
        if epoch == 0:
            log(f"  >>> KEY METRIC: vis_loss initial = {avg_v:.2f} (expect 5-10 for success, <1 = memorization)")

    s1_time = time.time() - t_stage
    log(f"  Stage 1 done: {s1_time/60:.1f} min")
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 's1_vis_decoder.pt'))
    torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 's1_lang_decoder.pt'))

    # ============================================================
    # STAGE 2: Joint (NO DETACH)
    # ============================================================
    log(f"\n{'='*60}")
    log(f"  STAGE 2: Joint Training ({STAGE_2_EPOCHS} epochs, NO DETACH)")
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
        traj_losses, vis_losses, lang_losses = [], [], []
        opt_s2.zero_grad()
        errors = 0
        for idx, sample in enumerate(samples):
            try:
                batch = build_inputs_with_latent(sample, processor, tokenizer)

                # Forward WITH gradient (no detach!)
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'],
                    mm_token_type_ids=batch['mm_token_type_ids'],
                    output_hidden_states=True,
                )
                traj_loss = outputs.loss
                if traj_loss is None or torch.isnan(traj_loss):
                    continue

                hidden = outputs.hidden_states[-1]

                # Latent positions (NOT detached -?gradients flow back!)
                vis_latent_h = hidden[:, batch['vis_lat_pos'], :]
                lang_latent_h = hidden[:, batch['lang_lat_pos'], :]

                # ViT embeddings (detached -?don't want to train ViT in Stage 2)
                with torch.no_grad():
                    img_path = sample.get('images', [None])[0]
                    image = Image.open(img_path).convert('RGB').resize((448, 448))
                    img_inputs = processor.image_processor(images=[image], return_tensors='pt')
                    pv = img_inputs['pixel_values'].to(device)
                    grid_thw = img_inputs['image_grid_thw'].to(device)
                    vit_out = vit_encoder(pv, grid_thw=grid_thw)
                    if hasattr(vit_out, 'last_hidden_state'):
                        vit_emb = vit_out.last_hidden_state.unsqueeze(0) if vit_out.last_hidden_state.dim() == 2 else vit_out.last_hidden_state
                    else:
                        vit_emb = vit_out.unsqueeze(0) if vit_out.dim() == 2 else vit_out

                # Decoders
                vis_loss = torch.tensor(0.0, device=device)
                if sample['_has_vis']:
                    vis_gt = subsample_tokens(vis_tokens_dict[sample['_scene']], NUM_VIS_QUERIES)
                    vis_gt = vis_gt.unsqueeze(0).to(device)
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
                continue
            except Exception as e:
                errors += 1
                if errors <= 5 and epoch == 0:
                    log(f"  S2 err {idx}: {type(e).__name__}: {str(e)[:120]}")
                continue

        avg_t = np.mean(traj_losses) if traj_losses else 0
        avg_v = np.mean(vis_losses) if vis_losses else 0
        avg_l = np.mean(lang_losses) if lang_losses else 0
        log(f"  S2 Epoch {epoch+1}/{STAGE_2_EPOCHS}: traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}, "
            f"samples={len(traj_losses)}, errors={errors}")
        torch.cuda.empty_cache()

        with open(os.path.join(OUTPUT_DIR, 'partial_results.json'), 'w') as f:
            json.dump({'stage': 2, 'epoch': epoch+1, 'traj': avg_t, 'vis': avg_v, 'lang': avg_l,
                      'samples': len(traj_losses)}, f, indent=2)

    s2_time = time.time() - t_stage
    log(f"  Stage 2 done: {s2_time/60:.1f} min")

    # Save final
    model.save_pretrained(os.path.join(OUTPUT_DIR, 'final'))
    torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'final_vis_decoder.pt'))
    torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'final_lang_decoder.pt'))

    # ============================================================
    # Summary
    # ============================================================
    total_time = pre_time + s0_time + s1_time + s2_time
    log(f"\n{'='*60}")

    # === SAVE ALL MODEL ARTIFACTS (BF-037 final run) ===
    log("\n=== Saving all model artifacts ===")
    SAVE_DIR = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    torch.save(vis_decoder.state_dict(), os.path.join(SAVE_DIR, "vis_decoder.pt"))
    torch.save(lang_decoder.state_dict(), os.path.join(SAVE_DIR, "lang_decoder.pt"))
    import json as _json
    _cfg = {"max_samples": MAX_SAMPLES, "grad_accum": GRAD_ACCUM, "lr": LR,
            "codebook_size": CODEBOOK_SIZE, "num_vis_queries": NUM_VIS_QUERIES,
            "lambda_vis": LAMBDA_VIS, "lambda_lang": LAMBDA_LANG,
            "stages": "Preliminary+S0+S1+S2", "vis_tokens": VIS_TOKENS_PATH}
    with open(os.path.join(SAVE_DIR, "training_config.json"), "w") as _f:
        _json.dump(_cfg, _f, indent=2)
    log(f"  Saved to {SAVE_DIR}")
    log(f"  Contents: {os.listdir(SAVE_DIR)}")

    log(f"  W9 COMPLETE")
    log(f"{'='*60}")
    log(f"  Pre: {pre_time/60:.1f}m | S0: {s0_time/60:.1f}m | S1: {s1_time/60:.1f}m | S2: {s2_time/60:.1f}m")
    log(f"  Total: {total_time/60:.1f} min")
    log(f"  Final: traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}")
    log(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    with open(os.path.join(OUTPUT_DIR, 'TRAINING_COMPLETE.txt'), 'w') as f:
        f.write(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {total_time/60:.1f} min\n")
        f.write(f"traj={avg_t:.4f}, vis={avg_v:.2f}, lang={avg_l:.4f}\n")

if __name__ == "__main__":
    main()
