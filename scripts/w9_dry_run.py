#!/opt/onevl-env/bin/python3
"""
W9 Dry Run �?Validate ALL components before full training.
Tests: model load, ViT dims, latent tokens, manual tokenization, 
       decoder forward, image token count, one full training step.
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
VIS_TOKENS_PATH = '/opt/onevl-experiment/data/emu3_visual_tokens_1000.pt'

VIS_LATENT_ID = 151662  # <|fim_pad|>
LANG_LATENT_ID = 151663  # <|repo_name|>
NUM_VIS_LATENT = 4
NUM_LANG_LATENT = 2
CODEBOOK_SIZE = 32768
NUM_VIS_QUERIES = 64

device = 'cuda'
errors = []

def check(name, condition, detail=""):
    if condition:
        print(f"  �?{name}")
    else:
        print(f"  �?{name}: {detail}")
        errors.append(f"{name}: {detail}")

print("=" * 60)
print("  W9 DRY RUN �?Validating all components")
print("=" * 60)

# === 1. Load model ===
print("\n[1/8] Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    trust_remote_code=True, attn_implementation="eager"
).to(device)

check("Model loaded", True)
check("Hidden dim = 2048", model.config.text_config.hidden_size == 2048,
      f"got {model.config.text_config.hidden_size}")

# === 2. ViT encoder path ===
print("\n[2/8] ViT encoder...")
try:
    vit_encoder = model.model.visual
    check("model.model.visual exists", True)
except AttributeError as e:
    check("model.model.visual exists", False, str(e))
    # Try alternatives
    for path in ['model.visual', 'model.model.model.visual']:
        try:
            vit_encoder = eval(path)
            check(f"  Found at {path}", True)
            break
        except:
            pass

# === 3. ViT output dim probe ===
print("\n[3/8] ViT output dimensions...")
probe_img = Image.new('RGB', (448, 448), color=(128, 128, 128))
img_inputs = processor.image_processor(images=[probe_img], return_tensors='pt')
pv = img_inputs['pixel_values'].to(device)
grid_thw = img_inputs['image_grid_thw'].to(device)
t, h, w = grid_thw[0].tolist()

with torch.no_grad():
    vit_out = vit_encoder(pv, grid_thw=grid_thw)

if hasattr(vit_out, 'last_hidden_state'):
    vit_emb = vit_out.last_hidden_state
else:
    vit_emb = vit_out
if vit_emb.dim() == 2:
    vit_emb = vit_emb.unsqueeze(0)

vit_dim = vit_emb.shape[-1]
vit_num_patches = vit_emb.shape[1]
print(f"  grid_thw: t={int(t)}, h={int(h)}, w={int(w)} �?t*h*w = {int(t*h*w)}")
print(f"  ViT output: shape={vit_emb.shape}, dim={vit_dim}, patches={vit_num_patches}")
print(f"  t*h*w // 4 = {int(t*h*w)//4}")

check("ViT patches == t*h*w (raw, before merge)", vit_num_patches == int(t*h*w),
      f"patches={vit_num_patches}, t*h*w={int(t*h*w)}")

# The number of image_pad tokens in manual tokenization should be t*h*w//4 (after merge)
num_img_tokens = int(t * h * w // 4)
check("num_img_tokens = t*h*w//4 (merged)", num_img_tokens == 196,
      f"tokens={num_img_tokens}")

# === 4. Latent token IDs ===
print("\n[4/8] Latent token IDs...")
vis_tok = tokenizer.convert_ids_to_tokens(VIS_LATENT_ID)
lang_tok = tokenizer.convert_ids_to_tokens(LANG_LATENT_ID)
print(f"  VIS_LATENT_ID={VIS_LATENT_ID} �?'{vis_tok}'")
print(f"  LANG_LATENT_ID={LANG_LATENT_ID} �?'{lang_tok}'")
check("VIS token exists in vocab", vis_tok is not None and vis_tok != tokenizer.unk_token,
      f"got '{vis_tok}'")
check("LANG token exists in vocab", lang_tok is not None and lang_tok != tokenizer.unk_token,
      f"got '{lang_tok}'")

# === 5. Manual tokenization test ===
print("\n[5/8] Manual tokenization...")
im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
vis_start = tokenizer.convert_tokens_to_ids('<|vision_start|>')
vis_end = tokenizer.convert_tokens_to_ids('<|vision_end|>')
img_pad = tokenizer.convert_tokens_to_ids('<|image_pad|>')
nl = tokenizer.encode('\n', add_special_tokens=False)

check("im_start token", im_start != tokenizer.unk_token_id, f"id={im_start}")
check("img_pad token", img_pad != tokenizer.unk_token_id, f"id={img_pad}")
check("vision_start token", vis_start != tokenizer.unk_token_id, f"id={vis_start}")

# Build a test sequence
sys_role = tokenizer.encode('system', add_special_tokens=False)
sys_text = tokenizer.encode('You are a helpful assistant.', add_special_tokens=False)
usr_role = tokenizer.encode('user', add_special_tokens=False)
usr_text = tokenizer.encode('Predict trajectory.', add_special_tokens=False)
ast_role = tokenizer.encode('assistant', add_special_tokens=False)
answer_ids = tokenizer.encode('[1.0, 2.0], [3.0, 4.0]', add_special_tokens=False)

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

latent_start = len(input_ids)
latent_ids = [VIS_LATENT_ID] * NUM_VIS_LATENT + [LANG_LATENT_ID] * NUM_LANG_LATENT
input_ids += latent_ids
labels += [-100] * len(latent_ids)

input_ids += answer_ids + [im_end]
labels += answer_ids + [im_end]

vis_lat_pos = list(range(latent_start, latent_start + NUM_VIS_LATENT))
lang_lat_pos = list(range(latent_start + NUM_VIS_LATENT, latent_start + NUM_VIS_LATENT + NUM_LANG_LATENT))

print(f"  Sequence length: {len(input_ids)}")
print(f"  Image tokens: {num_img_tokens}")
print(f"  Latent positions: vis={vis_lat_pos}, lang={lang_lat_pos}")
check("Sequence length reasonable", 100 < len(input_ids) < 1000, f"len={len(input_ids)}")

# Construct mm_token_type_ids (0=text, 1=image)
mm_token_type_ids = [0] * len(input_ids)
img_start_in_seq = len(sys_part) + 1 + len(usr_role) + len(nl) + 1  # im_start + usr_role + nl + vis_start
for i in range(img_start_in_seq, img_start_in_seq + num_img_tokens):
    if i < len(mm_token_type_ids):
        mm_token_type_ids[i] = 1

# === 6. Apply LoRA + full model forward ===
print("\n[6/8] LoRA + model forward...")
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["embed_tokens"],
    task_type=TaskType.CAUSAL_LM, bias="none",
)
model = get_peft_model(model, lora_config)
model.train()

input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
labels_t = torch.tensor([labels], dtype=torch.long, device=device)
attn_mask = torch.ones_like(input_t)
mm_t = torch.tensor([mm_token_type_ids], dtype=torch.long, device=device)

try:
    outputs = model(
        input_ids=input_t,
        attention_mask=attn_mask,
        labels=labels_t,
        pixel_values=pv,
        image_grid_thw=grid_thw,
        mm_token_type_ids=mm_t,
        output_hidden_states=True,
    )
    traj_loss = outputs.loss
    hidden = outputs.hidden_states[-1]
    print(f"  Forward pass OK! loss={traj_loss.item():.4f}")
    print(f"  Hidden states shape: {hidden.shape}")
    
    vis_latent_h = hidden[:, vis_lat_pos, :]
    lang_latent_h = hidden[:, lang_lat_pos, :]
    print(f"  Vis latent: {vis_latent_h.shape}")
    print(f"  Lang latent: {lang_latent_h.shape}")
    
    check("Forward pass succeeds", True)
    check("Traj loss is valid", not torch.isnan(traj_loss), f"loss={traj_loss.item()}")
    check("Hidden dim correct", hidden.shape[-1] == 2048, f"got {hidden.shape[-1]}")
    check("Vis latent shape", vis_latent_h.shape == (1, 4, 2048), f"got {vis_latent_h.shape}")
    check("Lang latent shape", lang_latent_h.shape == (1, 2, 2048), f"got {lang_latent_h.shape}")
except Exception as e:
    check("Forward pass succeeds", False, f"{type(e).__name__}: {e}")

# === 7. Decoder forward ===
print("\n[7/8] Decoder forward test...")

class VisualDecoderV2(nn.Module):
    def __init__(self, vit_dim, hidden_dim=2048, codebook_size=32768,
                 num_queries=64, num_layers=2, num_heads=8, inner_dim=512):
        super().__init__()
        self.proj_vit = nn.Linear(vit_dim, inner_dim)
        self.proj_lat = nn.Linear(hidden_dim, inner_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, inner_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=inner_dim, nhead=num_heads,
            dim_feedforward=inner_dim * 2, batch_first=True, dropout=0.1)
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

vis_decoder = VisualDecoderV2(
    vit_dim=vit_dim, hidden_dim=2048, codebook_size=CODEBOOK_SIZE,
    num_queries=NUM_VIS_QUERIES
).to(device).to(torch.bfloat16)

# Load visual tokens
vis_tokens_dict = torch.load(VIS_TOKENS_PATH, weights_only=False)
sample_scene = list(vis_tokens_dict.keys())[0]
full_tokens = vis_tokens_dict[sample_scene]  # (96, 170)
flat = full_tokens.flatten()
indices = torch.linspace(0, len(flat)-1, NUM_VIS_QUERIES).long()
gt_ids = flat[indices].unsqueeze(0).to(device)

try:
    # Preliminary mode (ViT only)
    vis_loss_pre = vis_decoder(vit_emb, latent_h=None, gt_ids=gt_ids)
    print(f"  Preliminary forward: vis_loss={vis_loss_pre.item():.2f}")
    check("Preliminary vis_loss near log(32768)�?0.4", 
          5 < vis_loss_pre.item() < 12,
          f"got {vis_loss_pre.item():.2f}")
    
    # Stage 1 mode (ViT + latent)
    vis_loss_s1 = vis_decoder(vit_emb, vis_latent_h.detach(), gt_ids)
    print(f"  Stage 1 forward: vis_loss={vis_loss_s1.item():.2f}")
    check("Stage 1 vis_loss near log(32768)", 
          5 < vis_loss_s1.item() < 12,
          f"got {vis_loss_s1.item():.2f}")
    
except Exception as e:
    check("Decoder forward", False, f"{type(e).__name__}: {e}")

# === 8. Backward pass ===
print("\n[8/8] Backward pass test...")
try:
    total_loss = traj_loss + 0.5 * vis_loss_s1
    total_loss.backward()
    check("Backward pass succeeds", True)
    
    # Check gradients exist on latent token embeddings
    embed_layer = None
    for name, p in model.named_parameters():
        if 'embed_tokens' in name and p.requires_grad:
            embed_layer = p
            break
    if embed_layer is not None and embed_layer.grad is not None:
        lat_grad = embed_layer.grad[VIS_LATENT_ID].abs().sum().item()
        check("Gradient flows to latent embeddings", lat_grad > 0, f"grad_sum={lat_grad}")
    else:
        check("Gradient flows to latent embeddings", False, "no grad on embed_tokens")
        
except Exception as e:
    check("Backward pass", False, f"{type(e).__name__}: {e}")

# === Summary ===
print(f"\n{'='*60}")
mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  GPU peak: {mem:.1f} GB")
if errors:
    print(f"\n  �?FAILED �?{len(errors)} errors:")
    for e in errors:
        print(f"    - {e}")
    print(f"\n  DO NOT proceed to full training until all checks pass!")
else:
    print(f"\n  �?ALL CHECKS PASSED �?Safe to run full training")
    print(f"  Key metrics:")
    print(f"    ViT dim: {vit_dim}, patches: {vit_num_patches}")
    print(f"    Image tokens in sequence: {num_img_tokens}")
    print(f"    Preliminary vis_loss: {vis_loss_pre.item():.2f} (expect ~10.4)")
    print(f"    Stage 1 vis_loss: {vis_loss_s1.item():.2f} (expect ~10.4)")
    print(f"    Traj loss: {traj_loss.item():.4f}")
    print(f"    GPU peak: {mem:.1f} GB / 22 GB")
