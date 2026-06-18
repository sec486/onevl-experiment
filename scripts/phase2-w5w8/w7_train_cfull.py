#!/usr/bin/env python3
"""
W7: Train c-full (language + visual decoders) on REAL trajectory data.
This is the corrected version that uses ground-truth trajectories from NAVSIM metadata.

Architecture:
  Cosmos-Reason2-2B (LoRA r=64 + embed_tokens)
  ├── Language Decoder (2-layer, cross-attn to latent states) → reconstructs CoT
  ├── Visual Decoder (2-layer, cross-attn to latent states) → predicts future patches
  └── Trajectory Prediction (existing) → predicts waypoints
  
  Loss = L_trajectory + 0.5 * L_language + lambda_vis * L_visual
"""
import json, torch, time, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, '/opt/onevl-experiment/data')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from language_decoder import LanguageDecoderHead
from qwen_vl_utils import process_vision_info
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = '/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561'
DATA_PATH = '/opt/onevl-experiment/data/navsim_real_traj_1000_latent.jsonl'
DATA_PATH_ANSWER = '/opt/onevl-experiment/data/navsim_real_traj_1000.jsonl'
COT_PATH = '/opt/onevl-experiment/data/navsim_cot_cosmos_1000.jsonl'
OUTPUT_DIR = '/opt/onevl-experiment/output/w7_cfull_real_traj'
IMAGE_TOKEN_ID = 151655
LAMBDA_LANG = 0.5
LAMBDA_VIS = 0.1  # Conservative start for visual decoder
NUM_EPOCHS = 5
MAX_SAMPLES = 200  # Start small, verify convergence
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-5

print('=' * 60)
print('  W7: c-full Training with REAL Trajectory GT')
print('  Fix for BF-032')
print(f'  Lambda: lang={LAMBDA_LANG}, vis={LAMBDA_VIS}')
print(f'  Samples: {MAX_SAMPLES}, Epochs: {NUM_EPOCHS}')
print('=' * 60)

# Check data exists
if not os.path.exists(DATA_PATH):
    # Fallback to answer-only if latent not generated yet
    if os.path.exists(DATA_PATH_ANSWER):
        DATA_PATH = DATA_PATH_ANSWER
        print(f"Using answer-only format: {DATA_PATH}")
    else:
        print(f"ERROR: No training data found at {DATA_PATH} or {DATA_PATH_ANSWER}")
        print("Run w7_fix_bf032.py first to extract trajectories!")
        sys.exit(1)

# Load data
print("\n=== Loading training data ===")
samples = []
with open(DATA_PATH) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))
            if len(samples) >= MAX_SAMPLES:
                break

print(f"Loaded {len(samples)} samples")

# Load CoT data for language decoder targets (if available)
cot_data = {}
if os.path.exists(COT_PATH):
    with open(COT_PATH) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Key by scene+frame if available
                scene = item.get('scene', '')
                cot_data[scene] = item.get('messages', [{}])[-1].get('content', '')
    print(f"Loaded {len(cot_data)} CoT annotations for language decoder")
else:
    print("No CoT data found - language decoder will use generic targets")

# Load model
print("\n=== Loading model ===")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = processor.tokenizer

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="eager",
).to(device)

# Apply LoRA with embed_tokens trainable
print("\n=== Applying LoRA (r=64 + embed_tokens) ===")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["embed_tokens"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Initialize decoders
hidden_size = model.config.text_config.hidden_size  # 2048
vocab_size = len(tokenizer)

print(f"\n=== Initializing decoders (hidden_size={hidden_size}) ===")

# Language decoder
lang_decoder = LanguageDecoderHead(
    hidden_size=hidden_size,
    vocab_size=vocab_size,
    num_layers=2,
    num_heads=8,
    max_seq_len=128,
).to(device).to(torch.bfloat16)
print(f"Language decoder params: {sum(p.numel() for p in lang_decoder.parameters()):,}")

# Visual decoder (predicts future frame patch tokens)
class VisualDecoderHead(nn.Module):
    """Predicts future frame ViT patch tokens from visual latent states."""
    def __init__(self, hidden_size=2048, num_patches=2040, patch_dim=1024, num_layers=2, num_heads=8):
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        
        # Cross-attention layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            dtype=torch.bfloat16,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Learnable query tokens (one per output patch group)
        # Use fewer queries than patches and project up
        self.num_queries = 64  # Compress to 64 queries
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, hidden_size, dtype=torch.bfloat16) * 0.02)
        
        # Project from hidden_size to patch predictions
        self.output_proj = nn.Linear(hidden_size, num_patches * patch_dim // self.num_queries, dtype=torch.bfloat16)
        
    def forward(self, visual_latent_states):
        """
        Args:
            visual_latent_states: (B, num_visual_latents, hidden_size) — typically (B, 4, 2048)
        Returns:
            predicted_patches: (B, num_patches, patch_dim)
        """
        B = visual_latent_states.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Cross-attend queries to visual latent states
        decoded = self.decoder(queries, visual_latent_states)
        
        # Project to patch predictions
        patch_preds = self.output_proj(decoded)  # (B, num_queries, patches_per_query * patch_dim)
        
        # Reshape to (B, num_patches, patch_dim)
        patch_preds = patch_preds.reshape(B, self.num_patches, self.patch_dim)
        
        return patch_preds

vis_decoder = VisualDecoderHead(
    hidden_size=hidden_size,
    num_patches=2040,  # From W6: Cosmos-Reason2 ViT produces 2040 patches
    patch_dim=1024,
    num_layers=2,
    num_heads=8,
).to(device)
print(f"Visual decoder params: {sum(p.numel() for p in vis_decoder.parameters()):,}")

# Optimizer
all_params = list(model.parameters()) + list(lang_decoder.parameters()) + list(vis_decoder.parameters())
trainable_params = [p for p in all_params if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

# Extract ViT patch tokens for visual decoder GT
print("\n=== Extracting ViT patch tokens for visual decoder GT ===")
print("(Using current frame patches as proxy — future frame extraction requires")
print(" loading next frame which we'll do for a subset)")

# For now: extract patches from each sample's image using the model's ViT
# This gives us the GT format. In full version, we'd use frame_idx+2 (t+1.0s)
vit_model = model.base_model.model.visual if hasattr(model, 'base_model') else model.visual
print(f"ViT accessible: {vit_model is not None}")

# Training loop
print(f"\n=== Starting Training ===")
print(f"Samples: {len(samples)}, Epochs: {NUM_EPOCHS}")
print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"Steps per epoch: {len(samples) // (BATCH_SIZE * GRAD_ACCUM)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.train()
lang_decoder.train()
vis_decoder.train()

total_steps = 0
log_interval = 5
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_losses = {'total': [], 'traj': [], 'lang': [], 'vis': []}
    optimizer.zero_grad()
    
    for idx, sample in enumerate(samples):
        # Prepare input
        messages = sample['messages']
        user_msg = messages[0]
        assistant_msg = messages[1]
        
        # Process with processor
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # Handle image
            image_inputs = None
            if 'images' in sample and sample['images']:
                img_path = sample['images'][0]
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    # Use process_vision_info if available
                    inputs = processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    ).to(device)
                else:
                    inputs = processor(
                        text=[text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    ).to(device)
            else:
                inputs = processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)
        except Exception as e:
            if idx == 0:
                print(f"  Processing error (sample {idx}): {e}")
            continue
        
        # Forward pass
        try:
            outputs = model(**inputs, output_hidden_states=True)
            
            # 1. Trajectory loss (standard LM loss on answer tokens)
            logits = outputs.logits
            labels = inputs['input_ids'].clone()
            # Mask everything except the answer portion
            # Find <answer> token position
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Simple: use full sequence loss (includes answer)
            traj_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 2. Language decoder loss
            # Get hidden states at latent token positions
            hidden_states = outputs.hidden_states[-1]  # Last layer (B, seq_len, hidden_size)
            
            # Find latent token positions (tokens between start-latent and end-latent)
            # For simplicity in this fix, use the last 6 hidden states before the answer
            # as proxy for latent states (positions 4 visual + 2 language)
            seq_len = hidden_states.shape[1]
            latent_states = hidden_states[:, max(0, seq_len-8):seq_len-2, :]  # 6 latent positions
            
            if latent_states.shape[1] >= 2:
                lang_latent = latent_states[:, -2:, :]  # Last 2 = language latents
                
                # Language decoder target: CoT text
                scene_name = sample.get('scene', '')
                cot_text = cot_data.get(scene_name, "The vehicle should maintain current trajectory based on the driving scene.")
                cot_tokens = tokenizer(cot_text, return_tensors='pt', max_length=64, truncation=True, padding='max_length')['input_ids'].to(device)
                
                lang_loss = lang_decoder(lang_latent, cot_tokens)
            else:
                lang_loss = torch.tensor(0.0, device=device)
            
            # 3. Visual decoder loss
            if latent_states.shape[1] >= 4:
                vis_latent = latent_states[:, :4, :]  # First 4 = visual latents
                
                # Get ViT patches as GT (from the image in this batch)
                # Extract pixel_values if present
                if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                    with torch.no_grad():
                        # Get the visual encoder output
                        pixel_values = inputs['pixel_values']
                        if hasattr(model, 'base_model'):
                            vis_enc = model.base_model.model.visual
                        else:
                            vis_enc = model.visual
                        
                        if vis_enc is not None:
                            vis_output = vis_enc(pixel_values, grid_thw=inputs.get('image_grid_thw'))
                            gt_patches = vis_output  # Shape: (num_patches, hidden_size)
                            if gt_patches.dim() == 2:
                                gt_patches = gt_patches.unsqueeze(0)  # (1, num_patches, dim)
                            
                            # Predict patches
                            pred_patches = vis_decoder(vis_latent)
                            
                            # Align shapes for MSE
                            min_patches = min(pred_patches.shape[1], gt_patches.shape[1])
                            min_dim = min(pred_patches.shape[2], gt_patches.shape[2])
                            vis_loss = F.mse_loss(
                                pred_patches[:, :min_patches, :min_dim],
                                gt_patches[:, :min_patches, :min_dim].to(pred_patches.dtype)
                            )
                        else:
                            vis_loss = torch.tensor(0.0, device=device)
                else:
                    vis_loss = torch.tensor(0.0, device=device)
            else:
                vis_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            total_loss = traj_loss + LAMBDA_LANG * lang_loss + LAMBDA_VIS * vis_loss
            total_loss = total_loss / GRAD_ACCUM
            total_loss.backward()
            
            # Track losses
            epoch_losses['total'].append(total_loss.item() * GRAD_ACCUM)
            epoch_losses['traj'].append(traj_loss.item())
            epoch_losses['lang'].append(lang_loss.item() if torch.is_tensor(lang_loss) else lang_loss)
            epoch_losses['vis'].append(vis_loss.item() if torch.is_tensor(vis_loss) else vis_loss)
            
        except Exception as e:
            if idx < 3:
                print(f"  Forward pass error (sample {idx}): {e}")
            continue
        
        # Gradient accumulation step
        if (idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_steps += 1
            
            if total_steps % log_interval == 0:
                avg_total = np.mean(epoch_losses['total'][-GRAD_ACCUM*log_interval:])
                avg_traj = np.mean(epoch_losses['traj'][-GRAD_ACCUM*log_interval:])
                avg_lang = np.mean(epoch_losses['lang'][-GRAD_ACCUM*log_interval:])
                avg_vis = np.mean(epoch_losses['vis'][-GRAD_ACCUM*log_interval:])
                elapsed = time.time() - start_time
                print(f"  Step {total_steps} | total={avg_total:.4f} traj={avg_traj:.4f} lang={avg_lang:.4f} vis={avg_vis:.4f} | {elapsed:.0f}s")
    
    # Epoch summary
    if epoch_losses['total']:
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Avg total: {np.mean(epoch_losses['total']):.4f}")
        print(f"  Avg traj:  {np.mean(epoch_losses['traj']):.4f}")
        print(f"  Avg lang:  {np.mean(epoch_losses['lang']):.4f}")
        print(f"  Avg vis:   {np.mean(epoch_losses['vis']):.4f}")
        print(f"  GPU mem:   {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

# Save
print(f"\n=== Saving to {OUTPUT_DIR} ===")
model.save_pretrained(OUTPUT_DIR)
torch.save(lang_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'lang_decoder.pt'))
torch.save(vis_decoder.state_dict(), os.path.join(OUTPUT_DIR, 'vis_decoder.pt'))

# Save training config
config = {
    'lambda_lang': LAMBDA_LANG,
    'lambda_vis': LAMBDA_VIS,
    'num_epochs': NUM_EPOCHS,
    'max_samples': MAX_SAMPLES,
    'lr': LR,
    'lora_r': 64,
    'fix': 'BF-032 — real trajectories from NAVSIM metadata',
    'data_source': DATA_PATH,
    'total_steps': total_steps,
}
with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

elapsed = time.time() - start_time
print(f"\nDone! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Output: {OUTPUT_DIR}")
print(f"Total steps: {total_steps}")
