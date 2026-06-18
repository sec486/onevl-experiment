"""
Training script for Phase 1B: Latent CoT with Auxiliary Language Decoder.

This implements the core OneVL training recipe (simplified):
- Base model: Cosmos-Reason2-2B with LoRA (r=64) + trainable embed_tokens
- Auxiliary decoder: 2-layer transformer that reconstructs CoT from latent hidden states
- Combined loss: L_total = L_trajectory + λ * L_language_decoder

Usage:
    python3 train_with_decoder.py [--lambda_decoder 0.5] [--epochs 3] [--r 64]
"""
import argparse
import json
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from language_decoder import LanguageDecoderHead, find_latent_token_positions


MODEL_PATH = "/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
DATASET_PATH = "/opt/onevl-experiment/OneVL_training/demo_data/navsim/navsim_vis4_text2_demo100.jsonl"
OUTPUT_DIR = "/opt/onevl-experiment/output/latent_cot_with_decoder"


class LatentCoTDataset(Dataset):
    """Dataset that provides both trajectory targets AND CoT text targets."""
    
    def __init__(self, jsonl_path, tokenizer, max_seq_len=2048, max_cot_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_cot_len = max_cot_len
        self.samples = []
        
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample.get("messages", [])
        think_steps = sample.get("think_steps", "")
        
        # Tokenize the full conversation (including latent tokens)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Labels for trajectory loss (same as input_ids, masked for padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # Tokenize CoT text separately (target for language decoder)
        cot_encoded = self.tokenizer(
            think_steps,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_cot_len,
            padding="max_length",
        )
        cot_ids = cot_encoded["input_ids"].squeeze(0)
        cot_mask = cot_encoded["attention_mask"].squeeze(0)
        
        # Mark padding in CoT as -100 for loss computation
        cot_labels = cot_ids.clone()
        cot_labels[cot_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "cot_ids": cot_ids,
            "cot_mask": cot_mask,
            "cot_labels": cot_labels,
        }


def train(args):
    print("=" * 60)
    print("  Phase 1B: Latent CoT with Auxiliary Language Decoder")
    print("=" * 60)
    print(f"  λ (decoder weight): {args.lambda_decoder}")
    print(f"  LoRA rank: {args.r}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {args.output_dir}")
    print()
    
    # 1. Load base model
    print("Loading base model...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    
    # 2. Apply LoRA with embed_tokens trainable
    print("Applying LoRA (r={}, embed_tokens trainable)...".format(args.r))
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.r * 4,  # Standard ratio
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 3. Create language decoder head
    print("Creating auxiliary language decoder...")
    decoder = LanguageDecoderHead(
        d_model=2048,  # Match Cosmos-Reason2-2B hidden_size
        n_heads=8,
        n_layers=2,
        vocab_size=tokenizer.vocab_size,
        max_cot_length=128,
        dropout=0.1,
    ).to(dtype=torch.bfloat16, device="cuda")
    
    decoder_params = decoder.get_parameter_count()
    print(f"  Decoder params: {decoder_params['total']:,}")
    print(f"  Decoder memory: ~{decoder_params['total'] * 2 / 1e9:.2f} GB (BF16)")
    
    # 4. Load dataset
    print("Loading dataset...")
    dataset = LatentCoTDataset(
        DATASET_PATH, tokenizer, 
        max_seq_len=args.max_seq_len, 
        max_cot_len=128
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0
    )
    
    # 5. Setup optimizer (both model LoRA params + decoder params)
    print("Setting up optimizer...")
    optimizer_params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": decoder.parameters(), "lr": args.lr * 2},  # Slightly higher LR for decoder
    ]
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)
    
    # 6. Training loop
    print(f"\nStarting training ({args.epochs} epochs, {len(dataset)} samples)...")
    print(f"  Effective batch size: {args.grad_accum} (grad accumulation)")
    print()
    
    model.train()
    decoder.train()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    global_step = 0
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_traj_loss = 0
        epoch_decoder_loss = 0
        epoch_total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")
            cot_ids = batch["cot_ids"].to("cuda")
            cot_mask = batch["cot_mask"].to("cuda")
            
            # Forward pass through base model (get trajectory loss + hidden states)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,  # Need hidden states for decoder
            )
            
            traj_loss = outputs.loss
            
            # Extract hidden states at latent token positions
            # Use last hidden state from the model
            last_hidden = outputs.hidden_states[-1]  # (B, seq_len, 2048)
            
            # Find latent token positions
            latent_positions = find_latent_token_positions(input_ids, tokenizer)
            latent_positions = latent_positions.to("cuda")
            
            # Handle case where no latent tokens found
            if latent_positions.max() < 0:
                # No latent tokens in this sample — skip decoder loss
                total_loss = traj_loss
                decoder_loss = torch.tensor(0.0, device="cuda")
            else:
                # Filter out -1 padding positions
                valid_mask = latent_positions >= 0
                
                # Gather hidden states at latent positions
                B, num_latent = latent_positions.shape
                # Clamp to avoid index errors (-1 → 0, will be masked anyway)
                safe_positions = latent_positions.clamp(min=0)
                positions_expanded = safe_positions.unsqueeze(-1).expand(B, num_latent, 2048)
                latent_hidden = last_hidden.gather(1, positions_expanded)
                
                # Zero out invalid positions
                latent_hidden = latent_hidden * valid_mask.unsqueeze(-1).float()
                
                # Decoder forward: reconstruct CoT from latent hidden states
                _, decoder_loss = decoder(
                    latent_hidden_states=latent_hidden,
                    target_ids=cot_ids,
                    target_mask=cot_mask,
                )
                
                # Combined loss
                total_loss = traj_loss + args.lambda_decoder * decoder_loss
            
            # Scale for gradient accumulation
            scaled_loss = total_loss / args.grad_accum
            scaled_loss.backward()
            
            # Track metrics
            epoch_traj_loss += traj_loss.item()
            epoch_decoder_loss += decoder_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
            
            # Optimizer step every grad_accum batches
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(decoder.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log every 10 steps
                if global_step % 10 == 0:
                    avg_traj = epoch_traj_loss / num_batches
                    avg_dec = epoch_decoder_loss / num_batches
                    avg_total = epoch_total_loss / num_batches
                    elapsed = time.time() - start_time
                    print(f"  Step {global_step}/{total_steps} | "
                          f"traj_loss={avg_traj:.3f} | "
                          f"decoder_loss={avg_dec:.3f} | "
                          f"total={avg_total:.3f} | "
                          f"elapsed={elapsed:.0f}s")
        
        # End of epoch
        avg_traj = epoch_traj_loss / num_batches
        avg_dec = epoch_decoder_loss / num_batches
        avg_total = epoch_total_loss / num_batches
        elapsed = time.time() - start_time
        print(f"\n  Epoch {epoch+1}/{args.epochs} complete: "
              f"traj={avg_traj:.3f}, decoder={avg_dec:.3f}, total={avg_total:.3f} "
              f"({elapsed:.0f}s elapsed)")
    
    # 7. Save
    print(f"\nSaving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save LoRA adapter
    model.save_pretrained(args.output_dir)
    
    # Save decoder separately
    decoder_path = os.path.join(args.output_dir, "language_decoder.pt")
    torch.save(decoder.state_dict(), decoder_path)
    
    # Save training config
    config = {
        "variant": "latent_cot_with_decoder",
        "lambda_decoder": args.lambda_decoder,
        "lora_r": args.r,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_traj_loss": avg_traj,
        "final_decoder_loss": avg_dec,
        "final_total_loss": avg_total,
        "training_time_s": elapsed,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Trajectory loss: {avg_traj:.4f}")
    print(f"  Decoder loss:    {avg_dec:.4f}")
    print(f"  Total loss:      {avg_total:.4f}")
    print(f"  Training time:   {elapsed:.1f}s")
    print(f"  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    print(f"{'='*60}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_decoder", type=float, default=0.5, help="Weight for decoder loss")
    parser.add_argument("--r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()
    
    train(args)
