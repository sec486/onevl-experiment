"""
Baseline training script for OneVL × Cosmos-Reason2 experiment.
Uses ms-swift with custom LoRA config that includes embed_tokens.

Approach: Since ms-swift CLI doesn't expose --lora_target_modules directly,
we use the Python API to configure LoRA before training.
"""
import sys
import os
import json

# Patch: set lora target modules via environment variable that swift respects,
# or use the swift Python API directly
def main():
    import subprocess
    
    variant = sys.argv[1] if len(sys.argv) > 1 else "answer_only"
    
    MODEL_PATH = "/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
    
    datasets = {
        "answer_only": "/opt/onevl-experiment/OneVL_training/demo_data/navsim/navsim_answer_demo100.jsonl",
        "answer_only_r64": "/opt/onevl-experiment/OneVL_training/demo_data/navsim/navsim_answer_demo100.jsonl",
        "explicit_cot": "/opt/onevl-experiment/data/navsim_cot_cosmos_100.jsonl",
        "explicit_cot_sonnet": "/opt/onevl-experiment/data/navsim_cot_sonnet_100.jsonl",
        "latent_cot": "/opt/onevl-experiment/OneVL_training/demo_data/navsim/navsim_vis4_text2_demo100.jsonl",
        "latent_cot_r64": "/opt/onevl-experiment/OneVL_training/demo_data/navsim/navsim_vis4_text2_demo100.jsonl",
    }
    
    dataset_path = datasets.get(variant)
    if not dataset_path:
        print(f"Unknown variant: {variant}. Available: {list(datasets.keys())}")
        sys.exit(1)
    
    output_dir = f"/opt/onevl-experiment/output/baseline_{variant}"
    
    print(f"=== Training Baseline: {variant} ===")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print()
    
    # Method: Use swift with a YAML config that specifies lora targets
    # Create a temporary config
    config = {
        "model": MODEL_PATH,
        "model_type": "qwen3_vl",
        "dataset": dataset_path,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "gradient_checkpointing": True,
        "output_dir": output_dir,
        "logging_steps": 10,
        "num_train_epochs": 3,
        "save_strategy": "epoch",
        # LoRA config with embed_tokens
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "q_proj k_proj v_proj o_proj embed_tokens",
    }
    
    # Try multiple approaches:
    
    # Approach 1: swift sft with --lora_target (space-separated, not comma)
    cmd = [
        "swift", "sft",
        "--model", MODEL_PATH,
        "--model_type", "qwen3_vl",
        "--dataset", dataset_path,
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "2e-5",
        "--gradient_checkpointing", "true",
        "--output_dir", output_dir,
        "--logging_steps", "10",
        "--num_train_epochs", "3",
        "--save_strategy", "epoch",
        "--lora_target", "q_proj k_proj v_proj o_proj embed_tokens",
    ]
    
    print(f"Trying: swift sft with --lora_target ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS with --lora_target")
        print(result.stdout[-2000:])
        return
    
    # Check if the error is about --lora_target being invalid
    if "remaining_argv" in result.stderr or "remaining_argv" in result.stdout:
        print(f"--lora_target not recognized. Trying alternative...")
        
        # Approach 2: Use --target_modules (another common name)
        cmd2 = cmd.copy()
        idx = cmd2.index("--lora_target")
        cmd2[idx] = "--target_modules"
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode == 0:
            print("SUCCESS with --target_modules")
            print(result2.stdout[-2000:])
            return
        
        # Approach 3: Use environment variable or write custom training script
        print("CLI approaches failed. Using direct peft API...")
        train_with_peft_api(variant, MODEL_PATH, dataset_path, output_dir)
    else:
        print(f"FAILED with different error:")
        print(result.stderr[-2000:])
        print(result.stdout[-2000:])


def train_with_peft_api(variant, model_path, dataset_path, output_dir):
    """Direct training using transformers + peft, bypassing ms-swift CLI."""
    import torch
    from transformers import (
        Qwen3VLForConditionalGeneration, 
        AutoProcessor,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset
    
    print("Loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Apply LoRA with embed_tokens included
    # For answer_only and explicit_cot: standard LoRA (no embed_tokens needed)
    # For latent_cot (variant c): add modules_to_save for embed_tokens
    needs_embed = "latent" in variant
    
    # Use higher rank for latent_cot_r64 ablation
    lora_rank = 64 if "r64" in variant else 8
    lora_alpha = 128 if "r64" in variant else 32
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens"] if needs_embed else None,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Verify embed_tokens is trainable
    for name, param in model.named_parameters():
        if "embed_tokens" in name and param.requires_grad:
            print(f"  TRAINABLE: {name} ({param.numel():,} params)")
    
    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    samples = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"  {len(samples)} samples loaded")
    
    # Simple tokenized dataset (text-only for now, images handled separately)
    class SimpleDataset(Dataset):
        def __init__(self, samples, processor):
            self.samples = samples
            self.processor = processor
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            messages = sample.get("messages", [])
            
            # Format as chat
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            # Tokenize
            encoded = self.processor.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding="max_length",
            )
            
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask padding in labels
            labels[attention_mask == 0] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
    
    train_dataset = SimpleDataset(samples, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving to {output_dir}")
    trainer.save_model(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
