# W9 Model Checkpoints

Organized per-stage checkpoints for the OneVL × Cosmos-Reason2 experiment.

## Checkpoint Overview

| Folder | Stage | What's trained | Key metric |
|--------|-------|---------------|------------|
| `stage_preliminary/` | Preliminary | Visual decoder only (ViT input) | vis: 6.94→4.21 |
| `stage0_traj_warmup/` | Stage 0 | Main model LoRA (traj only) | traj: 0.716→0.631 |
| `stage1_decoder_warmup/` | Stage 1 | Both decoders (detached) | vis: 4.06→3.93, lang: 2.98→1.56 |
| `stage2_joint_final/` | Stage 2 | Everything (no detach) | traj: 0.624→0.608 |

## Training Pipeline

```
[Base Cosmos-Reason2-2B]
        │
        ▼ Preliminary (vis decoder learns future frames alone)
[stage_preliminary/vis_decoder.pt]
        │
        ▼ Stage 0 (main model learns trajectory)
[stage0_traj_warmup/ — NOT SAVED, see README]
        │
        ▼ Stage 1 (decoders learn to read latent states, model frozen)
[stage1_decoder_warmup/vis_decoder.pt + lang_decoder.pt]
        │
        ▼ Stage 2 (joint, no-detach, decoder gradients help main model)
[stage2_joint_final/ — PRODUCTION CHECKPOINT]
```

## Quick Reference

- **For inference:** Use `stage2_joint_final/adapter_*` only. Decoders discarded.
- **For world model demo:** Use `stage2_joint_final/vis_decoder.pt` to generate future frames
- **For CoT analysis:** Use `stage2_joint_final/lang_decoder.pt` to reconstruct reasoning
- **For ablation study:** Compare Stage 0 traj=0.631 vs Stage 2 traj=0.608

## Base Model
```
nvidia/Cosmos-Reason2-2B
Local: /opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561/
```
