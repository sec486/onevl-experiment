# Ablation Study: Complete OneVL Recipe Comparison

**Purpose:** Reproduce OneVL's ablation table with unified ADE/FDE metrics on Cosmos-Reason2-2B.
**Reference:** https://github.com/GeorgeLuImmortal/OneVL_training (Training section)
**Data:** 250 NAVSIM scenes, 1250 training samples, 50 held-out eval samples (same for all variants)

---

## Ablation Matrix

| # | Variant | Latent Tokens | Vis Decoder | Lang Decoder | Staged Training | Script |
|:-:|---------|:---:|:---:|:---:|:---:|------|
| 1 | AR Answer (baseline) | ✗ | ✗ | ✗ | — | `v1_ar_answer.py` |
| 2 | AR CoT+Answer | ✗ | ✗ | ✗ | — | `v2_ar_cot.py` |
| 3 | OneVL w/o vis dec | ✓ | ✗ | ✓ | ✓ (3-stage) | `v3_lang_only.py` |
| 4 | OneVL w/o lang dec | ✓ | ✓ | ✗ | ✓ (3-stage) | `v4_vis_only.py` |
| 5 | OneVL w/o staged | ✓ | ✓ | ✓ | ✗ (direct joint) | `v5_no_staged.py` |
| 6 | **OneVL full** | ✓ | ✓ | ✓ | ✓ (4-stage) | `v6_full.py` (= w9_train_correct.py) |

## OneVL Paper Results (reference)

| Variant | PDM-score |
|---------|:---------:|
| AR Answer | 87.47 |
| AR CoT+Answer | 88.29 |
| OneVL w/o vis dec | 87.97 |
| OneVL w/o lang dec | 88.53 |
| OneVL w/o staged | 67.13 |
| **OneVL full** | **88.84** |

## Our Results (完成 ✅)

| 变体 | ADE (m) ↓ | FDE (m) ↓ | Valid | Traj Loss | Vis Loss | Lang Loss | 训练时间 |
|------|:---------:|:---------:|:-----:|:---------:|:--------:|:---------:|:--------:|
| 1. AR Answer (基线) | 7.14 | 12.40 | 50/50 | 0.604 | — | — | 86 min |
| 2. **AR CoT+Answer** | **6.76** | **11.63** | 50/50 | 0.569 | — | — | 84 min |
| 3. Lang 解码器 only | 8.29 | 14.75 | 50/50 | 0.498 | — | 1.40 | 61 min |
| 4. Vis 解码器 only | 8.58 | 15.34 | 50/50 | 0.629 | 3.92 | — | 108 min |
| 5. 无分阶段训练 | 7.74 | 13.64 | 50/50 | 0.606 | 6.32 | 2.37 | 95 min |
| 6. **OneVL 完整版** | **7.63** | **13.39** | 50/50 | 0.608 | 3.90 | 1.43 | 93 min |

### 核心发现

1. **显式 CoT (V2: 6.76m) 在我们的设置下仍然最优** — 原因: 固定 CoT 文本 + LoRA + 250 scenes 的局限性
2. **OneVL 完整版 (V6: 7.63m) 优于任何单独解码器** — V3: 8.29m, V4: 8.58m → 双解码器协同 > 任一单独
3. **分阶段训练有效** — V6 (7.63m) 优于 V5 (7.74m)，且 V5 的 vis_loss=6.32 几乎没有收敛 (vs V6 的 3.90)
4. **单独解码器反而损害性能** — V3/V4 (8.29/8.58m) 都比基线 V1 (7.14m) 差 → latent tokens 如果没有充分联合训练会干扰推理
5. **Training loss ≠ 泛化能力** — V3 traj_loss 最低 (0.498) 但 ADE 最差 (8.29m) = 过拟合。V6 的视觉解码器充当正则化器

### 与 OneVL 论文对比

| 趋势 | OneVL 论文 | 我们的实验 | 一致? |
|------|:---:|:---:|:---:|
| 完整版 > 单解码器 | ✅ (88.84 > 87.97/88.53) | ✅ (7.63 < 8.29/8.58) | ✅ |
| 分阶段训练关键 | ✅ (88.84 vs 67.13) | ✅ (7.63 vs 7.74, vis_loss 3.90 vs 6.32) | ✅ |
| 完整版 > AR CoT | ✅ (88.84 > 88.29) | ❌ (7.63 > 6.76) | ❌ |
| 完整版 > AR Answer | ✅ (88.84 > 87.47) | ❌ (7.63 > 7.14) | ❌ |

### 为什么完整版在我们的设置下没有超过 AR CoT?

| 因素 | OneVL 论文 | 我们的实验 | 影响 |
|------|:---:|:---:|------|
| CoT 标注质量 | AdaThinkDrive (每样本不同, 场景特定) | 固定一句话 (所有样本相同) | 语言解码器学不到有意义的场景推理 |
| 训练数据量 | >10K scenes | 250 scenes | latent tokens 难以学到足够丰富的表征 |
| 训练方式 | 全量微调 | LoRA (12% 参数) | latent embedding 学习能力受限 |
| Latent tokens 数量 | 35 vis + 20 lang | 4 vis + 2 lang | 信息瓶颈过窄 |
| Visual tokenizer | Emu3.5 IBQ 131K | Emu3 32K (subsampled 64) | 重建目标精度不足 |

**结论:** OneVL 方法在 Cosmos-Reason2 上的相对趋势 (完整>单独, 分阶段>直接) 得到验证，但由于上述 5 个缩减因素，完整版尚未超过简单的显式 CoT 基线。这是预期内的结果 — 扩大规模 (数据, 模型, latent tokens 数量) 是下一步。

## Execution Plan

1. All variants use IDENTICAL: data, eval set, LoRA config (r=64), base model, image size
2. Only difference: which decoders are active + training schedule
3. Each variant saves per-stage checkpoints
4. Eval: greedy decoding with proper prompt format, 50 held-out samples
5. Estimated total: ~6-8 hours GPU, ~$5-7

## File Structure

```
ablation/
├── README.md              (this file)
├── config.py              (shared config for all variants)
├── decoders.py            (VisualDecoderV2 + LangDecoder classes)
├── data_utils.py          (build_inputs, parse_trajectory, eval functions)
├── v1_ar_answer.py        (variant 1: answer-only SFT)
├── v2_ar_cot.py           (variant 2: CoT+answer SFT)
├── v3_lang_only.py        (variant 3: latent + lang decoder, no vis)
├── v4_vis_only.py         (variant 4: latent + vis decoder, no lang)
├── v5_no_staged.py        (variant 5: full but skip staged training)
├── v6_full.py             (variant 6: full 4-stage = our final model)
├── eval_all.py            (unified evaluation for all variants)
└── run_all.sh             (orchestrator: trains all 6, evals all 6)
```
