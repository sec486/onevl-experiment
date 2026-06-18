# OneVL × Cosmos-Reason2 实验完整代码库

**实验周期:** 2026-06-01 ~ 2026-06-12 (Phase 1: W1-W4, Phase 2: W5-W9)
**总费用:** ~$35
**最终结果:** vis_loss 6.94→3.95 (非记忆化), traj 0.624→0.608 (2.6% improvement), 0 errors

---

## 文件夹结构

```
onevl-experiment-code/
│
├── Phase1-Report/              # Phase 1 完整文档 (W1-W4)
│   ├── Phase1-完整报告.md       # ★ Phase 1 总结
│   ├── W1-execution-log.md      # 环境搭建
│   ├── W2-execution-log.md      # CoT 生成 + baseline
│   ├── W3-execution-log.md      # Latent CoT + decoder
│   ├── W4-evaluation-report.md  # ADE/FDE 评估
│   └── latent_cot_analysis_report.html  # 可视化报告
│
├── Phase2-Report/              # Phase 2 完整文档 (W5-W9)
│   ├── Phase2-完整报告.md       # ★ Phase 2 总结
│   ├── OneVL-visual-decoder-deep-analysis.md  # 深度分析 + 教训
│   ├── Phase 2 计划 (Final).md  # 实验计划
│   ├── W8C Final Report (Quick Reviewed).md
│   ├── W9-execution-plan.md
│   └── latent_cot_analysis_report.html  # 可视化报告 (updated)
│
├── Phase3-Plan.md              # Phase 3 AlpaSim 集成计划
│
├── model/                      # ★ 最终训练产出 (896MB)
│   ├── adapter_model.safetensors  # LoRA adapter (692MB)
│   ├── adapter_config.json
│   ├── vis_decoder.pt            # 视觉解码器 (48MB)
│   ├── lang_decoder.pt           # 语言解码器 (157MB)
│   └── training_config.json      # 超参数
│
├── scripts/                    # W8-W9 训练脚本 (instance /scripts/)
│   ├── w9_train_correct.py       # ★ 最终版 4 阶段训练
│   ├── w9_encode_250.py          # Emu3 编码 (BF-037 修复)
│   ├── w9_expand_data.py         # 数据扩展 35→250 scenes
│   ├── w9_dry_run_v2.py          # 训练前验证
│   ├── w8c_encode_emu3.py        # Emu3 编码器 (原版)
│   └── w8c_train_three_stage.py  # W8 三阶段训练
│
├── data-scripts/               # W1-W8 实验脚本 (instance /data/)
│   ├── cot_generator_cosmos.py   # W2: Cosmos-Reason2 CoT 生成
│   ├── cot_generator.py          # W2: Haiku/Sonnet CoT 生成
│   ├── train_baseline.py         # W2: Answer-only + explicit CoT
│   ├── train_with_decoder.py     # W3: Latent CoT + 语言解码器
│   ├── language_decoder.py       # W3: 解码器头架构
│   ├── eval_all_variants.py      # W4: ADE/FDE 评估
│   ├── w6_visual_decoder_train.py  # W6: 视觉解码器 (MSE, 失败)
│   ├── w7_extract_and_train.py   # W7: 真实轨迹提取
│   ├── w7_eval_ade.py            # W7: ADE 评估
│   ├── w8_build_codebook.py      # W8: k-means codebook
│   ├── w8_three_stage_train.py   # W8: 三阶段训练
│   ├── w8_train_clang_1000.py    # W8: c-lang 1000 样本
│   └── ... (32 files total)
│
├── training-data/              # 训练数据 (JSONL)
│   ├── navsim_real_traj_expanded.jsonl  # ★ 最终训练数据 (250 scenes, 1250 samples)
│   ├── navsim_cot_cosmos_100.jsonl      # Phase 1 CoT 标注
│   ├── navsim_cot_cosmos_1000.jsonl     # 1000 样本 CoT
│   ├── navsim_real_traj_1000.jsonl      # W7 真实轨迹
│   └── ... (11 files)
│
├── results/                    # 训练日志 (18 files)
│   ├── w9_FINAL_log.txt          # ★ 最终成功运行
│   ├── w9_final_train_log.txt    # 早期成功运行
│   ├── bf037_step1_result.txt    # BF-037 调查结果
│   └── ...
│
└── bugfixes/                   # Bug 文档
    └── BF-037-emu3-encode-api-breakage.md
```

---

## 关键文件速查

### 要理解实验做了什么 → 读报告

| 文件 | 内容 |
|------|------|
| `Phase1-Report/Phase1-完整报告.md` | Phase 1 (100 samples): 语言解码器验证, ADE 2.10m |
| `Phase2-Report/Phase2-完整报告.md` | Phase 2 (1250 samples): 视觉解码器验证, vis 6.94→3.95 |
| `Phase2-Report/OneVL-visual-decoder-deep-analysis.md` | 7 条关键教训 + 5 个架构组件详解 |

### 要理解代码怎么写的 → 读核心脚本

| 文件 | 内容 | 阶段 |
|------|------|:----:|
| `data-scripts/train_with_decoder.py` | Latent CoT + 语言解码器训练 | W3 |
| `data-scripts/language_decoder.py` | 语言解码器架构定义 | W3 |
| `data-scripts/w6_visual_decoder_train.py` | 视觉解码器 v1 (MSE, 失败) | W6 |
| `scripts/w8c_train_three_stage.py` | 三阶段训练 (Emu3 codebook) | W8 |
| **`scripts/w9_train_correct.py`** | **★ 最终版: 4 阶段 + 5 组件完整实现** | W9 |

### 要复现实验 → 用最终脚本

```bash
# 1. 编码 (需 transformers 4.51.3)
/opt/onevl-env/bin/python3 scripts/w9_encode_250.py

# 2. 训练 (需 transformers 5.4.0, ~93 min on L4 24GB)
/opt/onevl-env/bin/python3 scripts/w9_train_correct.py
```

---

## 实验演进 (W1→W9)

| Week | 做了什么 | 核心脚本 | 结果 |
|:----:|---------|---------|------|
| W1 | 环境搭建 + 训练循环验证 | (inline commands) | 确认 LoRA 可训 |
| W2 | CoT 生成 + baseline 训练 | `cot_generator_cosmos.py`, `train_baseline.py` | 5 variant 训练完成 |
| W3 | Latent CoT + 语言解码器 | `train_with_decoder.py`, `language_decoder.py` | ADE 2.10m (3.5× improvement) |
| W4 | ADE/FDE 评估 | `eval_all_variants.py` | Phase 1 结论确认 |
| W5 | NAVSIM 数据验证 | (ops agent) | 1192 scenes, 142K frames |
| W6 | 视觉解码器 v1 (MSE) | `w6_visual_decoder_train.py` | 失败: 梯度爆炸 ×140 |
| W7 | BF-032 修复 + 公平对比 | `w7_extract_and_train.py`, `w7_eval_ade.py` | c-lang 1.17m vs c-full 2.81m |
| W8 | 根因分析 + 方案 B/C | `w8_three_stage_train.py`, `w8c_encode_emu3.py` | k-means 不学, Emu3 记忆化 |
| W9 | 正确实现 (5 组件) | **`w9_train_correct.py`** | ✅ vis 6.94→3.95, traj 0.608, 0 errors |

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|:----:|------|
| Python | 3.10 | `/opt/onevl-env/bin/python3` |
| transformers | 5.4.0 | 训练 (Qwen3-VL support) |
| transformers | 4.51.3 | Emu3 编码 (兼容性) |
| torch | 2.6.0+cu124 | GPU 计算 |
| peft | 0.19.1 | LoRA |
| GPU | NVIDIA L4 24GB | g6.xlarge ($0.80/hr) |

**注意:** Emu3 编码和训练需要不同的 transformers 版本。编码先做 (4.51.3)，然后切回 5.4.0 训练。
