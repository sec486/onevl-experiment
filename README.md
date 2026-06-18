# OneVL × Cosmos-Reason2 实验代码库

**实验周期:** 2026-06-01 ~ 2026-06-13 (Phase 1 W1-W4 + Phase 2 W5-W9 + Ablation)
**总费用:** ~$42
**核心结论:** AR CoT 最优 (ADE 6.76m); OneVL 完整版 (7.63m) 优于单解码器 (8.29/8.58m)

---

## 文件夹结构

```
onevl-experiment-code/
│
├── README.md                   # 英文 README (研究视角, 对标 OneVL 论文格式)
├── README_CN.md                # 本文件 — 中文 README (项目管理视角)
├── CURRENT_STATUS.md           # 运维状态 (instance IDs, S3 paths, 命令记录)
├── Phase3-Plan.md              # Phase 3 AlpaSim 闭环集成计划
│
├── scripts/                    # 所有实验脚本 (按阶段分类)
│   ├── phase1-w1w4/            # Phase 1: CoT 生成, baseline 训练, 评估
│   │   ├── cot_generator_cosmos.py    # W2: 用 Cosmos-Reason2 生成 CoT
│   │   ├── cot_generator.py           # W2: 用 Bedrock Claude 生成 CoT
│   │   ├── train_baseline.py          # W2: Answer-only + explicit CoT 训练
│   │   ├── train_with_decoder.py      # W3: Latent CoT + 语言解码器
│   │   ├── language_decoder.py        # W3: 语言解码器架构定义
│   │   └── eval_all_variants.py       # W4: ADE/FDE 评估
│   ├── phase2-w5w8/            # Phase 2: 视觉解码器尝试 (MSE→CE 演进)
│   │   ├── w6_visual_decoder_train.py # W6: 视觉解码器 v1 (MSE, 失败)
│   │   ├── w7_extract_and_train.py    # W7: 真实轨迹提取 + 训练
│   │   ├── w8_three_stage_train.py    # W8: k-means 三阶段训练
│   │   ├── w8c_encode_emu3.py         # W8c: Emu3 编码 (35 scenes)
│   │   ├── w8c_train_three_stage.py   # W8c: Emu3 三阶段训练
│   │   └── ... (12 files)
│   └── w9-final/               # W9: 正确的 OneVL 实现 (最终版)
│       ├── w9_train_correct.py        # ★ 4 阶段完整训练脚本
│       ├── w9_encode_250.py           # BF-037 修复后的 Emu3 编码
│       ├── w9_expand_data.py          # 数据扩展 35→250 scenes
│       ├── w9_dry_run_v2.py           # 训练前 12 项验证
│       ├── bf037_investigate.py       # BF-037 调查
│       └── ... (14 files)
│
├── ablation/                   # Ablation Study (6 变体对比)
│   ├── README.md               # 结果表 + 分析 (中文)
│   ├── TODO.md                 # 执行状态记录
│   ├── config.py               # 统一配置
│   ├── decoders.py             # 解码器架构
│   ├── data_utils.py           # 数据加载/评估工具
│   ├── v1_ar_answer.py         # V1: Answer-only 基线
│   ├── v2_ar_cot.py            # V2: 显式 CoT (当前最优)
│   ├── v3_lang_only.py         # V3: 语言解码器 only
│   ├── v4_vis_only_memfix.py   # V4: 视觉解码器 only (需 grad checkpoint)
│   ├── v5_no_staged.py         # V5: 无分阶段训练
│   ├── v6_full.py              # V6: OneVL 完整版
│   ├── run_all.sh              # 编排脚本
│   └── results/                # V1-V6 的 checkpoints + logs
│       ├── v1_ar_answer/       #   checkpoint_final/ + results.json + train_log.txt
│       ├── v2_ar_cot/
│       ├── v3_lang_only/
│       ├── v4_vis_only/
│       ├── v5_no_staged/
│       └── v6_full/
│
├── w9-checkpoints/             # W9 完整训练的分阶段 checkpoints
│   ├── README.md               # 各阶段说明 + 加载方法
│   ├── stage_preliminary/      # 视觉解码器独立预训练后
│   ├── stage0_traj_warmup/     # 轨迹 warmup 后 (⚠️ LoRA 未保存)
│   ├── stage1_decoder_warmup/  # 解码器 warmup 后 (detach)
│   └── stage2_joint_final/     # ★ 最终联合训练后 (692MB LoRA + 解码器)
│
├── model/                      # W9 最终模型 (= stage2_joint_final 的副本)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  # LoRA 权重 (692MB)
│   ├── vis_decoder.pt             # 视觉解码器 (47MB)
│   ├── lang_decoder.pt            # 语言解码器 (156MB)
│   └── training_config.json
│
├── training-data/              # JSONL 训练数据
│   ├── navsim_real_traj_expanded.jsonl  # ★ 最终: 250 scenes, 1250 samples
│   ├── navsim_cot_cosmos_100.jsonl      # Phase 1 CoT (Cosmos 生成)
│   ├── navsim_real_traj_1000.jsonl      # W7 真实轨迹 (1000 samples)
│   └── ... (11 files)
│
├── logs/                       # 训练日志
│   └── w9-training/            # W9 各次训练的完整日志
│
├── bugfixes/                   # Bug 文档
│   └── BF-037-emu3-encode-api-breakage.md
│
├── Phase1-Report/              # Phase 1 完整文档 (W1-W4)
│   ├── Phase1-完整报告.md       # ★ 总结
│   └── ... (9 files)
│
└── Phase2-Report/              # Phase 2 完整文档 (W5-W9)
    ├── Phase2-完整报告.md       # ★ 总结
    ├── latent_cot_analysis_report.html  # 可视化 HTML 报告
    └── ... (14 files)
```

---

## Ablation 最终结果

| # | 变体 | ADE (m) ↓ | FDE (m) ↓ | 说明 |
|:-:|------|:---------:|:---------:|------|
| 1 | AR Answer | 7.14 | 12.40 | 无 latent, 无解码器 |
| 2 | **AR CoT+Answer** | **6.76** | **11.63** | ★ 当前最优 — 显式推理文本 |
| 3 | Lang 解码器 only | 8.29 | 14.75 | 语言解码器, 3 阶段训练 |
| 4 | Vis 解码器 only | 8.58 | 15.34 | 视觉解码器, 4 阶段训练 |
| 5 | 无分阶段训练 | 7.74 | 13.64 | 完整架构, 直接联合训练 |
| 6 | **OneVL 完整版** | **7.63** | **13.39** | 双解码器 + 4 阶段分步训练 |

---

## 关键文件速查

| 目的 | 文件 |
|------|------|
| 理解实验做了什么 | `Phase1-Report/Phase1-完整报告.md`, `Phase2-Report/Phase2-完整报告.md` |
| 看最终对比结果 | `ablation/README.md` |
| 看 HTML 可视化 | `Phase2-Report/latent_cot_analysis_report.html` |
| 复现 W9 训练 | `scripts/w9-final/w9_train_correct.py` |
| 复现 Ablation | `ablation/v1_ar_answer.py` ~ `v6_full.py` |
| 加载最终模型 | `model/adapter_model.safetensors` + `PeftModel.from_pretrained()` |
| 查看数据格式 | `training-data/navsim_real_traj_expanded.jsonl` |
| 了解 Bug 历史 | `bugfixes/BF-037-emu3-encode-api-breakage.md` |

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|:----:|------|
| Python | 3.10 | 训练/推理 |
| transformers | 5.4.0 | 训练 (Qwen3-VL) |
| transformers | 4.51.3 | Emu3 编码 (兼容性) |
| torch | 2.6.0+cu124 | GPU 计算 |
| peft | 0.19.1 | LoRA |
| GPU | L4 24GB 或 L40S 48GB | 训练 (V4 Stage 2 需要 grad checkpoint on 24GB) |

---

## 如何复现

```bash
# 1. 编码视觉 tokens (需 transformers 4.51.3)
/opt/onevl-env/bin/python3 scripts/w9-final/w9_encode_250.py

# 2. 切回训练版本
pip install transformers==5.4.0

# 3. 运行完整 4 阶段训练 (~93 min on L4)
/opt/onevl-env/bin/python3 scripts/w9-final/w9_train_correct.py

# 4. 或运行 Ablation 某个变体
cd ablation && python v1_ar_answer.py
```
