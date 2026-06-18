# OneVL × Cosmos-Reason2 实验代码库

**核心结论:** AR CoT 最优 (ADE 6.76m); OneVL 完整版 (7.63m) 优于单解码器 (8.29/8.58m)

---

## 项目简介

将 [OneVL](https://arxiv.org/abs/2604.18486)（小米）的双辅助解码器训练方法迁移到 [NVIDIA Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)，在 [NAVSIM](https://github.com/autonomousvision/navsim) 自动驾驶数据上进行轨迹预测。通过隐式 Chain-of-Thought 训练实现推理零开销的性能提升。

---

## 文件夹结构

```
onevl-experiment/
│
├── README_CN.md                # 本文件 — 中文 README (项目管理视角)
├── CURRENT_STATUS.md           # 运维状态
├── Phase3-Plan.md              # Phase 3 AlpaSim 闭环集成计划
│
├── scripts/                    # 所有实验脚本 (按阶段分类)
│   ├── INDEX.md                # 代码索引 (★/✅/❌/🔧 分类)
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
│   │   └── ... (24 files)
│   └── w9-final/               # W9: 正确的 OneVL 实现 (最终版)
│       ├── w9_train_correct.py        # ★ 4 阶段完整训练脚本
│       ├── w9_reencode_emu3.py        # BF-037 修复后的 Emu3 编码
│       ├── w9_expand_data.py          # 数据扩展 35→250 scenes
│       └── w9_dry_run_v2.py           # 训练前验证
│
├── ablation/                   # Ablation Study (6 变体对比)
│   ├── README.md               # 结果表 + 分析 (中文)
│   ├── config.py               # 统一配置
│   ├── decoders.py             # 解码器架构
│   ├── data_utils.py           # 数据加载/评估工具
│   ├── v1_ar_answer.py         # V1: Answer-only 基线
│   ├── v2_ar_cot.py            # V2: 显式 CoT (当前最优)
│   ├── v3_lang_only.py         # V3: 语言解码器 only
│   ├── v4_vis_only.py          # V4: 视觉解码器 only
│   ├── v5_no_staged.py         # V5: 无分阶段训练
│   ├── v6_full.py              # V6: OneVL 完整版
│   └── results/                # V1-V6 的 checkpoints + logs
│       ├── v1_ar_answer/       #   checkpoint_final/ + results.json + train_log.txt
│       ├── v2_ar_cot/
│       ├── v3_lang_only/
│       ├── v4_vis_only/
│       ├── v5_no_staged/
│       └── v6_full/
│
├── model/                      # W9 最终模型权重
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  # LoRA 权重 (692MB)
│   ├── vis_decoder.pt             # 视觉解码器 (47MB)
│   ├── lang_decoder.pt            # 语言解码器 (156MB)
│   └── training_config.json
│
├── w9-checkpoints/             # W9 完整训练的分阶段 checkpoints
│   └── README.md               # 各阶段说明 + 加载方法
│
├── training-data/              # JSONL 训练数据
│   ├── navsim_real_traj_expanded.jsonl  # ★ 最终: 250 scenes, 1250 samples
│   ├── emu3_visual_tokens_expanded.pt   # Emu3 编码后的视觉 token
│   ├── navsim_cot_cosmos_100.jsonl      # Phase 1 CoT (Cosmos 生成)
│   ├── navsim_real_traj_1000.jsonl      # W7 真实轨迹 (1000 samples)
│   └── ... (12 files)
│
├── demo/                       # 客户 PoC 演示包 (可直接分享)
│   ├── README.md               # 演示说明 (中文)
│   ├── QUICK_SETUP.md          # 15 分钟快速部署
│   ├── scripts/                # 清理版核心脚本
│   │   ├── train_full.py       # 4 阶段训练
│   │   ├── encode_emu3.py      # Emu3 编码
│   │   └── eval.py             # ADE/FDE 评估
│   ├── ablation/               # 消融实验 (6 变体)
│   └── results/                # 预计算结果 JSON
│
├── bugfixes/                   # Bug 文档
│   └── BF-037-emu3-encode-api-breakage.md
│
├── models/                     # 基座模型 (Cosmos-Reason2-2B, .gitignore)
├── onevl/                      # OneVL 论文原始代码 (参考, .gitignore)
├── navsim/                     # NAVSIM 代码库 (.gitignore)
├── navtrain_data/              # NAVSIM 传感器图像 (.gitignore)
├── output/                     # 所有历史训练输出 (.gitignore)
└── archive/                    # 旧文件存档 (.gitignore)
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

## 核心发现

1. **显式 CoT (V2: 6.76m) 在当前设置下最优** — 原因: 固定 CoT 文本 + LoRA + 250 场景 + 4+2 隐式 token
2. **OneVL 完整版 (V6: 7.63m) 优于任何单独解码器** — V3: 8.29m, V4: 8.58m → 双解码器协同有效
3. **分阶段训练关键** — V6 的 vis_loss=3.90（收敛）vs V5 的 vis_loss=6.32（几乎未收敛）
4. **单解码器反而损害基线** — V3/V4 都比 V1 (7.14m) 差 → 不充分的联合训练会干扰推理
5. **Training loss ≠ 泛化能力** — V3 的 traj_loss 最低 (0.498) 但 ADE 最差 (8.29m) = 过拟合

### 与 OneVL 论文对比

| 趋势 | OneVL 论文 | 我们的实验 | 一致? |
|------|:---:|:---:|:---:|
| 完整版 > 单解码器 | ✅ | ✅ | ✅ |
| 分阶段训练关键 | ✅ | ✅ | ✅ |
| 完整版 > AR CoT | ✅ | ❌ | ❌ |

差距原因: CoT 标注质量 (固定单句 vs 逐样本)、数据规模 (250 vs >10K)、LoRA vs 全量微调、latent token 数量 (6 vs 55)

---

## 关键文件速查

| 目的 | 文件 |
|------|------|
| 快速了解项目 | 本文件 `README_CN.md` |
| 看最终对比结果 | `ablation/README.md` |
| 复现 W9 训练 | `scripts/w9-final/w9_train_correct.py` |
| 复现 Ablation | `ablation/v1_ar_answer.py` ~ `v6_full.py` |
| 加载最终模型 | `model/adapter_model.safetensors` + `PeftModel.from_pretrained()` |
| 查看数据格式 | `training-data/navsim_real_traj_expanded.jsonl` |
| 给客户演示 | `demo/` 文件夹（自包含, 可直接分享） |
| 了解 Bug 历史 | `bugfixes/BF-037-emu3-encode-api-breakage.md` |
| 代码索引 | `scripts/INDEX.md` |

---

## 环境要求

| 依赖 | 版本 | 用途 |
|------|:----:|------|
| Python | 3.10 | 训练/推理 |
| transformers | 5.4.0 | 训练 (Qwen3-VL / Cosmos-Reason2) |
| transformers | 4.51.3 | Emu3 编码 (兼容性, 单独步骤) |
| torch | 2.6.0+cu124 | GPU 计算 |
| peft | 0.19.1 | LoRA |
| GPU | L4 24GB 或 L40S 48GB | 训练 (V4 Stage 2 需要 grad checkpoint on 24GB) |

---

## 如何复现

```bash
# 1. 编码视觉 tokens (需 transformers 4.51.3)
pip install transformers==4.51.3
python scripts/w9-final/w9_reencode_emu3.py

# 2. 切回训练版本
pip install transformers==5.4.0

# 3. 运行完整 4 阶段训练 (~93 min on L4)
python scripts/w9-final/w9_train_correct.py

# 4. 或运行 Ablation 某个变体
cd ablation && python v1_ar_answer.py
```

---

## 许可说明

基于以下开源项目:
- Cosmos-Reason2-2B — NVIDIA Open Model License（允许商业使用）
- Emu3 VisionTokenizer — Apache 2.0（允许商业使用）
- NAVSIM 数据集 — 基于 nuPlan，需确认 nuPlan Terms of Use（数据可能限非商业用途，可用自有数据替换）

---

## 参考资料

- [OneVL 论文 (arXiv:2604.18486)](https://arxiv.org/abs/2604.18486)
- [OneVL 训练代码](https://github.com/GeorgeLuImmortal/OneVL_training)
- [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
- [NAVSIM](https://github.com/autonomousvision/navsim)
- [Emu3 VisionTokenizer](https://huggingface.co/BAAI/Emu3-VisionTokenizer)
