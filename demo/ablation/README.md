# 消融实验

复现 OneVL 论文 Table 3 的消融对比，使用统一的 ADE/FDE 指标，基座模型为 Cosmos-Reason2-2B。

## 变体矩阵

| # | 变体 | 隐式 Token | 视觉解码器 | 语言解码器 | 分阶段训练 | 脚本 |
|:-:|------|:---:|:---:|:---:|:---:|--------|
| 1 | AR Answer（基线）| ✗ | ✗ | ✗ | — | `v1_ar_answer.py` |
| 2 | AR CoT+Answer | ✗ | ✗ | ✗ | — | `v2_ar_cot.py` |
| 3 | 仅语言解码器 | ✓ | ✗ | ✓ | ✓ | `v3_lang_only.py` |
| 4 | 仅视觉解码器 | ✓ | ✓ | ✗ | ✓ | `v4_vis_only.py` |
| 5 | 无分阶段训练 | ✓ | ✓ | ✓ | ✗ | `v5_no_staged.py` |
| 6 | OneVL 完整版 | ✓ | ✓ | ✓ | ✓ | `v6_full.py` |

## 实验结果

| 变体 | ADE (m) ↓ | FDE (m) ↓ | 训练时长 |
|------|:---------:|:---------:|:--------:|
| V1: AR Answer | 7.14 | 12.40 | 86 分钟 |
| **V2: AR CoT** | **6.76** | **11.63** | 84 分钟 |
| V3: 仅语言解码器 | 8.29 | 14.75 | 61 分钟 |
| V4: 仅视觉解码器 | 8.58 | 15.34 | 108 分钟 |
| V5: 无分阶段训练 | 7.74 | 13.64 | 95 分钟 |
| **V6: 完整版** | **7.63** | **13.39** | 93 分钟 |

## 核心发现

1. **显式 CoT (V2: 6.76m) 在当前设置下最优** — 原因: 固定 CoT 文本 + LoRA + 250 场景的局限性
2. **完整版 (V6: 7.63m) 优于任何单独解码器** — V3: 8.29m, V4: 8.58m → 双解码器协同有效
3. **分阶段训练关键** — V6 的 vis_loss=3.90（收敛）vs V5 的 vis_loss=6.32（几乎未收敛）
4. **单解码器反而损害性能** — V3/V4 都比基线 V1 (7.14m) 差 → 不充分的联合训练会干扰推理
5. **Training loss ≠ 泛化能力** — V3 的 traj_loss 最低 (0.498) 但 ADE 最差 (8.29m) = 过拟合

## 使用方法

```bash
# 逐个运行
python v1_ar_answer.py
python v2_ar_cot.py
python v3_lang_only.py
python v4_vis_only.py    # 24GB 需要 gradient checkpointing
python v5_no_staged.py
python v6_full.py        # = ../scripts/train_full.py
```

预计算结果: `../results/ablation_results.json`

## 共享文件

- `config.py` — 统一配置（路径、超参数、架构参数）
- `decoders.py` — VisualDecoderV2 + LangDecoder 架构定义
- `data_utils.py` — 数据加载、tokenization、评估指标计算
