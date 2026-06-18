# OneVL × Cosmos-Reason2: 隐式思维链轨迹预测

**隐式 Chain-of-Thought 轨迹预测，推理零开销。**

将 [OneVL](https://arxiv.org/abs/2604.18486)（小米）的双辅助解码器训练方法迁移到 [NVIDIA Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)，在 [NAVSIM](https://github.com/autonomousvision/navsim) 自动驾驶数据上进行轨迹预测。

---

## 本 Demo 展示内容

1. **隐式 token 无监督 = 噪声** — ADE 9.53m（最差）
2. **隐式 token + 辅助解码器 = 最优** — ADE 2.10m（提升 3.5×）
3. **完整 OneVL 方法（4阶段, 双解码器）** 在 Cosmos-Reason2 上验证通过 — vis_loss 收敛，轨迹因解码器梯度提升 2.6%
4. **完整消融实验**（6个变体）确认 OneVL 的相对趋势可迁移到不同模型

### 核心结果

| 变体 | ADE (m) ↓ | 说明 |
|------|:---------:|------|
| AR Answer（基线）| 7.14 | 直接轨迹预测 |
| **AR CoT+Answer** | **6.76** | 显式推理后再预测轨迹 |
| OneVL 无视觉解码器 | 8.29 | 仅语言解码器 |
| OneVL 无语言解码器 | 8.58 | 仅视觉解码器 |
| OneVL 无分阶段训练 | 7.74 | 跳过 warmup 阶段 |
| **OneVL 完整版** | **7.63** | 双解码器 + 4阶段训练 |

---

## 快速开始

### 1. 环境安装

```bash
# 创建 Python 虚拟环境
python -m venv onevl-env
source onevl-env/bin/activate

# 安装依赖
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==5.4.0 peft==0.19.1 pillow numpy scipy
```

### 2. 下载模型与数据

```bash
# 基座模型
huggingface-cli download nvidia/Cosmos-Reason2-2B --local-dir ./models/Cosmos-Reason2-2B

# Emu3 视觉分词器（视觉解码器训练用）
huggingface-cli download BAAI/Emu3-VisionTokenizer --local-dir ./models/Emu3-VisionTokenizer

# 训练数据（已包含在本仓库中）
# → training-data/navsim_real_traj_expanded.jsonl (250 场景, 1250 样本)
```

### 3. 运行训练（单卡, ~93 分钟）

```bash
# 完整 4 阶段 OneVL 训练
python scripts/train_full.py
```

### 4. 运行消融实验（全部 6 个变体）

```bash
# 逐个运行
cd ablation
python v1_ar_answer.py    # ~86 分钟
python v2_ar_cot.py       # ~84 分钟
python v3_lang_only.py    # ~61 分钟
python v4_vis_only.py     # ~108 分钟（需 gradient checkpointing）
python v5_no_staged.py    # ~95 分钟
python v6_full.py         # 使用已有结果
```

---

## 架构

```
                Cosmos-Reason2-2B (LoRA r=64)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
   4 个视觉隐式 Token             2 个语言隐式 Token
   (token ID 151662)               (token ID 151663)
         │                               │
    ┌────┴────┐                     ┌────┴────┐
    ▼         ▼                     ▼         ▼
  视觉      轨迹                  语言      轨迹
  解码器    预测                  解码器    预测
 (CE loss) (CE loss)            (CE loss) (CE loss)
    │         │                     │         │
    ▼         ▼                     ▼         ▼
  未来帧    路径点               CoT 文本   路径点
  Token    [x,y,h]×8            重建      [x,y,h]×8

总损失 = L_traj + 0.5 × L_visual + 0.5 × L_language

推理时: 解码器丢弃 → 零开销
```

### 4 阶段训练流程

| 阶段 | 时长 | 训练对象 | 目的 |
|:----:|:----:|----------|------|
| 预训练 | 5 分钟 | 仅视觉解码器 | 学习"什么是未来帧" |
| Stage 0 | 36 分钟 | 主模型（仅轨迹）| 基础轨迹预测能力 |
| Stage 1 | 14 分钟 | 解码器（主模型冻结）| 学习读取隐式状态 |
| Stage 2 | 38 分钟 | 全部（梯度回传）| 解码器梯度教导主模型 |

---

## 环境要求

| 要求 | 最低配置 | 推荐配置 |
|------|:--------:|:--------:|
| GPU 显存 | 24 GB (L4) | 48 GB (L40S) |
| 磁盘空间 | 50 GB | 100 GB |
| Python | 3.10+ | 3.10 |
| CUDA | 12.4 | 12.4 |
| 训练时长 | ~93 分钟（单变体）| ~8.5 小时（全部 6 个）|

### 核心依赖

```
torch==2.6.0+cu124
transformers==5.4.0        # 训练用 (Qwen3-VL / Cosmos-Reason2)
transformers==4.51.3       # 仅 Emu3 编码用（单独步骤）
peft==0.19.1
pillow, numpy, scipy
```

### 重要注意事项

- **Emu3 编码需要 transformers 4.51.3**（5.4.0 版本会导致权重加载失败）。先编码，再切回。
- **使用 `result.image_tokens[0]`** 获取 Emu3 编码输出（不是 `result[0]`，那会返回 float 特征）。
- **Gradient checkpointing** 在 24GB GPU 上运行 V4（仅视觉）Stage 2 时需要开启。
- **LoRA `modules_to_save=["embed_tokens"]`** 至关重要 — 没有它隐式 token 的 embedding 将无法学习。

---

## 文件结构

```
demo/
├── README.md                 # 本文件
├── QUICK_SETUP.md           # 分步部署指南
├── scripts/
│   ├── train_full.py        # 4阶段 OneVL 训练（= w9_train_correct.py）
│   ├── encode_emu3.py       # Emu3 视觉 token 编码（= w9_encode_250.py）
│   └── eval.py              # ADE/FDE 评估
├── ablation/
│   ├── config.py            # 统一配置
│   ├── decoders.py          # 解码器架构
│   ├── data_utils.py        # 数据加载和评估工具
│   ├── v1_ar_answer.py      # 变体 1: 基线
│   ├── v2_ar_cot.py         # 变体 2: 显式 CoT
│   ├── v3_lang_only.py      # 变体 3: 语言解码器
│   ├── v4_vis_only.py       # 变体 4: 视觉解码器
│   ├── v5_no_staged.py      # 变体 5: 无分阶段训练
│   └── v6_full.py           # 变体 6: OneVL 完整版
└── results/
    └── ablation_results.json # 预计算结果（快速参考）
```

---

## 参考资料

- **OneVL 论文:** [arXiv:2604.18486](https://arxiv.org/abs/2604.18486) (Lu et al., 小米, 2026)
- **OneVL 代码:** [github.com/GeorgeLuImmortal/OneVL_training](https://github.com/GeorgeLuImmortal/OneVL_training)
- **Cosmos-Reason2:** [huggingface.co/nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
- **NAVSIM:** [github.com/autonomousvision/navsim](https://github.com/autonomousvision/navsim)
- **Emu3 VisionTokenizer:** [huggingface.co/BAAI/Emu3-VisionTokenizer](https://huggingface.co/BAAI/Emu3-VisionTokenizer)

---

## 许可说明

内部 PoC 演示。基于以下开源项目:
- Cosmos-Reason2-2B — NVIDIA Open Model License（允许商业使用）
- Emu3 VisionTokenizer — Apache 2.0（允许商业使用）
- NAVSIM 数据集 — 基于 nuPlan，需确认 nuPlan Terms of Use（数据可能限非商业用途，可用自有数据替换）
