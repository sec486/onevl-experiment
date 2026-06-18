# 快速部署指南

15 分钟内运行 OneVL × Cosmos-Reason2 演示。

---

## 第 1 步：申请 GPU 实例

**AWS（推荐）：**
```bash
# g6.xlarge = NVIDIA L4 24GB ($0.80/小时 按需)
aws ec2 run-instances \
  --instance-type g6.xlarge \
  --image-id ami-0abcdef1234567890 \  # Deep Learning AMI (Ubuntu)
  --key-name your-key \
  --region us-east-1
```

**可选机型：**
| 机型 | GPU | 显存 | 费用 | 备注 |
|------|-----|:----:|:----:|------|
| g6.xlarge | L4 | 24 GB | $0.80/hr | ✅ 所有变体均可运行 |
| g6e.2xlarge | L40S | 48 GB | $2.73/hr | 无需 gradient checkpointing |
| g5.xlarge | A10G | 24 GB | $1.21/hr | L4 的替代方案 |

---

## 第 2 步：安装环境

```bash
# SSH 登录实例
ssh -i your-key.pem ubuntu@<instance-ip>

# 创建虚拟环境
python3 -m venv /opt/onevl-env
source /opt/onevl-env/bin/activate

# 安装 PyTorch + 依赖
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==5.4.0 peft==0.19.1 pillow numpy scipy huggingface_hub

# 验证安装
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 第 3 步：下载模型

```bash
# 基座模型 (~5GB)
huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --local-dir /opt/onevl/models/Cosmos-Reason2-2B

# Emu3 视觉分词器 (~2GB)
huggingface-cli download BAAI/Emu3-VisionTokenizer \
  --local-dir /opt/onevl/models/Emu3-VisionTokenizer
```

---

## 第 4 步：准备训练数据

将 `training-data/` 文件夹上传到实例：
```bash
# 从本地机器执行
scp -r training-data/ ubuntu@<instance>:/opt/onevl/data/

# 核心文件: navsim_real_traj_expanded.jsonl (250 场景, 1250 样本)
```

---

## 第 5 步：编码视觉 Token（一次性，~17 分钟）

```bash
# 重要: Emu3 编码需要 transformers 4.51.3
pip install transformers==4.51.3

python scripts/encode_emu3.py

# 编码完成后切回训练版本
pip install transformers==5.4.0
```

---

## 第 6 步：运行训练

```bash
# 完整 4 阶段 OneVL 训练 (~93 分钟)
python scripts/train_full.py

# 输出: checkpoint 保存到 /opt/onevl/output/
```

---

## 第 7 步：评估

```bash
python scripts/eval.py --checkpoint /opt/onevl/output/full_training/final_model --latent
# 输出: 50 个测试样本的 ADE 和 FDE
```

---

## 第 8 步：运行消融实验（可选，总计 ~8.5 小时）

```bash
cd ablation

# 逐个运行各变体
python v1_ar_answer.py    # 基线
python v2_ar_cot.py       # 显式 CoT（当前最优）
python v3_lang_only.py    # 仅语言解码器
python v4_vis_only.py     # 仅视觉解码器
python v5_no_staged.py    # 无分阶段训练
python v6_full.py         # OneVL 完整版（使用已有结果）
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| V4 Stage 2 报 `CUDA out of memory` | 在 Stage 2 循环前添加 `model.gradient_checkpointing_enable()` |
| Emu3 返回 float 而非 int64 | 使用 `result.image_tokens[0]`，不是 `result[0]` |
| 报错 `KeyError: 'qwen3_vl'` | 升级 transformers: `pip install transformers==5.4.0` |
| `embed_tokens` 冻结（隐式 token 无法学习）| LoRA 配置中添加 `modules_to_save=["embed_tokens"]` |
| Visual loss = 0（ViT 未被调用）| LoRA 包装后路径变为 `model.base_model.model.model.visual` |

---

## 预期结果

运行全部 6 个变体后，应看到如下结果：

```
V1 AR Answer:      ADE ≈ 7.1m
V2 AR CoT:         ADE ≈ 6.8m  ← 最优
V3 Lang Only:      ADE ≈ 8.3m
V4 Vis Only:       ADE ≈ 8.6m
V5 No Staged:      ADE ≈ 7.7m
V6 OneVL Full:     ADE ≈ 7.6m
```

核心结论: 完整版 > 单解码器 ✅，分阶段 > 直接联合 ✅

---

## 费用估算

| 活动 | 时长 | 费用 (g6.xlarge) |
|------|:----:|:----------------:|
| 环境搭建 + 编码 | 30 分钟 | $0.40 |
| 单次训练 (V6 完整版) | 93 分钟 | $1.24 |
| 完整消融实验 (全部 6 个) | 8.5 小时 | $6.80 |
| **总计（完整演示）** | **~10 小时** | **~$8** |
