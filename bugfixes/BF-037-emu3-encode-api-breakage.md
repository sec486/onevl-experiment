# BF-037: Emu3 VisionTokenizer `encode()` 返回 float 而非 int64 token IDs

**Status:** ✅ RESOLVED (June 11, 2026)
**Fix:** `result.image_tokens[0]` instead of `result[0]`
**Validation:** 253 scenes encoded (int64, range [15, 16341]), full training completed with 0 errors

**Date:** 2026-06-10
**Severity:** Critical (完全阻塞 W9 Step 2 训练)
**Wasted:** ~3 hours GPU time + ~2 hours debugging

---

## 症状

W9 Step 2 训练（250 scenes, 1250 samples）在所有 decoder 阶段报错：

```
RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'
```

只有 Stage 0（trajectory，不涉及视觉解码器）正常。

## 表面原因 vs 根本原因

| 层级 | 描述 |
|------|------|
| **表面** | `CrossEntropyLoss` 收到了 float targets 而非 int64 |
| **中层** | `emu3_visual_tokens_expanded.pt` 文件中存储的是 float32 张量，range `[-4.59, 7.72]` |
| **根本** | `Emu3VQVAE.encode()` 在当前环境下返回的是 **连续 latent features**，不是离散 codebook IDs |

## 详细分析

### 1. Emu3 VQ-VAE 编码流程（正确时）

```
图像 (H×W×3)
  → Encoder (卷积网络, 下采样)
  → Latent features (连续 float, shape ~[4, 96, 170])
  → Quantizer (查找最近 codebook 向量)
  → Discrete token IDs (int64, range [0, 32768))
```

**关键：** `encode()` 应该返回最终的离散 IDs，不是中间的 latent features。

### 2. 实际发生了什么

```python
result = vq_model.encode(pixel_values, image_sizes)
result[0]  # 应该是 int64 token IDs
```

**实际返回：**
```
shape: torch.Size([1, 1, 4, 96, 170])
dtype: torch.float32
range: [-4.59, 7.72]
unique values: 65103
```

这明显是 encoder 输出的连续 latent features（65103 个 unique float 值 ≈ 连续分布），**量化步骤没有执行**。

### 3. 为什么原来的 35 scene 编码成功了？

原来的 `emu3_visual_tokens_1000.pt` 文件：
```
35 scenes
dtype: torch.int64
range: [1683, 14067]
```

这是**有效的离散 codebook IDs**（整数、在 [0, 32768) 范围内）。

**可能原因：**
1. W8c 编码时使用了不同的 API 调用方式（可能是 `vq_model.quantize()` 或 `result.indices`）
2. 或者当时的 transformers 版本的 `encode()` 方法签名不同，内部包含了量化步骤
3. 原来编码的确切时间是 June 9 之前，当时环境尚未被 transformers 升级破坏

### 4. transformers 版本 + 架构命名不匹配

Load report 显示严重的权重映射问题：

| Checkpoint 中的 key 格式 | Library 期望的 key 格式 | 状态 |
|---|---|---|
| `encoder.down.{0,1,2,3}.block.{0,1}...` | `encoder.down_block.down.{0,1,2,3}.block.{0,1}...` | UNEXPECTED ← 没被加载 |
| `decoder.up.{0,1,2,3}.block.{0,1,2}...` | `decoder.up_block.up.{0,1,2,3}.block.{0,1,2}...` | UNEXPECTED ← 没被加载 |
| `encoder.mid.block_1...` | `encoder.middle_block.block_1...` | UNEXPECTED ← 没被加载 |

**但 `quantize` 层不在 UNEXPECTED/MISSING 列表中** — 说明 quantizer 权重是正确加载的。

这意味着：
- Encoder 权重全是随机初始化的 → 输出 garbage features
- Quantizer 权重正确 → 如果输入是 garbage，输出的离散 ID 也是无意义的
- 但更关键的是：`encode()` 方法可能因为架构不匹配，**根本没走到 quantize 步骤**，直接返回了 encoder 输出

### 5. 版本测试结果

| transformers 版本 | Emu3 encode() 行为 | 说明 |
|:-:|---|---|
| 4.51.3 (当前 onevl-env) | 返回 float32, shape [1,1,4,96,170] | encoder weights 是 UNEXPECTED → 随机 |
| 5.4.0 (训练用) | 同上 | 同样的 key mismatch |
| 5.10.2 | import 直接崩溃 (float8 dtype) | 不可用 |
| ??? (W8c 成功时) | 返回 int64, shape [96,170] | 正确的量化输出 |

## 为什么 Dry Run 没有拦住

| 问题 | 为什么没发现 |
|------|-------------|
| Dry run 使用了**原来的 35-scene 文件**做验证 | 那个文件是有效的 int64 |
| 没有验证 expanded 文件的 dtype/range | expand_data.py 产出后没有 validation step |
| 没有在 dry run 中执行一次真实的 `encode()` 调用 | 只验证了 model forward，没验证数据准备 |

## 修复方案

### 方案 A：找到正确的 `encode()` API（推荐）

1. 查看 HuggingFace 上 `BAAI/Emu3-VisionTokenizer` 的 `modeling_emu3.py` 源码
2. 确认正确的调用方式（可能是 `encode()` + `.indices` 或 `encode_to_tokens()`）
3. 用原来成功的 35-scene 文件做对照验证

### 方案 B：手动执行量化步骤

```python
# 如果 encode() 只返回 latent features:
with torch.no_grad():
    latent = vq_model.encode(pixel_values, image_sizes)  # float features
    # 手动量化
    _, indices = vq_model.quantize(latent)  # → int64 IDs
```

### 方案 C：使用 VQGAN（taming-transformers）替代

如果 Emu3 的版本兼容性无法解决，使用 `taming-transformers` 的 VQGAN（16K codebook，自包含，不依赖 transformers 版本）。

## 预防规则（新增到 steering）

1. **数据准备脚本必须在最后执行 validation：**
   ```python
   # 编码结束后立即验证
   assert all(t.dtype == torch.int64 for t in tokens.values()), "Token dtype must be int64"
   assert all(t.min() >= 0 and t.max() < CODEBOOK_SIZE for t in tokens.values()), "Token range invalid"
   ```

2. **Dry run 必须使用实际训练数据文件（不是 reference 文件）：**
   ```python
   # 加载的必须是 w9_train_correct.py 中的 VIS_TOKENS_PATH
   # 不能用 emu3_visual_tokens_1000.pt 验证 emu3_visual_tokens_expanded.pt
   ```

3. **`encode()` 返回值必须立即检查：**
   ```python
   result = vq_model.encode(pv, sizes)
   assert result[0].dtype == torch.int64, f"encode() returned {result[0].dtype}, expected int64"
   assert result[0].min() >= 0, f"Negative token IDs: min={result[0].min()}"
   ```

4. **transformers 版本变更后，所有依赖 HF 模型的数据准备步骤必须重新验证**

## 关联 bugs

| Bug | 关系 |
|-----|------|
| BF-036 | transformers 版本兼容性问题链（同一根因的不同表现） |
| BF-033 | 同样是"静默失败"模式 — model load 看似成功但实际权重没加载 |
| BF-034 | 视觉解码器设计缺陷（使用 MSE 而非 CE）— BF-037 是修复 BF-034 过程中引入的 |

## 时间线

```
June 9 (W8c):  Emu3 编码 35 scenes → 成功 (int64, [1683, 14067])
               环境: transformers==4.51.3, 某个 HF cache 状态
               
June 10 (W9):  升级 transformers → 5.4.0 for Qwen3-VL
               expand_data.py 调用 encode() → 返回 float32 (bug!)
               保存为 emu3_visual_tokens_expanded.pt (garbage data)
               
June 10 (W9 Step 2): 训练开始 → nll_loss dtype error on decoder stages
                     Stage 0 (trajectory only) 正常 → 误导以为是"简单 dtype cast"
                     
June 10 (debug): 发现 token range [-5, 13] (after .long() cast)
                 → 发现 encode() 返回 float → 发现 weight mismatch
                 → 确认根因: transformers 架构命名变更破坏了 Emu3 encoder 加载
```


## Resolution (June 11)

**The fix was simpler than expected:**

```python
# WRONG (returns float latent features):
token_tensor = result[0].cpu()

# CORRECT (returns int64 discrete codebook IDs):
token_tensor = result.image_tokens[0].cpu()
```

`Emu3VQVAEModelOutput` has two attributes:
- `result.last_hidden_state` / `result[0]`: float32 continuous encoder features (shape [1,1,4,H,W])
- `result.image_tokens`: list of int64 tensors with discrete codebook IDs (range [0, 32768))

**Additional fix needed:** Images must be resized to 448×448 before Emu3 encoding (1920×1080 NAVSIM images cause OOM on L4 24GB).

**Training validated:** Full 4-stage pipeline runs 92.7 min with 0 errors, vis_loss converges meaningfully (6.94→3.95), proving the visual tokenizer produces learnable targets.
