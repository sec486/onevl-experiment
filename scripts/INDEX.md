# 实验代码索引

按实验时间线和功能分类。每个文件标注了用途和状态。

---

## phase1-w1w4/ — Phase 1 核心代码 (W1-W4, Jun 1-3)

| 文件 | 用途 | 阶段 | 状态 |
|------|------|:----:|:----:|
| `cot_generator_cosmos.py` | 用 Cosmos-Reason2 + 真实图像生成 CoT 标注 | W2 | ✅ 正式 |
| `cot_generator.py` | 用 Bedrock Claude 生成 CoT 标注 (baseline) | W2 | ✅ 正式 |
| `train_baseline.py` | 训练 Answer-only + Explicit CoT 变体 | W2 | ✅ 正式 |
| `train_with_decoder.py` | 训练 Latent CoT + 语言解码器 | W3 | ★ 核心 |
| `language_decoder.py` | 语言解码器 (LanguageDecoderHead) 架构定义 | W3 | ★ 核心 |
| `eval_all_variants.py` | Phase 1 ADE/FDE 评估 (5 variants) | W4 | ✅ 正式 |
| `eval_cfull.py` | c-full 评估 (Phase 2 用, 误放此处) | W7 | ⚠️ 应在 phase2 |
| `eval_w6.py` | W6 评估 (Phase 2 用, 误放此处) | W6 | ⚠️ 应在 phase2 |

---

## phase2-w5w8/ — Phase 2 视觉解码器实验 (W5-W8c, Jun 3-9)

### W6: MSE 视觉解码器 (失败路径)
| 文件 | 用途 | 状态 |
|------|------|:----:|
| `w6_visual_decoder_train.py` | 首次视觉解码器训练 (MSE loss) | ❌ 方案失败 |
| `w6_cfull.py` | c-full 联合训练 (MSE, 合成数据) | ❌ BF-032 |
| `w6_vd_pooled.py` | 视觉解码器 pooled 版本 | ❌ 实验性 |
| `w6_vd_lambda1.py` | λ_vis=1.0 实验 | ❌ 实验性 |
| `w6_debug_model.py` | 模型调试 | 🔧 调试 |

### W7: 真实轨迹提取 + 公平对比
| 文件 | 用途 | 状态 |
|------|------|:----:|
| `w7_extract_and_train.py` | 从 NAVSIM pkl 提取真实轨迹 | ✅ 正式 |
| `w7_eval_ade.py` | ADE/FDE 评估 | ✅ 正式 |
| `w7_fix_bf032.py` | BF-032 修复 | 🔧 修复 |
| `w7_train_cfull.py` | c-full 训练 v1 | ❌ 被 v2/v3 替代 |
| `w7_train_v2.py` | c-full 训练 v2 | ❌ 被 v3 替代 |
| `w7_train_v3.py` | c-full 训练 v3 | ✅ 正式 (最终版) |
| `w7_train_v4_1000.py` | 1000 样本训练 | ✅ 正式 |

### W8: CE Loss + 三阶段训练
| 文件 | 用途 | 状态 |
|------|------|:----:|
| `w8_build_codebook.py` | k-means codebook 构建 | ✅ 方案 B |
| `w8_three_stage_train.py` | 三阶段训练 (k-means) | ✅ 方案 B |
| `w8_train_clang_1000.py` | c-lang 1000 样本训练 (ADE 1.17m) | ★ 核心 |
| `w8_cfull_fixed.py` | BF-034 修复尝试 v1 | ❌ 被替代 |
| `w8_cfull_fixed_v2.py` | BF-034 修复尝试 v2 | ❌ 被替代 |
| `w8_cfull_fixed_v3.py` | BF-034 修复尝试 v3 | ❌ 被替代 |
| `w8_cfull_pragmatic.py` | 实用主义修复 | 🔧 调试 |
| `w8_emu3_encode.py` | Emu3 编码 (早期版本) | ❌ 被 w8c 替代 |
| `w8_eval_fixed.py` | 修复后评估 | ✅ |
| `w8_p0_fix_and_train.py` | P0 优先级修复 | 🔧 修复 |
| `w8_validate_diagnosis.py` | 诊断脚本 | 🔧 调试 |
| `w8_validate2.py` | 验证脚本 v2 | 🔧 调试 |
| `w8c_encode_emu3.py` | ★ Emu3 VisionTokenizer 编码 (35 scenes) | ★ 核心 |
| `w8c_train_three_stage.py` | ★ Emu3 三阶段训练 | ★ 核心 |

---

## w9-final/ — W9 正确实现 + BF-037 修复 (Jun 10-12)

### 正式代码 (最终可用)
| 文件 | 用途 | 状态 |
|------|------|:----:|
| `w9_train_correct.py` | ★★ 4 阶段完整训练 (最终版, 0 errors) | ★★ 核心 |
| `w9_encode_250.py` | Emu3 编码 250 scenes (BF-037 修复后) | ★ 核心 |
| `w9_expand_data.py` | 数据扩展 35→250 scenes | ✅ 正式 |
| `w9_dry_run_v2.py` | 训练前 12 项验证 | ✅ 正式 |

### BF-037 调试过程
| 文件 | 用途 | 状态 |
|------|------|:----:|
| `bf037_investigate.py` | BF-037 根因调查 | 🔧 调试 |
| `bf037_step1.py` | 找到正确 API (result.image_tokens[0]) | 🔧 调试 |
| `w9_fix_dtype.py` | dtype 修复尝试 | 🔧 调试 |
| `w9_reencode_emu3.py` | 重新编码脚本 | 🔧 调试 |

### 运维/部署脚本
| 文件 | 用途 | 状态 |
|------|------|:----:|
| `w9_dry_run.py` | 早期 dry run (被 v2 替代) | ❌ |
| `w9_bf037_complete_fix.sh` | 一键修复 shell 脚本 | 🔧 |
| `w9_fix_and_validate.sh` | 修复 + 验证 | 🔧 |
| `w9_ops_monitor.sh` | ops agent 监控 | 🔧 |
| `w9_patch_and_run.sh` | patch 后运行 | 🔧 |
| `w9_reencode_fix.sh` | 重编码修复 v1 | 🔧 |
| `w9_reencode_fix_v2.sh` | 重编码修复 v2 | 🔧 |
| `w9_run_per_stage.sh` | 分阶段运行 | 🔧 |
| `w9_save_patch.py` | 添加模型保存 | 🔧 |
| `w9_stage2_epoch5.py` | Stage 2 续跑 (LoRA 无法续) | ❌ |

---

## 状态说明

| 标记 | 含义 |
|:----:|------|
| ★★ | 最终正式版, 可直接复现 |
| ★ | 核心功能代码 |
| ✅ | 正式使用过, 结果有效 |
| ❌ | 已被替代/方案失败 (保留供参考) |
| 🔧 | 调试/修复/运维脚本 (一次性使用) |
| ⚠️ | 放错位置 |

---

## 快速定位

**想复现完整实验?**
→ `w9-final/w9_train_correct.py` (4 阶段训练) + `w9-final/w9_encode_250.py` (Emu3 编码)

**想复现 Ablation?**
→ `../ablation/v1_ar_answer.py` ~ `v6_full.py`

**想理解语言解码器架构?**
→ `phase1-w1w4/language_decoder.py` + `phase1-w1w4/train_with_decoder.py`

**想看 BF-037 怎么调的?**
→ `w9-final/bf037_investigate.py` → `bf037_step1.py` → `w9_encode_250.py`

**想看 MSE→CE loss 的演进?**
→ `phase2-w5w8/w6_visual_decoder_train.py` (MSE 失败) → `w8c_train_three_stage.py` (CE 正确)
