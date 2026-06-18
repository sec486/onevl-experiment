#!/usr/bin/env python3
"""
变体 6: OneVL 完整版（4阶段, 双解码器）
=========================================
完整 OneVL 方法: 视觉和语言两个辅助解码器，
完整 4 阶段训练流程。

等同于 scripts/train_full.py — 在消融对比中保留此文件仅为完整性。

预期结果: ADE ~7.6m, 训练 ~93 分钟 (L4)。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# V6 is identical to the main train_full.py script.
# To run V6 in the ablation context, use:
#   python ../scripts/train_full.py
#
# Or symlink: ln -s ../scripts/train_full.py v6_full.py
#
# The ablation results for V6 are pre-computed in results/ablation_results.json

print("="*60)
print("  V6 (OneVL 完整版) = scripts/train_full.py")
print("  运行: python ../scripts/train_full.py")
print("  或使用预计算结果: results/ablation_results.json")
print("="*60)
print()
print("V6 使用与 train_full.py 完全相同的 4 阶段训练流程:")
print("  1. 预训练: 视觉解码器预训练（仅 ViT）")
print("  2. Stage 0: 轨迹 warmup")
print("  3. Stage 1: 解码器 warmup（梯度截断）")
print("  4. Stage 2: 联合训练（梯度回传）")
print()
print("预期结果: ADE=7.63m, FDE=13.39m, L4 上 ~93 分钟")
