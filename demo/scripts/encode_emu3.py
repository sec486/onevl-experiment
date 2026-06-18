# -*- coding: utf-8 -*-
"""
Emu3 视觉 Token 编码
=====================
使用 BAAI/Emu3-VisionTokenizer 将 NAVSIM 驾驶场景图像编码为离散视觉 token。
这些 token 作为训练过程中视觉辅助解码器的 ground truth 目标。

重要提示:
  - 需要 transformers==4.51.3（不是 5.4.0），否则 Emu3 模型加载失败
  - 使用 result.image_tokens[0] 获取 token ID（不是 result[0]，那会返回 float 特征）
  - 编码完成后切回: pip install transformers==5.4.0

用法:
    # 先安装正确的 transformers 版本
    pip install transformers==4.51.3
    python encode_emu3.py
    # 然后切回训练版本
    pip install transformers==5.4.0

输出:
    emu3_visual_tokens_expanded.pt — dict[场景名称 → int64 tensor of token IDs]
"""
import torch, json, time, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from PIL import Image
from transformers import Emu3VQVAE, Emu3ImageProcessor

# ============================================================
# 配置（根据你的环境修改路径）
# ============================================================
DATA_JSONL = '/opt/onevl/data/navsim_real_traj_expanded.jsonl'
OUTPUT_FILE = '/opt/onevl/data/emu3_visual_tokens_expanded.pt'
SENSOR_DIR = Path('/opt/onevl/navtrain_data/trainval_sensor_blobs/trainval')
MODEL_PATH = 'BAAI/Emu3-VisionTokenizer'  # 或本地路径
FUTURE_OFFSET = 2  # 编码哪个未来帧（相对于中点的偏移）

# ============================================================
# Main
# ============================================================
def main():
    print("Loading Emu3 VisionTokenizer...")
    vq = Emu3VQVAE.from_pretrained(MODEL_PATH, trust_remote_code=True).cuda().eval()
    iproc = Emu3ImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Loaded")

    # Get unique scenes from training JSONL
    scenes = set()
    with open(DATA_JSONL) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                scenes.add(s.get('scene', ''))
    scenes.discard('')
    print(f"Unique scenes in JSONL: {len(scenes)}")

    # Load existing valid encodings (incremental encoding)
    encoded = {}
    if os.path.exists(OUTPUT_FILE):
        encoded = torch.load(OUTPUT_FILE, weights_only=False)
        valid = {}
        for k, v in encoded.items():
            if v.dtype == torch.int64 and v.min().item() >= 0 and v.max().item() < 32768:
                valid[k] = v
        encoded = valid
        print(f"Loaded {len(encoded)} existing valid encodings")

    # Find scenes that need encoding
    available_dirs = {d.name for d in SENSOR_DIR.iterdir() if d.is_dir()}
    to_encode = [s for s in scenes if s in available_dirs and s not in encoded]
    print(f"To encode: {len(to_encode)} scenes")

    if not to_encode:
        print("Nothing to encode. All scenes already have valid tokens.")
        return

    # Encode
    errors = 0
    t0 = time.time()
    for i, scene in enumerate(sorted(to_encode)):
        try:
            cam_dir = SENSOR_DIR / scene / 'CAM_F0'
            frames = sorted(cam_dir.glob('*.jpg'))
            if len(frames) < 10:
                continue

            # Select future frame
            future_idx = min(len(frames) - 1, len(frames) // 2 + FUTURE_OFFSET)
            img = Image.open(frames[future_idx]).convert("RGB")
            processed = iproc(img, return_tensors="pt")
            pv = processed["pixel_values"].cuda()
            sizes = processed["image_sizes"]

            with torch.no_grad():
                result = vq.encode(pv, sizes)
                # BF-037 FIX: use .image_tokens[0] (int64 token IDs)
                # NOT result[0] which gives float latent features
                tokens = result.image_tokens[0].cpu()

            # Validate
            assert tokens.dtype == torch.int64, f"Expected int64, got {tokens.dtype}"
            assert tokens.min().item() >= 0, f"Negative token ID: {tokens.min()}"
            assert tokens.max().item() < 32768, f"Token ID out of range: {tokens.max()}"

            encoded[scene] = tokens

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(to_encode)}] {elapsed:.0f}s, {len(encoded)} total encoded")
                torch.save(encoded, OUTPUT_FILE)
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors += 1
            if errors <= 3:
                print(f"  OOM on {scene}, skipping")
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error {scene}: {type(e).__name__}: {str(e)[:80]}")

    # Final save
    torch.save(encoded, OUTPUT_FILE)
    elapsed = time.time() - t0

    # Validation summary
    all_min = min(t.min().item() for t in encoded.values())
    all_max = max(t.max().item() for t in encoded.values())
    dtypes = set(str(t.dtype) for t in encoded.values())

    print(f"\n{'='*50}")
    print(f"DONE: {len(encoded)} scenes encoded in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Errors: {errors}")
    print(f"dtype: {dtypes}, range: [{all_min}, {all_max}]")
    if dtypes == {'torch.int64'} and all_min >= 0 and all_max < 32768:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED — check token extraction API")


if __name__ == "__main__":
    main()
