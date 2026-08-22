"""
Ascend NPU adaptation demo for openai/clip-vit-base-patch32.

CLIP 双塔模型：图像编码器（ViT）与文本编码器（Transformer）分别产出
特征并投影到共享空间，用余弦相似度做图文匹配。验证方式：程序内合成
一张红色方块图像（PIL，无需外部数据），与候选文本逐一打分，真实权重
下要求颜色匹配的文本得分高于不匹配文本。

- Supports --dry-run: random weights + shrunken layers, no weight download.
- Full run: loads pretrained weights onto the selected accelerator.
- Model cache is pinned to this adaptation's models/ directory.

保存输出: uv run python demo.py > output.txt 2>&1
"""

import argparse
import os
import time
from pathlib import Path

# 国内网络环境默认走 HF 镜像（外部已设置的环境变量优先）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModel, CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"


def shrink_config_for_dry_run(config):
    """保守缩小：仅将层数减为 2，其余不变，保证通用兼容。

    CLIP 为复合 config：需同时收缩 text_config 与 vision_config。
    """

    def _shrink_one(cfg):
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, key):
                old = getattr(cfg, key)
                if old is None:
                    continue
                new = max(1, min(old, 2))
                setattr(cfg, key, new)
                if old != new:
                    print(f"[Setup] Shrunk {key}: {old} -> {new}")

    _shrink_one(config)
    for sub in ("text_config", "vision_config"):
        if getattr(config, sub, None) is not None:
            print(f"[Setup] Shrinking {sub}:")
            _shrink_one(getattr(config, sub))


def select_idle_npu() -> int:
    """选择空闲 HBM 最多的 NPU 并设为当前设备。

    本机禁止设置 ASCEND_RT_VISIBLE_DEVICES（会导致 aclInit error 107001），
    因此通过 torch.npu.set_device() 在多卡中选定单卡。
    """
    count = torch.npu.device_count()
    best_idx, best_free = 0, -1
    for i in range(count):
        try:
            free, _total = torch.npu.mem_get_info(i)
        except Exception:
            free = 0
        print(f"[Device] NPU {i}: free HBM {free / 1024**3:.1f} GiB")
        if free > best_free:
            best_idx, best_free = i, free
    torch.npu.set_device(best_idx)
    return best_idx


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def get_device():
    """Detects best device (NPU > CUDA > CPU). Returns device string."""
    # 1. Check for Ascend NPU
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            idx = select_idle_npu()
            print(f"[Device] Huawei Ascend NPU detected, selected npu:{idx}")
            return f"npu:{idx}"
    except ImportError:
        pass

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0"

    # 3. Fallback to CPU
    print("[Device] No accelerator detected, using CPU.")
    return "cpu"


def make_synthetic_image() -> Image.Image:
    """合成测试图像：白色背景上的红色方块（无需外部数据下载）。"""
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 184, 184], fill=(255, 0, 0))
    return img


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ascend Adaptation Demo (openai/clip-vit-base-patch32, image-text matching)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    args = parser.parse_args()

    # 缓存固定在 adaptation 目录内，避免污染项目根 models/
    cache_dir = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {cache_dir}")

    device = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # Step 1: Processor (tokenizer + image processor)
    print(f"[Setup] Loading processor for {MODEL_ID}...")
    processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)

    # Step 2: Model loading (Dry Run vs Real)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    vlayers = getattr(config.vision_config, "num_hidden_layers", "?")
    tlayers = getattr(config.text_config, "num_hidden_layers", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, vision_layers={vlayers}, text_layers={tlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers).")
        # transformers 5.x: from_config 在 Auto 类上；AutoModel 对 CLIPConfig 解析为 CLIPModel
        model = AutoModel.from_config(config)
        model = model.to(device)
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = CLIPModel.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            cache_dir=cache_dir,
        )
        model = model.to(device)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")

    model.eval()

    # Step 3: Image-text matching
    image = make_synthetic_image()
    texts = [
        "a solid red square",
        "a solid blue square",
        "a photo of a mountain landscape",
    ]
    print(f"[Run] Input: synthetic image (red square on white background, 224x224)")
    print(f"[Run] Candidate texts: {texts}")

    print("[Run] Encoding image and texts...")
    start_time = time.time()

    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        image_outputs = model.get_image_features(**image_inputs)
        text_outputs = model.get_text_features(**text_inputs)

    # transformers 5.x: get_*_features 返回完整输出对象，投影后的特征在 pooler_output；
    # 4.x 直接返回张量，这里做双向兼容。
    image_features = getattr(image_outputs, "pooler_output", image_outputs)
    text_features = getattr(text_outputs, "pooler_output", text_outputs)

    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

    assert torch.isfinite(image_features).all() and torch.isfinite(text_features).all(), "Features contain NaN/Inf"

    scores = (image_features @ text_features.T).squeeze(0).tolist()
    match_time = time.time() - start_time

    for text, score in zip(texts, scores):
        print(f"[Run] score('{text}') = {score:.4f}")
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    print(f"[Run] Output: best match = '{texts[best_idx]}' (score={scores[best_idx]:.4f})")
    print(f"[Run] Matching time: {match_time:.4f} seconds")

    if args.dry_run:
        # 随机权重下分数无意义，只验证双塔前向与相似度路径可跑通
        print("[Run] Dry-run: random weights, scores are meaningless (path validation only).")
    else:
        # 真实权重：红色方块应与“红色”文本更匹配，而不是蓝色文本
        assert scores[0] > scores[1], (
            f"Image-text matching failed with pretrained weights: "
            f"score(red)={scores[0]:.4f} <= score(blue)={scores[1]:.4f}"
        )
        print("[Run] Image-text matching correct: red square matched red text over blue text.")

    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
