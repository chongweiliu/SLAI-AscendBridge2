"""
Ascend NPU adaptation demo for timm/mobilenetv3_small_100.lamb_in1k (image classification).

timm 图像分类模型：验证方式为随机图像前向 -> logits / Top-1。
Supports DRY RUN (random weights, no download) and Full Run (real weights).
保存输出: uv run python demo.py --dry-run > output.txt 2>&1

Host notes (2x Ascend910):
- 本机严禁设置 ASCEND_RT_VISIBLE_DEVICES（会导致 aclInit error 107001），
  选卡一律使用 torch.npu.set_device()。
"""

import argparse
import os
import time
from pathlib import Path

import torch

# 缓存固定在本 adaptation 内，避免污染项目根 models/
CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()

# HuggingFace 镜像（国内直连 huggingface.co 不通）；timm hf-hub 权重下载也走这里
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", CACHE_DIR)

import timm  # noqa: E402

MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"
ARCH = "mobilenetv3_small_100"  # timm registry 架构名（.lamb_in1k 是预训练 tag）
NUM_CLASSES = 1000
INPUT_SIZE = (3, 224, 224)
# ImageNet 归一化（来自模型 pretrained_cfg: mean/std）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def get_device(preferred_index: int = 0):
    """Detects best device. Returns device string like 'npu:0' / 'cuda:0' / 'cpu'."""
    # 1. Check for Ascend NPU
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            print("[Device] Huawei Ascend NPU detected.")
            print(f"[Device] NPU count: {count}")
            index = preferred_index if preferred_index < count else 0
            # 本机严禁 ASCEND_RT_VISIBLE_DEVICES，用 set_device 选卡
            torch.npu.set_device(index)
            return f"npu:{index}"
    except ImportError:
        pass

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0"

    # 3. Fallback to CPU
    print("[Device] No accelerator detected, using CPU.")
    return "cpu"


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MobileNetV3-Small (timm) Ascend Adaptation Demo")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    parser.add_argument("--npu-index", type=int, default=0, help="Preferred NPU index (选卡前先 npu-smi info)")
    args = parser.parse_args()

    print(f"[Setup] Model cache: {CACHE_DIR}")

    device = get_device(preferred_index=args.npu_index)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Model Loading (Dry Run vs Real)
    # ============================================================
    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        print(f"[Setup] Creating timm model: {ARCH} (pretrained=False, no download)")
        model = timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES)
        model = model.to(device)
        print(f"[Setup] Model placed on {device}")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        print(f"[Setup] Creating timm model: hf-hub:{MODEL_ID} (pretrained=True)")
        model = timm.create_model(f"hf-hub:{MODEL_ID}", pretrained=True, num_classes=NUM_CLASSES)
        model = model.to(device)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Setup] Model: {ARCH}, params: {n_params:.2f}M, num_classes={NUM_CLASSES}")

    model.eval()

    # ============================================================
    # Step 2: Forward with random image -> logits / Top-1
    # ============================================================
    torch.manual_seed(0)
    x = torch.rand(1, *INPUT_SIZE, device=device)  # 随机图像 [0,1]
    x = normalize_imagenet(x)
    print(f"[Run] Input: random image tensor {tuple(x.shape)} (ImageNet-normalized)")

    print("[Run] Forward pass...")
    start_time = time.time()

    with torch.no_grad():
        logits = model(x)

    assert logits is not None and logits.dim() == 2, "Classification forward returned invalid logits"
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"
    fwd_time = time.time() - start_time
    print(f"[Run] Logits shape: {tuple(logits.shape)} (batch, num_classes), all finite")

    top5_vals, top5_ids = torch.topk(logits[0], k=5)
    top1_id = int(top5_ids[0])
    top1_val = float(top5_vals[0])

    print(f"[Run] Output: Top-1 class index={top1_id} (logit={top1_val:.4f})")
    print(f"[Run] Top-5 class indices: {[int(i) for i in top5_ids]}")
    print(f"[Run] Forward time: {fwd_time:.4f} seconds")
    if args.dry_run:
        print("[Run] Note: dry-run uses random weights, predicted class is meaningless by design.")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
