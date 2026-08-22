"""
Ascend NPU adaptation demo for Qwen/Qwen2.5-1.5B-Instruct.

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

# HuggingFace 镜像（国内直连 huggingface.co 不通）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def shrink_config_for_dry_run(config):
    """保守缩小：仅将层数减为 2，其余不变，保证通用兼容。"""
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, key):
            old = getattr(config, key)
            new = max(1, min(old, 2))
            setattr(config, key, new)
            if old != new:
                print(f"[Setup] Shrunk {key}: {old} -> {new}")


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


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B-Instruct Ascend Adaptation Demo")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    parser.add_argument("--npu-index", type=int, default=0, help="Preferred NPU index (选卡前先 npu-smi info)")
    args = parser.parse_args()

    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    # 缓存固定在本 adaptation 内，避免污染项目根 models/
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    device = get_device(preferred_index=args.npu_index)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Tokenizer
    # ============================================================
    print(f"[Setup] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)

    # ============================================================
    # Step 2: Model Loading (Dry Run vs Real)
    # ============================================================
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    nlayers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, layers={nlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers).")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(device)
        print(f"[Setup] Model placed on {device}")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        model = model.to(device)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")

    model.eval()

    # ============================================================
    # Step 3: Inference Test
    # ============================================================
    input_text = "Hello, this is a test run on Huawei Ascend NPU."
    print(f"[Run] Input: {input_text}")

    inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)

    print("[Run] Generating...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    assert outputs is not None and outputs.shape[0] > 0, "generate() returned empty output"
    gen_time = time.time() - start_time
    print(f"[Run] Generated {outputs.shape[1]} tokens")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"[Run] Output: {generated_text}")
    print(f"[Run] Generation time: {gen_time:.4f} seconds")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
