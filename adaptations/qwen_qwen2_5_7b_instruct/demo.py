"""
Ascend NPU adaptation demo for Qwen/Qwen2.5-7B-Instruct.

Modes:
  - DRY RUN (--dry-run): random weights, layers shrunk to 2, no weight download.
  - Full Run (default): loads pretrained weights with device_map="auto".

Save output: uv run python demo.py --dry-run > output.txt 2>&1
"""

import argparse
import os
import time
from pathlib import Path

# Domestic HF mirror defaults (override via env if needed). Set before HF imports.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

try:  # register torch.npu when torch_npu is installed
    import torch_npu  # noqa: F401
except ImportError:
    pass

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Cache is pinned inside this adaptation directory (adaptation_path/models/).
CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()


def shrink_config_for_dry_run(config):
    """Conservative shrink: reduce layer count to 2 only, keep everything else."""
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, key):
            old = getattr(config, key)
            new = max(1, min(old, 2))
            setattr(config, key, new)
            if old != new:
                print(f"[Setup] Shrunk {key}: {old} -> {new}")


def pick_npu_device() -> str:
    """Pick the NPU with the most free HBM and select it via torch.npu.set_device().

    NOTE: on this host ASCEND_RT_VISIBLE_DEVICES must NOT be set (breaks aclInit),
    so card selection happens here at runtime instead.
    """
    device_count = torch.npu.device_count()
    best_idx, best_free = 0, -1
    for idx in range(device_count):
        try:
            free, total = torch.npu.mem_get_info(idx)
        except Exception:
            free, total = 0, 0
        print(f"[Device] npu:{idx} free HBM: {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB")
        if free > best_free:
            best_idx, best_free = idx, free
    torch.npu.set_device(best_idx)
    return f"npu:{best_idx}"


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def get_device():
    """Detects best device. Returns (device_str, device_count)."""
    # 1. Check for Ascend NPU
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            print("[Device] Huawei Ascend NPU detected.")
            npu_count = torch.npu.device_count()
            print(f"[Device] NPU count: {npu_count}")
            return pick_npu_device(), npu_count
    except Exception as exc:
        print(f"[Device] NPU init failed: {exc}")

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0", torch.cuda.device_count()

    # 3. Fallback to CPU (validation will reject this)
    print("[Device] No accelerator detected, using CPU.")
    return "cpu", 0


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (Qwen2.5-7B-Instruct)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (random weights, no download)")
    args = parser.parse_args()

    print(f"[Setup] Model: {MODEL_ID}")
    print(f"[Setup] Model cache: {CACHE_DIR}")

    device, _ = get_device()
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
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        finally:
            torch.set_default_dtype(old_dtype)
        model.to(device)
        print(f"[Setup] Random-weight model placed on {device}")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype="auto",
            cache_dir=CACHE_DIR,
            device_map="auto",
        )
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

    inputs = tokenizer(input_text, return_tensors="pt").to(first_device)

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
