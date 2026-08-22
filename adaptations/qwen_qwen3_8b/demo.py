"""
Ascend NPU adaptation demo for Qwen/Qwen3-8B.

Generated from .claude/skills/ascend-adaptation/templates/demo.py.j2 with
host-specific adjustments:
- HF mirror defaults (direct huggingface.co is unreachable on this host)
- NPU card selection via torch.npu.set_device() (this host MUST NOT set
  ASCEND_RT_VISIBLE_DEVICES; it triggers aclInit error 107001)

Supports DRY RUN (random weights) and Full Run (real weights).
保存输出: uv run python demo.py --dry-run > output.txt 2>&1
"""

import argparse
import os
import time
from pathlib import Path

import torch

# HuggingFace mirror defaults for CN environments (overridable).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


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
def pick_npu_index(count):
    """选卡：优先 NPU_DEVICE_ID 环境变量；否则选空闲 HBM 最多的卡（避免写死 0 号卡）。"""
    env = os.environ.get("NPU_DEVICE_ID")
    if env is not None and env.strip() != "":
        idx = int(env)
        if 0 <= idx < count:
            print(f"[Device] NPU_DEVICE_ID={idx} requested via env")
            return idx
    best, best_free = 0, -1
    for i in range(count):
        try:
            free, total = torch.npu.mem_get_info(i)
        except Exception:
            free, total = 0, 0
        print(f"[Device] NPU {i}: free HBM {free / 1024**3:.1f} GiB / total {total / 1024**3:.1f} GiB")
        if free > best_free:
            best, best_free = i, free
    return best


def get_device():
    """Detects best device (NPU > CUDA > CPU). Returns (device_str, device_count)."""
    # 1. Check for Ascend NPU
    try:
        import torch_npu  # noqa: F401  # registers torch.npu

        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            print("[Device] Huawei Ascend NPU detected.")
            print(f"[Device] NPU count: {count}")
            idx = pick_npu_index(count)
            torch.npu.set_device(idx)
            print(f"[Device] Selected NPU device: npu:{idx}")
            return f"npu:{idx}", count
    except ImportError:
        pass

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0", torch.cuda.device_count()

    # 3. Fallback to CPU
    print("[Device] No accelerator detected, using CPU.")
    return "cpu", 0


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (Qwen/Qwen3-8B)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    args = parser.parse_args()

    MODEL_ID = "Qwen/Qwen3-8B"
    # Best practice: cache under this adaptation so downloads stay local and reproducible
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    from accelerate import dispatch_model, infer_auto_device_map
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device, _count = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Tokenizer (Always needed)
    # ============================================================
    print(f"[Setup] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)

    # ============================================================
    # Step 2: Config + Model Loading (Dry Run vs Real)
    # ============================================================
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    nlayers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, layers={nlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers/size).")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        device_map = infer_auto_device_map(model)
        # 多卡环境限制单卡：本机严禁 ASCEND_RT_VISIBLE_DEVICES，直接强制所有模块落在所选卡上
        device_map = {name: device for name in device_map}
        model = dispatch_model(model, device_map=device_map)
        print(f"[Setup] Model dispatched to single device: {device}")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype="auto",
            cache_dir=CACHE_DIR,
            device_map="auto",
        )
        if device.startswith("npu"):
            # 限制只使用所选单卡（其余 NPU 设 0GiB，避免 device_map="auto" 跨卡）
            idx = int(device.split(":")[1])
            count = torch.npu.device_count()
            max_memory = {i: ("0GiB" if i != idx else "60GiB") for i in range(count)}
            max_memory["cpu"] = "64GiB"
            load_kwargs["max_memory"] = max_memory
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
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
    end_time = time.time()
    gen_time = end_time - start_time
    print(f"[Run] Generated {outputs.shape[1]} tokens")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"[Run] Output: {generated_text}")
    print(f"[Run] Generation time: {gen_time:.4f} seconds")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
