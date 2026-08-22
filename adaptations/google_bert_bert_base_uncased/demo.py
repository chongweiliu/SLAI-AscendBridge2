"""
Ascend NPU adaptation demo for google-bert/bert-base-uncased (masked LM, 非生成式).

验证方式：[MASK] token 填空（masked token prediction）+ 前向 logits 校验。

Host-specific adjustments:
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

MODEL_ID = "google-bert/bert-base-uncased"


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
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (google-bert/bert-base-uncased, masked LM)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    args = parser.parse_args()

    # Best practice: cache under this adaptation so downloads stay local and reproducible
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    from accelerate import dispatch_model, infer_auto_device_map
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    device, _count = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Tokenizer (Always needed)
    # ============================================================
    print(f"[Setup] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # ============================================================
    # Step 2: Config + Model Loading (Dry Run vs Real)
    # ============================================================
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    nlayers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, layers={nlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers/size).")
        model = AutoModelForMaskedLM.from_config(config)
        device_map = infer_auto_device_map(model)
        # 多卡环境限制单卡：本机严禁 ASCEND_RT_VISIBLE_DEVICES，直接强制所有模块落在所选卡上
        device_map = {name: device for name in device_map}
        model = dispatch_model(model, device_map=device_map)
        print(f"[Setup] Model dispatched to single device: {device}")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        load_kwargs = dict(
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
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, **load_kwargs)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")

    model.eval()

    # ============================================================
    # Step 3: Masked Token Prediction (fill-mask)
    # ============================================================
    input_text = "The capital of France is [MASK]."
    print(f"[Run] Input: {input_text}")

    inputs = tokenizer(input_text, return_tensors="pt").to(first_device)
    mask_pos = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)

    print("[Run] Forward pass + masked token prediction...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    end_time = time.time()
    fwd_time = end_time - start_time

    # 前向 logits 校验
    expected_shape = (inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], config.vocab_size)
    assert tuple(logits.shape) == expected_shape, f"Unexpected logits shape: {logits.shape} vs {expected_shape}"
    print(f"[Run] Logits shape OK: {tuple(logits.shape)}")

    top_k = 5
    mask_logits = logits[0, mask_pos]
    top = torch.topk(mask_logits, k=top_k)
    preds = [(tokenizer.decode(tok), float(v)) for tok, v in zip(top.indices, top.values)]
    best_token = preds[0][0]

    print(f"[Run] Top-{top_k} predictions for [MASK]:")
    for rank, (tok, val) in enumerate(preds, 1):
        print(f"[Run]   {rank}. {tok!r} (logit={val:.4f})")

    print(f"[Run] Output: The capital of France is {best_token}.")
    print(f"[Run] Forward time: {fwd_time:.4f} seconds")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
