"""
Ascend NPU adaptation demo for google-t5/t5-small (seq2seq).

T5 是 text2text (encoder-decoder) 模型：输入必须带任务前缀（如
"translate English to German: ..."），用 AutoModelForSeq2SeqLM 加载，
generate() 由 decoder 自回归产出目标序列。

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
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "google-t5/t5-small"


def shrink_config_for_dry_run(config):
    """保守缩小：仅将层数减为 2，其余不变，保证通用兼容。"""
    for key in ("num_hidden_layers", "n_layer", "num_layers", "num_decoder_layers"):
        if hasattr(config, key):
            old = getattr(config, key)
            if old is None:
                continue
            new = max(1, min(old, 2))
            setattr(config, key, new)
            if old != new:
                print(f"[Setup] Shrunk {key}: {old} -> {new}")


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


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (google-t5/t5-small, seq2seq)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    args = parser.parse_args()

    # 缓存固定在 adaptation 目录内，避免污染项目根 models/
    cache_dir = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {cache_dir}")

    device = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # Step 1: Tokenizer
    print(f"[Setup] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)

    # Step 2: Model loading (Dry Run vs Real)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    nlayers = getattr(config, "num_layers", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, encoder/decoder layers={nlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers).")
        model = AutoModelForSeq2SeqLM.from_config(config)
        model = model.to(device)
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
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

    # Step 3: Inference test (seq2seq: 任务前缀 + 源句)
    input_text = "translate English to German: The house is wonderful."
    print(f"[Run] Input: {input_text}")

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    print("[Run] Generating...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

    assert outputs is not None and outputs.shape[0] > 0, "generate() returned empty output"
    gen_time = time.time() - start_time

    print(f"[Run] Generated {outputs.shape[1]} tokens")
    print(f"[Run] Output token ids: {outputs[0].tolist()}")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not generated_text.strip():
        # 随机权重 dry-run 下 decoder 可能只产出 eos/pad，属正常现象
        generated_text = "<empty after skip_special_tokens; expected with random weights in dry-run>"

    print(f"[Run] Output: {generated_text}")
    print(f"[Run] Generation time: {gen_time:.4f} seconds")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
