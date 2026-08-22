"""
Ascend NPU adaptation demo for Qwen/Qwen3.5-9B (multimodal, qwen3_5).

架构: Qwen3_5ForConditionalGeneration — 视觉编码器 + 混合注意力文本主干
(3/4 linear_attention + 1/4 full_attention, mrope)。
本 demo 验证文本生成路径（text-only generation），覆盖文本主干全部算子。

Supports DRY RUN (random weights, no weight download) and Full Run (real weights).
保存输出: uv run python demo.py --dry-run > output.txt 2>&1

Note (本机环境): 不设置 ASCEND_RT_VISIBLE_DEVICES（本机会触发 aclInit error 107001），
改用 torch.npu.set_device() 选择空闲单卡（先用 `npu-smi info` 查看占用）。
"""

import argparse
import os
import time
from pathlib import Path

# 国内网络环境：HuggingFace 镜像 + 禁用 Xet（直连 huggingface.co 会挂死）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoConfig, AutoTokenizer

# Qwen3.5 是多模态 (image-text-to-text) 模型；主类为 AutoModelForImageTextToText
# （映射到 Qwen3_5ForConditionalGeneration）。旧版本回退到 AutoModelForVision2Seq。
try:
    from transformers import AutoModelForImageTextToText as AutoModelCls
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoModelCls


def shrink_config_for_dry_run(config):
    """保守缩小：文本主干层数减为 4（保留 1 个 full_attention 层），视觉塔深度减为 2。"""
    text_cfg = getattr(config, "text_config", None)
    targets = [text_cfg] if text_cfg is not None else [config]
    for cfg in targets:
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, key):
                old = getattr(cfg, key)
                new = max(1, min(old, 4))
                setattr(cfg, key, new)
                if old != new:
                    print(f"[Setup] Shrunk text {key}: {old} -> {new}")
        # layer_types 与层数保持一致（混合注意力逐层类型），截断到前 new 层
        layer_types = getattr(cfg, "layer_types", None)
        n = getattr(cfg, "num_hidden_layers", None)
        if isinstance(layer_types, list) and n is not None and len(layer_types) > n:
            setattr(cfg, "layer_types", layer_types[:n])
            print(f"[Setup] Truncated layer_types: {len(layer_types)} -> {n}")
    vision_cfg = getattr(config, "vision_config", None)
    if vision_cfg is not None and hasattr(vision_cfg, "depth"):
        old = vision_cfg.depth
        vision_cfg.depth = max(1, min(old, 2))
        if old != vision_cfg.depth:
            print(f"[Setup] Shrunk vision depth: {old} -> {vision_cfg.depth}")


# ============================================================
# Device Selection Logic
# ============================================================
def get_device(device_index: int = 0):
    """Detects best device (NPU > CUDA > CPU). Returns (device_str, device_count)."""
    # 1. Check for Ascend NPU
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            npu_count = torch.npu.device_count()
            print("[Device] Huawei Ascend NPU detected.")
            print(f"[Device] NPU count: {npu_count}")
            # 多卡共享环境：限制单卡，用 set_device 选卡（不用 ASCEND_RT_VISIBLE_DEVICES）
            torch.npu.set_device(device_index)
            return f"npu:{device_index}", npu_count
    except ImportError:
        pass

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        torch.cuda.set_device(device_index)
        return f"cuda:{device_index}", torch.cuda.device_count()

    # 3. Fallback to CPU
    print("[Device] No accelerator detected, using CPU.")
    return "cpu", 0


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo: Qwen/Qwen3.5-9B")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    parser.add_argument("--device-index", type=int, default=0, help="NPU/CUDA device index (default: 0)")
    args = parser.parse_args()

    MODEL_ID = "Qwen/Qwen3.5-9B"
    # 缓存固定在 adaptation 目录内，避免污染项目根 models/
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    device, _ = get_device(args.device_index)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Tokenizer（文本路径；视觉输入需 AutoProcessor + pillow）
    # ============================================================
    print(f"[Setup] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)

    # ============================================================
    # Step 2: Model Loading (Dry Run vs Real)
    # ============================================================
    # --dry-run: 随机权重 + 缩小层数，快速验证架构与代码路径，不下载权重。
    # 无 --dry-run: 加载预训练权重（9B, bf16 约 18GB，单卡可载）。
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    text_cfg = getattr(config, "text_config", config)
    nlayers = getattr(text_cfg, "num_hidden_layers", None) or getattr(text_cfg, "n_layer", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, "
          f"arch={getattr(config, 'architectures', '?')}, text_layers={nlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers).")
        model = AutoModelCls.from_config(config, trust_remote_code=True)
        model = model.to(device)
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = AutoModelCls.from_pretrained(
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
    # Step 3: Inference Test（文本生成路径）
    # ============================================================
    input_text = "Hello, this is a test run on Huawei Ascend NPU."
    print(f"[Run] Input: {input_text}")

    inputs = tokenizer(input_text, return_tensors="pt").to(first_device)

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
