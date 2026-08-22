"""
Ascend NPU adaptation demo for Qwen/Qwen2.5-VL-7B-Instruct.

VLM (image-text-to-text): Qwen2_5_VLForConditionalGeneration + AutoProcessor.
Supports DRY RUN (random weights, no weight download) and Full Run (real weights).

保存输出: uv run python demo.py --dry-run > output.txt 2>&1
"""

import argparse
import os
import time
from pathlib import Path

# 国内网络：HuggingFace 官方源直连不通，统一走镜像；禁用 Xet 协议（镜像不支持）。
# 使用 setdefault，外部显式设置的环境变量优先。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch

try:
    import torch_npu  # noqa: F401  # 注册 torch.npu
except ImportError:
    pass

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def shrink_config_for_dry_run(config):
    """保守缩小：文本塔层数与视觉塔深度均减为 2，其余不变，保证架构路径一致。"""
    old = getattr(config, "num_hidden_layers", None)
    if old is not None:
        new = max(1, min(old, 2))
        if new != old:
            config.num_hidden_layers = new
            print(f"[Setup] Shrunk num_hidden_layers: {old} -> {new}")
    vision = getattr(config, "vision_config", None)
    if vision is not None and hasattr(vision, "depth"):
        old_v = vision.depth
        new_v = max(1, min(old_v, 2))
        if new_v != old_v:
            vision.depth = new_v
            print(f"[Setup] Shrunk vision_config.depth: {old_v} -> {new_v}")


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def get_device():
    """Detects best device. Returns (device_str, device_count).

    本机约定：严禁设置 ASCEND_RT_VISIBLE_DEVICES（会导致 aclInit error 107001），
    多卡时用 torch.npu.set_device() 选卡；挑当前空闲显存最多的卡，避免与其他任务抢卡。
    """
    # 1. Ascend NPU
    if hasattr(torch, "npu") and torch.npu.is_available():
        count = torch.npu.device_count()
        print("[Device] Huawei Ascend NPU detected.")
        print(f"[Device] NPU count: {count}")
        best, best_free = 0, -1
        for i in range(count):
            free, total = torch.npu.mem_get_info(i)
            print(f"[Device] npu:{i} free={free / 1e9:.1f}GB / total={total / 1e9:.1f}GB")
            if free > best_free:
                best, best_free = i, free
        torch.npu.set_device(best)
        return f"npu:{best}", count

    # 2. NVIDIA CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0", torch.cuda.device_count()

    # 3. CPU fallback（验收不允许，后续断言会拦截）
    print("[Device] No accelerator detected, using CPU.")
    return "cpu", 0


def make_demo_image() -> Image.Image:
    """生成一张本地测试图（红色方块 + 渐变背景），不依赖任何外部下载。"""
    img = Image.new("RGB", (448, 448))
    px = img.load()
    for x in range(448):
        for y in range(448):
            px[x, y] = (x % 256, y % 256, (x + y) % 256)
    for x in range(150, 300):
        for y in range(150, 300):
            px[x, y] = (255, 0, 0)
    return img


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (Qwen2.5-VL)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    # 缓存固定在本 adaptation 的 models/ 下，避免写入项目根或其他目录
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    device, _ = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Processor (tokenizer + image processor, VLM 必备)
    # ============================================================
    print(f"[Setup] Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # ============================================================
    # Step 2: Model Loading (Dry Run vs Real)
    # ============================================================
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print(
        f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, "
        f"layers={getattr(config, 'num_hidden_layers', '?')}, "
        f"vision_depth={getattr(config.vision_config, 'depth', '?')}"
    )

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        # transformers>=4.57 将公开的 from_config 改名为 _from_config，做兼容
        if hasattr(Qwen2_5_VLForConditionalGeneration, "from_config"):
            model = Qwen2_5_VLForConditionalGeneration.from_config(config, torch_dtype=torch.bfloat16)
        else:
            model = Qwen2_5_VLForConditionalGeneration._from_config(config, torch_dtype=torch.bfloat16)
        model.to(device)
        print("[Setup] Random-weight model created and moved to device.")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        # 7B bf16 ~16GB，单卡 64GB 足够；不用 device_map="auto"，避免跨卡与并行任务互相干扰
        model.to(device)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")

    model.eval()

    # ============================================================
    # Step 3: Multimodal Inference Test (image + text -> text)
    # ============================================================
    image = make_demo_image()
    user_text = "Describe this image briefly."
    print(f"[Run] Input: image {image.size} (RGB gradient + red square) + text: {user_text}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(first_device)
    print(f"[Run] Processed inputs: input_ids={tuple(inputs.input_ids.shape)}, pixel_values present={'pixel_values' in inputs}")

    print("[Run] Generating...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_time = time.time() - start_time

    assert outputs is not None and outputs.shape[0] > 0, "generate() returned empty output"
    new_tokens = outputs[0, inputs.input_ids.shape[1]:]
    assert new_tokens.numel() > 0, "generate() produced no new tokens"
    generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"[Run] Generated {new_tokens.numel()} new tokens")
    print(f"[Run] Output: {generated_text}")
    print(f"[Run] Generation time: {gen_time:.4f} seconds")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
