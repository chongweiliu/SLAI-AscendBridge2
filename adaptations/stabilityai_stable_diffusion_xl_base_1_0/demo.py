"""
Ascend NPU adaptation demo for stabilityai/stable-diffusion-xl-base-1.0.

diffusers 文生图 pipeline: StableDiffusionXLPipeline
(unet + text_encoder + text_encoder_2 + vae + scheduler + tokenizers)。

- DRY RUN: 随机权重（仅下载各组件 config.json 与 tokenizer 小文件），
  64x64、1-2 步走通完整 pipeline；不下载 6.9GB 权重。
- Full Run: 加载真实权重（建议 variant="fp16"），1 步小图验证。

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

from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


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


def build_dry_run_pipeline(cache_dir: str, dtype: torch.dtype):
    """随机权重构建 SDXL pipeline：仅下载各组件 config.json 与 tokenizer 小文件。"""
    print("[Setup] DRY RUN: loading component configs (no weight download)...")
    unet_cfg = UNet2DConditionModel.load_config(MODEL_ID, subfolder="unet", cache_dir=cache_dir)
    unet = UNet2DConditionModel.from_config(unet_cfg)

    vae_cfg = AutoencoderKL.load_config(MODEL_ID, subfolder="vae", cache_dir=cache_dir)
    vae = AutoencoderKL.from_config(vae_cfg)

    te_cfg = CLIPTextConfig.from_pretrained(MODEL_ID, subfolder="text_encoder", cache_dir=cache_dir)
    text_encoder = CLIPTextModel(te_cfg)

    te2_cfg = CLIPTextConfig.from_pretrained(MODEL_ID, subfolder="text_encoder_2", cache_dir=cache_dir)
    text_encoder_2 = CLIPTextModelWithProjection(te2_cfg)

    scheduler_cfg = EulerDiscreteScheduler.load_config(MODEL_ID, subfolder="scheduler", cache_dir=cache_dir)
    scheduler = EulerDiscreteScheduler.from_config(scheduler_cfg)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", cache_dir=cache_dir)
    tokenizer_2 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2", cache_dir=cache_dir)

    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
    )
    return pipe


def build_full_pipeline(cache_dir: str, dtype: torch.dtype):
    """真实权重加载（优先 fp16 variant，体积约 6.9GB）。"""
    print("[Setup] Attempting to load pre-trained weights (variant=fp16)...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            variant="fp16",
            cache_dir=cache_dir,
        )
    except Exception as exc:  # 个别镜像缺 fp16 variant 时退回默认权重
        print(f"[Setup] fp16 variant unavailable ({exc}); falling back to default weights")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
    return pipe


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (SDXL text-to-image)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    parser.add_argument("--steps", type=int, default=None, help="num_inference_steps (default: 2 dry / 1 full)")
    parser.add_argument("--height", type=int, default=None, help="output height (default: 64 dry / 512 full)")
    parser.add_argument("--width", type=int, default=None, help="output width (default: 64 dry / 512 full)")
    args = parser.parse_args()

    if os.environ.get("DRY_RUN") == "1":
        args.dry_run = True

    # 缓存固定在本 adaptation 的 models/ 下，避免写入项目根或其他目录
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    device, _ = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    dtype = torch.float16

    # ============================================================
    # Step 1: Build Pipeline
    # ============================================================
    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: random weights, minimal steps/size.")
        pipe = build_dry_run_pipeline(CACHE_DIR, dtype)
        steps, height, width = args.steps or 2, args.height or 64, args.width or 64
    else:
        pipe = build_full_pipeline(CACHE_DIR, dtype)
        steps, height, width = args.steps or 1, args.height or 512, args.width or 512

    # 位置参数 (device, dtype) 在新旧版 diffusers 的 Pipeline.to() 中均生效；
    # 注意新版 (>=0.39) 关键字参数是 dtype=/device=，torch_dtype= 会被静默忽略。
    pipe.to(device, dtype)
    pipe.set_progress_bar_config(disable=True)

    unet_param = next(pipe.unet.parameters())
    assert unet_param.device.type in ("npu", "cuda"), f"UNet must be on NPU or CUDA, got {unet_param.device}"
    assert unet_param.dtype == dtype, f"UNet dtype mismatch: {unet_param.dtype} != {dtype}"
    print(f"[Setup] Pipeline on device: {unet_param.device} (dtype={unet_param.dtype})")

    # ============================================================
    # Step 2: Generation
    # ============================================================
    prompt = "a photo of an astronaut riding a horse on mars"
    print(f"[Run] Input: prompt='{prompt}', steps={steps}, size={height}x{width}")

    # offload/多设备场景下 CPU generator 更稳（本例整管线单卡，保持一致口径）
    generator = torch.Generator(device="cpu").manual_seed(42)

    print("[Run] Generating...")
    start_time = time.time()
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            generator=generator,
        )
    gen_time = time.time() - start_time

    image = output.images[0]
    assert image is not None and image.size == (width, height), f"Unexpected image size: {getattr(image, 'size', None)}"
    out_path = Path(__file__).resolve().parent / ("sample_dry_run.png" if args.dry_run else "sample_full.png")
    image.save(out_path)

    print(f"[Run] Output: image {image.size} saved to {out_path.name}")
    print(f"[Run] Generation time: {gen_time:.4f} seconds")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
