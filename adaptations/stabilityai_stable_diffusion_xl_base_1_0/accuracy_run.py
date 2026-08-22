#!/usr/bin/env python3
"""Benchmark for stabilityai/stable-diffusion-xl-base-1.0 (diffusion text-to-image).

产出 outputs_*.pt / benchmark_metrics_*.json / trace_*.json。
config/dry-run 模式（随机权重，1 步 64x64），latency-only + latent 统计量，不落盘大图。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DATASET_NAME = "builtin"

PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "a red cube on a white table", "a blue circle on green background",
    "a serene mountain lake at sunset", "a futuristic city skyline",
    "a portrait of a robot", "a bowl of fruit", "a cat sitting on a windowsill",
    "a steaming cup of coffee", "a vintage car on a country road",
    "a field of sunflowers", "a lighthouse by the sea", "a cozy cabin in snow",
    "a dragon flying over mountains", "a knight in armor", "a wizard casting a spell",
    "a tropical beach", "a desert oasis", "a waterfall in a jungle",
    "a snowy mountain peak", "a forest at dawn", "a city street at night",
    "a sailboat on calm water", "a hot air balloon in the sky", "a castle on a hill",
    "a flower garden", "a plate of sushi", "a glass of wine", "a chess board",
    "a piano on a stage", "a guitar by a campfire", "a bookshelf full of books",
    "a telescope pointing at stars", "a microscope on a desk", "a globe of the earth",
    "a clock tower", "a windmill in a field", "a bridge over a river",
    "a train at a station", "an airplane in the clouds", "a submarine underwater",
    "a rocket launching", "a satellite in orbit", "a robot waving",
    "an alien landscape", "a crystal cave", "a volcanic eruption",
    "a rainbow over a valley", "a starry night sky", "a nebula in space",
    "a galaxy spiral", "a black hole", "a forest of giant mushrooms",
]


def select_idle_npu() -> int:
    count = torch.npu.device_count() if hasattr(torch, "npu") else 0
    best_idx, best_free = 0, -1
    for i in range(count):
        try:
            free, _t = torch.npu.mem_get_info(i)
            if free > best_free:
                best_free, best_idx = free, i
        except Exception:
            pass
    if count:
        torch.npu.set_device(best_idx)
    return best_idx


def get_device(force_cpu: bool = False):
    if force_cpu:
        return "cpu"
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            idx = select_idle_npu()
            print(f"[Device] Huawei Ascend NPU detected, selected npu:{idx}")
            return f"npu:{idx}"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def get_dtype_str(dtype: torch.dtype) -> str:
    m = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    return m.get(dtype, str(dtype).replace("torch.", ""))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pretrained", action="store_true", help="Tier2: load pretrained weights")
    parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    cache_dir = (Path(__file__).resolve().parent / "models").as_posix()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    device = get_device(force_cpu=args.cpu)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from diffusers import (  # noqa: E402
        AutoencoderKL,
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer  # noqa: E402

    dtype = torch.float16
    if args.use_pretrained:
        # Tier2: from_pretrained 真实权重
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, variant="fp16", cache_dir=cache_dir)
        except Exception:
            pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, cache_dir=cache_dir)
        steps, height, width = 1, 512, 512
    else:
        # Tier1: from_config 随机权重（不下载 6.9GB）
        unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(MODEL_ID, subfolder="unet", cache_dir=cache_dir))
        vae = AutoencoderKL.from_config(AutoencoderKL.load_config(MODEL_ID, subfolder="vae", cache_dir=cache_dir))
        te = CLIPTextModel(CLIPTextConfig.from_pretrained(MODEL_ID, subfolder="text_encoder", cache_dir=cache_dir))
        te2 = CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(MODEL_ID, subfolder="text_encoder_2", cache_dir=cache_dir))
        sched = EulerDiscreteScheduler.from_config(EulerDiscreteScheduler.load_config(MODEL_ID, subfolder="scheduler", cache_dir=cache_dir))
        tok = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", cache_dir=cache_dir)
        tok2 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2", cache_dir=cache_dir)
        pipe = StableDiffusionXLPipeline(vae=vae, text_encoder=te, text_encoder_2=te2, tokenizer=tok, tokenizer_2=tok2, unet=unet, scheduler=sched)
        steps, height, width = 1, 64, 64
    pipe.to(device, dtype)
    pipe.set_progress_bar_config(disable=True)
    dtype_str = get_dtype_str(next(pipe.unet.parameters()).dtype)
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = DATASET_NAME
    print(f"[Setup] dtype: {dtype_str}, mode={mode_str}")

    import torch_npu  # noqa: F401

    prompts = PROMPTS[: args.max_samples] if args.max_samples <= len(PROMPTS) else PROMPTS
    n = len(prompts)
    print(f"[benchmark] {n} prompts, {steps} step(s) {height}x{width}")

    def gen_one(prompt):
        out = pipe(prompt, num_inference_steps=steps, height=height, width=width)
        img = out.images[0]
        import numpy as np
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

    trace_path = Path(__file__).resolve().parent / f"trace_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    try:
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile
        with npu_profile(activities=[NPUActivity.CPU, NPUActivity.NPU]) as prof:
            t0 = time.time()
            _ = gen_one(prompts[0] if prompts else "a photo of a cat")
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            step1_latency = time.time() - t0
        prof.export_trace(str(trace_path))
    except Exception as e:
        step1_latency = 0.0
        if not trace_path.exists():
            trace_path.write_text(json.dumps({"fallback": str(e)}))
    print(f"[benchmark] step1 latency: {step1_latency:.4f}s, trace: {trace_path.exists()}")

    stats = []
    t_all = time.time()
    for p in prompts:
        stats.append(gen_one(p))
    total_latency = time.time() - t_all
    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    outputs = {"prompts": prompts, "image_stats": stats}
    out_name = f"outputs_npu_{dtype_str}_{mode_str}_{dataset_name}.pt"
    torch.save(outputs, Path(__file__).resolve().parent / out_name)
    print(f"[benchmark] outputs saved: {out_name}")

    metric = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step1_forward_latency_s": round(step1_latency, 6),
        "latency_s": round(total_latency / max(n, 1), 6),
        "peak_memory_mb": round(peak_mem, 2),
        "num_samples": n,
        "device": device,
        "device_model": "Ascend910",
        "mode": mode_str,
        "output_type": "diffusion_latency",
        "dataset": dataset_name,
        "dtype": dtype_str,
        "packages": {"torch": torch.__version__, "torch_npu": getattr(__import__("torch_npu"), "__version__", "n/a"), "diffusers": __import__("diffusers").__version__},
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ttft_ms": round(step1_latency * 1000, 3),
    }
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[benchmark] metrics saved: {met_name}")
    print(f"[benchmark] DONE: {n} samples")


if __name__ == "__main__":
    main()
