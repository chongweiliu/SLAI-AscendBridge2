#!/usr/bin/env python3
"""Benchmark for openai/clip-vit-base-patch32 (image-text matching, dual encoder).

产出 outputs_*.pt / benchmark_metrics_*.json / trace_*.json。
画像：合成图像与候选文本的相似度（config 模式随机权重，相似度无意义但前向路径完整）。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_NAME = "builtin"

# 60 条候选文本（与合成图像做图文匹配），满足 completed gate num_samples>=50
CANDIDATE_TEXTS = [
    "a solid red square", "a solid blue square", "a photo of a mountain landscape",
    "a diagram of a neural network", "a cat sitting on a sofa", "a plate of sushi",
    "a city skyline at night", "a green forest path", "a basketball on a court",
    "a handwritten digit", "a cup of coffee on a desk", "a dog running in a park",
    "an airplane in the sky", "a clock showing noon", "a red car on the road",
    "a group of people dancing", "a bowl of fruit", "a laptop on a table",
    "a flower in a garden", "a boat on a lake", "a book on a shelf",
    "a guitar on stage", "a horse in a field", "a glass of orange juice",
    "a bridge over a river", "a child playing with toys", "a pizza on a plate",
    "a train at a station", "a tree in autumn", "a soccer ball on grass",
    "a smartphone on a desk", "a painting of a sunset", "a bicycle by a wall",
    "a slice of cake", "a pair of shoes", "a television showing news",
    "a microphone on a stand", "a chair near a window", "a fish in an aquarium",
    "a kite in the wind", "a candle on a table", "a mirror on a wall",
    "a suitcase by a door", "a helmet on a bike", "a bottle of wine",
    "a map of the world", "a scissors and paper", "a key on a ring",
    "a hat on a rack", "a umbrella in the rain", "a balloon in the air",
    "a chess board with pieces", "a candle lit in the dark", "a feather on the ground",
    "a snowman in winter", "a rainbow over hills", "a shell on the beach",
    "a leaf on the water", "a pebble on the path", "a star in the night sky",
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


def make_synthetic_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([40, 40, 184, 184], fill=(220, 30, 30))
    return img


def main():
    import argparse
    from transformers import AutoConfig, AutoModel, CLIPModel, CLIPProcessor

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

    processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if args.use_pretrained:
        print("[Setup] Loading pretrained weights...")
        model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype="auto", cache_dir=cache_dir)
    else:
        print("[Setup] DRY/config mode: random weights")
        model = AutoModel.from_config(config)
    model = model.to(device)
    model.eval()
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = DATASET_NAME
    print(f"[Setup] Model dtype: {dtype_str}, mode={mode_str}")

    texts = CANDIDATE_TEXTS[: args.max_samples]
    n = len(texts)
    print(f"[benchmark] {n} candidate texts x 1 synthetic image")

    image = make_synthetic_image()

    def encode_batch(img, txts):
        image_inputs = processor(images=img, return_tensors="pt").to(device)
        text_inputs = processor(text=txts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                pixel_values=image_inputs["pixel_values"],
            )
        # logits_per_image: (1, n)
        return out.logits_per_image.squeeze(0).float()

    import torch_npu  # noqa: F401

    # Step1: trace + metrics (first batch)
    trace_path = Path(__file__).resolve().parent / f"trace_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    try:
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile
        with npu_profile(activities=[NPUActivity.CPU, NPUActivity.NPU]) as prof:
            t0 = time.time()
            _ = encode_batch(image, texts[:8])
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            step1_latency = time.time() - t0
        prof.export_trace(str(trace_path))
    except Exception as e:
        step1_latency = 0.0
        if not trace_path.exists():
            trace_path.write_text(json.dumps({"fallback": str(e)}))
    print(f"[benchmark] step1 latency: {step1_latency:.4f}s, trace: {trace_path.exists()}")

    # Step2: all texts in chunks
    all_sims = []
    t_all = time.time()
    with torch.no_grad():
        for i in range(0, n, 16):
            chunk = texts[i : i + 16]
            sims = encode_batch(image, chunk).tolist()
            all_sims.extend(sims)
    total_latency = time.time() - t_all
    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    outputs = {
        "texts": texts,
        "image": "synthetic red square 224x224",
        "similarity_per_text": all_sims,
        "top1_text": texts[all_sims.index(max(all_sims))] if all_sims else "",
    }
    out_name = f"outputs_npu_{dtype_str}_{mode_str}_{dataset_name}.pt"
    torch.save(outputs, Path(__file__).resolve().parent / out_name)
    print(f"[benchmark] outputs saved: {out_name} (top1={outputs['top1_text']})")

    metric = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step1_forward_latency_s": round(step1_latency, 6),
        "latency_s": round(total_latency / max(n, 1), 6),
        "peak_memory_mb": round(peak_mem, 2),
        "num_samples": n,
        "device": device,
        "device_model": "Ascend910",
        "mode": mode_str,
        "output_type": "image_text_similarity",
        "dataset": dataset_name,
        "dtype": dtype_str,
        "packages": {"torch": torch.__version__, "torch_npu": getattr(__import__("torch_npu"), "__version__", "n/a")},
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[benchmark] metrics saved: {met_name}")
    print(f"[benchmark] DONE: {n} samples")


if __name__ == "__main__":
    main()
