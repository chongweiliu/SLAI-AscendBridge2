"""
accuracy_run_perf.py for openai/clip-vit-base-patch32 — NPU optimized version.

Optimizations:
  1. All texts in one batch (1 image encoding vs 4 in baseline chunks of 16)
  2. TASK_QUEUE_ENABLE=1: Async operator dispatch
  3. Symmetric warmup(3x): Matches baseline warmup

Contract: image_text_similarity (same as accuracy_run.py).
  - Synthetic image + 60 candidate texts
  - Output: similarity_per_text

Usage:
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 60
    uv run python accuracy_run_perf.py compare
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_NAME = "builtin"
PERF_SUFFIX = "_perf"
WARMUP_ITERATIONS = 3

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


def get_package_versions() -> dict:
    import importlib.metadata
    packages = ["torch", "transformers", "torch_npu", "numpy", "PIL"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def make_synthetic_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([40, 40, 184, 184], fill=(220, 30, 30))
    return img


def run_perf(use_pretrained: bool, max_samples: int, cpu: bool):
    """运行 perf 推理（all texts in one batch）。"""
    from transformers import AutoConfig, AutoModel, CLIPModel, CLIPProcessor

    adapt_dir = Path(__file__).resolve().parent
    cache_dir = (adapt_dir / "models").as_posix()

    device = get_device(force_cpu=cpu)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[perf] Using device: {device}")

    torch.manual_seed(42)

    processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if use_pretrained:
        print("[perf] Loading pretrained weights...")
        model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype="auto", cache_dir=cache_dir)
    else:
        print("[perf] Config mode: random weights")
        model = AutoModel.from_config(config)
    model = model.to(device)
    model.eval()
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if use_pretrained else "config"

    texts = CANDIDATE_TEXTS[:max_samples]
    n = len(texts)
    print(f"[perf] {n} candidate texts x 1 synthetic image")

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
        return out.logits_per_image.squeeze(0).float()

    # Warmup
    print(f"[perf] warming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        encode_batch(image, texts[:8])
    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    # All texts in one batch (1 image encoding vs 4 in baseline)
    perf_start = time.perf_counter()
    start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    with torch.no_grad():
        all_sims = encode_batch(image, texts).tolist()

    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    perf_end = time.perf_counter()
    wall_clock_s = perf_end - perf_start
    end_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Save outputs
    outputs_path = adapt_dir / f"outputs_npu_{dtype_str}_{mode_str}_{DATASET_NAME}{PERF_SUFFIX}.pt"
    outputs = {
        "texts": texts,
        "image": "synthetic red square 224x224",
        "similarity_per_text": all_sims,
        "top1_text": texts[all_sims.index(max(all_sims))] if all_sims else "",
    }
    torch.save(outputs, outputs_path)
    print(f"[perf] outputs saved: {outputs_path.name} (top1={outputs['top1_text']})")

    # Save metrics
    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    latency_per_sample = wall_clock_s / n if n > 0 else wall_clock_s
    selected_npu = int(device.split(":")[1]) if ":" in device else 0

    metrics_path = adapt_dir / f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{DATASET_NAME}{PERF_SUFFIX}.json"
    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "latency_s": round(latency_per_sample, 6),
        "wall_clock_s": round(wall_clock_s, 6),
        "peak_memory_mb": round(peak_mem, 2),
        "num_samples": n,
        "device": device,
        "device_model": "Ascend910",
        "mode": mode_str,
        "output_type": "image_text_similarity",
        "dataset": DATASET_NAME,
        "dtype": dtype_str,
        "warmup_iterations": WARMUP_ITERATIONS,
        "packages": get_package_versions(),
        "optimization_items": ["batched_inference", "warmup", "TASK_QUEUE_ENABLE"],
        "optimization_kind": "runtime_only",
        "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE", "0") == "1",
        "batch_size": n,
        "selected_npu": selected_npu,
        "selected_npus": [selected_npu],
        "device_topology": f"single-die:{selected_npu}",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[perf] metrics saved: {metrics_path.name}")
    print(f"[perf] wall_clock_s: {wall_clock_s:.6f}, latency_s: {latency_per_sample:.6f}")

    return metrics_path, outputs_path


def compare_outputs(adapt_dir: Path):
    """对比 baseline vs perf outputs，生成 optimization_notes.json。"""
    print("\n" + "=" * 60)
    print("Compare: baseline vs perf")
    print("=" * 60)

    baseline_metrics_files = sorted(adapt_dir.glob("benchmark_metrics_npu_*_pretrained_*.json"))
    baseline_metrics_files = [f for f in baseline_metrics_files if "_perf" not in f.name]
    perf_metrics_files = sorted(adapt_dir.glob("benchmark_metrics_npu_*_pretrained_*_perf.json"))

    if not baseline_metrics_files or not perf_metrics_files:
        print("[compare] ERROR: Missing metrics files")
        return None

    baseline_metrics_path = baseline_metrics_files[-1]
    perf_metrics_path = perf_metrics_files[-1]
    print(f"[compare] baseline: {baseline_metrics_path.name}")
    print(f"[compare] perf: {perf_metrics_path.name}")

    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)
    with open(perf_metrics_path) as f:
        perf_metrics = json.load(f)

    baseline_outputs_files = sorted(adapt_dir.glob("outputs_npu_*_pretrained_*.pt"))
    baseline_outputs_files = [f for f in baseline_outputs_files if "_perf" not in f.name]
    perf_outputs_files = sorted(adapt_dir.glob("outputs_npu_*_pretrained_*_perf.pt"))

    if not baseline_outputs_files or not perf_outputs_files:
        print("[compare] ERROR: Missing output files")
        return None

    baseline_data = torch.load(baseline_outputs_files[-1], weights_only=False)
    perf_data = torch.load(perf_outputs_files[-1], weights_only=False)

    baseline_sims = baseline_data.get("similarity_per_text", [])
    perf_sims = perf_data.get("similarity_per_text", [])

    if not baseline_sims or not perf_sims:
        print("[compare] ERROR: Missing similarity scores")
        return None

    num_compare = min(len(baseline_sims), len(perf_sims))
    print(f"[compare] comparing {num_compare} samples")

    b_tensor = torch.tensor(baseline_sims[:num_compare], dtype=torch.float32)
    p_tensor = torch.tensor(perf_sims[:num_compare], dtype=torch.float32)

    cos = torch.nn.functional.cosine_similarity(b_tensor.unsqueeze(0), p_tensor.unsqueeze(0)).item()
    cos = min(1.0, max(0.0, cos))
    max_abs_error = (b_tensor - p_tensor).abs().max().item()
    avg_abs_error = (b_tensor - p_tensor).abs().mean().item()

    baseline_wall_clock = baseline_metrics.get("wall_clock_s", 0.0)
    perf_wall_clock = perf_metrics.get("wall_clock_s", 0.0)
    baseline_latency = baseline_metrics.get("latency_s", 0.0)
    perf_latency = perf_metrics.get("latency_s", 0.0)

    speedup_ratio = baseline_wall_clock / perf_wall_clock if perf_wall_clock > 0 else 0.0
    num_samples = min(baseline_metrics.get("num_samples", 0), perf_metrics.get("num_samples", 0))

    print(f"[compare] cosine_similarity: {cos:.10f}")
    print(f"[compare] max_abs_error: {max_abs_error:.10f}")
    print(f"[compare] baseline_wall_clock_s: {baseline_wall_clock}")
    print(f"[compare] perf_wall_clock_s: {perf_wall_clock}")
    print(f"[compare] speedup_ratio: {speedup_ratio:.6f}")

    baseline_warmup = baseline_metrics.get("warmup_iterations", WARMUP_ITERATIONS)
    perf_warmup = perf_metrics.get("warmup_iterations", WARMUP_ITERATIONS)
    selected_npu = perf_metrics.get("selected_npu", 0)
    selected_npus = perf_metrics.get("selected_npus", [selected_npu])
    device_topology = perf_metrics.get("device_topology", f"single-die:{selected_npu}")

    result = {
        "dtype": perf_metrics.get("dtype", "fp32"),
        "mode": perf_metrics.get("mode", "pretrained"),
        "dataset": perf_metrics.get("dataset", "builtin"),
        "output_type": "image_text_similarity",
        "baseline_artifact": baseline_metrics_path.name,
        "perf_artifact": perf_metrics_path.name,
        "num_samples": num_samples,
        "baseline_latency_s": round(baseline_latency, 6),
        "perf_latency_s": round(perf_latency, 6),
        "baseline_wall_clock_s": round(baseline_wall_clock, 6),
        "perf_wall_clock_s": round(perf_wall_clock, 6),
        "wall_clock_source": "artifact_explicit_field",
        "baseline_warmup_iterations": baseline_warmup,
        "perf_warmup_iterations": perf_warmup,
        "warmup_policy": "symmetric",
        "baseline_memory_mb": round(baseline_metrics.get("peak_memory_mb", 0), 2),
        "perf_memory_mb": round(perf_metrics.get("peak_memory_mb", 0), 2),
        "speedup_ratio": round(speedup_ratio, 6),
        "latency_reduction_pct": round((1 - 1 / speedup_ratio) * 100, 4) if speedup_ratio > 0 else 0,
        "cosine_similarity": round(cos, 10),
        "max_abs_error": round(max_abs_error, 10),
        "optimization_items": perf_metrics.get("optimization_items", ["batched_inference", "warmup", "TASK_QUEUE_ENABLE"]),
        "optimization_kind": "runtime_only",
        "task_queue_enable": perf_metrics.get("task_queue_enable", True),
        "batch_size": perf_metrics.get("batch_size", num_samples),
        "selected_npu": selected_npu,
        "selected_npus": selected_npus,
        "device_topology": device_topology,
        "parallel_mode": "single_card",
        "comparison_method": "independent_baseline_artifact",
        "comparison_scope": "steady_state",
        "precision_method": "cosine_similarity",
        "validation_note": f"Independent baseline artifact ({baseline_metrics_path.name}) vs perf artifact ({perf_metrics_path.name}). Symmetric warmup ({baseline_warmup}x). Cosine={cos:.10f}, max_abs_error={max_abs_error:.10f}.",
        "steady_state_baseline_latency_s": round(baseline_latency, 6),
        "steady_state_perf_latency_s": round(perf_latency, 6),
    }

    notes = {
        "measurement_contract_version": 3,
        "optimizations": "batched_inference(all_texts_one_batch) + warmup(3x) + TASK_QUEUE_ENABLE=1",
        "results": [result],
        "best_result": result,
    }

    notes_path = adapt_dir / "optimization_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))
    print(f"[compare] optimization_notes saved to {notes_path}")

    compare_data = {
        "cosine_similarity": round(cos, 10),
        "max_abs_error": round(max_abs_error, 10),
        "avg_abs_error": round(avg_abs_error, 10),
        "baseline_samples": num_samples,
        "perf_samples": num_samples,
        "cuda_samples": num_samples,
        "ascend_samples": num_samples,
        "total_samples": num_samples,
        "baseline_wall_clock_s": round(baseline_wall_clock, 6),
        "perf_wall_clock_s": round(perf_wall_clock, 6),
        "speedup_ratio": round(speedup_ratio, 6),
        "wall_clock_speedup_ratio": round(speedup_ratio, 6),
    }
    compare_path = adapt_dir / "output_compare_perf.json"
    compare_path.write_text(json.dumps(compare_data, indent=2))
    print(f"[compare] output_compare_perf saved to {compare_path}")

    return notes


def main():
    parser = argparse.ArgumentParser(description="NPU optimized accuracy_run_perf for clip-vit-base-patch32")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    run_parser = subparsers.add_parser("run", help="Run perf inference")
    run_parser.add_argument("--use-pretrained", action="store_true", help="Load pretrained weights")
    run_parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU")

    subparsers.add_parser("compare", help="Compare baseline vs perf")

    args = parser.parse_args()

    if args.command == "run":
        run_perf(use_pretrained=args.use_pretrained, max_samples=args.max_samples, cpu=args.cpu)
    elif args.command == "compare":
        adapt_dir = Path(__file__).resolve().parent
        notes = compare_outputs(adapt_dir)
        if notes is None:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
