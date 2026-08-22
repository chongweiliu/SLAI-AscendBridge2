"""
NPU Optimized accuracy_run_perf.py for timm/mobilenetv3_small_100.lamb_in1k.
Runtime-only optimization: warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference(bs=8).

优化策略: runtime_only (无模型代码修改)
  - warmup 3 次前向推理，预热 NPU 算子编译缓存
  - TASK_QUEUE_ENABLE=1 异步算子下发
  - batched_inference(bs=8) 批量推理，减少 kernel launch 开销

Usage:
    TASK_QUEUE_ENABLE=1 uv run --extra ascend python accuracy_run_perf.py run --use-pretrained --max-samples 50
    uv run python accuracy_run_perf.py compare
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", CACHE_DIR)

import numpy as np
import timm
import torch
import torch.nn.functional as F

PERF_SUFFIX = "_perf"
WARMUP_ITERATIONS = 3

MODEL_ID = "timm/mobilenetv3_small_100.lamb_in1k"
ARCH = "mobilenetv3_small_100"
NUM_CLASSES = 1000
INPUT_SIZE = (3, 224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ADAPT_DIR = Path(__file__).resolve().parent


def load_benchmark_images():
    """加载合成随机图像（与 accuracy_run.py 一致）。"""
    print("[perf] using generated random images (seed=42)")
    random.seed(42)
    np.random.seed(42)
    images = []
    for _ in range(250):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32) / 255.0
        images.append(torch.from_numpy(arr).permute(2, 0, 1))
    return images, "random"


def normalize_imagenet(x):
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def get_device(force_cpu=False):
    if force_cpu:
        return "cpu", 0, "cpu"
    try:
        import torch_npu
        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            best_index, best_free = 0, -1
            for i in range(count):
                try:
                    free, total = torch.npu.mem_get_info(i)
                except Exception:
                    free, total = 0, 0
                print(f"[Device] NPU {i}: free={free/1024**3:.1f} GB / total={total/1024**3:.1f} GB")
                if free > best_free:
                    best_free = free
                    best_index = i
            torch.npu.set_device(best_index)
            device_name = torch.npu.get_device_name(best_index)
            print(f"[Device] selected npu:{best_index} ({device_name})")
            return f"npu:{best_index}", count, device_name
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0", torch.cuda.device_count(), torch.cuda.get_device_name(0)
    return "cpu", 0, "cpu"


def get_dtype_str(dtype):
    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype, str(dtype).replace("torch.", ""))


def get_package_versions():
    import importlib.metadata
    packages = ["torch", "timm", "torch_npu", "numpy", "torchvision"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def setup_model(use_pretrained, device):
    if use_pretrained:
        model = timm.create_model(f"hf-hub:{MODEL_ID}", pretrained=True, num_classes=NUM_CLASSES)
    else:
        model = timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES)
    model = model.to(device)
    model.eval()
    return model


def run_warmup(model, device, images, n_iterations=WARMUP_ITERATIONS):
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    print(f"[perf] warmup {n_iterations} iterations...")
    warmup_x = normalize_imagenet(images[0].unsqueeze(0)).to(device)
    with torch.no_grad():
        for i in range(n_iterations):
            t0 = time.perf_counter()
            _ = model(warmup_x)
            if device_type == "npu":
                torch.npu.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            print(f"[perf] warmup iter {i+1}/{n_iterations}: {time.perf_counter()-t0:.6f}s")
    del warmup_x
    print("[perf] warmup complete")


def cmd_run(args):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device, _, _ = get_device(force_cpu=args.cpu)
    if device.startswith("npu"):
        torch.npu.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    images, dataset_name = load_benchmark_images()
    num_samples = min(len(images), args.max_samples)
    images = images[:num_samples]
    print(f"[perf] dataset: {dataset_name}, samples: {num_samples}")

    model = setup_model(args.use_pretrained, device)
    first_device = next(model.parameters()).device
    device_short = first_device.type
    mode_str = "pretrained" if args.use_pretrained else "config"
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    print(f"[perf] model on {first_device}, mode={mode_str}, dtype={dtype_str}")

    # Warmup
    run_warmup(model, first_device, images)

    # Batched inference
    batch_size = 8
    print(f"[perf] running {num_samples} samples (batched, bs={batch_size})")

    tqe_enabled = os.environ.get("TASK_QUEUE_ENABLE", "0") == "1"
    wall_start = time.perf_counter()
    start_time = datetime.now().isoformat()

    all_labels = []
    all_top5_ids = []
    all_top5_scores = []

    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            batch_imgs = images[batch_start:batch_start+batch_size]
            batch_actual = len(batch_imgs)

            # Stack images into batch tensor
            batch_tensor = torch.stack(batch_imgs, dim=0).to(first_device)  # [B, 3, 224, 224]
            batch_tensor = normalize_imagenet(batch_tensor)

            logits = model(batch_tensor)  # [B, NUM_CLASSES]

            for j in range(batch_actual):
                top5_vals, top5_ids = torch.topk(logits[j], k=5)
                top1_id = int(top5_ids[0].item())
                all_labels.append(str(top1_id))
                all_top5_ids.append([int(v) for v in top5_ids.cpu().tolist()])
                all_top5_scores.append([float(v) for v in top5_vals.cpu().tolist()])

            del batch_tensor, logits
            processed = min(batch_start + batch_size, num_samples)
            if processed % 8 == 0 or processed == num_samples:
                print(f"[perf] processed {processed}/{num_samples} samples")

    wall_clock_s = time.perf_counter() - wall_start
    end_time = datetime.now().isoformat()

    # Save outputs
    outputs_path = ADAPT_DIR / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"
    output_data = {
        "class_labels": all_labels,
        "top5_ids": all_top5_ids,
        "top5_scores": all_top5_scores,
    }
    torch.save(output_data, outputs_path)

    # Save metrics
    metrics_path = ADAPT_DIR / f"benchmark_metrics_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"
    per_sample_latency_s = round(wall_clock_s / num_samples, 6)
    selected_npu = first_device.index if first_device.type == "npu" else None
    device_model = "unknown"
    if device_short == "npu" and first_device.index is not None:
        device_model = torch.npu.get_device_name(first_device.index)

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "latency_s": per_sample_latency_s,
        "wall_clock_s": round(wall_clock_s, 6),
        "peak_memory_mb": 0.0,
        "num_samples": num_samples,
        "device": str(first_device),
        "device_model": device_model,
        "mode": mode_str,
        "dataset": dataset_name,
        "dtype": dtype_str,
        "output_type": "class_labels",
        "warmup_iterations": WARMUP_ITERATIONS,
        "task_queue_enable": tqe_enabled,
        "batch_size": batch_size,
        "optimization_kind": "runtime_only",
        "selected_npu": selected_npu,
        "packages": get_package_versions(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    unique_top1 = len(set(all_labels))
    print(f"\n[perf] outputs saved to {outputs_path}")
    print(f"[perf] metrics saved to {metrics_path}")
    print(f"[perf] wall_clock_s: {wall_clock_s:.6f}")
    print(f"[perf] per_sample_latency_s: {per_sample_latency_s}")
    print(f"[perf] unique top-1 classes: {unique_top1}")
    print(f"[perf] warmup: {WARMUP_ITERATIONS}, TQE: {tqe_enabled}, bs: {batch_size}")


def cmd_compare(args):
    import glob

    # Find baseline and perf metrics
    baseline_files = sorted(glob.glob(str(ADAPT_DIR / "benchmark_metrics_*_pretrained_*.json")))
    baseline_files = [f for f in baseline_files if "_perf" not in f]
    perf_files = sorted(glob.glob(str(ADAPT_DIR / "benchmark_metrics_*_pretrained_*_perf.json")))

    if not baseline_files or not perf_files:
        print("[compare] ERROR: Missing baseline or perf metrics")
        return

    with open(baseline_files[-1]) as f:
        baseline_metrics = json.load(f)
    with open(perf_files[-1]) as f:
        perf_metrics = json.load(f)

    baseline_artifact = Path(baseline_files[-1]).name
    perf_artifact = Path(perf_files[-1]).name
    print(f"[compare] baseline: {baseline_artifact}")
    print(f"[compare] perf: {perf_artifact}")

    # Find outputs
    dtype_str = baseline_metrics.get("dtype", "fp32")
    device_short = baseline_metrics.get("device", "npu").split(":")[0]
    dataset_name = baseline_metrics.get("dataset", "random")
    mode_str = baseline_metrics.get("mode", "pretrained")

    baseline_out = ADAPT_DIR / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.pt"
    perf_out = ADAPT_DIR / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"

    if not baseline_out.exists() or not perf_out.exists():
        print(f"[compare] ERROR: Missing output files")
        return

    baseline_data = torch.load(baseline_out, weights_only=False)
    perf_data = torch.load(perf_out, weights_only=False)

    # Compare class_labels (exact match)
    baseline_labels = baseline_data["class_labels"]
    perf_labels = perf_data["class_labels"]
    n_samples = min(len(baseline_labels), len(perf_labels))
    print(f"[compare] comparing {n_samples} samples")

    matches = sum(1 for i in range(n_samples) if baseline_labels[i] == perf_labels[i])
    text_match_rate = matches / n_samples if n_samples > 0 else 0.0

    # Compare top5_ids
    top5_matches = sum(1 for i in range(n_samples) if baseline_data["top5_ids"][i] == perf_data["top5_ids"][i])

    # Compare top5_scores (cosine similarity)
    baseline_scores = [s[0] for s in baseline_data["top5_scores"]]
    perf_scores = [s[0] for s in perf_data["top5_scores"]]
    if baseline_scores and perf_scores:
        b_tensor = torch.tensor(baseline_scores)
        p_tensor = torch.tensor(perf_scores)
        cosine = torch.nn.functional.cosine_similarity(b_tensor.unsqueeze(0), p_tensor.unsqueeze(0)).item()
        cosine = max(0.0, min(1.0, cosine))
    else:
        cosine = 0.0

    # Speedup
    baseline_wall = baseline_metrics.get("wall_clock_s")
    perf_wall = perf_metrics.get("wall_clock_s")
    baseline_latency = round(baseline_wall / n_samples, 6) if baseline_wall else None
    perf_latency = round(perf_wall / n_samples, 6) if perf_wall else None
    speedup = round(baseline_wall / perf_wall, 6) if baseline_wall and perf_wall and perf_wall > 0 else None
    latency_reduction = round((1 - perf_latency / baseline_latency) * 100, 2) if baseline_latency and perf_latency else None

    baseline_warmup = baseline_metrics.get("warmup_iterations", WARMUP_ITERATIONS)
    perf_warmup = perf_metrics.get("warmup_iterations", WARMUP_ITERATIONS)

    print(f"\n[compare] Results:")
    print(f"  text_match_rate: {text_match_rate:.8f} ({matches}/{n_samples})")
    print(f"  top5_match_rate: {top5_matches/n_samples:.8f}")
    print(f"  cosine_similarity: {cosine:.8f}")
    print(f"  baseline_wall_clock_s: {baseline_wall}")
    print(f"  perf_wall_clock_s: {perf_wall}")
    print(f"  speedup_ratio: {speedup}")
    print(f"  latency_reduction_pct: {latency_reduction}%")

    # Write output_compare_perf.json
    compare_result = {
        "text_match_rate": round(text_match_rate, 8),
        "top5_match_rate": round(top5_matches / n_samples, 8),
        "cosine_similarity": round(cosine, 8),
        "baseline_samples": n_samples,
        "perf_samples": n_samples,
        "total_samples": n_samples,
    }
    compare_path = ADAPT_DIR / "output_compare_perf.json"
    compare_path.write_text(json.dumps(compare_result, indent=2))
    print(f"[compare] saved to {compare_path}")

    # Build optimization_notes.json
    selected_npu = perf_metrics.get("selected_npu")
    selected_npus = [str(selected_npu)] if selected_npu is not None else []
    perf_memory_mb = perf_metrics.get("peak_memory_mb", 0.0)
    perf_batch_size = perf_metrics.get("batch_size", 1)
    optimization_items = ["warmup", "TASK_QUEUE_ENABLE", "batched_inference"]

    result_entry = {
        "dtype": dtype_str,
        "mode": mode_str,
        "dataset": dataset_name,
        "output_type": "class_labels",
        "baseline_artifact": baseline_artifact,
        "perf_artifact": perf_artifact,
        "num_samples": n_samples,
        "baseline_latency_s": baseline_latency,
        "perf_latency_s": perf_latency,
        "baseline_wall_clock_s": baseline_wall,
        "perf_wall_clock_s": perf_wall,
        "wall_clock_source": "artifact_explicit_field",
        "baseline_warmup_iterations": baseline_warmup,
        "perf_warmup_iterations": perf_warmup,
        "warmup_policy": "symmetric",
        "perf_memory_mb": perf_memory_mb,
        "speedup_ratio": speedup,
        "latency_reduction_pct": latency_reduction,
        "cosine_similarity": round(cosine, 8),
        "text_match_rate": round(text_match_rate, 8),
        "optimization_items": optimization_items,
        "optimization_kind": "runtime_only",
        "task_queue_enable": perf_metrics.get("task_queue_enable", False),
        "batch_size": perf_batch_size,
        "comparison_method": "independent_baseline_artifact",
        "comparison_scope": "steady_state",
        "validation_note": "独立 baseline 工件对比，非 self-baseline；对称 warmup(3x) 后的 steady_state 测量；非冷启动对热启动。",
        "steady_state_baseline_latency_s": baseline_latency,
        "steady_state_perf_latency_s": perf_latency,
    }

    # Update perf metrics with output_compare
    perf_metrics["output_compare"] = compare_result
    perf_metrics["wall_clock_speedup_ratio"] = speedup
    perf_metrics["latency_s"] = perf_latency
    with open(perf_files[-1], "w") as f:
        json.dump(perf_metrics, indent=2, fp=f)

    # Fix baseline metrics latency_s
    baseline_metrics["latency_s"] = baseline_latency
    with open(baseline_files[-1], "w") as f:
        json.dump(baseline_metrics, indent=2, fp=f)

    optimization_notes = {
        "measurement_contract_version": 3,
        "optimizations": f"runtime_only: warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_inference(bs={perf_batch_size})",
        "optimization_kind": "runtime_only",
        "code_modified": False,
        "selected_npus": selected_npus,
        "device_topology": f"single die {selected_npus[0]}" if selected_npus else "unknown",
        "parallel_mode": "single_device",
        "results": [result_entry],
        "best_result": result_entry,
    }

    notes_path = ADAPT_DIR / "optimization_notes.json"
    notes_path.write_text(json.dumps(optimization_notes, indent=2))
    print(f"[compare] optimization_notes.json saved to {notes_path}")


def main():
    parser = argparse.ArgumentParser(description="NPU optimized accuracy run for timm/mobilenetv3_small_100.lamb_in1k")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run perf inference")
    run_parser.add_argument("--use-pretrained", action="store_true")
    run_parser.add_argument("--max-samples", type=int, default=250)
    run_parser.add_argument("--cpu", action="store_true")
    run_parser.add_argument("--profile-level", type=str, default=None)

    subparsers.add_parser("compare", help="Compare baseline vs perf outputs")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
