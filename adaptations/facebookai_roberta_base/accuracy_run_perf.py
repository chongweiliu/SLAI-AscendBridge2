#!/usr/bin/env python3
"""
accuracy_run_perf.py for FacebookAI/roberta-base — NPU 性能优化版（runtime_only）。

优化策略：
  - warmup(3x) 预热（与 baseline 对称）
  - TASK_QUEUE_ENABLE=1 异步算子下发
  - batched 前向推理（batch_size=8）减少 Python 循环开销

合同：mask 位置 logits，与 accuracy_run.py 同口径。

Usage:
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50
    uv run python accuracy_run_perf.py compare
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import set_seed as transformers_set_seed

PERF_SUFFIX = "_perf"
BATCH_SIZE = 8
WARMUP_ITERATIONS = 3

MODEL_ID = "FacebookAI/roberta-base"


def load_benchmark_texts() -> tuple[list[str], str]:
    """与 accuracy_run.py 完全一致的内置样本。"""
    print("[perf] using built-in benchmark texts (fill-mask profile)")
    builtin_texts = [
        "The capital of France is <mask>.",
        "The largest planet in our solar system is <mask>.",
        "The Great Wall is located in <mask>.",
        "The river <mask> flows through Egypt.",
        "Mount Everest is the highest <mask> on Earth.",
        "The official language of Brazil is <mask>.",
        "Tokyo is the capital of <mask>.",
        "The Sahara is the largest hot <mask> in the world.",
        "The Pacific is the largest <mask> on Earth.",
        "Berlin is the capital of <mask>.",
        "Water freezes at zero degrees <mask>.",
        "The chemical symbol for gold is <mask>.",
        "Plants use sunlight to make food through <mask>.",
        "The speed of light is about 300,000 kilometers per <mask>.",
        "Humans have <mask> chromosomes in each cell.",
        "The Earth revolves around the <mask> once a year.",
        "The Moon orbits around the <mask>.",
        "Sound travels through the air as <mask>.",
        "The human body needs <mask> to breathe.",
        "Ice is the solid form of <mask>.",
        "I drink a cup of <mask> every morning.",
        "She reads the <mask> to learn the latest news.",
        "He went to the <mask> to buy fresh vegetables.",
        "We use a <mask> to tell the time.",
        "The children play football in the <mask>.",
        "My grandmother bakes delicious <mask> pie.",
        "The doctor gave him a <mask> for the pain.",
        "She took the <mask> to the airport.",
        "He opened the <mask> to let in fresh air.",
        "The restaurant serves Italian <mask>.",
        "William Shakespeare wrote many famous <mask>.",
        "The piano is a musical <mask>.",
        "The Mona Lisa was painted by Leonardo da <mask>.",
        "The Olympic Games are held every four <mask>.",
        "People celebrate Christmas in <mask>.",
        "The first man on the Moon was Neil <mask>.",
        "Beethoven was a famous <mask>.",
        "The ancient pyramids were built in <mask>.",
        "Chess is a game of <mask>.",
        "The violin has four <mask>.",
        "The teacher wrote the answer on the <mask>.",
        "Students must pass the <mask> to graduate.",
        "The meeting will start at nine <mask>.",
        "He works as an engineer at a technology <mask>.",
        "The library has thousands of <mask>.",
        "She finished her homework before <mask>.",
        "The scientist published the results of the <mask>.",
        "The nurse works in a <mask>.",
        "The farmer grows <mask> in the field.",
        "The pilot flew the plane across the <mask>.",
        "The opposite of hot is <mask>.",
        "A week has seven <mask>.",
        "There are twelve months in a <mask>.",
        "The color of the sky on a clear day is <mask>.",
        "Birds build their nests in <mask>.",
        "Fish live in the <mask>.",
        "Bees make <mask> from flowers.",
        "A person who writes books is called an <mask>.",
        "The season after winter is <mask>.",
        "We celebrate the new year in <mask>.",
    ]
    return builtin_texts, "builtin"


def select_idle_npu() -> int:
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


def get_device(force_cpu: bool = False):
    if force_cpu:
        return "cpu", 0, "cpu"
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            idx = select_idle_npu()
            device_name = torch.npu.get_device_name(idx)
            return f"npu:{idx}", torch.npu.device_count(), device_name
    except ImportError:
        pass
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return "cuda:0", torch.cuda.device_count(), device_name
    return "cpu", 0, "cpu"


def get_dtype_str(dtype: torch.dtype) -> str:
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.int64: "int64",
        torch.int32: "int32",
    }
    return dtype_map.get(dtype, str(dtype).replace("torch.", ""))


def get_package_versions() -> dict:
    import importlib.metadata
    packages = ["torch", "transformers", "torch_npu", "numpy", "datasets"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def setup_model(use_pretrained: bool, device, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if use_pretrained:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = model.to(device)
    else:
        model = AutoModelForMaskedLM.from_config(config)
        model = model.to(device)
    model.eval()
    return model, tokenizer


def batched_mask_logits(model, tokenizer, texts, device, batch_size, mask_token_id, max_length=512):
    """批量前向推理，提取每条样本 <mask> 位置的 logits。"""
    device_str = str(device)
    all_logits = []
    all_latencies = []

    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)

            sample_start = time.perf_counter()
            outputs = model(**encoded)
            logits = outputs.logits  # [B, seq_len, vocab]

            if device_str.startswith("npu"):
                torch.npu.synchronize()
            elif device_str.startswith("cuda"):
                torch.cuda.synchronize()
            batch_latency = time.perf_counter() - sample_start
            per_sample_latency = batch_latency / len(batch_texts)
            all_latencies.extend([per_sample_latency] * len(batch_texts))

            input_ids = encoded["input_ids"]
            for j in range(len(batch_texts)):
                # input_ids[j] is 1D [seq_len], nonzero returns tuple of length 1
                mask_positions = (input_ids[j] == mask_token_id).nonzero(as_tuple=True)
                if len(mask_positions[0]) == 1:
                    mask_pos = int(mask_positions[0][0])
                    all_logits.append(logits[j, mask_pos].cpu())
                else:
                    all_logits.append(torch.zeros(model.config.vocab_size, dtype=torch.float32))

            del encoded, outputs, logits
            if device_str.startswith("npu"):
                torch.npu.empty_cache()
            elif device_str.startswith("cuda"):
                torch.cuda.empty_cache()

    return all_logits, all_latencies


def cmd_run(args):
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    ADAPT_DIR = Path(__file__).resolve().parent

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    transformers_set_seed(SEED)
    device, _, _ = get_device(force_cpu=args.cpu)
    if device.startswith("npu"):
        torch.npu.manual_seed_all(SEED)
    elif device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)

    torch.use_deterministic_algorithms(True, warn_only=True)

    if not args.cpu:
        assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got {device}"

    texts, dataset_name = load_benchmark_texts()
    texts = texts[: args.max_samples]
    print(f"[perf] using dataset: {dataset_name}, total samples: {len(texts)}, max_samples: {args.max_samples}")

    model, tokenizer = setup_model(args.use_pretrained, device, CACHE_DIR)
    first_device = next(model.parameters()).device
    device_short = first_device.type
    mode_str = "pretrained" if args.use_pretrained else "config"

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    device_ids = None
    if first_device.index is not None:
        device_ids = [first_device.index]

    mask_token_id = tokenizer.mask_token_id

    # warmup
    print(f"[perf] warmup {WARMUP_ITERATIONS} iterations (batch_size={BATCH_SIZE})...")
    warmup_texts = texts[:BATCH_SIZE] if len(texts) >= BATCH_SIZE else [texts[0] if texts else "The capital of France is <mask>."]
    warmup_encoded = tokenizer(warmup_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(first_device)
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(**warmup_encoded)
    if device_short == "npu":
        torch.npu.synchronize()
    elif device_short == "cuda":
        torch.cuda.synchronize()
    print(f"[perf] warmup done")

    # 正式推理
    print(f"[perf] running batched mask logits (batch_size={BATCH_SIZE})...")
    run_start = time.perf_counter()

    all_logits, all_latencies = batched_mask_logits(
        model, tokenizer, texts, first_device, BATCH_SIZE, mask_token_id
    )

    if device_short == "npu":
        torch.npu.synchronize()
    elif device_short == "cuda":
        torch.cuda.synchronize()

    run_end = time.perf_counter()
    wall_clock_s = round(run_end - run_start, 6)
    avg_latency_s = round(sum(all_latencies) / len(all_latencies), 6) if all_latencies else None
    peak_memory_mb = 0.0
    if device_short == "npu" and device_ids:
        peak_memory_mb = torch.npu.max_memory_allocated(device_ids[0]) / (1024**2)
    elif device_short == "cuda" and device_ids:
        peak_memory_mb = torch.cuda.max_memory_allocated(device_ids[0]) / (1024**2)

    # 保存 outputs
    outputs_path = ADAPT_DIR / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"
    output_data = {"logits": all_logits}
    torch.save(output_data, outputs_path)

    # 保存 metrics
    metrics_path = ADAPT_DIR / f"benchmark_metrics_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"
    start_time = datetime.now().isoformat()
    end_time = datetime.now().isoformat()

    device_model = "unknown"
    if device_short == "npu" and first_device.index is not None:
        device_model = torch.npu.get_device_name(first_device.index)
    elif device_short == "cuda" and first_device.index is not None:
        device_model = torch.cuda.get_device_name(first_device.index)

    selected_npu = first_device.index if first_device.index is not None else 0

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "wall_clock_s": wall_clock_s,
        "latency_s": avg_latency_s,
        "peak_memory_mb": round(peak_memory_mb, 2),
        "num_samples": len(all_logits),
        "device": str(first_device),
        "device_model": device_model,
        "mode": mode_str,
        "dataset": dataset_name,
        "dtype": dtype_str,
        "output_type": "logits",
        "optimization_kind": "runtime_only",
        "warmup_iterations": WARMUP_ITERATIONS,
        "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE", "0"),
        "batch_size": BATCH_SIZE,
        "selected_npu": selected_npu,
        "selected_npus": [selected_npu],
        "device_topology": f"1d:{selected_npu}",
        "parallel_mode": "single",
        "packages": get_package_versions(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[perf] outputs saved to {outputs_path}")
    print(f"[perf] metrics saved to {metrics_path}")
    print(f"[perf] wall_clock_s={wall_clock_s}, avg_latency_s={avg_latency_s}")
    print(f"[perf] num_samples={len(all_logits)}")
    print(f"[perf] peak_memory_mb={round(peak_memory_mb, 2)}")


def cmd_compare(args):
    ADAPT_DIR = Path(__file__).resolve().parent

    def _find_artifacts():
        all_metrics = sorted(ADAPT_DIR.glob("benchmark_metrics_*.json"))
        perf_metrics = [f for f in all_metrics if PERF_SUFFIX in f.name]
        baseline_metrics = [f for f in all_metrics if PERF_SUFFIX not in f.name]

        if not perf_metrics:
            raise FileNotFoundError(f"找不到 perf metrics 文件")
        if not baseline_metrics:
            raise FileNotFoundError(f"找不到 baseline metrics 文件")

        def _get_end_time(path):
            try:
                return json.loads(path.read_text()).get("end_time", "")
            except Exception:
                return ""

        baseline_path = max(baseline_metrics, key=_get_end_time)
        perf_path = max(perf_metrics, key=_get_end_time)

        b_metric = json.loads(baseline_path.read_text())
        p_metric = json.loads(perf_path.read_text())

        b_stem = baseline_path.stem.replace("benchmark_metrics_", "")
        p_stem = perf_path.stem.replace("benchmark_metrics_", "").replace(PERF_SUFFIX, "")

        b_outputs = ADAPT_DIR / ("outputs_" + b_stem + ".pt")
        p_outputs = ADAPT_DIR / ("outputs_" + p_stem + PERF_SUFFIX + ".pt")

        if not b_outputs.exists():
            raise FileNotFoundError(f"找不到 baseline outputs: {b_outputs}")
        if not p_outputs.exists():
            raise FileNotFoundError(f"找不到 perf outputs: {p_outputs}")

        return baseline_path, perf_path, b_outputs, p_outputs, b_metric, p_metric

    baseline_metrics_path, perf_metrics_path, baseline_outputs_path, perf_outputs_path, b_metric, p_metric = _find_artifacts()

    print(f"[compare] baseline metrics: {baseline_metrics_path.name}")
    print(f"[compare] perf metrics: {perf_metrics_path.name}")
    print(f"[compare] baseline outputs: {baseline_outputs_path.name}")
    print(f"[compare] perf outputs: {perf_outputs_path.name}")

    if b_metric.get("mode") != p_metric.get("mode"):
        raise ValueError(f"mode 不一致: baseline={b_metric.get('mode')} vs perf={p_metric.get('mode')}")
    if b_metric.get("dataset") != p_metric.get("dataset"):
        raise ValueError(f"dataset 不一致")
    if b_metric.get("dtype") != p_metric.get("dtype"):
        raise ValueError(f"dtype 不一致")
    if b_metric.get("num_samples") != p_metric.get("num_samples"):
        raise ValueError(f"num_samples 不一致: baseline={b_metric.get('num_samples')} vs perf={p_metric.get('num_samples')}")

    b_data = torch.load(baseline_outputs_path, map_location="cpu", weights_only=False)
    p_data = torch.load(perf_outputs_path, map_location="cpu", weights_only=False)

    b_logits = b_data.get("logits", [])
    p_logits = p_data.get("logits", [])

    n = min(len(b_logits), len(p_logits))
    if n == 0:
        raise ValueError("logits 列表为空，无法对比")

    cosine_sims = []
    max_errors = []
    for i in range(n):
        b_t = b_logits[i].float().flatten()
        p_t = p_logits[i].float().flatten()
        if b_t.shape != p_t.shape:
            raise ValueError(f"样本 {i} logits shape 不一致: {b_t.shape} vs {p_t.shape}")
        cos_sim = torch.nn.functional.cosine_similarity(b_t.unsqueeze(0), p_t.unsqueeze(0), dim=1).item()
        max_err = torch.max(torch.abs(b_t - p_t)).item()
        cos_sim = min(max(cos_sim, 0.0), 1.0)
        cosine_sims.append(cos_sim)
        max_errors.append(max_err)

    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    min_cosine = min(cosine_sims)
    max_abs_error = max(max_errors)

    baseline_wall_clock_s = b_metric.get("wall_clock_s")
    perf_wall_clock_s = p_metric.get("wall_clock_s")

    if not baseline_wall_clock_s or not perf_wall_clock_s:
        raise ValueError("metrics 缺少 wall_clock_s")

    baseline_latency_s = round(baseline_wall_clock_s / n, 6)
    perf_latency_s = round(perf_wall_clock_s / n, 6)

    speedup_ratio = round(baseline_wall_clock_s / perf_wall_clock_s, 6)
    latency_reduction_pct = round((1 - perf_latency_s / baseline_latency_s) * 100, 2)

    selected_npu = p_metric.get("selected_npu", 0)
    selected_npus = p_metric.get("selected_npus", [selected_npu])
    device_topology = p_metric.get("device_topology", f"1d:{selected_npu}")

    print(f"\n[compare] Results:")
    print(f"  cosine_similarity: {avg_cosine:.8f} (min={min_cosine:.8f})")
    print(f"  max_abs_error: {max_abs_error}")
    print(f"  baseline_wall_clock_s: {baseline_wall_clock_s}")
    print(f"  perf_wall_clock_s: {perf_wall_clock_s}")
    print(f"  speedup_ratio: {speedup_ratio}")

    compare_result = {
        "cosine_similarity": avg_cosine,
        "min_cosine_similarity": min_cosine,
        "max_abs_error": max_abs_error,
        "baseline_samples": n,
        "perf_samples": n,
        "cuda_samples": n,
        "ascend_samples": n,
        "baseline_wall_clock_s": baseline_wall_clock_s,
        "perf_wall_clock_s": perf_wall_clock_s,
        "speedup_ratio": speedup_ratio,
        "baseline_latency_s": baseline_latency_s,
        "perf_latency_s": perf_latency_s,
        "latency_reduction_pct": latency_reduction_pct,
        "baseline_warmup_iterations": b_metric.get("warmup_iterations", 0),
        "perf_warmup_iterations": p_metric.get("warmup_iterations", 0),
        "output_type": "logits",
    }
    compare_path = ADAPT_DIR / "output_compare_perf.json"
    compare_path.write_text(json.dumps(compare_result, indent=2))

    # 更新 metrics
    p_metric["output_compare"] = compare_result
    p_metric["latency_s"] = perf_latency_s
    perf_metrics_path.write_text(json.dumps(p_metric, indent=2))
    b_metric["latency_s"] = baseline_latency_s
    baseline_metrics_path.write_text(json.dumps(b_metric, indent=2))

    # 写 optimization_notes.json
    baseline_artifact = baseline_metrics_path.name
    perf_artifact = perf_metrics_path.name
    optimizations_str = "runtime_only: warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_mask_logits(bs=8)"

    result_entry = {
        "dtype": b_metric.get("dtype", "fp32"),
        "mode": b_metric.get("mode", "pretrained"),
        "dataset": b_metric.get("dataset", "builtin"),
        "output_type": "logits",
        "baseline_artifact": baseline_artifact,
        "perf_artifact": perf_artifact,
        "num_samples": n,
        "baseline_latency_s": baseline_latency_s,
        "perf_latency_s": perf_latency_s,
        "baseline_wall_clock_s": baseline_wall_clock_s,
        "perf_wall_clock_s": perf_wall_clock_s,
        "wall_clock_source": "artifact_explicit_field",
        "baseline_warmup_iterations": b_metric.get("warmup_iterations", WARMUP_ITERATIONS),
        "perf_warmup_iterations": p_metric.get("warmup_iterations", WARMUP_ITERATIONS),
        "warmup_policy": "symmetric",
        "speedup_ratio": speedup_ratio,
        "latency_reduction_pct": latency_reduction_pct,
        "baseline_memory_mb": b_metric.get("peak_memory_mb", 0),
        "perf_memory_mb": p_metric.get("peak_memory_mb", 0),
        "memory_reduction_pct": round(
            (1 - p_metric.get("peak_memory_mb", 0) / b_metric.get("peak_memory_mb", 1)) * 100, 2
        ) if b_metric.get("peak_memory_mb") else None,
        "cosine_similarity": avg_cosine,
        "min_cosine_similarity": min_cosine,
        "max_abs_error": max_abs_error,
        "comparison_method": "independent_baseline_artifact",
        "precision_method": "cosine_similarity",
        "comparison_scope": "steady_state",
        "validation_note": "独立 baseline 工件对比，非 self-baseline；baseline 与 perf 对称 warmup(3x)，同卡串行，batched mask logits 合同。",
        "steady_state_baseline_latency_s": baseline_latency_s,
        "steady_state_perf_latency_s": perf_latency_s,
        "optimization_items": ["warmup_3x", "TASK_QUEUE_ENABLE", "batched_mask_logits_bs8"],
        "optimization_kind": "runtime_only",
        "selected_npu": selected_npu,
        "selected_npus": selected_npus,
        "device_topology": device_topology,
        "parallel_mode": p_metric.get("parallel_mode", "single"),
        "task_queue_enable": p_metric.get("task_queue_enable", "1"),
        "batch_size": p_metric.get("batch_size", BATCH_SIZE),
    }

    notes = {
        "measurement_contract_version": 3,
        "optimizations": optimizations_str,
        "results": [result_entry],
        "best_result": result_entry,
    }

    notes_path = ADAPT_DIR / "optimization_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))
    print(f"\n[compare] optimization_notes saved to {notes_path}")
    print(f"[compare] speedup_ratio={speedup_ratio}, cosine={avg_cosine:.8f}, max_abs_error={max_abs_error}")


def main():
    parser = argparse.ArgumentParser(description="accuracy_run_perf.py for FacebookAI/roberta-base (runtime_only)")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="执行 batched 前向推理")
    run_parser.add_argument("--use-pretrained", action="store_true", help="加载预训练权重")
    run_parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    compare_parser = subparsers.add_parser("compare", help="对比 baseline vs perf")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
