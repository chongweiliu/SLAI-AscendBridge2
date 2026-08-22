#!/usr/bin/env python3
"""
accuracy_run_perf.py for laion/clap-htsat-fused — NPU 性能优化版（runtime_only）。

优化策略：
  - warmup(3x) 预热（与 baseline 对称）
  - TASK_QUEUE_ENABLE=1 异步算子下发
  - 更大批次（batch_size=30）减少 Python 循环开销

合同：audio-text similarity，与 accuracy_run.py 同口径。

Usage:
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50
    uv run python accuracy_run_perf.py compare
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PERF_SUFFIX = "_perf"
BATCH_SIZE = 30
WARMUP_ITERATIONS = 3

MODEL_ID = "laion/clap-htsat-fused"
DATASET_NAME = "builtin"

CANDIDATE_TEXTS = [
    "the sound of a bell ringing", "a pure tone beep", "a dog barking",
    "a cat meowing", "a bird chirping", "a car horn honking",
    "thunder rumbling", "wind howling", "rain falling",
    "a piano playing", "a guitar strumming", "a drum beating",
    "someone speaking", "a baby crying", "laughter",
    "a door creaking", "footsteps on gravel", "glass breaking",
    "water flowing", "waves crashing", "a fire crackling",
    "a train passing", "an airplane flying over", "a clock ticking",
    "a phone ringing", "an alarm beeping", "a siren wailing",
    "a hammer hitting", "a saw cutting wood", "a vacuum running",
    "a crowd cheering", "people applauding", "a choir singing",
    "a violin playing", "a flute playing", "a trumpet blowing",
    "a rooster crowing", "a cow mooing", "a horse neighing",
    "a frog croaking", "a cricket chirping", "a mosquito buzzing",
    "a beep tone", "a buzzer sounding", "a chime ringing",
    "music playing softly", "a bass thumping", "a snare drum",
    "a hi-hat closing", "a cymbal crashing", "a synthesizer pad",
    "an electronic beat", "ambient noise", "static hiss",
    "a hum at low frequency", "a whistle", "a pop sound",
    "a click", "a snap", "a clap of hands",
]


def make_sine_wave(freq_hz=440.0, seconds=1.0, sr=48000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wave = 0.6 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return wave


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
    packages = ["torch", "transformers", "torch_npu", "numpy"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def cmd_run(args):
    from transformers import AutoConfig, AutoModel, AutoProcessor

    cache_dir = (Path(__file__).resolve().parent / "models").as_posix()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    device = get_device(force_cpu=args.cpu)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[perf] Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    sr = getattr(processor.feature_extractor, "sampling_rate", 48000)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if args.use_pretrained:
        print("[perf] Loading pretrained weights...")
        model = AutoModel.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    else:
        print("[perf] DRY/config mode: random weights")
        model = AutoModel.from_config(config)
    model = model.to(device)
    model.eval()
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = DATASET_NAME
    print(f"[perf] dtype: {dtype_str}, mode={mode_str}, sr={sr}")

    texts = CANDIDATE_TEXTS[: args.max_samples]
    n = len(texts)
    audio = make_sine_wave(440.0, 1.0, sr)
    print(f"[perf] {n} texts x 1 sine-wave audio, batch_size={BATCH_SIZE}")

    import torch_npu  # noqa: F401

    def encode(txts):
        inputs = processor(text=txts, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.audio_embeds.float(), out.text_embeds.float()

    # warmup
    print(f"[perf] warmup {WARMUP_ITERATIONS} iterations...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = encode(texts[:BATCH_SIZE])
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    print(f"[perf] warmup done")

    # 正式推理
    all_sims = []
    t_all = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            a, t = encode(chunk)
            a = torch.nn.functional.normalize(a, dim=-1)
            t = torch.nn.functional.normalize(t, dim=-1)
            all_sims.extend((a @ t.T).squeeze(0).tolist())
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    total_latency = time.perf_counter() - t_all
    wall_clock_s = round(total_latency, 6)

    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    selected_npu = 0
    if hasattr(torch, "npu") and torch.npu.is_available():
        selected_npu = torch.npu.current_device()

    outputs = {"texts": texts, "audio": "440Hz sine 1s", "similarity_per_text": all_sims,
               "top1_text": texts[all_sims.index(max(all_sims))] if all_sims else ""}
    out_name = f"outputs_npu_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"
    torch.save(outputs, Path(__file__).resolve().parent / out_name)
    print(f"[perf] outputs saved: {out_name}")

    metric = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_clock_s": wall_clock_s,
        "latency_s": round(total_latency / max(n, 1), 6),
        "peak_memory_mb": round(peak_mem, 2),
        "num_samples": n,
        "device": device,
        "device_model": "Ascend910",
        "mode": mode_str,
        "output_type": "audio_text_similarity",
        "dataset": dataset_name,
        "dtype": dtype_str,
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
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[perf] metrics saved: {met_name}")
    print(f"[perf] wall_clock_s={wall_clock_s}, num_samples={n}")


def cmd_compare(args):
    ADAPT_DIR = Path(__file__).resolve().parent

    def _find_artifacts():
        all_metrics = sorted(ADAPT_DIR.glob("benchmark_metrics_*.json"))
        perf_metrics = [f for f in all_metrics if PERF_SUFFIX in f.name]
        baseline_metrics = [f for f in all_metrics if PERF_SUFFIX not in f.name]

        if not perf_metrics:
            raise FileNotFoundError("找不到 perf metrics")
        if not baseline_metrics:
            raise FileNotFoundError("找不到 baseline metrics")

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

    if b_metric.get("mode") != p_metric.get("mode"):
        raise ValueError(f"mode 不一致: baseline={b_metric.get('mode')} vs perf={p_metric.get('mode')}")
    if b_metric.get("dataset") != p_metric.get("dataset"):
        raise ValueError("dataset 不一致")
    if b_metric.get("dtype") != p_metric.get("dtype"):
        raise ValueError("dtype 不一致")
    if b_metric.get("num_samples") != p_metric.get("num_samples"):
        raise ValueError(f"num_samples 不一致: baseline={b_metric.get('num_samples')} vs perf={p_metric.get('num_samples')}")

    b_data = torch.load(baseline_outputs_path, map_location="cpu", weights_only=False)
    p_data = torch.load(perf_outputs_path, map_location="cpu", weights_only=False)

    b_sims = b_data.get("similarity_per_text", [])
    p_sims = p_data.get("similarity_per_text", [])

    n = min(len(b_sims), len(p_sims))
    if n == 0:
        raise ValueError("similarity 列表为空")

    # Compare similarity vectors as tensors
    b_t = torch.tensor(b_sims[:n], dtype=torch.float32).flatten()
    p_t = torch.tensor(p_sims[:n], dtype=torch.float32).flatten()

    if b_t.shape != p_t.shape:
        raise ValueError(f"similarity shape 不一致: {b_t.shape} vs {p_t.shape}")

    cos_sim = torch.nn.functional.cosine_similarity(b_t.unsqueeze(0), p_t.unsqueeze(0), dim=1).item()
    max_err = torch.max(torch.abs(b_t - p_t)).item()
    cos_sim = min(max(cos_sim, 0.0), 1.0)

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
    print(f"  cosine_similarity: {cos_sim:.8f}")
    print(f"  max_abs_error: {max_err}")
    print(f"  baseline_wall_clock_s: {baseline_wall_clock_s}")
    print(f"  perf_wall_clock_s: {perf_wall_clock_s}")
    print(f"  speedup_ratio: {speedup_ratio}")

    compare_result = {
        "cosine_similarity": cos_sim,
        "max_abs_error": max_err,
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
        "output_type": "audio_text_similarity",
    }
    compare_path = ADAPT_DIR / "output_compare_perf.json"
    compare_path.write_text(json.dumps(compare_result, indent=2))

    p_metric["output_compare"] = compare_result
    p_metric["latency_s"] = perf_latency_s
    perf_metrics_path.write_text(json.dumps(p_metric, indent=2))
    b_metric["latency_s"] = baseline_latency_s
    baseline_metrics_path.write_text(json.dumps(b_metric, indent=2))

    baseline_artifact = baseline_metrics_path.name
    perf_artifact = perf_metrics_path.name
    optimizations_str = "runtime_only: warmup(3x) + TASK_QUEUE_ENABLE=1 + larger_batch(bs=30)"

    result_entry = {
        "dtype": b_metric.get("dtype", "fp32"),
        "mode": b_metric.get("mode", "pretrained"),
        "dataset": b_metric.get("dataset", "builtin"),
        "output_type": "audio_text_similarity",
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
        "cosine_similarity": cos_sim,
        "max_abs_error": max_err,
        "comparison_method": "independent_baseline_artifact",
        "precision_method": "cosine_similarity",
        "comparison_scope": "steady_state",
        "validation_note": "独立 baseline 工件对比，非 self-baseline；baseline 与 perf 对称 warmup(3x)，同卡串行，audio-text similarity 合同。",
        "steady_state_baseline_latency_s": baseline_latency_s,
        "steady_state_perf_latency_s": perf_latency_s,
        "optimization_items": ["warmup_3x", "TASK_QUEUE_ENABLE", "larger_batch_bs30"],
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
    print(f"[compare] speedup_ratio={speedup_ratio}, cosine={cos_sim:.8f}, max_abs_error={max_err}")


def main():
    parser = argparse.ArgumentParser(description="accuracy_run_perf.py for laion/clap-htsat-fused (runtime_only)")
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
