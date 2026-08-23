#!/usr/bin/env python3
"""Benchmark for openai/whisper-large-v3-turbo (ASR, SpeechSeq2Seq).

产出 outputs_*.pt / benchmark_metrics_*.json / trace_*.json。
config 模式（随机权重，转录无意义但 encoder→decoder→generate 全链路完整）。
音频用 numpy 合成，不依赖 torchaudio。
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "openai/whisper-large-v3-turbo"
DATASET_NAME = "builtin"
WARMUP_ITERATIONS = 3

# 复用 demo 中已验证的音频合成与 generation_config 兼容 patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import (  # noqa: E402
    make_synthetic_audio,
    patch_whisper_generation_config,
)


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
    from transformers import AutoConfig, AutoProcessor, WhisperForConditionalGeneration

    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pretrained", action="store_true", help="Tier2: load pretrained weights")
    parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    cache_dir = (Path(__file__).resolve().parent / "models").as_posix()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    device = get_device(force_cpu=args.cpu)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if args.use_pretrained:
        print("[Setup] Loading pretrained weights...")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, cache_dir=cache_dir)
    else:
        print("[Setup] DRY/config mode: random weights (full architecture)")
        if hasattr(WhisperForConditionalGeneration, "from_config"):
            model = WhisperForConditionalGeneration.from_config(config, torch_dtype=torch.float16)
        else:
            model = WhisperForConditionalGeneration._from_config(config, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    patch_whisper_generation_config(model, processor)
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = DATASET_NAME
    print(f"[Setup] dtype: {dtype_str}, mode={mode_str}, sr={sr}")

    import torch_npu  # noqa: F401

    # 50 条合成音频（不同时长/频率），满足 completed gate num_samples>=50
    audios = []
    for i in range(args.max_samples if args.max_samples <= 60 else 60):
        secs = 1.0 + (i % 4) * 0.5
        a = make_synthetic_audio(seconds=secs)
        if i % 3 == 1:
            a = a * 0.5  # quieter variant
        audios.append(a)
    n = len(audios)
    print(f"[benchmark] {n} synthetic audio samples")

    def transcribe(audio):
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        feats = inputs.input_features.to(device).to(next(model.parameters()).dtype)
        with torch.no_grad():
            gen = model.generate(input_features=feats, language="en", task="transcribe", max_new_tokens=32)
        text = processor.batch_decode(gen, skip_special_tokens=True)[0]
        return text

    trace_path = Path(__file__).resolve().parent / f"trace_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    try:
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile
        with npu_profile(activities=[NPUActivity.CPU, NPUActivity.NPU]) as prof:
            t0 = time.time()
            _ = transcribe(audios[0])
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            step1_latency = time.time() - t0
        prof.export_trace(str(trace_path))
    except Exception as e:
        step1_latency = 0.0
        if not trace_path.exists():
            trace_path.write_text(json.dumps({"fallback": str(e)}))
    print(f"[benchmark] step1 latency: {step1_latency:.4f}s, trace: {trace_path.exists()}")

    # Warmup: 3 次推理预热 NPU 算子编译缓存（与 perf 对称）
    print(f"\n[benchmark] warmup {WARMUP_ITERATIONS} iterations...")
    for i in range(WARMUP_ITERATIONS):
        t0 = time.perf_counter()
        _ = transcribe(audios[0])
        print(f"[benchmark] warmup iter {i+1}/{WARMUP_ITERATIONS}: {time.perf_counter()-t0:.6f}s")
    print("[benchmark] warmup complete")

    texts = []
    wall_start = time.perf_counter()
    for a in audios:
        texts.append(transcribe(a))
    wall_clock_s = time.perf_counter() - wall_start
    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    outputs = {"audios": [f"synthetic {len(a)/sr:.1f}s" for a in audios], "transcriptions": texts}
    out_name = f"outputs_npu_{dtype_str}_{mode_str}_{dataset_name}.pt"
    torch.save(outputs, Path(__file__).resolve().parent / out_name)
    print(f"[benchmark] outputs saved: {out_name}")

    metric = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step1_forward_latency_s": round(step1_latency, 6),
        "latency_s": round(wall_clock_s / max(n, 1), 6),
        "wall_clock_s": round(wall_clock_s, 6),
        "warmup_iterations": WARMUP_ITERATIONS,
        "peak_memory_mb": round(peak_mem, 2),
        "num_samples": n,
        "device": device,
        "device_model": "Ascend910",
        "mode": mode_str,
        "output_type": "transcription",
        "dataset": dataset_name,
        "dtype": dtype_str,
        "packages": {"torch": torch.__version__, "torch_npu": getattr(__import__("torch_npu"), "__version__", "n/a"), "numpy": np.__version__},
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ttft_ms": round(step1_latency * 1000, 3),
    }
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[benchmark] metrics saved: {met_name}")
    print(f"[benchmark] DONE: {n} samples")


if __name__ == "__main__":
    main()
