#!/usr/bin/env python3
"""Benchmark for laion/clap-htsat-fused (audio-text contrastive embedding, ClapModel).

产出 outputs_*.pt / benchmark_metrics_*.json / trace_*.json。
音频用 numpy 正弦波合成，不依赖 torchaudio/librosa。config 模式随机权重（相似度无意义但前向完整）。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

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


def main():
    import argparse
    from transformers import AutoConfig, AutoModel, AutoProcessor

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

    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    sr = getattr(processor.feature_extractor, "sampling_rate", 48000)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if args.use_pretrained:
        print("[Setup] Loading pretrained weights...")
        model = AutoModel.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    else:
        print("[Setup] DRY/config mode: random weights")
        model = AutoModel.from_config(config)
    model = model.to(device)
    model.eval()
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = DATASET_NAME
    print(f"[Setup] dtype: {dtype_str}, mode={mode_str}, sr={sr}")

    texts = CANDIDATE_TEXTS[: args.max_samples]
    n = len(texts)
    audio = make_sine_wave(440.0, 1.0, sr)
    print(f"[benchmark] {n} texts x 1 sine-wave audio ({len(audio)} samples @ {sr}Hz)")

    import torch_npu  # noqa: F401

    def encode(txts):
        inputs = processor(text=txts, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.audio_embeds.float(), out.text_embeds.float()

    trace_path = Path(__file__).resolve().parent / f"trace_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    try:
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile
        with npu_profile(activities=[NPUActivity.CPU, NPUActivity.NPU]) as prof:
            t0 = time.time()
            _ = encode(texts[:8])
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            step1_latency = time.time() - t0
        prof.export_trace(str(trace_path))
    except Exception as e:
        step1_latency = 0.0
        if not trace_path.exists():
            trace_path.write_text(json.dumps({"fallback": str(e)}))
    print(f"[benchmark] step1 latency: {step1_latency:.4f}s, trace: {trace_path.exists()}")

    all_sims = []
    t_all = time.time()
    with torch.no_grad():
        for i in range(0, n, 16):
            chunk = texts[i : i + 16]
            a, t = encode(chunk)
            a = torch.nn.functional.normalize(a, dim=-1)
            t = torch.nn.functional.normalize(t, dim=-1)
            all_sims.extend((a @ t.T).squeeze(0).tolist())
    total_latency = time.time() - t_all
    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    outputs = {"texts": texts, "audio": "440Hz sine 1s", "similarity_per_text": all_sims,
               "top1_text": texts[all_sims.index(max(all_sims))] if all_sims else ""}
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
        "output_type": "audio_text_similarity",
        "dataset": dataset_name,
        "dtype": dtype_str,
        "packages": {"torch": torch.__version__, "torch_npu": getattr(__import__("torch_npu"), "__version__", "n/a"), "numpy": np.__version__},
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[benchmark] metrics saved: {met_name}")
    print(f"[benchmark] DONE: {n} samples")


if __name__ == "__main__":
    main()
