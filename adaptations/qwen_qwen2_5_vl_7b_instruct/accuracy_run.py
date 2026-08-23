#!/usr/bin/env python3
"""Benchmark for Qwen/Qwen2.5-VL-7B-Instruct (text-only teacher-forcing path).

Produces outputs_*.pt / benchmark_metrics_*.json / trace_*.json.
Teacher-forcing: forward pass on text, extract last-token logits + perplexity.
"""
import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer
from transformers import set_seed as transformers_set_seed

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
WARMUP_ITERATIONS = 3
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"


def load_benchmark_texts() -> tuple[list[str], str]:
    """Load benchmark texts from wikitext dataset, return (texts, dataset_name)."""
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        print(f"[benchmark] loading dataset from {wikitext_path}")
        ds = load_from_disk(str(wikitext_path))
        if hasattr(ds, "keys"):
            ds = ds["test"]
        texts = sorted([sample["text"] for sample in ds if sample.get("text", "").strip()])
        print(f"[benchmark] loaded {len(texts)} samples from wikitext")
        return texts, "wikitext"
    # fallback builtin
    print("[benchmark] using built-in benchmark texts")
    builtin_texts = [
        "Hello, this is a benchmark run on an Ascend NPU.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized the field of deep learning.",
        "PyTorch is an open-source machine learning framework.",
        "The attention mechanism allows models to focus on relevant parts of input.",
        "Language models can generate coherent and contextually relevant text.",
        "Huawei Ascend NPUs are designed for AI workloads.",
        "Benchmarking measures the latency and throughput of inference systems.",
    ]
    return builtin_texts, "builtin"


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

    packages = ["torch", "transformers", "torch_npu", "numpy", "datasets"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def run_warmup(model, tokenizer, device, n_iterations: int = WARMUP_ITERATIONS):
    """Warmup: n forward passes to prime NPU operator compilation cache."""
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    print(f"[benchmark] warmup {n_iterations} iterations...")
    warmup_text = "Warmup: Hello, this is a warmup forward pass."
    warmup_inputs = tokenizer(warmup_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        for i in range(n_iterations):
            t0 = time.perf_counter()
            _ = model(**warmup_inputs)
            if device_type == "npu":
                torch.npu.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"[benchmark] warmup iter {i+1}/{n_iterations}: {dt:.6f}s")
    del warmup_inputs
    print("[benchmark] warmup complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pretrained", action="store_true", help="Tier2: load pretrained weights")
    parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    transformers_set_seed(SEED)

    cache_dir = (Path(__file__).resolve().parent / "models").as_posix()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    device = get_device(force_cpu=args.cpu)
    if not args.cpu:
        assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    if device.startswith("npu"):
        torch.npu.manual_seed_all(SEED)
    elif device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)

    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls  # noqa: F401

    if args.use_pretrained:
        print("[Setup] Loading pretrained weights...")
        model = AutoModelCls.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto", cache_dir=cache_dir)
    else:
        print("[Setup] DRY/config mode: random weights (full text backbone)")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = AutoModelCls.from_config(config, trust_remote_code=True)
        finally:
            torch.set_default_dtype(old_dtype)
    model = model.to(device)
    model.eval()

    import torch_npu  # noqa: F401

    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if args.use_pretrained else "config"
    texts, dataset_name = load_benchmark_texts()
    num_samples = min(len(texts), args.max_samples)
    texts = texts[:num_samples]
    print(f"[Setup] dtype: {dtype_str}, mode={mode_str}, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(f"[benchmark] {num_samples} samples from {dataset_name}")

    # Warmup (symmetric with perf script)
    run_warmup(model, tokenizer, device, WARMUP_ITERATIONS)

    # Teacher-forcing forward pass (sequential, bs=1)
    all_logits = []
    all_ppl = []

    wall_start = time.perf_counter()
    start_time = datetime.now().isoformat()

    with torch.no_grad():
        for idx, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [1, seq_len, vocab]

            seq_len = attention_mask.sum().item()
            last_token_logits = logits[0, seq_len - 1, :].cpu()
            all_logits.append(last_token_logits)

            # Perplexity on real tokens
            real_logits = logits[0, :seq_len, :]
            real_labels = input_ids[0, :seq_len]
            shift_logits = real_logits[:-1, :].contiguous()
            shift_labels = real_labels[1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ppl = torch.exp(loss).item()
            all_ppl.append(ppl)

            del input_ids, attention_mask, out, logits
            if (idx + 1) % 16 == 0:
                if hasattr(torch, "npu"):
                    torch.npu.empty_cache()
                print(f"[benchmark] processed {idx+1}/{num_samples} samples")

    wall_clock_s = time.perf_counter() - wall_start
    end_time = datetime.now().isoformat()

    # Save outputs
    outputs = {"logits": all_logits, "perplexity": all_ppl}
    out_name = f"outputs_npu_{dtype_str}_{mode_str}_{dataset_name}.pt"
    torch.save(outputs, Path(__file__).resolve().parent / out_name)
    print(f"[benchmark] outputs saved: {out_name}")

    # Peak memory
    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    device_short = device.split(":")[0] if ":" in device else device
    device_model = "unknown"
    if device_short == "npu":
        dev_idx = int(device.split(":")[1]) if ":" in device else 0
        device_model = torch.npu.get_device_name(dev_idx)
    elif device_short == "cuda":
        device_model = torch.cuda.get_device_name(0)

    per_sample_latency_s = round(wall_clock_s / max(num_samples, 1), 6)
    ppl_avg = round(sum(all_ppl) / len(all_ppl), 2) if all_ppl else None

    metric = {
        "start_time": start_time,
        "end_time": end_time,
        "latency_s": per_sample_latency_s,
        "wall_clock_s": round(wall_clock_s, 6),
        "peak_memory_mb": round(peak_mem, 2),
        "num_samples": num_samples,
        "device": device,
        "device_model": device_model,
        "mode": mode_str,
        "output_type": "logits",
        "dataset": dataset_name,
        "dtype": dtype_str,
        "warmup_iterations": WARMUP_ITERATIONS,
        "packages": get_package_versions(),
    }
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[benchmark] metrics saved: {met_name}")
    print(f"[benchmark] wall_clock_s: {wall_clock_s:.6f}")
    print(f"[benchmark] avg per-sample latency: {per_sample_latency_s}s")
    print(f"[benchmark] perplexity: avg={ppl_avg}")
    print(f"[benchmark] DONE: {num_samples} samples")


if __name__ == "__main__":
    main()
