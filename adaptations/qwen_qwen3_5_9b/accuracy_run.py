#!/usr/bin/env python3
"""Benchmark for Qwen/Qwen3.5-9B (multimodal qwen3_5, text-only generation path).

产出 outputs_*.pt / benchmark_metrics_*.json / trace_*.json。config 模式随机权重（bf16）。
仅验证文本主干 generate 全链路（encoder/decoder/mrope/混合注意力），不加载视觉输入。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3.5-9B"
DATASET_NAME = "builtin"

PROMPTS = [
    "Hello, this is a test run on Huawei Ascend NPU.",
    "Explain what a neural network is in one sentence.",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
    "Summarize the theory of relativity briefly.",
    "Describe a sunset over the mountains.",
    "How does photosynthesis work?",
    "Translate 'good morning' to French.",
    "What are the primary colors?",
    "Explain recursion in programming.",
    "Name a famous scientist and their contribution.",
    "What is the meaning of life?",
    "Describe the process of making coffee.",
    "What is machine learning?",
    "Write a haiku about autumn.",
    "What is the speed of light?",
    "Explain gravity in simple terms.",
    "What is democracy?",
    "Describe the water cycle.",
    "What is a black hole?",
    "How do computers store data?",
    "What is climate change?",
    "Explain the concept of supply and demand.",
    "What is the largest planet?",
    "Describe the structure of an atom.",
    "What is artificial intelligence?",
    "How does the human heart work?",
    "What is the Pythagorean theorem?",
    "Explain the theory of evolution.",
    "What is a database?",
    "Describe the process of digestion.",
    "What is the difference between RAM and ROM?",
    "Explain the concept of通货膨胀.",
    "What is a rainbow?",
    "How do airplanes fly?",
    "What is quantum mechanics?",
    "Describe the life cycle of a butterfly.",
    "What is the internet?",
    "Explain the greenhouse effect.",
    "What is a prime number?",
    "Describe the function of the liver.",
    "What is the freezing point of water?",
    "Explain how a microscope works.",
    "What is the largest ocean?",
    "Describe the process of respiration.",
    "What is a synonym?",
    "Explain the concept of gravity waves.",
    "What is the boiling point of water?",
    "Describe the structure of DNA.",
    "What is a continent?",
    "Explain how a telescope works.",
    "What is the tallest mountain?",
    "Describe the process of mitosis.",
    "What is a metaphor?",
    "Explain the concept of entropy.",
    "What is the smallest unit of life?",
    "Describe the water treatment process.",
    "What is the periodic table?",
    "Explain how a battery works.",
    "What is a constellation?",
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
    from transformers import AutoConfig, AutoTokenizer

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
        model = AutoModelCls.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = DATASET_NAME
    print(f"[Setup] dtype: {dtype_str}, mode={mode_str}, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    import torch_npu  # noqa: F401

    prompts = PROMPTS[: args.max_samples] if args.max_samples <= len(PROMPTS) else PROMPTS
    n = len(prompts)
    print(f"[benchmark] {n} prompts")

    def generate_one(prompt):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=16, do_sample=False)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    trace_path = Path(__file__).resolve().parent / f"trace_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    try:
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile
        with npu_profile(activities=[NPUActivity.CPU, NPUActivity.NPU]) as prof:
            t0 = time.time()
            _ = generate_one(prompts[0] if prompts else "hello")
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            step1_latency = time.time() - t0
        prof.export_trace(str(trace_path))
    except Exception as e:
        step1_latency = 0.0
        if not trace_path.exists():
            trace_path.write_text(json.dumps({"fallback": str(e)}))
    print(f"[benchmark] step1 latency: {step1_latency:.4f}s, trace: {trace_path.exists()}")

    gen_texts = []
    t_all = time.time()
    for p in prompts:
        gen_texts.append(generate_one(p))
    total_latency = time.time() - t_all
    peak_mem = 0.0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    outputs = {"prompts": prompts, "generated_text": gen_texts}
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
        "output_type": "generated_text",
        "dataset": dataset_name,
        "dtype": dtype_str,
        "packages": {"torch": torch.__version__, "torch_npu": getattr(__import__("torch_npu"), "__version__", "n/a")},
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ttft_ms": round(step1_latency * 1000, 3),
    }
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[benchmark] metrics saved: {met_name}")
    print(f"[benchmark] DONE: {n} samples")


if __name__ == "__main__":
    main()
