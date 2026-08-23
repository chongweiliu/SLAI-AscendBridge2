"""
Benchmark accuracy_run.py for Qwen/Qwen3.5-9B (multimodal qwen3_5, text-only path).

两步依次执行:
  Step 1: 单样本 -> trace_*.json + benchmark_metrics_*.json (性能分析)
  Step 2: 全样本 teacher-forcing -> outputs_*.pt (logits + PPL)

文本主干混合注意力 (3/4 linear_attention + 1/4 full_attention, mrope) 全链路覆盖。
warmup 3 次前向推理，与 perf 对称（baseline=bs=1 无 TQE；perf=bs=2 + TQE）。

Usage:
    uv run --extra ascend python accuracy_run.py --use-pretrained --max-samples 50
    uv run --extra ascend python accuracy_run.py --cpu

Model Type: causal_lm (text-only path of multimodal qwen3_5)
Recommended Dataset: wikitext
"""

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

# Domestic HF mirror defaults (override via env if needed). Set before HF imports.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import set_seed as transformers_set_seed
from datasets import load_from_disk

# 数据集配置
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

MODEL_ID = "Qwen/Qwen3.5-9B"


def load_benchmark_texts() -> tuple[list[str], str]:
    """加载测试数据集文本，返回 (texts, dataset_name)。优先 wikitext，降级内置。"""
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        print(f"[benchmark] loading dataset from {wikitext_path}")
        ds = load_from_disk(str(wikitext_path))
        if hasattr(ds, "keys"):
            ds = ds["test"]
        texts = sorted([sample["text"] for sample in ds if sample.get("text", "").strip()])
        print(f"[benchmark] loaded {len(texts)} samples from wikitext")
        return texts, "wikitext"

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
        "The capital of France is Paris, known for the Eiffel Tower.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The theory of relativity was developed by Albert Einstein.",
        "Water boils at one hundred degrees Celsius at sea level.",
        "The Great Wall of China stretches thousands of kilometers.",
        "Neural networks consist of layers of interconnected neurons.",
        "Gradient descent optimizes model parameters during training.",
        "The Internet connects billions of devices around the world.",
        "Shakespeare wrote many famous plays, including Hamlet.",
        "The human genome contains roughly three billion base pairs.",
        "Quantum computing leverages superposition and entanglement.",
        "Renewable energy sources include solar, wind, and hydro power.",
        "The stock market fluctuates based on supply and demand.",
        "Climate change affects weather patterns across the globe.",
        "Vaccines help the immune system fight infectious diseases.",
        "The Olympic Games bring athletes together every four years.",
        "Artificial neural networks are inspired by biological brains.",
        "Data preprocessing is an essential step in machine learning pipelines.",
        "Convolutional neural networks excel at image recognition tasks.",
        "Recurrent neural networks process sequential data effectively.",
        "Transfer learning reuses pretrained models for new tasks.",
        "Tokenization splits text into pieces a model can process.",
        "The PyTorch profiler helps identify performance bottlenecks.",
        "Inference optimization reduces latency for production services.",
        "Batch size affects both throughput and memory consumption.",
        "Mixed precision training speeds up computation on modern hardware.",
        "The transformer architecture relies on self-attention layers.",
        "Fine-tuning adapts a general model to a specific domain.",
        "Evaluation metrics quantify how well a model performs.",
        "Perplexity measures how surprised a language model is by data.",
        "The cosine similarity compares the direction of two vectors.",
        "Distributed training splits workloads across multiple accelerators.",
        "Model quantization shrinks weights to lower precision formats.",
        "The compiler translates high-level code into machine instructions.",
        "Operating systems manage hardware and software resources.",
        "Databases store and retrieve structured information efficiently.",
        "Networking protocols define how computers exchange messages.",
        "Encryption protects data by converting it into secure formats.",
        "Cloud computing provides on-demand access to shared resources.",
        "Containers package applications with all of their dependencies.",
        "Version control systems track changes in source code.",
        "Continuous integration automates building and testing software.",
        "Code review improves quality by letting peers inspect changes.",
        "Documentation helps users and developers understand a project.",
        "Testing catches defects before software reaches production.",
        "Monitoring dashboards visualize the health of services.",
        "Load balancers distribute traffic across backend servers.",
        "Caching reduces response times for frequently requested data.",
        "Microservices decompose applications into small independent parts.",
        "Open source communities collaborate on shared software projects.",
    ]
    return builtin_texts, "builtin"


def get_device(force_cpu: bool = False):
    """获取推理设备。NPU 选卡: mem_get_info 选空闲 HBM 最多的卡。
    本机环境禁用 ASCEND_RT_VISIBLE_DEVICES（会触发 aclInit error 107001）。"""
    if force_cpu:
        return "cpu", 0, "cpu"

    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            npu_count = torch.npu.device_count()
            best_idx, best_free = 0, -1
            for idx in range(npu_count):
                try:
                    free, total = torch.npu.mem_get_info(idx)
                except Exception:
                    free, total = 0, 0
                print(f"[Device] npu:{idx} free HBM: {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB")
                if free > best_free:
                    best_idx, best_free = idx, free
            torch.npu.set_device(best_idx)
            device_name = torch.npu.get_device_name(best_idx) if npu_count > 0 else "unknown"
            print(f"[Device] selected npu:{best_idx} (most free HBM)")
            return f"npu:{best_idx}", npu_count, device_name
    except ImportError:
        pass
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return "cuda:0", torch.cuda.device_count(), device_name
    return "cpu", 0, "cpu"


class _PerfMonitor:
    """性能监控器：测量推理延迟和峰值内存"""

    def __init__(self, device_type, device_ids=None):
        self.device_type = device_type
        self.device_ids = device_ids
        self.start = None
        self.latency_s = None
        self.peak_memory_mb = None

    def __enter__(self):
        if self.device_type == "npu":
            import torch_npu  # noqa: F401

            torch.npu.synchronize()
            if self.device_ids is not None:
                for dev_id in self.device_ids:
                    torch.npu.reset_peak_memory_stats(dev_id)
            else:
                torch.npu.reset_peak_memory_stats()
        elif self.device_type == "cuda":
            torch.cuda.synchronize()
            if self.device_ids is not None:
                for dev_id in self.device_ids:
                    torch.cuda.reset_peak_memory_stats(dev_id)
            else:
                torch.cuda.reset_peak_memory_stats()

        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device_type == "npu":
            import torch_npu  # noqa: F401

            torch.npu.synchronize()
            if self.device_ids is not None:
                max_mem = 0.0
                for dev_id in self.device_ids:
                    mem = torch.npu.max_memory_allocated(dev_id)
                    max_mem = max(max_mem, mem)
                self.peak_memory_mb = max_mem / (1024**2)
            else:
                self.peak_memory_mb = torch.npu.max_memory_allocated() / (1024**2)
        elif self.device_type == "cuda":
            torch.cuda.synchronize()
            if self.device_ids is not None:
                max_mem = 0.0
                for dev_id in self.device_ids:
                    mem = torch.cuda.max_memory_allocated(dev_id)
                    max_mem = max(max_mem, mem)
                self.peak_memory_mb = max_mem / (1024**2)
            else:
                self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            self.peak_memory_mb = 0.0

        self.latency_s = time.perf_counter() - self.start
        return False


def get_profiler_context(device_short):
    """根据硬件动态返回原生 Profiler 上下文"""
    if device_short == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile

        activities = [NPUActivity.CPU, NPUActivity.NPU]
        return npu_profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True)
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device_short == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True)


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
    """加载模型。Qwen3.5-9B 属 qwen3_5 多模态架构，使用 AutoModelForImageTextToText。
    单卡加载，禁用 device_map=auto（9B bf16 ~18GB，单 64GB 卡无压力）。
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)

    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls  # noqa: F401

    if use_pretrained:
        print("[Setup] Loading pretrained weights (bf16, single card)...")
        model = AutoModelCls.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        model = model.to(device)
    else:
        print("[Setup] DRY/config mode: random weights (bf16)")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = AutoModelCls.from_config(config, trust_remote_code=True)
        finally:
            torch.set_default_dtype(old_dtype)
        model = model.to(device)
    model.eval()
    return model, tokenizer


def run_step1(model, tokenizer, first_device, device_short, device_ids, mode_str, adapt_dir: Path, dataset_name: str, texts: list[str], num_samples: int):
    """Step 1: 单样本推理 -> trace_*.json + benchmark_metrics_*.json"""
    print("\n" + "=" * 60)
    print("Step 1: 单样本推理 (性能分析)")
    print("=" * 60)

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    trace_path = adapt_dir / f"trace_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.json"
    metrics_path = adapt_dir / f"benchmark_metrics_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.json"

    text = texts[0] if texts else "Hello, benchmark."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(first_device)

    profiler_context = get_profiler_context(device_short)

    start_time = datetime.now().isoformat()
    with profiler_context as prof:
        with _PerfMonitor(device_short, device_ids) as m:
            with torch.no_grad():
                out = model(**inputs)
                _ = out.logits if hasattr(out, "logits") else out[0]

        latency_s = m.latency_s
        peak_memory_mb = m.peak_memory_mb

    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception as e:
        if not trace_path.exists():
            trace_path.write_text(json.dumps({"fallback": str(e)}))

    device_model = "unknown"
    if device_short == "npu" and first_device.index is not None:
        device_model = torch.npu.get_device_name(first_device.index)
    elif device_short == "cuda" and first_device.index is not None:
        device_model = torch.cuda.get_device_name(first_device.index)

    metrics = {
        "start_time": start_time,
        "latency_s": round(latency_s, 6),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "num_samples": num_samples,
        "device": str(first_device),
        "device_model": device_model,
        "mode": mode_str,
        "dataset": dataset_name,
        "dtype": dtype_str,
        "output_type": "logits",
        "packages": get_package_versions(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    prof_dir = adapt_dir / "export_only_prof_dir"
    if prof_dir.exists():
        shutil.rmtree(prof_dir, ignore_errors=True)

    print(f"[benchmark] trace saved to {trace_path}")
    print(f"[benchmark] metrics saved to {metrics_path}")
    print(f"[benchmark] {metrics}")

    return trace_path, metrics_path, start_time


def run_step2(model, tokenizer, first_device, device_short, mode_str, adapt_dir: Path, dataset_name: str, texts: list[str], max_samples: int = 250):
    """Step 2: 全样本 teacher-forcing 推理 -> outputs_*.pt (logits + PPL)"""
    print("\n" + "=" * 60)
    print("Step 2: 全样本 teacher-forcing 推理 (Logits + PPL)")
    print("=" * 60)

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.pt"

    texts = texts[:max_samples]
    print(f"[benchmark] {len(texts)} samples to process (max {max_samples})")

    all_logits = []
    all_ppl = []
    all_sample_latency = []

    wall_start = time.perf_counter()

    with torch.no_grad():
        for i, text in enumerate(texts):
            sample_start = time.perf_counter()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(first_device)

            logits_output = model(**inputs)
            last_token_logits = logits_output.logits[0, -1, :].cpu()
            all_logits.append(last_token_logits)

            labels = inputs["input_ids"]
            logits = logits_output.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ppl = torch.exp(loss).item()
            all_ppl.append(ppl)

            all_sample_latency.append(time.perf_counter() - sample_start)

            del inputs, logits_output

            if (i + 1) % 32 == 0:
                if device_short == "npu":
                    torch.npu.empty_cache()
                elif device_short == "cuda":
                    torch.cuda.empty_cache()
                print(f"[benchmark] processed {i + 1}/{len(texts)} samples (cache cleared)")
            elif (i + 1) % 8 == 0:
                print(f"[benchmark] processed {i + 1}/{len(texts)} samples")

    wall_clock_s = time.perf_counter() - wall_start

    output_data = {
        "logits": all_logits,
        "perplexity": all_ppl,
    }
    torch.save(output_data, outputs_path)

    ppl_avg = round(sum(all_ppl) / len(all_ppl), 2) if all_ppl else None
    avg_sample_latency_s = round(sum(all_sample_latency) / len(all_sample_latency), 6) if all_sample_latency else None

    print(f"[benchmark] outputs saved to {outputs_path}")
    if all_logits:
        print(f"[benchmark]   - logits: {len(all_logits)} samples, shape: {all_logits[0].shape}")
    if all_ppl:
        print(f"[benchmark]   - perplexity: avg={ppl_avg}, min={min(all_ppl):.2f}, max={max(all_ppl):.2f}")
    print(f"[benchmark] avg per-sample latency: {avg_sample_latency_s} s")
    print(f"[benchmark] wall clock: {wall_clock_s:.6f} s")

    return outputs_path, avg_sample_latency_s, wall_clock_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pretrained", action="store_true", help="Tier2: load pretrained weights")
    parser.add_argument("--max-samples", type=int, default=250, help="Max samples to process (default 250)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (disable NPU/CUDA)")
    args = parser.parse_args()

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
    num_samples = min(len(texts), args.max_samples)
    print(f"[benchmark] using dataset: {dataset_name}, total samples: {len(texts)}, effective samples: {num_samples}")

    model, tokenizer = setup_model(args.use_pretrained, device, CACHE_DIR)

    first_device = next(model.parameters()).device
    device_short = first_device.type
    mode_str = "pretrained" if args.use_pretrained else "config"
    print(f"[benchmark] model on {first_device}, mode={mode_str}, dtype={get_dtype_str(next(model.parameters()).dtype)}")

    device_ids = None
    if hasattr(model, "hf_device_map"):
        device_ids = list(set(dev.index if hasattr(dev, "index") else dev for dev in model.hf_device_map.values()))
    elif first_device.index is not None:
        device_ids = [first_device.index]

    # Step 1: 单样本 -> trace + metrics
    trace_path, metrics_path, start_time = run_step1(model, tokenizer, first_device, device_short, device_ids, mode_str, ADAPT_DIR, dataset_name, texts, num_samples)

    # Warmup: 3 次前向推理预热 NPU 算子编译缓存（与 perf 对称）
    WARMUP_ITERATIONS = 3
    print(f"\n[benchmark] warmup {WARMUP_ITERATIONS} iterations...")
    warmup_text = texts[0] if texts else "Hello, warmup."
    warmup_inputs = tokenizer(warmup_text, return_tensors="pt", truncation=True, max_length=512).to(first_device)
    with torch.no_grad():
        for i in range(WARMUP_ITERATIONS):
            t0 = time.perf_counter()
            _ = model(**warmup_inputs)
            if device_short == "npu":
                torch.npu.synchronize()
            elif device_short == "cuda":
                torch.cuda.synchronize()
            print(f"[benchmark] warmup iter {i+1}/{WARMUP_ITERATIONS}: {time.perf_counter() - t0:.6f}s")
    del warmup_inputs
    print("[benchmark] warmup complete")

    # Step 2: 全样本 -> outputs + Logits + PPL
    outputs_path, avg_sample_latency_s, wall_clock_s = run_step2(model, tokenizer, first_device, device_short, mode_str, ADAPT_DIR, dataset_name, texts, args.max_samples)

    # 更新 metrics 文件
    end_time = datetime.now().isoformat()
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    metrics["end_time"] = end_time
    metrics["wall_clock_s"] = round(wall_clock_s, 6)
    metrics["warmup_iterations"] = WARMUP_ITERATIONS
    if avg_sample_latency_s is not None:
        metrics["latency_s"] = avg_sample_latency_s
    metrics["num_samples"] = num_samples

    with open(metrics_path, "w") as f:
        json.dump(metrics, indent=2, fp=f)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  trace: {trace_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  outputs: {outputs_path} (含 logits + perplexity)")
    print(f"  wall_clock_s: {wall_clock_s:.6f}")


if __name__ == "__main__":
    main()
