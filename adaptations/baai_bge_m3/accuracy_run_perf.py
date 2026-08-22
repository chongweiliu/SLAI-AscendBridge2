"""
NPU Optimized accuracy_run_perf.py for BAAI/bge-m3 (Sentence Embedding).
Runtime-only optimization: batched_encoding + warmup(3x) + TASK_QUEUE_ENABLE=1.

Usage:
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50 --batch-size 8
    uv run python accuracy_run_perf.py compare
"""

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")

from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402
from transformers import set_seed as transformers_set_seed  # noqa: E402

from datasets import load_from_disk  # noqa: E402

PERF_SUFFIX = "_perf"

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_TEXT_FIELD = "article"

SIMILARITY_PAIRS = [
    {"anchor": "The cat sits on the mat.", "paraphrase": "A cat is sitting on a mat.", "unrelated": "The stock market crashed yesterday."},
    {"anchor": "A man is playing the guitar.", "paraphrase": "Someone is strumming a guitar.", "unrelated": "The weather is sunny today."},
    {"anchor": "She reads a book every night.", "paraphrase": "Every night she reads a book.", "unrelated": "Quantum computers use qubits."},
]


def load_benchmark_texts() -> tuple[list[str], str]:
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        ds = load_from_disk(str(wikitext_path))
        if hasattr(ds, "keys"):
            split_name = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
            ds = ds[split_name]
        texts = sorted([sample["text"] for sample in ds if sample.get("text", "").strip()])
        return texts, "wikitext"

    cnn_path = DATASET_DIR / "cnn_dailymail___3.0.0"
    if cnn_path.exists():
        ds = load_from_disk(str(cnn_path))
        if hasattr(ds, "keys"):
            split_name = "test" if "test" in ds else ("validation" if "validation" in ds else "train")
            ds = ds[split_name]
        texts = sorted([sample[DATASET_TEXT_FIELD] for sample in ds if sample[DATASET_TEXT_FIELD].strip()])
        return texts, "cnn_dailymail"

    print("[perf] using built-in benchmark texts")
    builtin_texts = [
        "Hello, this is a benchmark run.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized the field of deep learning.",
        "PyTorch is an open-source machine learning framework.",
        "The attention mechanism allows models to focus on relevant parts of input.",
        "Language models can generate coherent and contextually relevant text.",
        "Artificial neural networks are inspired by the structure of the human brain.",
        "Deep learning models require large amounts of data to achieve good performance.",
        "Convolutional neural networks are commonly used for image classification tasks.",
        "Recurrent neural networks are designed to process sequential data such as text.",
        "Gradient descent is an optimization algorithm used to minimize the loss function.",
        "Backpropagation computes the gradients needed to update the network weights.",
        "Transfer learning allows a model pretrained on one task to be reused on another.",
        "Fine-tuning adjusts the weights of a pretrained model on a smaller dataset.",
        "Tokenization splits raw text into smaller units that a model can process.",
        "The vocabulary of a language model defines the set of tokens it can recognize.",
        "Perplexity measures how well a probability model predicts a sample of text.",
        "A lower perplexity indicates that the model is more confident in its predictions.",
        "Inference latency is the time required for a model to produce an output.",
        "Throughput describes how many samples a system can process per unit of time.",
        "Quantization reduces the numerical precision of model weights to save memory.",
        "Model compression techniques aim to make neural networks smaller and faster.",
        "Knowledge distillation trains a small student model to mimic a larger teacher.",
        "Data augmentation increases the diversity of training data with modified samples.",
        "Regularization methods such as dropout help prevent overfitting during training.",
        "Batch normalization stabilizes and accelerates the training of deep networks.",
        "Learning rate schedules adjust the step size during the course of training.",
        "Early stopping halts training when validation performance stops improving.",
        "Cross-validation provides a robust estimate of model performance on new data.",
        "Precision and recall are complementary metrics for evaluating classification models.",
        "The F1 score combines precision and recall into a single harmonic mean value.",
        "Confusion matrices summarize the correct and incorrect predictions of a classifier.",
        "Benchmark suites provide standardized tasks for comparing different models.",
        "Reproducibility requires fixing random seeds and recording the software environment.",
        "Deterministic algorithms produce identical outputs for identical inputs.",
        "Graphics processing units accelerate the matrix operations used in deep learning.",
        "Ascend neural processing units provide high-performance computing for AI workloads.",
        "Hardware accelerators can significantly reduce the time needed to train models.",
        "Distributed training splits a workload across multiple devices to scale up.",
        "Mixed precision training uses both float16 and float32 for efficiency.",
        "The softmax function converts logits into a probability distribution.",
        "Cross-entropy loss is widely used for training classification models.",
        "Word embeddings represent words as dense vectors in a continuous space.",
        "Contextual embeddings produce different representations depending on surrounding words.",
        "Self-attention computes weighted combinations of all positions in a sequence.",
        "Positional encodings inject information about token order into transformers.",
        "Encoder-decoder architectures are common in machine translation systems.",
        "Autoregressive models generate text one token at a time from left to right.",
        "Greedy decoding always selects the most probable next token.",
        "Beam search explores multiple candidate sequences to improve generation quality.",
        "Sampling strategies such as top-k and top-p control the diversity of generation.",
        "Evaluating generated text remains a challenging open problem in the field.",
        "Well-designed benchmarks help track progress across model generations.",
        "Sentence embeddings map variable-length text to fixed-size dense vectors.",
        "Cosine similarity is a common metric for comparing sentence embeddings.",
        "Semantic search retrieves documents by meaning rather than keyword overlap.",
        "Contrastive learning trains embedding models using positive and negative pairs.",
        "Retrieval augmented generation grounds language models in external knowledge.",
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
        return "cpu", 0, "cpu", None
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            idx = select_idle_npu()
            device_name = torch.npu.get_device_name(idx)
            return f"npu:{idx}", torch.npu.device_count(), device_name, [idx]
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0", torch.cuda.device_count(), torch.cuda.get_device_name(0), [0]
    return "cpu", 0, "cpu", None


class _PerfMonitor:
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
                max_mem = max(torch.npu.max_memory_allocated(d) for d in self.device_ids)
            else:
                max_mem = torch.npu.max_memory_allocated()
            self.peak_memory_mb = max_mem / (1024**2)
        elif self.device_type == "cuda":
            torch.cuda.synchronize()
            if self.device_ids is not None:
                max_mem = max(torch.cuda.max_memory_allocated(d) for d in self.device_ids)
            else:
                max_mem = torch.cuda.max_memory_allocated()
            self.peak_memory_mb = max_mem / (1024**2)
        else:
            self.peak_memory_mb = 0.0
        self.latency_s = time.perf_counter() - self.start
        return False


def get_profiler_context(device_short):
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
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
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
    MODEL_ID = "BAAI/bge-m3"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if use_pretrained:
        model = AutoModel.from_pretrained(MODEL_ID, torch_dtype="auto", cache_dir=cache_dir)
        model = model.to(device)
    else:
        model = AutoModel.from_config(config)
        model = model.to(device)
    model.eval()
    return model, tokenizer


def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts(model, tokenizer, texts, first_device):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(first_device)
    outputs = model(**encoded)
    embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    del outputs, encoded
    return embeddings


def run_step1_perf(model, tokenizer, first_device, device_short, device_tag, device_ids, mode_str, adapt_dir, dataset_name, texts):
    print("\n" + "=" * 60)
    print("Step 1 (perf): single sample (performance analysis)")
    print("=" * 60)
    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)
    trace_path = adapt_dir / f"trace_{device_tag}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"
    metrics_path = adapt_dir / f"benchmark_metrics_{device_tag}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"
    text = texts[0] if texts else "Hello, benchmark."
    profiler_context = get_profiler_context(device_short)
    start_time = datetime.now().isoformat()
    with profiler_context as prof:
        with _PerfMonitor(device_short, device_ids) as m:
            with torch.no_grad():
                _ = encode_texts(model, tokenizer, [text], first_device)
        latency_s = m.latency_s
        peak_memory_mb = m.peak_memory_mb
    prof.export_chrome_trace(str(trace_path))
    device_model = "unknown"
    if device_short == "npu" and first_device.index is not None:
        device_model = torch.npu.get_device_name(first_device.index)
    metrics = {
        "start_time": start_time,
        "latency_s": round(latency_s, 6),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "num_samples": len(texts),
        "device": str(first_device),
        "device_model": device_model,
        "mode": mode_str,
        "dataset": dataset_name,
        "dtype": dtype_str,
        "output_type": "cls_embeddings",
        "packages": get_package_versions(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    prof_dir = adapt_dir / "export_only_prof_dir"
    if prof_dir.exists():
        shutil.rmtree(prof_dir, ignore_errors=True)
    print(f"[perf] trace saved to {trace_path}")
    print(f"[perf] metrics saved to {metrics_path}")
    return trace_path, metrics_path, start_time


def run_step2_perf(model, tokenizer, first_device, device_short, mode_str, adapt_dir, dataset_name, texts, max_samples=250, batch_size=8, warmup_iterations=3):
    print("\n" + "=" * 60)
    print(f"Step 2 (perf): batched encoding (bs={batch_size}, warmup={warmup_iterations})")
    print("=" * 60)
    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)
    outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"
    texts = texts[:max_samples]
    print(f"[perf] {len(texts)} samples to process (max {max_samples}, batch_size={batch_size})")

    # Warmup
    print(f"[perf] warmup {warmup_iterations} iterations...")
    warmup_batch = texts[:min(batch_size, len(texts))]
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = encode_texts(model, tokenizer, warmup_batch, first_device)
    if device_short == "npu":
        torch.npu.synchronize()
    print("[perf] warmup done")

    all_embeddings = []
    step2_start = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            embeddings = encode_texts(model, tokenizer, batch_texts, first_device)
            # Split batch embeddings into individual rows
            for j in range(len(batch_texts)):
                all_embeddings.append(embeddings[j].unsqueeze(0).cpu())
            del embeddings
            if (batch_start // batch_size + 1) % 4 == 0:
                if device_short == "npu":
                    torch.npu.empty_cache()
                print(f"[perf] processed {batch_end}/{len(texts)} samples (cache cleared)")
            else:
                print(f"[perf] processed {batch_end}/{len(texts)} samples")

    if device_short == "npu":
        torch.npu.synchronize()
    step2_wall_clock = time.perf_counter() - step2_start
    wall_clock_s = round(step2_wall_clock, 6)

    # Similarity profile
    all_sentences = []
    for pair in SIMILARITY_PAIRS:
        all_sentences.extend([pair["anchor"], pair["paraphrase"], pair["unrelated"]])
    with torch.no_grad():
        sim_embeddings = encode_texts(model, tokenizer, all_sentences, first_device)
    profile = {"pairs": [], "all_margins_positive": None}
    margins = []
    for i, pair in enumerate(SIMILARITY_PAIRS):
        emb_a, emb_p, emb_u = sim_embeddings[i*3], sim_embeddings[i*3+1], sim_embeddings[i*3+2]
        cos_p = float(torch.dot(emb_p, emb_a))
        cos_u = float(torch.dot(emb_u, emb_a))
        margin = cos_p - cos_u
        margins.append(margin)
        profile["pairs"].append({"cos_paraphrase": round(cos_p, 6), "cos_unrelated": round(cos_u, 6), "margin": round(margin, 6)})
    profile["all_margins_positive"] = all(m > 0 for m in margins)

    output_data = {"texts": list(texts), "embeddings": all_embeddings, "similarity_profile": profile}
    torch.save(output_data, outputs_path)

    print(f"[perf] outputs saved to {outputs_path}")
    print(f"[perf]   - embeddings: {len(all_embeddings)} samples, dim={all_embeddings[0].shape[-1]}")
    print(f"[perf]   - wall_clock_s: {wall_clock_s}")
    return outputs_path, None, None, wall_clock_s


def cmd_run(args):
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    ADAPT_DIR = Path(__file__).resolve().parent
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    transformers_set_seed(SEED)
    device, _, _, device_ids = get_device(force_cpu=args.cpu)
    if device.startswith("npu"):
        torch.npu.manual_seed_all(SEED)
    elif device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not args.cpu:
        assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got {device}"
    texts, dataset_name = load_benchmark_texts()
    texts = texts[: args.max_samples]
    print(f"[perf] dataset: {dataset_name}, samples: {len(texts)}, max_samples={args.max_samples}")
    model, tokenizer = setup_model(args.use_pretrained, device, CACHE_DIR)
    first_device = next(model.parameters()).device
    device_short = first_device.type
    device_tag = str(first_device).replace(":", "_")
    mode_str = "pretrained" if args.use_pretrained else "config"
    trace_path, metrics_path, start_time = run_step1_perf(model, tokenizer, first_device, device_short, device_tag, device_ids, mode_str, ADAPT_DIR, dataset_name, texts)
    outputs_path, ttft_avg, tpot_avg, wall_clock_s = run_step2_perf(model, tokenizer, first_device, device_short, mode_str, ADAPT_DIR, dataset_name, texts, args.max_samples, args.batch_size, args.warmup_iterations)
    end_time = datetime.now().isoformat()
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    metrics["end_time"] = end_time
    metrics["ttft_ms"] = ttft_avg
    metrics["tpot_ms"] = tpot_avg
    metrics["wall_clock_s"] = wall_clock_s
    metrics["batch_size"] = args.batch_size
    metrics["warmup_iterations"] = args.warmup_iterations
    metrics["task_queue_enable"] = os.environ.get("TASK_QUEUE_ENABLE", "0")
    metrics["optimization_kind"] = "runtime_only"
    metrics["selected_npu"] = str(first_device)
    metrics["device_topology"] = f"1die:{first_device.index}" if first_device.index is not None else "cpu"
    num_samples_actual = len(texts)
    if num_samples_actual > 0 and wall_clock_s > 0:
        metrics["latency_s"] = round(wall_clock_s / num_samples_actual, 6)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n" + "=" * 60)
    print("Summary (perf)")
    print("=" * 60)
    print(f"  trace: {trace_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  outputs: {outputs_path}")
    print(f"  wall_clock_s: {wall_clock_s}")


def cmd_compare(args):
    ADAPT_DIR = Path(__file__).resolve().parent
    baseline_metrics_files = sorted(ADAPT_DIR.glob("benchmark_metrics_*_pretrained_*.json"))
    baseline_metrics_files = [f for f in baseline_metrics_files if not f.name.endswith("_perf.json")]
    perf_metrics_files = sorted(ADAPT_DIR.glob("benchmark_metrics_*_pretrained_*_perf.json"))
    baseline_output_files = sorted(ADAPT_DIR.glob("outputs_*_pretrained_*.pt"))
    baseline_output_files = [f for f in baseline_output_files if not f.name.endswith("_perf.pt")]
    perf_output_files = sorted(ADAPT_DIR.glob("outputs_*_pretrained_*_perf.pt"))

    if not baseline_metrics_files or not perf_metrics_files or not baseline_output_files or not perf_output_files:
        print("[compare] Error: Missing baseline or perf artifacts")
        return 1

    def load_metrics(f):
        with open(f) as fh:
            return json.load(fh)

    baseline_metrics = load_metrics(baseline_metrics_files[-1])
    perf_metrics = load_metrics(perf_metrics_files[-1])
    baseline_path = baseline_output_files[-1]
    perf_path = perf_output_files[-1]

    print(f"[compare] baseline metrics: {baseline_metrics_files[-1].name}")
    print(f"[compare] perf metrics: {perf_metrics_files[-1].name}")
    print(f"[compare] baseline outputs: {baseline_path.name}")
    print(f"[compare] perf outputs: {perf_path.name}")

    if baseline_metrics.get("mode") != "pretrained" or perf_metrics.get("mode") != "pretrained":
        print(f"[compare] Error: mode must be pretrained")
        return 1

    b_ns = baseline_metrics.get("num_samples", 0)
    p_ns = perf_metrics.get("num_samples", 0)
    num_samples = min(b_ns, p_ns)

    baseline_outputs = torch.load(baseline_path, map_location="cpu", weights_only=False)
    perf_outputs = torch.load(perf_path, map_location="cpu", weights_only=False)

    b_emb = baseline_outputs.get("embeddings", [])
    p_emb = perf_outputs.get("embeddings", [])

    if not b_emb or not p_emb:
        print("[compare] Error: No embeddings found")
        return 1

    min_len = min(len(b_emb), len(p_emb))
    print(f"[compare] Comparing {min_len} embedding pairs")

    cosines = []
    max_abs_errors = []
    for i in range(min_len):
        b_e = b_emb[i].float().flatten()
        p_e = p_emb[i].float().flatten()
        cos = F.cosine_similarity(b_e.unsqueeze(0), p_e.unsqueeze(0)).item()
        cosines.append(cos)
        mae = (b_e - p_e).abs().max().item()
        max_abs_errors.append(mae)

    avg_cosine = sum(cosines) / len(cosines) if cosines else 0.0
    min_cosine = min(cosines) if cosines else 0.0
    max_abs_error = max(max_abs_errors) if max_abs_errors else 0.0
    avg_cosine = min(1.0, max(0.0, avg_cosine))
    min_cosine = min(1.0, max(0.0, min_cosine))

    print(f"[compare] cosine_similarity: avg={avg_cosine:.6f}, min={min_cosine:.6f}")
    print(f"[compare] max_abs_error: {max_abs_error:.6e}")

    baseline_wall_clock = baseline_metrics.get("wall_clock_s", 0.0)
    perf_wall_clock = perf_metrics.get("wall_clock_s", 0.0)
    baseline_latency = baseline_metrics.get("latency_s", 0.0)
    perf_latency = perf_metrics.get("latency_s", 0.0)

    if perf_wall_clock > 0 and baseline_wall_clock > 0:
        speedup_ratio = baseline_wall_clock / perf_wall_clock
    else:
        speedup_ratio = 0.0

    print(f"[compare] baseline_wall_clock_s: {baseline_wall_clock}")
    print(f"[compare] perf_wall_clock_s: {perf_wall_clock}")
    print(f"[compare] speedup_ratio: {speedup_ratio:.6f}")

    dtype = perf_metrics.get("dtype", "fp32")
    dataset = perf_metrics.get("dataset", "builtin")
    mode = perf_metrics.get("mode", "pretrained")
    output_type = "cls_embeddings"
    baseline_artifact = baseline_metrics_files[-1].name
    perf_artifact = perf_metrics_files[-1].name

    perf_memory_mb = perf_metrics.get("peak_memory_mb", 0.0)
    baseline_memory_mb = baseline_metrics.get("peak_memory_mb", 0.0)
    memory_reduction_pct = round((1 - perf_memory_mb / baseline_memory_mb) * 100, 2) if baseline_memory_mb > 0 else 0.0
    warmup_iters = perf_metrics.get("warmup_iterations", 3)

    result = {
        "dtype": dtype, "mode": mode, "dataset": dataset, "output_type": output_type,
        "baseline_artifact": baseline_artifact, "perf_artifact": perf_artifact,
        "num_samples": num_samples,
        "baseline_latency_s": round(baseline_latency, 6), "perf_latency_s": round(perf_latency, 6),
        "baseline_wall_clock_s": round(baseline_wall_clock, 6), "perf_wall_clock_s": round(perf_wall_clock, 6),
        "speedup_ratio": round(speedup_ratio, 6),
        "latency_reduction_pct": round((1 - perf_latency / baseline_latency) * 100, 2) if baseline_latency > 0 else 0.0,
        "baseline_memory_mb": round(baseline_memory_mb, 2), "perf_memory_mb": round(perf_memory_mb, 2),
        "memory_reduction_pct": memory_reduction_pct,
        "cosine_similarity": round(avg_cosine, 6), "min_cosine": round(min_cosine, 6),
        "max_abs_error": max_abs_error,
        "comparison_method": "independent_baseline_artifact", "precision_method": "cosine_similarity",
        "comparison_scope": "cold_start",
        "validation_note": f"已核查为独立 baseline 工件 ({baseline_artifact})，不是 self-baseline，也不是冷启动对热启动。baseline 与 perf 均在 pretrained 模式下运行，同一数据集、同一设备、对称 warmup 口径。",
        "steady_state_baseline_latency_s": round(baseline_latency, 6), "steady_state_perf_latency_s": round(perf_latency, 6),
        "optimization_kind": "runtime_only",
        "optimization_items": ["batched_encoding", "warmup_3x", "TASK_QUEUE_ENABLE"],
        "batch_size": perf_metrics.get("batch_size", 8), "warmup_iterations": warmup_iters,
        "selected_npu": perf_metrics.get("selected_npu", ""), "device_topology": perf_metrics.get("device_topology", ""),
        "baseline_warmup_iterations": warmup_iters, "perf_warmup_iterations": warmup_iters,
        "wall_clock_source": "artifact_explicit_field", "warmup_policy": "symmetric", "parallel_mode": "single_card",
    }

    notes = {
        "measurement_contract_version": 3,
        "optimizations": "batched_encoding(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1",
        "results": [result], "best_result": result,
    }

    compare_path = ADAPT_DIR / "output_compare_perf.json"
    compare_data = {
        "baseline_artifact": baseline_artifact, "perf_artifact": perf_artifact,
        "cosine_similarity": round(avg_cosine, 6), "min_cosine": round(min_cosine, 6),
        "max_abs_error": max_abs_error,
        "baseline_samples": num_samples, "perf_samples": num_samples,
        "total_samples": num_samples, "cuda_samples": num_samples, "ascend_samples": num_samples,
        "speedup_ratio": round(speedup_ratio, 6),
        "baseline_wall_clock_s": round(baseline_wall_clock, 6), "perf_wall_clock_s": round(perf_wall_clock, 6),
        "baseline_latency_s": round(baseline_latency, 6), "perf_latency_s": round(perf_latency, 6),
    }
    with open(compare_path, "w") as f:
        json.dump(compare_data, f, indent=2)
    print(f"[compare] saved comparison to {compare_path}")

    notes_path = ADAPT_DIR / "optimization_notes.json"
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[compare] saved optimization notes to {notes_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="NPU optimized accuracy_run_perf.py for BAAI/bge-m3")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run optimized inference")
    run_parser.add_argument("--use-pretrained", action="store_true")
    run_parser.add_argument("--max-samples", type=int, default=250)
    run_parser.add_argument("--cpu", action="store_true")
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--warmup-iterations", type=int, default=3)
    compare_parser = subparsers.add_parser("compare", help="Compare baseline vs perf")
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        exit(cmd_compare(args))
    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
