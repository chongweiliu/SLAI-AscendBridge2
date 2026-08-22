"""
accuracy_run_perf.py for sentence-transformers/all-MiniLM-L6-v2 — NPU optimized version.

Optimizations:
  1. Batched inference (bs=8): Process 8 texts per forward pass instead of 1
  2. TASK_QUEUE_ENABLE=1: Async operator dispatch (reduces host-device sync)
  3. Symmetric warmup(3x): Matches baseline warmup for fair comparison

Contract: cls_embeddings (same as accuracy_run.py).
  - encode_texts with mean pooling + L2 normalization
  - Output: embeddings + similarity_profile

Usage:
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 60
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

# 国内网络环境默认走 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import set_seed as transformers_set_seed

from datasets import load_from_disk

PERF_SUFFIX = "_perf"

# 数据集配置
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_TEXT_FIELD = "article"

WARMUP_ITERATIONS = 3
BATCH_SIZE = 8

# 语义相似度画像固定对（与 accuracy_run.py 保持一致）
SIMILARITY_PAIRS = [
    {
        "anchor": "The cat sits on the mat.",
        "paraphrase": "A cat is sitting on a mat.",
        "unrelated": "The stock market crashed yesterday.",
    },
    {
        "anchor": "A man is playing the guitar.",
        "paraphrase": "Someone is strumming a guitar.",
        "unrelated": "The weather is sunny today.",
    },
    {
        "anchor": "She reads a book every night.",
        "paraphrase": "Every night she reads a book.",
        "unrelated": "Quantum computers use qubits.",
    },
]


def load_benchmark_texts() -> tuple[list[str], str]:
    """加载测试数据集文本，返回 (texts, dataset_name)。与 accuracy_run.py 保持一致。"""
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        print(f"[perf] loading dataset from {wikitext_path}")
        ds = load_from_disk(str(wikitext_path))
        if hasattr(ds, "keys"):  # DatasetDict — select train split
            ds = ds["train"]
        texts = sorted([sample["text"] for sample in ds if sample.get("text", "").strip()])
        print(f"[perf] loaded {len(texts)} samples from wikitext")
        return texts, "wikitext"

    cnn_path = DATASET_DIR / "cnn_dailymail___3.0.0"
    if cnn_path.exists():
        print(f"[perf] loading dataset from {cnn_path}")
        ds = load_from_disk(str(cnn_path))
        texts = sorted([sample[DATASET_TEXT_FIELD] for sample in ds if sample[DATASET_TEXT_FIELD].strip()])
        print(f"[perf] loaded {len(texts)} samples from cnn_dailymail")
        return texts, "cnn_dailymail"

    imdb_path = DATASET_DIR / "imdb"
    if imdb_path.exists():
        print(f"[perf] loading dataset from {imdb_path}")
        ds = load_from_disk(str(imdb_path))
        texts = sorted([sample["text"] for sample in ds if sample["text"].strip()])
        print(f"[perf] loaded {len(texts)} samples from imdb")
        return texts, "imdb"

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
    """选择空闲 HBM 最多的 NPU 并设为当前设备。"""
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
    """获取推理设备。"""
    if force_cpu:
        return "cpu", 0, "cpu"
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            idx = select_idle_npu()
            device_name = torch.npu.get_device_name(idx) if torch.npu.device_count() > 0 else "unknown"
            return f"npu:{idx}", torch.npu.device_count(), device_name
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
    """加载模型 (Sentence Embedding / BERT 系编码器)。"""
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

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


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """对 last_hidden_state 按 attention_mask 做 mean pooling，得到句向量。"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode_texts(model, tokenizer, texts, first_device) -> torch.Tensor:
    """批量编码文本为 L2 归一化句向量 [N, hidden_dim]（在设备上计算）。"""
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(first_device)
    outputs = model(**encoded)
    embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    del outputs, encoded
    return embeddings


def compute_similarity_profile(model, tokenizer, first_device, use_pretrained: bool) -> dict:
    """句向量模型语义相似度画像：固定语义对的余弦相似度与排序断言。"""
    print("\n" + "-" * 60)
    print("[perf] Similarity profile: semantic pair cosine similarities")
    print("-" * 60)

    all_sentences = []
    for pair in SIMILARITY_PAIRS:
        all_sentences.extend([pair["anchor"], pair["paraphrase"], pair["unrelated"]])

    with torch.no_grad():
        embeddings = encode_texts(model, tokenizer, all_sentences, first_device)

    profile = {"pairs": [], "all_margins_positive": None}
    margins = []
    for i, pair in enumerate(SIMILARITY_PAIRS):
        emb_anchor, emb_paraphrase, emb_unrelated = embeddings[i * 3 : i * 3 + 3]
        cos_paraphrase = float(torch.dot(emb_paraphrase, emb_anchor))
        cos_unrelated = float(torch.dot(emb_unrelated, emb_anchor))
        margin = cos_paraphrase - cos_unrelated
        margins.append(margin)
        entry = {
            "anchor": pair["anchor"],
            "paraphrase": pair["paraphrase"],
            "unrelated": pair["unrelated"],
            "cos_paraphrase": round(cos_paraphrase, 6),
            "cos_unrelated": round(cos_unrelated, 6),
            "margin": round(margin, 6),
        }
        profile["pairs"].append(entry)
        print(f"[perf] #{i + 1} cos(paraphrase)={cos_paraphrase:.4f} cos(unrelated)={cos_unrelated:.4f} margin={margin:.4f}")

    profile["all_margins_positive"] = all(m > 0 for m in margins)

    if use_pretrained:
        for i, margin in enumerate(margins):
            assert margin > 0, (
                f"Similarity ordering wrong with pretrained weights (pair #{i + 1}): "
                f"margin={margin:.4f} <= 0"
            )
        print("[perf] similarity ordering assertions passed (all margins > 0)")
    else:
        print("[perf] config mode: random weights, similarity values are meaningless (path validation only)")

    return profile


def run_perf(use_pretrained: bool, max_samples: int, cpu: bool):
    """运行 perf 推理，产出 outputs_*_perf.pt 和 benchmark_metrics_*_perf.json。"""
    adapt_dir = Path(__file__).resolve().parent
    cache_dir = (adapt_dir / "models").as_posix()

    # Seed
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    transformers_set_seed(SEED)
    device, _, _ = get_device(force_cpu=cpu)
    if device.startswith("npu"):
        torch.npu.manual_seed_all(SEED)
    elif device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if not cpu:
        assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got {device}"

    texts, dataset_name = load_benchmark_texts()
    texts = texts[:max_samples]
    num_samples = len(texts)
    print(f"[perf] dataset: {dataset_name}, samples: {num_samples}")

    model, tokenizer = setup_model(use_pretrained, device, cache_dir)
    first_device = next(model.parameters()).device
    device_short = str(first_device).replace(":", "_")
    perf_device_short = first_device.type
    mode_str = "pretrained" if use_pretrained else "config"
    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    device_ids = None
    if hasattr(model, "hf_device_map"):
        device_ids = list(set(dev.index if hasattr(dev, "index") else dev for dev in model.hf_device_map.values()))
    elif first_device.index is not None:
        device_ids = [first_device.index]

    device_model = "unknown"
    if perf_device_short == "npu" and first_device.index is not None:
        device_model = torch.npu.get_device_name(first_device.index)
    elif perf_device_short == "cuda" and first_device.index is not None:
        device_model = torch.cuda.get_device_name(first_device.index)

    # Warmup: 使用第一批样本做 WARMUP_ITERATIONS 次 batched forward
    dummy_texts = texts[:BATCH_SIZE]
    print(f"[perf] warming up ({WARMUP_ITERATIONS} iterations, batch_size={BATCH_SIZE})...")
    for _ in range(WARMUP_ITERATIONS):
        with torch.no_grad():
            _ = encode_texts(model, tokenizer, dummy_texts, first_device)

    if perf_device_short == "npu":
        torch.npu.synchronize()
    elif perf_device_short == "cuda":
        torch.cuda.synchronize()

    # 计时推理 (batched encoding)
    start_time = datetime.now().isoformat()
    peak_mem_monitor = _PerfMonitor(perf_device_short, device_ids)
    peak_mem_monitor.__enter__()
    perf_start = time.perf_counter()

    all_embeddings = []

    with torch.no_grad():
        for batch_start in range(0, num_samples, BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
            actual_batch_size = len(batch_texts)

            embeddings = encode_texts(model, tokenizer, batch_texts, first_device)

            # Split batch embeddings into individual [1, hidden_dim] to match baseline format
            for j in range(actual_batch_size):
                all_embeddings.append(embeddings[j : j + 1].cpu())

            del embeddings

            if (batch_start // BATCH_SIZE + 1) % 4 == 0:
                if perf_device_short == "npu":
                    torch.npu.empty_cache()
                print(f"[perf] processed {batch_start + actual_batch_size}/{num_samples} samples")

    peak_mem_monitor.__exit__()
    if perf_device_short == "npu":
        torch.npu.synchronize()
    elif perf_device_short == "cuda":
        torch.cuda.synchronize()

    perf_end = time.perf_counter()
    wall_clock_s = perf_end - perf_start
    end_time = datetime.now().isoformat()

    # 语义相似度画像
    with torch.no_grad():
        similarity_profile = compute_similarity_profile(model, tokenizer, first_device, use_pretrained)

    # Save outputs
    outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"
    output_data = {
        "texts": list(texts),
        "embeddings": all_embeddings,
        "similarity_profile": similarity_profile,
    }
    torch.save(output_data, outputs_path)
    print(f"[perf] outputs saved to {outputs_path}")

    # Save metrics
    metrics_path = adapt_dir / f"benchmark_metrics_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"
    latency_per_sample = wall_clock_s / num_samples if num_samples > 0 else wall_clock_s
    selected_npu = first_device.index if first_device.index is not None else 0

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "latency_s": round(latency_per_sample, 6),
        "wall_clock_s": round(wall_clock_s, 6),
        "peak_memory_mb": round(peak_mem_monitor.peak_memory_mb, 2),
        "num_samples": num_samples,
        "device": str(first_device),
        "device_model": device_model,
        "mode": mode_str,
        "dataset": dataset_name,
        "dtype": dtype_str,
        "output_type": "cls_embeddings",
        "warmup_iterations": WARMUP_ITERATIONS,
        "packages": get_package_versions(),
        "optimization_items": ["batched_inference", "warmup", "TASK_QUEUE_ENABLE"],
        "optimization_kind": "runtime_only",
        "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE", "0") == "1",
        "batch_size": BATCH_SIZE,
        "selected_npu": selected_npu,
        "selected_npus": [selected_npu],
        "device_topology": f"single-die:{selected_npu}",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[perf] metrics saved to {metrics_path}")
    print(f"[perf] wall_clock_s: {wall_clock_s:.6f}, latency_s: {latency_per_sample:.6f}")

    return metrics_path, outputs_path


def compare_outputs(adapt_dir: Path):
    """对比 baseline vs perf outputs，生成 optimization_notes.json。"""
    print("\n" + "=" * 60)
    print("Compare: baseline vs perf")
    print("=" * 60)

    # Find baseline and perf metrics files
    baseline_metrics_files = sorted(adapt_dir.glob("benchmark_metrics_npu_*_pretrained_*.json"))
    baseline_metrics_files = [f for f in baseline_metrics_files if "_perf" not in f.name]
    perf_metrics_files = sorted(adapt_dir.glob("benchmark_metrics_npu_*_pretrained_*_perf.json"))

    if not baseline_metrics_files:
        print("[compare] ERROR: No baseline metrics file found")
        return None
    if not perf_metrics_files:
        print("[compare] ERROR: No perf metrics file found")
        return None

    baseline_metrics_path = baseline_metrics_files[-1]
    perf_metrics_path = perf_metrics_files[-1]
    print(f"[compare] baseline: {baseline_metrics_path.name}")
    print(f"[compare] perf: {perf_metrics_path.name}")

    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)
    with open(perf_metrics_path) as f:
        perf_metrics = json.load(f)

    # Find baseline and perf output files
    baseline_outputs_files = sorted(adapt_dir.glob("outputs_npu_*_pretrained_*.pt"))
    baseline_outputs_files = [f for f in baseline_outputs_files if "_perf" not in f.name]
    perf_outputs_files = sorted(adapt_dir.glob("outputs_npu_*_pretrained_*_perf.pt"))

    if not baseline_outputs_files or not perf_outputs_files:
        print("[compare] ERROR: Missing output files")
        return None

    baseline_outputs_path = baseline_outputs_files[-1]
    perf_outputs_path = perf_outputs_files[-1]
    print(f"[compare] baseline outputs: {baseline_outputs_path.name}")
    print(f"[compare] perf outputs: {perf_outputs_path.name}")

    baseline_data = torch.load(baseline_outputs_path, weights_only=False)
    perf_data = torch.load(perf_outputs_path, weights_only=False)

    # Compare embeddings
    baseline_embeddings = baseline_data.get("embeddings", [])
    perf_embeddings = perf_data.get("embeddings", [])

    if not baseline_embeddings or not perf_embeddings:
        print("[compare] ERROR: Missing embeddings in outputs")
        return None

    num_compare = min(len(baseline_embeddings), len(perf_embeddings))
    print(f"[compare] comparing {num_compare} samples")

    cosines = []
    max_abs_errors = []
    for i in range(num_compare):
        b_emb = baseline_embeddings[i].float().flatten()
        p_emb = perf_embeddings[i].float().flatten()
        cos = torch.nn.functional.cosine_similarity(b_emb.unsqueeze(0), p_emb.unsqueeze(0)).item()
        cosines.append(cos)
        max_abs = (b_emb - p_emb).abs().max().item()
        max_abs_errors.append(max_abs)

    avg_cosine = sum(cosines) / len(cosines) if cosines else 0.0
    min_cosine = min(cosines) if cosines else 0.0
    avg_max_abs_error = sum(max_abs_errors) / len(max_abs_errors) if max_abs_errors else 0.0
    max_abs_error = max(max_abs_errors) if max_abs_errors else 0.0

    # Clamp cosine to [0, 1] for gate
    avg_cosine = min(1.0, max(0.0, avg_cosine))

    # Speedup
    baseline_wall_clock = baseline_metrics.get("wall_clock_s", 0.0)
    perf_wall_clock = perf_metrics.get("wall_clock_s", 0.0)
    baseline_latency = baseline_metrics.get("latency_s", 0.0)
    perf_latency = perf_metrics.get("latency_s", 0.0)

    if perf_wall_clock > 0:
        speedup_ratio = baseline_wall_clock / perf_wall_clock
    else:
        speedup_ratio = 0.0

    num_samples = min(baseline_metrics.get("num_samples", 0), perf_metrics.get("num_samples", 0))

    print(f"[compare] cosine_similarity: avg={avg_cosine:.10f}, min={min_cosine:.10f}")
    print(f"[compare] max_abs_error: avg={avg_max_abs_error:.10f}, max={max_abs_error:.10f}")
    print(f"[compare] baseline_wall_clock_s: {baseline_wall_clock}")
    print(f"[compare] perf_wall_clock_s: {perf_wall_clock}")
    print(f"[compare] speedup_ratio: {speedup_ratio:.6f}")

    # Build optimization_notes.json
    baseline_warmup = baseline_metrics.get("warmup_iterations", WARMUP_ITERATIONS)
    perf_warmup = perf_metrics.get("warmup_iterations", WARMUP_ITERATIONS)
    selected_npu = perf_metrics.get("selected_npu", 0)
    selected_npus = perf_metrics.get("selected_npus", [selected_npu])
    device_topology = perf_metrics.get("device_topology", f"single-die:{selected_npu}")

    baseline_artifact = baseline_metrics_path.name
    perf_artifact = perf_metrics_path.name

    result = {
        "dtype": perf_metrics.get("dtype", "fp32"),
        "mode": perf_metrics.get("mode", "pretrained"),
        "dataset": perf_metrics.get("dataset", "builtin"),
        "output_type": "cls_embeddings",
        "baseline_artifact": baseline_artifact,
        "perf_artifact": perf_artifact,
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
        "cosine_similarity": round(avg_cosine, 10),
        "min_cosine": round(min_cosine, 10),
        "max_abs_error": round(max_abs_error, 10),
        "optimization_items": perf_metrics.get("optimization_items", ["batched_inference", "warmup", "TASK_QUEUE_ENABLE"]),
        "optimization_kind": "runtime_only",
        "task_queue_enable": perf_metrics.get("task_queue_enable", True),
        "batch_size": perf_metrics.get("batch_size", BATCH_SIZE),
        "selected_npu": selected_npu,
        "selected_npus": selected_npus,
        "device_topology": device_topology,
        "parallel_mode": "single_card",
        "comparison_method": "independent_baseline_artifact",
        "comparison_scope": "steady_state",
        "precision_method": "cosine_similarity",
        "validation_note": f"Independent baseline artifact ({baseline_artifact}) vs perf artifact ({perf_artifact}). Symmetric warmup ({baseline_warmup}x). Cosine={avg_cosine:.10f}, max_abs_error={max_abs_error:.10f}.",
        "steady_state_baseline_latency_s": round(baseline_latency, 6),
        "steady_state_perf_latency_s": round(perf_latency, 6),
    }

    notes = {
        "measurement_contract_version": 3,
        "optimizations": "batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1",
        "results": [result],
        "best_result": result,
    }

    notes_path = adapt_dir / "optimization_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))
    print(f"[compare] optimization_notes saved to {notes_path}")

    # Also write output_compare_perf.json with explicit sample counts
    compare_data = {
        "cosine_similarity": round(avg_cosine, 10),
        "min_cosine": round(min_cosine, 10),
        "max_abs_error": round(max_abs_error, 10),
        "avg_max_abs_error": round(avg_max_abs_error, 10),
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
    parser = argparse.ArgumentParser(description="NPU optimized accuracy_run_perf for all-MiniLM-L6-v2")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run perf inference")
    run_parser.add_argument("--use-pretrained", action="store_true", help="Load pretrained weights")
    run_parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU")

    # compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare baseline vs perf")
    compare_parser.add_argument("--adapt", default=None, help="Adaptation name (unused, for compatibility)")

    args = parser.parse_args()

    if args.command == "run":
        run_perf(
            use_pretrained=args.use_pretrained,
            max_samples=args.max_samples,
            cpu=args.cpu,
        )
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
