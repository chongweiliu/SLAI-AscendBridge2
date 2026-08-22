#!/usr/bin/env python3
"""
accuracy_run_perf.py for openai-community/gpt2 — NPU 性能优化版（runtime_only）。

优化策略：
  - warmup(3x) 预热（与 baseline 对称）
  - TASK_QUEUE_ENABLE=1 异步算子下发
  - batched teacher-forcing（batch_size=4）减少 Python 循环开销

合同：teacher-forcing last_token_logits + perplexity，与 accuracy_run.py 同口径。

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

# 国内网络环境默认走 HF 镜像（外部已设置的环境变量优先）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as transformers_set_seed

from datasets import load_from_disk

PERF_SUFFIX = "_perf"
BATCH_SIZE = 4
WARMUP_ITERATIONS = 3

# 数据集配置（与 accuracy_run.py 完全一致）
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_TEXT_FIELD = "article"


def load_benchmark_texts() -> tuple[list[str], str]:
    """加载测试数据集文本，返回 (texts, dataset_name)。与 accuracy_run.py 同口径。"""
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        print(f"[perf] loading dataset from {wikitext_path}")
        ds = load_from_disk(str(wikitext_path))
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
    """加载模型 (CausalLM) — 与 accuracy_run.py 完全一致"""
    MODEL_ID = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)

    if use_pretrained:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto", cache_dir=cache_dir)
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(device)
    model.eval()
    return model, tokenizer


def batched_teacher_forcing(model, tokenizer, texts, device, batch_size, max_length=512):
    """批量 teacher-forcing 前向推理。

    对每个 batch 做一次前向，提取每条样本最后一个 token 的 logits，
    并计算每条样本的 perplexity。
    """
    device_str = str(device)
    all_logits = []
    all_ppl = []
    all_latencies = []

    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")

    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]

            # Tokenize batch（padding 到 batch 内最长）
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)

            sample_start = time.perf_counter()
            outputs = model(**encoded)
            logits = outputs.logits  # [B, seq_len, vocab_size]

            if device_str.startswith("npu"):
                torch.npu.synchronize()
            elif device_str.startswith("cuda"):
                torch.cuda.synchronize()
            batch_latency = time.perf_counter() - sample_start
            # 转换为 per-sample latency 以与 baseline 口径一致
            per_sample_latency = batch_latency / len(batch_texts)
            all_latencies.extend([per_sample_latency] * len(batch_texts))

            # 提取每条样本最后一个非 pad token 的 logits
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            for j in range(len(batch_texts)):
                # 找到第 j 条样本的最后一个非 pad 位置
                seq_len_j = int(attention_mask[j].sum().item())
                last_logits = logits[j, seq_len_j - 1, :].cpu()
                all_logits.append(last_logits)

                # 计算第 j 条样本的 perplexity
                sample_ids = input_ids[j, :seq_len_j].unsqueeze(0)  # [1, seq_len_j]
                sample_logits = logits[j, :seq_len_j, :].unsqueeze(0)  # [1, seq_len_j, vocab_size]
                shift_logits = sample_logits[..., :-1, :].contiguous()
                shift_labels = sample_ids[..., 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                ppl = torch.exp(loss).item()
                all_ppl.append(ppl)

            del encoded, outputs, logits
            if device_str.startswith("npu"):
                torch.npu.empty_cache()
            elif device_str.startswith("cuda"):
                torch.cuda.empty_cache()

    return all_logits, all_ppl, all_latencies


def cmd_run(args):
    """run 子命令：执行 batched teacher-forcing 前向推理，产出 perf 工件。"""
    MODEL_ID = "openai-community/gpt2"
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
    # GPT-2 tokenizer 没有 pad_token，批量 padding 需要设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    first_device = next(model.parameters()).device
    device_short = first_device.type
    mode_str = "pretrained" if args.use_pretrained else "config"

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    device_ids = None
    if first_device.index is not None:
        device_ids = [first_device.index]

    # --- warmup（与 baseline 对称：WARMUP_ITERATIONS 次） ---
    print(f"[perf] warmup {WARMUP_ITERATIONS} iterations (batch_size={BATCH_SIZE})...")
    warmup_texts = texts[:BATCH_SIZE] if len(texts) >= BATCH_SIZE else [texts[0] if texts else "Hello, benchmark."]
    warmup_encoded = tokenizer(
        warmup_texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(first_device)
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(**warmup_encoded)
    if device_short == "npu":
        torch.npu.synchronize()
    elif device_short == "cuda":
        torch.cuda.synchronize()
    print(f"[perf] warmup done")

    # --- 正式推理 ---
    print(f"[perf] running batched teacher-forcing (batch_size={BATCH_SIZE})...")
    run_start = time.perf_counter()

    all_logits, all_ppl, all_latencies = batched_teacher_forcing(
        model, tokenizer, texts, first_device, BATCH_SIZE
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
    output_data = {
        "logits": all_logits,
        "perplexity": all_ppl,
    }
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

    # selected_npu 信息
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

    ppl_avg = round(sum(all_ppl) / len(all_ppl), 2) if all_ppl else None
    print(f"[perf] outputs saved to {outputs_path}")
    print(f"[perf] metrics saved to {metrics_path}")
    print(f"[perf] wall_clock_s={wall_clock_s}, avg_latency_s={avg_latency_s}")
    print(f"[perf] num_samples={len(all_logits)}, ppl_avg={ppl_avg}")
    print(f"[perf] peak_memory_mb={round(peak_memory_mb, 2)}")


def cmd_compare(args):
    """compare 子命令：对比 baseline vs perf 的 logits 和 perplexity，写出 optimization_notes.json。"""
    ADAPT_DIR = Path(__file__).resolve().parent

    # 按 metrics JSON 内 start/end_time 选最近一轮 baseline/perf 工件
    def _find_artifacts():
        """查找 baseline 和 perf 的 metrics + outputs 工件。"""
        all_metrics = sorted(ADAPT_DIR.glob("benchmark_metrics_*.json"))
        perf_metrics = [f for f in all_metrics if PERF_SUFFIX in f.name]
        baseline_metrics = [f for f in all_metrics if PERF_SUFFIX not in f.name]

        if not perf_metrics:
            raise FileNotFoundError(f"找不到 perf metrics 文件 ({ADAPT_DIR}/benchmark_metrics_*{PERF_SUFFIX}.json)")
        if not baseline_metrics:
            raise FileNotFoundError(f"找不到 baseline metrics 文件 ({ADAPT_DIR}/benchmark_metrics_*.json)")

        # 选最近一轮：按 end_time
        def _get_end_time(path):
            try:
                return json.loads(path.read_text()).get("end_time", "")
            except Exception:
                return ""

        baseline_path = max(baseline_metrics, key=_get_end_time)
        perf_path = max(perf_metrics, key=_get_end_time)

        b_metric = json.loads(baseline_path.read_text())
        p_metric = json.loads(perf_path.read_text())

        # 找对应的 outputs 文件
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

    # 校验 mode / dataset / dtype 一致
    if b_metric.get("mode") != p_metric.get("mode"):
        raise ValueError(f"mode 不一致: baseline={b_metric.get('mode')} vs perf={p_metric.get('mode')}")
    if b_metric.get("dataset") != p_metric.get("dataset"):
        raise ValueError(f"dataset 不一致: baseline={b_metric.get('dataset')} vs perf={p_metric.get('dataset')}")
    if b_metric.get("dtype") != p_metric.get("dtype"):
        raise ValueError(f"dtype 不一致: baseline={b_metric.get('dtype')} vs perf={p_metric.get('dtype')}")
    if b_metric.get("num_samples") != p_metric.get("num_samples"):
        raise ValueError(f"num_samples 不一致: baseline={b_metric.get('num_samples')} vs perf={p_metric.get('num_samples')}，请重跑使样本数一致")

    # 加载 outputs
    b_data = torch.load(baseline_outputs_path, map_location="cpu", weights_only=False)
    p_data = torch.load(perf_outputs_path, map_location="cpu", weights_only=False)

    b_logits = b_data.get("logits", [])
    p_logits = p_data.get("logits", [])
    b_ppl = b_data.get("perplexity", [])
    p_ppl = p_data.get("perplexity", [])

    n = min(len(b_logits), len(p_logits))
    if n == 0:
        raise ValueError("logits 列表为空，无法对比")

    # 对比 logits
    cosine_sims = []
    max_errors = []
    for i in range(n):
        b_t = b_logits[i].float().flatten()
        p_t = p_logits[i].float().flatten()
        if b_t.shape != p_t.shape:
            raise ValueError(f"样本 {i} logits shape 不一致: {b_t.shape} vs {p_t.shape}")
        cos_sim = torch.nn.functional.cosine_similarity(b_t.unsqueeze(0), p_t.unsqueeze(0), dim=1).item()
        max_err = torch.max(torch.abs(b_t - p_t)).item()
        # 夹紧 cosine 到 [0, 1]
        cos_sim = min(max(cos_sim, 0.0), 1.0)
        cosine_sims.append(cos_sim)
        max_errors.append(max_err)

    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    min_cosine = min(cosine_sims)
    max_abs_error = max(max_errors)

    # 对比 perplexity
    ppl_diffs = []
    for i in range(min(len(b_ppl), len(p_ppl))):
        c, a = b_ppl[i], p_ppl[i]
        if c != 0:
            ppl_diffs.append(abs(c - a) / c)
        else:
            ppl_diffs.append(0.0)
    ppl_avg_rel_diff = sum(ppl_diffs) / len(ppl_diffs) if ppl_diffs else 0.0
    ppl_max_rel_diff = max(ppl_diffs) if ppl_diffs else 0.0

    # 计算 speedup
    baseline_wall_clock_s = b_metric.get("wall_clock_s")
    perf_wall_clock_s = p_metric.get("wall_clock_s")

    if not baseline_wall_clock_s or not perf_wall_clock_s:
        raise ValueError("metrics 缺少 wall_clock_s，无法计算 speedup")

    # 非生成类任务：latency_s 必须与 wall_clock_s / num_samples 一致
    baseline_latency_s = round(baseline_wall_clock_s / n, 6)
    perf_latency_s = round(perf_wall_clock_s / n, 6)

    speedup_ratio = round(baseline_wall_clock_s / perf_wall_clock_s, 6)
    latency_reduction_pct = round((1 - perf_latency_s / baseline_latency_s) * 100, 2) if baseline_latency_s and perf_latency_s else None

    # selected_npu 信息（从 perf 工件继承）
    selected_npu = p_metric.get("selected_npu", 0)
    selected_npus = p_metric.get("selected_npus", [selected_npu])
    device_topology = p_metric.get("device_topology", f"1d:{selected_npu}")

    print(f"\n[compare] Results:")
    print(f"  cosine_similarity: {avg_cosine:.8f} (min={min_cosine:.8f})")
    print(f"  max_abs_error: {max_abs_error}")
    print(f"  ppl_avg_rel_diff: {ppl_avg_rel_diff:.6f}")
    print(f"  baseline_wall_clock_s: {baseline_wall_clock_s}")
    print(f"  perf_wall_clock_s: {perf_wall_clock_s}")
    print(f"  speedup_ratio: {speedup_ratio}")
    print(f"  baseline_latency_s: {baseline_latency_s}")
    print(f"  perf_latency_s: {perf_latency_s}")

    # 写 output_compare_perf.json
    compare_result = {
        "cosine_similarity": avg_cosine,
        "min_cosine_similarity": min_cosine,
        "max_abs_error": max_abs_error,
        "ppl_avg_rel_diff": ppl_avg_rel_diff,
        "ppl_max_rel_diff": ppl_max_rel_diff,
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
    print(f"\n[compare] compare result saved to {compare_path}")

    # 更新 perf metrics，写入 output_compare + 修正 latency_s
    p_metric["output_compare"] = compare_result
    p_metric["latency_s"] = perf_latency_s
    perf_metrics_path.write_text(json.dumps(p_metric, indent=2))

    # 同步修正 baseline metrics 的 latency_s
    b_metric["latency_s"] = baseline_latency_s
    baseline_metrics_path.write_text(json.dumps(b_metric, indent=2))

    # 写 optimization_notes.json
    baseline_artifact = baseline_metrics_path.name
    perf_artifact = perf_metrics_path.name

    optimizations_str = "runtime_only: warmup(3x) + TASK_QUEUE_ENABLE=1 + batched_teacher_forcing(bs=4)"

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
        "ppl_avg_rel_diff_pct": round(ppl_avg_rel_diff * 100, 4),
        "ppl_max_rel_diff_pct": round(ppl_max_rel_diff * 100, 4),
        "comparison_method": "independent_baseline_artifact",
        "precision_method": "cosine_similarity",
        "comparison_scope": "steady_state",
        "validation_note": "独立 baseline 工件对比，非 self-baseline；baseline 与 perf 对称 warmup(3x)，同卡串行，batched teacher-forcing 合同。",
        "steady_state_baseline_latency_s": baseline_latency_s,
        "steady_state_perf_latency_s": perf_latency_s,
        "optimization_items": ["warmup_3x", "TASK_QUEUE_ENABLE", "batched_teacher_forcing_bs4"],
        "optimization_kind": "runtime_only",
        "selected_npu": selected_npu,
        "selected_npus": selected_npus,
        "device_topology": device_topology,
        "parallel_mode": p_metric.get("parallel_mode", "single"),
        "task_queue_enable": p_metric.get("task_queue_enable", "1"),
        "batch_size": p_metric.get("batch_size", BATCH_SIZE),
        "baseline_warmup_iterations": b_metric.get("warmup_iterations", WARMUP_ITERATIONS),
        "perf_warmup_iterations": p_metric.get("warmup_iterations", WARMUP_ITERATIONS),
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
    parser = argparse.ArgumentParser(description="accuracy_run_perf.py for openai-community/gpt2 (runtime_only)")
    subparsers = parser.add_subparsers(dest="command")

    # run 子命令
    run_parser = subparsers.add_parser("run", help="执行 batched teacher-forcing 前向推理")
    run_parser.add_argument("--use-pretrained", action="store_true", help="加载预训练权重")
    run_parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # compare子命令
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
