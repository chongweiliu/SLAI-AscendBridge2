"""
accuracy_run_perf.py for google-t5/t5-small — NPU optimized version.

Optimizations:
  1. npu_rms_norm: Replaces T5LayerNorm.forward with torch_npu.npu_rms_norm (fused kernel)
  2. Batched teacher-forcing: Processes samples in batches of 4 (better NPU utilization)
  3. TASK_QUEUE_ENABLE=1: Async operator dispatch (reduces host-device sync)
  4. Symmetric warmup(3x): Matches baseline warmup for fair comparison

Contract: teacher-forcing logits (same as modified accuracy_run.py).
  - encoder_input = text tokens
  - decoder_input = model._shift_right(encoder_ids)
  - labels = encoder_ids (reconstruction task)
  - Output: last_token_logits + perplexity per sample

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

# Ensure model_files is on sys.path for npu_patches import
_MODEL_FILES_DIR = Path(__file__).resolve().parent / "model_files"
if str(_MODEL_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_FILES_DIR))

# 国内网络环境默认走 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import set_seed as transformers_set_seed

from datasets import load_from_disk

PERF_SUFFIX = "_perf"

# 数据集配置
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_TEXT_FIELD = "article"

MODEL_ID = "google-t5/t5-small"

WARMUP_ITERATIONS = 3
BATCH_SIZE = 4


def load_benchmark_texts() -> tuple[list[str], str]:
    """加载测试数据集文本，返回 (texts, dataset_name)。

    与 accuracy_run.py 保持一致。
    """
    cnn_path = DATASET_DIR / "cnn_dailymail___3.0.0"
    if cnn_path.exists():
        print(f"[perf] loading dataset from {cnn_path}")
        ds = load_from_disk(str(cnn_path))
        texts = sorted(["summarize: " + sample[DATASET_TEXT_FIELD] for sample in ds if sample[DATASET_TEXT_FIELD].strip()])
        print(f"[perf] loaded {len(texts)} samples from cnn_dailymail")
        return texts, "cnn_dailymail"

    print("[perf] using built-in benchmark texts (translate/summarize profile)")
    builtin_texts = [
        "translate English to German: The house is wonderful.",
        "translate English to German: I drink coffee every morning.",
        "translate English to German: She reads a book in the garden.",
        "translate English to German: The weather is nice today.",
        "translate English to German: We go to school by bus.",
        "translate English to German: He plays football with his friends.",
        "translate English to German: The cat sleeps on the sofa.",
        "translate English to German: My brother works in a hospital.",
        "translate English to German: They travel to Berlin every summer.",
        "translate English to German: The children sing a happy song.",
        "translate English to German: I like bread with butter and jam.",
        "translate English to German: The train arrives at noon.",
        "translate English to German: She writes a letter to her grandmother.",
        "translate English to German: The mountains are covered with snow.",
        "translate English to German: We eat dinner at seven o'clock.",
        "translate English to German: The dog runs quickly through the park.",
        "translate English to German: He buys fresh fruit at the market.",
        "translate English to German: The students learn French at school.",
        "translate English to German: My mother bakes a chocolate cake.",
        "translate English to German: The river flows through the old town.",
        "translate English to German: I watch the stars at night.",
        "translate English to German: The teacher explains the lesson clearly.",
        "translate English to German: They build a new bridge near the city.",
        "translate English to German: She dances to the music happily.",
        "translate English to German: The farmer feeds the animals every day.",
        "translate English to German: We visit our friends on Sunday.",
        "translate English to German: The library has many old books.",
        "translate English to German: He rides his bicycle to work.",
        "translate English to German: The soup tastes very good.",
        "translate English to German: I open the window in the morning.",
        "translate English to German: The birds fly over the lake.",
        "translate English to German: She wears a red dress to the party.",
        "translate English to German: The shop closes at six o'clock.",
        "translate English to German: We listen to music in the evening.",
        "translate English to German: The doctor helps sick people.",
        "translate English to German: He drinks tea with lemon.",
        "translate English to German: The garden is full of flowers.",
        "translate English to German: They celebrate the festival together.",
        "translate English to German: I walk to the market with my father.",
        "translate English to German: The moon shines brightly tonight.",
        "translate English to French: Good morning, my friend.",
        "translate English to French: The city is very beautiful.",
        "translate English to French: I love this small restaurant.",
        "translate English to French: She speaks three languages.",
        "translate English to French: The museum opens at nine.",
        "translate English to French: We need more time to decide.",
        "translate English to French: The sea is calm today.",
        "translate English to French: He cooks dinner for the family.",
        "translate English to French: The flowers smell wonderful.",
        "translate English to French: I lost my umbrella yesterday.",
        "translate English to French: The hotel is near the beach.",
        "translate English to French: They arrive tomorrow morning.",
        "translate English to French: The concert starts at eight.",
        "translate English to French: She smiles at the camera.",
        "summarize: The quick brown fox jumped over the lazy dog. The dog did not notice and kept sleeping. The fox ran away into the forest.",
        "summarize: Tom wanted to buy a new bicycle. He saved money for six months. Finally he bought a red bicycle and rode it to school.",
        "summarize: The city built a new park near the river. Families come to play and relax. The park is open every day.",
        "summarize: Scientists studied the sleep of students. They found that enough sleep improves memory. Students should sleep eight hours.",
        "summarize: A small bird built a nest in the tree. It laid three eggs in spring. The eggs hatched after two weeks.",
        "summarize: The farmer planted wheat in the field. It rained often during the summer. The harvest was very good this year.",
        "summarize: Anna learned to play the piano. She practiced every day after school. After one year she played her first concert.",
        "summarize: The old bridge was damaged by the flood. Workers repaired it during the winter. Now cars can cross the river again.",
        "summarize: A new library opened in the town center. It has books, computers and a reading room. Many people visit it every week.",
        "summarize: The team trained hard for the final match. They won the game two to one. The fans celebrated in the streets.",
        "summarize: Maria traveled to Italy last summer. She visited Rome and Venice. She enjoyed the food and the art.",
        "summarize: The school started a garden project. The children grow vegetables and flowers. They learn about nature and teamwork.",
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
            device_name = torch.npu.get_device_name(idx)
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
    """加载模型 (Seq2Seq, T5)。加载后应用 NPU patch。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)

    if use_pretrained:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype="auto", cache_dir=cache_dir)
        model = model.to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_config(config)
        model = model.to(device)
    model.eval()

    # Apply NPU patches (npu_rms_norm replacing T5LayerNorm)
    try:
        from npu_patches import apply_npu_patches

        apply_npu_patches()
    except ImportError:
        print("[perf] WARNING: npu_patches not found, running without fusion ops")

    return model, tokenizer


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

    pad_token_id = tokenizer.pad_token_id
    vocab_size = model.config.vocab_size

    # Warmup: 使用第一个样本做 WARMUP_ITERATIONS 次 batched forward
    dummy_texts = texts[:BATCH_SIZE]
    dummy_inputs = tokenizer(dummy_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(first_device)
    dummy_decoder = model._shift_right(dummy_inputs.input_ids)
    print(f"[perf] warming up ({WARMUP_ITERATIONS} iterations, batch_size={BATCH_SIZE})...")
    for _ in range(WARMUP_ITERATIONS):
        with torch.no_grad():
            _ = model(input_ids=dummy_inputs.input_ids, decoder_input_ids=dummy_decoder, attention_mask=dummy_inputs.attention_mask)

    if perf_device_short == "npu":
        torch.npu.synchronize()
    elif perf_device_short == "cuda":
        torch.cuda.synchronize()

    # 计时推理 (batched teacher-forcing)
    start_time = datetime.now().isoformat()
    perf_start = time.perf_counter()
    peak_mem_monitor = _PerfMonitor(perf_device_short, device_ids)
    peak_mem_monitor.__enter__()

    all_logits = []
    all_ppl = []

    with torch.no_grad():
        for batch_start in range(0, num_samples, BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
            actual_batch_size = len(batch_texts)

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(first_device)

            encoder_ids = inputs.input_ids  # [batch, max_len]
            attention_mask = inputs.attention_mask  # [batch, max_len]
            decoder_input_ids = model._shift_right(encoder_ids)  # [batch, max_len]
            labels = encoder_ids.clone()  # [batch, max_len]
            labels[labels == pad_token_id] = -100  # ignore padding in loss

            outputs = model(
                input_ids=encoder_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )

            # Extract per-sample results
            for j in range(actual_batch_size):
                actual_len = attention_mask[j].sum().item()
                # Last token logits: position actual_len - 1 (same as baseline's -1 for non-padded)
                last_logits = outputs.logits[j, actual_len - 1, :].cpu()  # [vocab_size]
                all_logits.append(last_logits)

                # PPL: CE over real positions only
                sample_logits = outputs.logits[j, :actual_len, :]  # [actual_len, V]
                sample_labels = labels[j, :actual_len]  # [actual_len] (no -100 since all real)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fct(sample_logits, sample_labels)
                ppl = torch.exp(loss).item()
                all_ppl.append(ppl)

            del outputs, inputs, encoder_ids, attention_mask, decoder_input_ids, labels

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

    # Save outputs
    outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"
    output_data = {
        "logits": all_logits,
        "perplexity": all_ppl,
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
        "output_type": "logits",
        "warmup_iterations": WARMUP_ITERATIONS,
        "packages": get_package_versions(),
        "optimization_items": ["npu_rms_norm", "batched_inference", "warmup", "TASK_QUEUE_ENABLE"],
        "optimization_kind": "fusion_operator",
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

    # Compare logits
    baseline_logits = baseline_data.get("logits", [])
    perf_logits = perf_data.get("logits", [])

    if not baseline_logits or not perf_logits:
        print("[compare] ERROR: Missing logits in outputs")
        return None

    num_compare = min(len(baseline_logits), len(perf_logits))
    print(f"[compare] comparing {num_compare} samples")

    cosines = []
    max_abs_errors = []
    for i in range(num_compare):
        b_log = baseline_logits[i].float().flatten()
        p_log = perf_logits[i].float().flatten()
        cos = torch.nn.functional.cosine_similarity(b_log.unsqueeze(0), p_log.unsqueeze(0)).item()
        cosines.append(cos)
        max_abs = (b_log - p_log).abs().max().item()
        max_abs_errors.append(max_abs)

    avg_cosine = sum(cosines) / len(cosines) if cosines else 0.0
    min_cosine = min(cosines) if cosines else 0.0
    avg_max_abs_error = sum(max_abs_errors) / len(max_abs_errors) if max_abs_errors else 0.0
    max_abs_error = max(max_abs_errors) if max_abs_errors else 0.0

    # Clamp cosine to [0, 1] for gate
    avg_cosine = min(1.0, max(0.0, avg_cosine))

    # Compare perplexity
    baseline_ppl = baseline_data.get("perplexity", [])
    perf_ppl = perf_data.get("perplexity", [])
    ppl_rel_diffs = []
    for i in range(min(len(baseline_ppl), len(perf_ppl))):
        b_ppl = baseline_ppl[i]
        p_ppl = perf_ppl[i]
        if b_ppl > 0 and not (isinstance(b_ppl, float) and (b_ppl != b_ppl)):  # not NaN
            rel_diff = abs(p_ppl - b_ppl) / b_ppl
            ppl_rel_diffs.append(rel_diff)
    avg_ppl_rel_diff = sum(ppl_rel_diffs) / len(ppl_rel_diffs) if ppl_rel_diffs else 0.0
    ppl_avg_rel_diff_pct = round(avg_ppl_rel_diff * 100, 4)

    # Speedup
    baseline_wall_clock = baseline_metrics.get("wall_clock_s", 0.0)
    perf_wall_clock = perf_metrics.get("wall_clock_s", 0.0)
    baseline_latency = baseline_metrics.get("latency_s", 0.0)
    perf_latency = perf_metrics.get("latency_s", 0.0)

    if perf_wall_clock > 0:
        speedup_ratio = baseline_wall_clock / perf_wall_clock
    else:
        speedup_ratio = 0.0

    if perf_latency > 0:
        latency_speedup = baseline_latency / perf_latency
    else:
        latency_speedup = 0.0

    num_samples = min(baseline_metrics.get("num_samples", 0), perf_metrics.get("num_samples", 0))

    print(f"[compare] cosine_similarity: avg={avg_cosine:.10f}, min={min_cosine:.10f}")
    print(f"[compare] max_abs_error: avg={avg_max_abs_error:.10f}, max={max_abs_error:.10f}")
    print(f"[compare] ppl_avg_rel_diff_pct: {ppl_avg_rel_diff_pct}%")
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
        "output_type": "logits",
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
        "ppl_avg_rel_diff_pct": ppl_avg_rel_diff_pct,
        "optimization_items": perf_metrics.get("optimization_items", ["npu_rms_norm", "batched_inference", "warmup", "TASK_QUEUE_ENABLE"]),
        "optimization_kind": "fusion_operator",
        "task_queue_enable": perf_metrics.get("task_queue_enable", True),
        "batch_size": perf_metrics.get("batch_size", BATCH_SIZE),
        "selected_npu": selected_npu,
        "selected_npus": selected_npus,
        "device_topology": device_topology,
        "comparison_method": "independent_baseline_artifact",
        "comparison_scope": "steady_state",
        "precision_method": "cosine_similarity",
        "validation_note": f"Independent baseline artifact ({baseline_artifact}) vs perf artifact ({perf_artifact}). Symmetric warmup ({baseline_warmup}x). Cosine={avg_cosine:.10f}, max_abs_error={max_abs_error:.10f}.",
        "steady_state_baseline_latency_s": round(baseline_latency, 6),
        "steady_state_perf_latency_s": round(perf_latency, 6),
    }

    notes = {
        "measurement_contract_version": 3,
        "optimizations": "npu_rms_norm + batched_inference(bs=4) + warmup(3x) + TASK_QUEUE_ENABLE=1",
        "results": [result],
        "best_result": result,
    }

    notes_path = adapt_dir / "optimization_notes.json"
    notes_path.write_text(json.dumps(notes, indent=2))
    print(f"[compare] optimization_notes saved to {notes_path}")

    # Also write output_compare_perf.json
    compare_data = {
        "cosine_similarity": round(avg_cosine, 10),
        "min_cosine": round(min_cosine, 10),
        "max_abs_error": round(max_abs_error, 10),
        "avg_max_abs_error": round(avg_max_abs_error, 10),
        "ppl_avg_rel_diff_pct": ppl_avg_rel_diff_pct,
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
    parser = argparse.ArgumentParser(description="NPU optimized accuracy_run_perf for t5-small")
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
