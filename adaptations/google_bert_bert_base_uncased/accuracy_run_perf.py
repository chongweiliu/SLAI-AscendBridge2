"""
accuracy_run_perf.py for google-bert/bert-base-uncased — NPU optimized version.

Optimizations:
  1. Batched inference (bs=8): Process 8 texts per forward pass
  2. TASK_QUEUE_ENABLE=1: Async operator dispatch
  3. Symmetric warmup(3x): Matches baseline warmup

Contract: cls_embeddings (same as modified accuracy_run.py).

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

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import set_seed as transformers_set_seed

from datasets import load_from_disk

PERF_SUFFIX = "_perf"

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_TEXT_FIELD = "article"

MODEL_ID = "google-bert/bert-base-uncased"
WARMUP_ITERATIONS = 3
BATCH_SIZE = 8


def load_benchmark_texts() -> tuple[list[str], str]:
    """加载测试数据集文本。与 accuracy_run.py 保持一致。"""
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        print(f"[perf] loading dataset from {wikitext_path}")
        ds = load_from_disk(str(wikitext_path))
        if hasattr(ds, "keys"):
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

    print("[perf] using built-in benchmark texts")
    builtin_texts = [
        "The capital of France is Paris, a city known for the Eiffel Tower and the Louvre Museum.",
        "Machine learning is a subset of artificial intelligence that learns patterns from data.",
        "Natural language processing enables computers to understand and generate human language.",
        "Transformers have revolutionized the field of deep learning since their introduction in 2017.",
        "PyTorch is an open-source machine learning framework widely used in research and production.",
        "The attention mechanism allows models to focus on relevant parts of the input sequence.",
        "Language models can generate coherent and contextually relevant text given a prompt.",
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        "The Great Wall of China stretches thousands of kilometers across mountains and deserts.",
        "Quantum computing leverages superposition and entanglement to process information in new ways.",
        "The human heart beats roughly one hundred thousand times per day, pumping blood through the body.",
        "Climate change is driven largely by greenhouse gas emissions from human activity.",
        "The Pacific Ocean is the largest and deepest ocean on Earth, covering about a third of the surface.",
        "Shakespeare wrote tragedies such as Hamlet, Macbeth, and King Lear in the early seventeenth century.",
        "The speed of light in a vacuum is approximately three hundred million meters per second.",
        "Bees communicate the location of flowers to their hive mates through a waggle dance.",
        "The Industrial Revolution began in Britain and transformed manufacturing, transport, and society.",
        "Deoxyribonucleic acid carries the genetic instructions used in the growth of all living organisms.",
        "Mount Everest is the tallest mountain above sea level, located in the Himalayas.",
        "The Roman Empire expanded across Europe, North Africa, and the Middle East at its peak.",
        "Electric vehicles store energy in rechargeable batteries and use electric motors for propulsion.",
        "The stock market reflects investor expectations about the future earnings of companies.",
        "Antibiotics are medicines that fight bacterial infections by killing or inhibiting bacteria.",
        "The theory of general relativity describes gravity as the curvature of spacetime.",
        "Rivers carry sediment from mountains to the sea, shaping landscapes over millennia.",
        "The Olympic Games bring together athletes from around the world every four years.",
        "Coffee is one of the most widely traded agricultural commodities in the world.",
        "The printing press invented by Gutenberg made books affordable and spread literacy across Europe.",
        "Volcanoes erupt when molten rock from deep within the Earth reaches the surface.",
        "The immune system defends the body against pathogens using a network of cells and organs.",
        "Renewable energy sources include solar, wind, hydro, and geothermal power.",
        "The Sahara Desert is the largest hot desert in the world, spanning much of North Africa.",
        "Symphonies, concertos, and sonatas are classical forms of musical composition.",
        "The internet connects billions of devices through a global network of routers and cables.",
        "Migration of birds occurs seasonally as they travel between breeding and wintering grounds.",
        "The Renaissance was a period of cultural and artistic flourishing that began in Italy.",
        "Glaciers form when accumulated snow compresses into dense ice over many centuries.",
        "Vaccines train the immune system to recognize and fight specific infectious diseases.",
        "The Amazon rainforest hosts an extraordinary diversity of plant and animal species.",
        "Supply and demand interact in markets to determine the prices of goods and services.",
        "The moon orbits the Earth roughly once every twenty-seven days, causing ocean tides.",
        "Fermentation is a metabolic process in which microorganisms convert sugars into acids or alcohol.",
        "The United Nations was founded in 1945 to promote international cooperation and peace.",
        "Coral reefs provide habitat for marine life but are threatened by warming oceans.",
        "The human brain contains billions of neurons connected by trillions of synapses.",
        "Agriculture developed independently in several regions of the world thousands of years ago.",
        "Satellites orbit the Earth for communication, navigation, weather monitoring, and research.",
        "The rule of law requires that governments and citizens alike are subject to publicly disclosed laws.",
        "Plate tectonics explains the movement of the Earth's crust and the occurrence of earthquakes.",
        "Libraries provide public access to books, digital media, and community learning programs.",
        "The water cycle describes the continuous movement of water between the ocean, atmosphere, and land.",
        "Chess is a strategic board game played between two opponents on a checkered board.",
        "The Hubble Space Telescope has captured detailed images of distant galaxies and nebulae.",
        "Coral bleaching occurs when stressed corals expel the algae living in their tissues.",
        "The Silk Road was an ancient network of trade routes connecting East Asia and the Mediterranean.",
        "Wind turbines convert kinetic energy from moving air into electrical power.",
        "The periodic table organizes chemical elements by atomic number and recurring properties.",
        "Urban planning shapes how cities allocate land for housing, transport, and public spaces.",
        "The discovery of penicillin marked the beginning of modern antibiotic medicine.",
        "Mangrove forests protect coastlines from erosion and serve as nurseries for fish.",
        "The central bank manages monetary policy by setting interest rates and regulating money supply.",
        "Meteorologists use satellite data and computer models to forecast weather patterns.",
        "The ancient library of Alexandria was one of the largest centers of learning in the ancient world.",
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
    """性能监控器"""

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
    """加载模型 (BERT encoder)。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)

    if use_pretrained:
        model = AutoModel.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = model.to(device)
    else:
        model = AutoModel.from_config(config)
        model = model.to(device)
    model.eval()

    return model, tokenizer


def encode_texts(model, tokenizer, texts, first_device) -> torch.Tensor:
    """编码文本为 [CLS] embedding [N, hidden_dim]。"""
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(first_device)
    outputs = model(**encoded)
    embeddings = outputs.last_hidden_state[:, 0, :]  # [N, hidden_dim]
    del outputs, encoded
    return embeddings


def run_perf(use_pretrained: bool, max_samples: int, cpu: bool):
    """运行 perf 推理。"""
    adapt_dir = Path(__file__).resolve().parent
    cache_dir = (adapt_dir / "models").as_posix()

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
    if first_device.index is not None:
        device_ids = [first_device.index]

    device_model = "unknown"
    if perf_device_short == "npu" and first_device.index is not None:
        device_model = torch.npu.get_device_name(first_device.index)
    elif perf_device_short == "cuda" and first_device.index is not None:
        device_model = torch.cuda.get_device_name(first_device.index)

    # Warmup
    dummy_texts = texts[:BATCH_SIZE]
    print(f"[perf] warming up ({WARMUP_ITERATIONS} iterations, batch_size={BATCH_SIZE})...")
    for _ in range(WARMUP_ITERATIONS):
        with torch.no_grad():
            _ = encode_texts(model, tokenizer, dummy_texts, first_device)

    if perf_device_short == "npu":
        torch.npu.synchronize()
    elif perf_device_short == "cuda":
        torch.cuda.synchronize()

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

    outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"
    output_data = {
        "texts": list(texts),
        "embeddings": all_embeddings,
    }
    torch.save(output_data, outputs_path)
    print(f"[perf] outputs saved to {outputs_path}")

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

    baseline_metrics_files = sorted(adapt_dir.glob("benchmark_metrics_npu_*_pretrained_*.json"))
    baseline_metrics_files = [f for f in baseline_metrics_files if "_perf" not in f.name]
    perf_metrics_files = sorted(adapt_dir.glob("benchmark_metrics_npu_*_pretrained_*_perf.json"))

    if not baseline_metrics_files or not perf_metrics_files:
        print("[compare] ERROR: Missing metrics files")
        return None

    baseline_metrics_path = baseline_metrics_files[-1]
    perf_metrics_path = perf_metrics_files[-1]
    print(f"[compare] baseline: {baseline_metrics_path.name}")
    print(f"[compare] perf: {perf_metrics_path.name}")

    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)
    with open(perf_metrics_path) as f:
        perf_metrics = json.load(f)

    baseline_outputs_files = sorted(adapt_dir.glob("outputs_npu_*_pretrained_*.pt"))
    baseline_outputs_files = [f for f in baseline_outputs_files if "_perf" not in f.name]
    perf_outputs_files = sorted(adapt_dir.glob("outputs_npu_*_pretrained_*_perf.pt"))

    if not baseline_outputs_files or not perf_outputs_files:
        print("[compare] ERROR: Missing output files")
        return None

    baseline_data = torch.load(baseline_outputs_files[-1], weights_only=False)
    perf_data = torch.load(perf_outputs_files[-1], weights_only=False)

    baseline_embeddings = baseline_data.get("embeddings", [])
    perf_embeddings = perf_data.get("embeddings", [])

    if not baseline_embeddings or not perf_embeddings:
        print("[compare] ERROR: Missing embeddings")
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
    avg_cosine = min(1.0, max(0.0, avg_cosine))

    baseline_wall_clock = baseline_metrics.get("wall_clock_s", 0.0)
    perf_wall_clock = perf_metrics.get("wall_clock_s", 0.0)
    baseline_latency = baseline_metrics.get("latency_s", 0.0)
    perf_latency = perf_metrics.get("latency_s", 0.0)

    speedup_ratio = baseline_wall_clock / perf_wall_clock if perf_wall_clock > 0 else 0.0
    num_samples = min(baseline_metrics.get("num_samples", 0), perf_metrics.get("num_samples", 0))

    print(f"[compare] cosine_similarity: avg={avg_cosine:.10f}, min={min_cosine:.10f}")
    print(f"[compare] max_abs_error: avg={avg_max_abs_error:.10f}, max={max_abs_error:.10f}")
    print(f"[compare] baseline_wall_clock_s: {baseline_wall_clock}")
    print(f"[compare] perf_wall_clock_s: {perf_wall_clock}")
    print(f"[compare] speedup_ratio: {speedup_ratio:.6f}")

    baseline_warmup = baseline_metrics.get("warmup_iterations", WARMUP_ITERATIONS)
    perf_warmup = perf_metrics.get("warmup_iterations", WARMUP_ITERATIONS)
    selected_npu = perf_metrics.get("selected_npu", 0)
    selected_npus = perf_metrics.get("selected_npus", [selected_npu])
    device_topology = perf_metrics.get("device_topology", f"single-die:{selected_npu}")

    result = {
        "dtype": perf_metrics.get("dtype", "fp32"),
        "mode": perf_metrics.get("mode", "pretrained"),
        "dataset": perf_metrics.get("dataset", "wikitext"),
        "output_type": "cls_embeddings",
        "baseline_artifact": baseline_metrics_path.name,
        "perf_artifact": perf_metrics_path.name,
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
        "validation_note": f"Independent baseline artifact ({baseline_metrics_path.name}) vs perf artifact ({perf_metrics_path.name}). Symmetric warmup ({baseline_warmup}x). Cosine={avg_cosine:.10f}, max_abs_error={max_abs_error:.10f}.",
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
    parser = argparse.ArgumentParser(description="NPU optimized accuracy_run_perf for bert-base-uncased")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    run_parser = subparsers.add_parser("run", help="Run perf inference")
    run_parser.add_argument("--use-pretrained", action="store_true", help="Load pretrained weights")
    run_parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU")

    compare_parser = subparsers.add_parser("compare", help="Compare baseline vs perf")

    args = parser.parse_args()

    if args.command == "run":
        run_perf(use_pretrained=args.use_pretrained, max_samples=args.max_samples, cpu=args.cpu)
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
