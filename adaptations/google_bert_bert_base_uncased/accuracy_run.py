"""
Benchmark accuracy_run.py for google-bert/bert-base-uncased.
Modified by npu-optimizer for cls_embeddings contract.

Model Type: encoder (BERT, non-generative)
评测画像: 逐样本 [CLS] embedding 提取

两步依次执行:
  Step 1: 单样本 -> trace_*.json + benchmark_metrics_*.json (性能分析)
  Step 2: 全样本 -> outputs_*.pt (cls_embeddings)

Usage:
    uv run python accuracy_run.py                    # 完整执行 (step1 -> step2)
    uv run python accuracy_run.py --use-pretrained   # Tier2: 加载预训练权重
    uv run python accuracy_run.py --cpu              # 强制 CPU 推理 (禁用 NPU/CUDA)
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

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402
from transformers import set_seed as transformers_set_seed  # noqa: E402

from datasets import load_from_disk  # noqa: E402

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_TEXT_FIELD = "article"

MODEL_ID = "google-bert/bert-base-uncased"
WARMUP_ITERATIONS = 3


def load_benchmark_texts() -> tuple[list[str], str]:
    """加载测试数据集文本，返回 (texts, dataset_name)。"""
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        print(f"[benchmark] loading dataset from {wikitext_path}")
        ds = load_from_disk(str(wikitext_path))
        if hasattr(ds, "keys"):  # DatasetDict — select train split
            ds = ds["train"]
        texts = sorted([sample["text"] for sample in ds if sample.get("text", "").strip()])
        print(f"[benchmark] loaded {len(texts)} samples from wikitext")
        return texts, "wikitext"

    cnn_path = DATASET_DIR / "cnn_dailymail___3.0.0"
    if cnn_path.exists():
        print(f"[benchmark] loading dataset from {cnn_path}")
        ds = load_from_disk(str(cnn_path))
        texts = sorted([sample[DATASET_TEXT_FIELD] for sample in ds if sample[DATASET_TEXT_FIELD].strip()])
        print(f"[benchmark] loaded {len(texts)} samples from cnn_dailymail")
        return texts, "cnn_dailymail"

    print("[benchmark] using built-in benchmark texts")
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
    """根据硬件动态返回原生 Profiler 上下文。"""
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
    """加载模型 (BERT encoder for cls_embeddings)。"""
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
    embeddings = outputs.last_hidden_state[:, 0, :]  # [N, hidden_dim] — [CLS] token
    del outputs, encoded
    return embeddings


def run_step1(model, tokenizer, first_device, device_short, device_tag, device_ids, mode_str, adapt_dir: Path, dataset_name: str, texts: list[str], num_samples: int):
    """Step 1: 单样本 forward -> trace + metrics"""
    print("\n" + "=" * 60)
    print("Step 1: 单样本 forward (性能分析)")
    print("=" * 60)

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    trace_path = adapt_dir / f"trace_{device_tag}_{dtype_str}_{mode_str}_{dataset_name}.json"
    metrics_path = adapt_dir / f"benchmark_metrics_{device_tag}_{dtype_str}_{mode_str}_{dataset_name}.json"

    text = texts[0] if texts else "The capital of France is Paris."

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
        "output_type": "cls_embeddings",
        "warmup_iterations": WARMUP_ITERATIONS,
        "packages": get_package_versions(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    prof_dir = adapt_dir / "export_only_prof_dir"
    if prof_dir.exists():
        shutil.rmtree(prof_dir, ignore_errors=True)

    print(f"[benchmark] trace saved to {trace_path}")
    print(f"[benchmark] metrics saved to {metrics_path}")
    return trace_path, metrics_path, start_time


def run_step2(model, tokenizer, first_device, device_short, mode_str, adapt_dir: Path, dataset_name: str, texts: list[str], max_samples: int = 250):
    """Step 2: 全样本 forward -> outputs (cls_embeddings)"""
    print("\n" + "=" * 60)
    print("Step 2: 全样本 forward (cls_embeddings)")
    print("=" * 60)

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)
    outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.pt"

    texts = texts[:max_samples]
    print(f"[benchmark] {len(texts)} samples to process (max {max_samples})")

    # Warmup
    dummy_text = texts[0] if texts else "Hello, benchmark."
    print(f"[benchmark] warming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        with torch.no_grad():
            _ = encode_texts(model, tokenizer, [dummy_text], first_device)

    if device_short == "npu":
        torch.npu.synchronize()
    elif device_short == "cuda":
        torch.cuda.synchronize()

    step2_start = time.perf_counter()
    all_embeddings = []

    with torch.no_grad():
        for i, text in enumerate(texts):
            embeddings = encode_texts(model, tokenizer, [text], first_device)
            all_embeddings.append(embeddings.cpu())  # [1, hidden_dim]
            del embeddings

            if (i + 1) % 32 == 0:
                if device_short == "npu":
                    torch.npu.empty_cache()
                elif device_short == "cuda":
                    torch.cuda.empty_cache()
                print(f"[benchmark] processed {i + 1}/{len(texts)} samples (cache cleared)")
            elif (i + 1) % 8 == 0:
                print(f"[benchmark] processed {i + 1}/{len(texts)} samples")

    if device_short == "npu":
        torch.npu.synchronize()
    elif device_short == "cuda":
        torch.cuda.synchronize()

    step2_end = time.perf_counter()
    wall_clock_s = step2_end - step2_start

    output_data = {
        "texts": list(texts),
        "embeddings": all_embeddings,
    }
    torch.save(output_data, outputs_path)

    avg_latency = wall_clock_s / len(texts) if texts else 0
    print(f"[benchmark] outputs saved to {outputs_path}")
    print(f"[benchmark]   - embeddings: {len(all_embeddings)} samples, dim={all_embeddings[0].shape[-1]}")
    print(f"[benchmark] wall_clock_s: {wall_clock_s:.6f}")

    return outputs_path, wall_clock_s, WARMUP_ITERATIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-pretrained", action="store_true", help="Tier2: load pretrained weights")
    parser.add_argument("--max-samples", type=int, default=250, help="Max samples to process (default 250, per R1 rule)")
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
    texts = texts[: args.max_samples]
    num_samples = len(texts)
    print(f"[benchmark] using dataset: {dataset_name}, samples: {num_samples}")

    model, tokenizer = setup_model(args.use_pretrained, device, CACHE_DIR)

    first_device = next(model.parameters()).device
    device_short = first_device.type
    device_tag = str(first_device).replace(":", "_")
    mode_str = "pretrained" if args.use_pretrained else "config"

    device_ids = None
    if first_device.index is not None:
        device_ids = [first_device.index]

    trace_path, metrics_path, start_time = run_step1(model, tokenizer, first_device, device_short, device_tag, device_ids, mode_str, ADAPT_DIR, dataset_name, texts, num_samples)
    outputs_path, wall_clock_s, warmup_iters = run_step2(model, tokenizer, first_device, device_short, mode_str, ADAPT_DIR, dataset_name, texts, args.max_samples)

    end_time = datetime.now().isoformat()
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    metrics["end_time"] = end_time
    metrics["wall_clock_s"] = round(wall_clock_s, 6)
    metrics["warmup_iterations"] = warmup_iters
    latency_per_sample = wall_clock_s / num_samples if num_samples > 0 else wall_clock_s
    metrics["latency_s"] = round(latency_per_sample, 6)

    with open(metrics_path, "w") as f:
        json.dump(metrics, indent=2, fp=f)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  trace: {trace_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  outputs: {outputs_path} (含 embeddings)")
    print(f"  wall_clock_s: {wall_clock_s:.6f}")
    print(f"  latency_s (per-sample): {latency_per_sample:.6f}")


if __name__ == "__main__":
    main()
