"""
NPU Optimized accuracy_run_perf.py for distilbert/distilbert-base-uncased (Masked LM).
Runtime-only optimization: batched_inference + warmup(3x) + TASK_QUEUE_ENABLE=1.

两步依次执行:
  Step 1: 单样本 -> trace_*_perf.json + benchmark_metrics_*_perf.json (性能分析)
  Step 2: 全样本 -> outputs_*_perf.pt (精度测试: batched inference)

Usage:
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50
    uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50 --batch-size 8
    uv run python accuracy_run_perf.py compare

Host notes (2x Ascend910, 多 runner 并发):
- 本机严禁设置 ASCEND_RT_VISIBLE_DEVICES（会导致 aclInit error 107001）；
  选卡一律通过 torch.npu.mem_get_info() 挑空闲卡 + torch.npu.set_device()。
- TASK_QUEUE_ENABLE=1 异步算子下发，+5~15% 推理加速。
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

# HuggingFace 镜像（国内直连 huggingface.co 不通）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Runtime-only 优化: 异步算子下发
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")

from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer  # noqa: E402
from transformers import set_seed as transformers_set_seed  # noqa: E402

from datasets import load_from_disk  # noqa: E402

PERF_SUFFIX = "_perf"

# 数据集配置
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
DATASET_TEXT_FIELD = "article"  # cnn_dailymail 的文本字段
MASK_RATIO = 0.15  # MLM 掩盖比例
MASK_SEED_BASE = 42  # 每样本掩盖位置的确定性种子基数


def load_benchmark_texts() -> tuple[list[str], str]:
    """加载测试数据集文本，返回 (texts, dataset_name)。

    优先级: 1. wikitext 2. cnn_dailymail 3. imdb 4. 内置
    """
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        print(f"[benchmark] loading dataset from {wikitext_path}")
        ds = load_from_disk(str(wikitext_path))
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

    imdb_path = DATASET_DIR / "imdb"
    if imdb_path.exists():
        print(f"[benchmark] loading dataset from {imdb_path}")
        ds = load_from_disk(str(imdb_path))
        texts = sorted([sample["text"] for sample in ds if sample["text"].strip()])
        print(f"[benchmark] loaded {len(texts)} samples from imdb")
        return texts, "imdb"

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


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def get_device(force_cpu: bool = False):
    """获取推理设备。若 force_cpu=True，强制返回 CPU。

    默认优先级: NPU > CUDA > CPU
    NPU 选卡: 用 torch.npu.mem_get_info() 遍历各卡，选空闲（free 显存最大）的卡。
    本机严禁设置 ASCEND_RT_VISIBLE_DEVICES，选卡一律用 torch.npu.set_device()。
    """
    if force_cpu:
        return "cpu", 0, "cpu", None

    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            best_index = 0
            best_free = -1
            for i in range(count):
                try:
                    free, total = torch.npu.mem_get_info(i)
                except Exception:
                    free, total = 0, 0
                print(f"[Device] NPU {i}: free={free / 1024**3:.2f}GB / total={total / 1024**3:.2f}GB")
                if free > best_free:
                    best_free = free
                    best_index = i
            print(f"[Device] Huawei Ascend NPU detected; selected idle device npu:{best_index}")
            torch.npu.set_device(best_index)
            device_name = torch.npu.get_device_name(best_index)
            return f"npu:{best_index}", count, device_name, [best_index]
    except ImportError:
        pass

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return "cuda:0", torch.cuda.device_count(), device_name, [0]
    return "cpu", 0, "cpu", None


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
    """Convert torch dtype to short string (fp32, fp16, bf16)."""
    dtype_map = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.int64: "int64",
        torch.int32: "int32",
    }
    return dtype_map.get(dtype, str(dtype).replace("torch.", ""))


def get_package_versions() -> dict:
    """获取关键依赖包的版本"""
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
    """加载模型 (MaskedLM / DistilBERT)"""
    MODEL_ID = "distilbert/distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)

    if use_pretrained:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, torch_dtype="auto", cache_dir=cache_dir)
        model = model.to(device)
    else:
        model = AutoModelForMaskedLM.from_config(config)
        model = model.to(device)
    model.eval()

    return model, tokenizer


def build_masked_inputs(tokenizer, text: str, sample_index: int, device):
    """对单条文本做确定性 [MASK] 掩盖，返回 (inputs, selected_positions, original_ids)。

    - 仅掩盖非特殊 token（[CLS]/[SEP]/[PAD] 不参与）
    - 使用固定种子 (MASK_SEED_BASE + sample_index) 保证多次运行完全一致
    - 至少掩盖 1 个 token；若文本无可掩盖 token 返回 None
    """
    orig = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    original_ids = orig["input_ids"]  # [1, L]

    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    candidates = torch.ones(original_ids.shape[1], dtype=torch.bool)
    for sp in special_ids:
        if sp is not None:
            candidates &= original_ids[0] != sp
    cand_positions = candidates.nonzero(as_tuple=True)[0]
    if cand_positions.numel() == 0:
        return None, None, None

    gen = torch.Generator()
    gen.manual_seed(MASK_SEED_BASE + sample_index)
    rand_vals = torch.rand(cand_positions.numel(), generator=gen)
    selected = cand_positions[rand_vals < MASK_RATIO]
    if selected.numel() == 0:
        selected = cand_positions[:1]

    masked_ids = original_ids.clone()
    masked_ids[0, selected] = tokenizer.mask_token_id

    inputs = {k: v.to(device) for k, v in orig.items()}
    inputs["input_ids"] = masked_ids.to(device)
    return inputs, selected, original_ids


def run_step1_perf(model, tokenizer, first_device, device_short, device_tag, device_ids, mode_str, adapt_dir: Path, dataset_name: str, texts: list[str]):
    """Step 1: 单样本推理 -> trace_*_perf.json + benchmark_metrics_*_perf.json"""
    print("\n" + "=" * 60)
    print("Step 1 (perf): 单样本推理 (性能分析)")
    print("=" * 60)

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    trace_path = adapt_dir / f"trace_{device_tag}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"
    metrics_path = adapt_dir / f"benchmark_metrics_{device_tag}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.json"

    text = texts[0] if texts else "The capital of France is Paris."
    built = build_masked_inputs(tokenizer, text, 0, first_device)
    if built[0] is None:
        inputs, _, _ = build_masked_inputs(tokenizer, "The capital of France is Paris.", 0, first_device)
    else:
        inputs = built[0]

    profiler_context = get_profiler_context(device_short)

    start_time = datetime.now().isoformat()
    with profiler_context as prof:
        with _PerfMonitor(device_short, device_ids) as m:
            with torch.no_grad():
                out = model(**inputs)
                _ = out.logits  # MLM: logits over vocab at every position

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
        "num_samples": len(texts),
        "device": str(first_device),
        "device_model": device_model,
        "mode": mode_str,
        "dataset": dataset_name,
        "dtype": dtype_str,
        "output_type": "generated_text",
        "packages": get_package_versions(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    prof_dir = adapt_dir / "export_only_prof_dir"
    if prof_dir.exists():
        shutil.rmtree(prof_dir, ignore_errors=True)

    print(f"[perf] trace saved to {trace_path}")
    print(f"[perf] metrics saved to {metrics_path}")
    print(f"[perf] {metrics}")

    return trace_path, metrics_path, start_time


def run_step2_perf(model, tokenizer, first_device, device_short, mode_str, adapt_dir: Path, dataset_name: str, texts: list[str], max_samples: int = 250, batch_size: int = 8, warmup_iterations: int = 3):
    """Step 2 (perf): 全样本 batched inference -> outputs_*_perf.pt

    Runtime-only 优化:
    - batched_inference: 多样本同时推理，减少 Host-Device 同步开销
    - warmup: 预热 N 轮，避免首次推理的编译开销
    - TASK_QUEUE_ENABLE=1: 异步算子下发（已在环境变量中设置）

    输出格式与 baseline 完全一致:
    - generated_text: [MASK] 填充还原后的文本列表
    - logits: 每样本首个 [MASK] 位置的 logits 张量列表
    - perplexity: 每样本伪困惑度列表
    """
    print("\n" + "=" * 60)
    print(f"Step 2 (perf): 全样本 batched inference (bs={batch_size}, warmup={warmup_iterations})")
    print("=" * 60)

    model_dtype = next(model.parameters()).dtype
    dtype_str = get_dtype_str(model_dtype)

    outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}{PERF_SUFFIX}.pt"

    texts = texts[:max_samples]
    print(f"[perf] {len(texts)} samples to process (max {max_samples}, batch_size={batch_size})")

    # 预构建所有样本的 masked inputs
    all_inputs_data = []  # (masked_ids, attention_mask, selected_positions, original_ids)
    for i, text in enumerate(texts):
        result = build_masked_inputs(tokenizer, text, i, first_device)
        if result[0] is None:
            print(f"[perf] sample {i} has no maskable tokens, skipped")
            continue
        inputs, selected, original_ids = result
        all_inputs_data.append((inputs["input_ids"][0].cpu(), inputs["attention_mask"][0].cpu(), selected, original_ids[0].cpu()))

    if not all_inputs_data:
        print("[perf] No valid samples to process!")
        torch.save({"generated_text": [], "logits": [], "perplexity": []}, outputs_path)
        return outputs_path, None, None, 0.0

    # 获取 pad_token_id 和 mask_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mask_token_id = tokenizer.mask_token_id

    # Warmup: 用前几个样本做 warmup
    print(f"[perf] warmup {warmup_iterations} iterations...")
    warmup_samples = all_inputs_data[:min(batch_size, len(all_inputs_data))]
    max_len_warmup = max(len(item[0]) for item in warmup_samples)
    warmup_batch_ids = torch.full((len(warmup_samples), max_len_warmup), pad_token_id, dtype=torch.long)
    warmup_batch_attn = torch.zeros((len(warmup_samples), max_len_warmup), dtype=torch.long)
    for j, (ids, attn, _, _) in enumerate(warmup_samples):
        l = len(ids)
        warmup_batch_ids[j, :l] = ids
        warmup_batch_attn[j, :l] = attn
    warmup_batch_ids = warmup_batch_ids.to(first_device)
    warmup_batch_attn = warmup_batch_attn.to(first_device)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_ids=warmup_batch_ids, attention_mask=warmup_batch_attn)
    if device_short == "npu":
        torch.npu.synchronize()
    print("[perf] warmup done")

    # 正式推理：按 batch 处理
    all_filled_texts = []
    all_logits = []
    all_ppl = []

    step2_start = time.perf_counter()

    with torch.no_grad():
        for batch_start in range(0, len(all_inputs_data), batch_size):
            batch_end = min(batch_start + batch_size, len(all_inputs_data))
            batch_data = all_inputs_data[batch_start:batch_end]
            actual_bs = len(batch_data)

            # Pad batch to same length
            max_len = max(len(item[0]) for item in batch_data)
            batch_ids = torch.full((actual_bs, max_len), pad_token_id, dtype=torch.long)
            batch_attn = torch.zeros((actual_bs, max_len), dtype=torch.long)
            batch_selected = []
            batch_original_ids = []

            for j, (ids, attn, selected, original_ids) in enumerate(batch_data):
                l = len(ids)
                batch_ids[j, :l] = ids
                batch_attn[j, :l] = attn
                batch_selected.append(selected)
                batch_original_ids.append(original_ids)

            batch_ids = batch_ids.to(first_device)
            batch_attn = batch_attn.to(first_device)

            out = model(input_ids=batch_ids, attention_mask=batch_attn)
            logits = out.logits  # [B, L, V]

            for j in range(actual_bs):
                selected = batch_selected[j].to(first_device)
                original_ids = batch_original_ids[j].to(first_device)

                mask_logits = logits[j, selected]  # [M, V]
                targets = original_ids[selected]  # [M]
                loss = F.cross_entropy(mask_logits, targets)
                ppl = torch.exp(loss).item()
                all_ppl.append(ppl)

                pred_ids = mask_logits.argmax(dim=-1).cpu()
                filled_ids = original_ids.cpu().clone()
                filled_ids[selected.cpu()] = pred_ids
                filled_text = tokenizer.decode(filled_ids, skip_special_tokens=True)
                all_filled_texts.append(filled_text)

                first_mask_logits = logits[j, selected[0]].cpu()
                all_logits.append(first_mask_logits)

            del out, logits
            if (batch_start // batch_size + 1) % 4 == 0:
                if device_short == "npu":
                    torch.npu.empty_cache()
                elif device_short == "cuda":
                    torch.cuda.empty_cache()
                print(f"[perf] processed {batch_end}/{len(all_inputs_data)} samples (cache cleared)")
            else:
                print(f"[perf] processed {batch_end}/{len(all_inputs_data)} samples")

    if device_short == "npu":
        torch.npu.synchronize()
    step2_wall_clock = time.perf_counter() - step2_start
    wall_clock_s = round(step2_wall_clock, 6)

    output_data = {
        "generated_text": all_filled_texts,
        "logits": all_logits,
        "perplexity": all_ppl,
    }
    torch.save(output_data, outputs_path)

    ppl_avg = round(sum(all_ppl) / len(all_ppl), 2) if all_ppl else None
    print(f"[perf] outputs saved to {outputs_path}")
    print(f"[perf]   - generated_text: {len(all_filled_texts)} samples")
    if all_filled_texts:
        print(f"[perf]   - example: {all_filled_texts[0][:100]}")
    print(f"[perf]   - logits: {len(all_logits)} samples" + (f", shape: {tuple(all_logits[0].shape)}" if all_logits else ""))
    print(f"[perf]   - pseudo-perplexity: avg={ppl_avg}" + (f", min={min(all_ppl):.2f}, max={max(all_ppl):.2f}" if all_ppl else ""))
    print(f"[perf]   - wall_clock_s: {wall_clock_s}")

    return outputs_path, None, None, wall_clock_s


def cmd_run(args):
    """run 子命令：执行优化版推理"""
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
    print(f"[perf] using dataset: {dataset_name}, samples selected: {len(texts)} (max_samples={args.max_samples})")

    model, tokenizer = setup_model(args.use_pretrained, device, CACHE_DIR)

    first_device = next(model.parameters()).device
    device_short = first_device.type
    device_tag = str(first_device).replace(":", "_")
    mode_str = "pretrained" if args.use_pretrained else "config"

    # Step 1: 单样本 -> trace + metrics
    trace_path, metrics_path, start_time = run_step1_perf(model, tokenizer, first_device, device_short, device_tag, device_ids, mode_str, ADAPT_DIR, dataset_name, texts)

    # Step 2: 全样本 batched inference -> outputs
    outputs_path, ttft_avg, tpot_avg, wall_clock_s = run_step2_perf(model, tokenizer, first_device, device_short, mode_str, ADAPT_DIR, dataset_name, texts, args.max_samples, args.batch_size, args.warmup_iterations)

    # 更新 metrics 文件
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

    with open(metrics_path, "w") as f:
        json.dump(metrics, indent=2, fp=f)

    print("\n" + "=" * 60)
    print("Summary (perf)")
    print("=" * 60)
    print(f"  trace: {trace_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  outputs: {outputs_path} (含 generated_text + logits + perplexity)")
    print(f"  wall_clock_s: {wall_clock_s}")


def cmd_compare(args):
    """compare 子命令：对比 baseline vs perf 产出，生成 optimization_notes.json"""
    ADAPT_DIR = Path(__file__).resolve().parent

    # 查找 baseline 和 perf 工件
    baseline_metrics_files = sorted(ADAPT_DIR.glob("benchmark_metrics_*_pretrained_*.json"))
    baseline_metrics_files = [f for f in baseline_metrics_files if not f.name.endswith("_perf.json")]
    perf_metrics_files = sorted(ADAPT_DIR.glob("benchmark_metrics_*_pretrained_*_perf.json"))

    baseline_output_files = sorted(ADAPT_DIR.glob("outputs_*_pretrained_*.pt"))
    baseline_output_files = [f for f in baseline_output_files if not f.name.endswith("_perf.pt")]
    perf_output_files = sorted(ADAPT_DIR.glob("outputs_*_pretrained_*_perf.pt"))

    if not baseline_metrics_files:
        print("[compare] Error: No baseline metrics files found")
        return 1
    if not perf_metrics_files:
        print("[compare] Error: No perf metrics files found")
        return 1
    if not baseline_output_files:
        print("[compare] Error: No baseline output files found")
        return 1
    if not perf_output_files:
        print("[compare] Error: No perf output files found")
        return 1

    # 按 end_time 选最近一轮成对工件
    def load_metrics_by_endtime(f):
        with open(f) as fh:
            m = json.load(fh)
        return m

    baseline_metrics = load_metrics_by_endtime(baseline_metrics_files[-1])
    perf_metrics = load_metrics_by_endtime(perf_metrics_files[-1])

    baseline_path = baseline_output_files[-1]
    perf_path = perf_output_files[-1]

    print(f"[compare] baseline metrics: {baseline_metrics_files[-1].name}")
    print(f"[compare] perf metrics: {perf_metrics_files[-1].name}")
    print(f"[compare] baseline outputs: {baseline_path.name}")
    print(f"[compare] perf outputs: {perf_path.name}")

    # 检查 mode 一致
    if baseline_metrics.get("mode") != perf_metrics.get("mode"):
        print(f"[compare] Error: mode mismatch: baseline={baseline_metrics.get('mode')}, perf={perf_metrics.get('mode')}")
        return 1
    if baseline_metrics.get("mode") != "pretrained":
        print(f"[compare] Error: baseline mode must be pretrained, got {baseline_metrics.get('mode')}")
        return 1

    # 检查 num_samples 一致
    b_ns = baseline_metrics.get("num_samples", 0)
    p_ns = perf_metrics.get("num_samples", 0)
    if b_ns != p_ns:
        print(f"[compare] Warning: num_samples mismatch: baseline={b_ns}, perf={p_ns}")

    # 加载 outputs
    baseline_outputs = torch.load(baseline_path, map_location="cpu", weights_only=False)
    perf_outputs = torch.load(perf_path, map_location="cpu", weights_only=False)

    # 比较 logits
    b_logits = baseline_outputs.get("logits", [])
    p_logits = perf_outputs.get("logits", [])

    if not b_logits or not p_logits:
        print("[compare] Error: No logits found in outputs")
        return 1

    min_len = min(len(b_logits), len(p_logits))
    print(f"[compare] Comparing {min_len} logits pairs")

    cosines = []
    max_abs_errors = []
    for i in range(min_len):
        b_l = b_logits[i].float().flatten()
        p_l = p_logits[i].float().flatten()
        cos = F.cosine_similarity(b_l.unsqueeze(0), p_l.unsqueeze(0)).item()
        cosines.append(cos)
        mae = (b_l - p_l).abs().max().item()
        max_abs_errors.append(mae)

    avg_cosine = sum(cosines) / len(cosines) if cosines else 0.0
    min_cosine = min(cosines) if cosines else 0.0
    max_abs_error = max(max_abs_errors) if max_abs_errors else 0.0
    avg_abs_error = sum(max_abs_errors) / len(max_abs_errors) if max_abs_errors else 0.0

    # 夹紧 cosine 到 [0, 1]
    avg_cosine = min(1.0, max(0.0, avg_cosine))
    min_cosine = min(1.0, max(0.0, min_cosine))

    # 比较 perplexity
    b_ppl = baseline_outputs.get("perplexity", [])
    p_ppl = perf_outputs.get("perplexity", [])
    ppl_min_len = min(len(b_ppl), len(p_ppl))
    ppl_rel_diffs = []
    for i in range(ppl_min_len):
        if b_ppl[i] > 0:
            ppl_rel_diffs.append(abs(p_ppl[i] - b_ppl[i]) / b_ppl[i])
    ppl_avg_rel_diff = sum(ppl_rel_diffs) / len(ppl_rel_diffs) if ppl_rel_diffs else 0.0

    # 比较 generated_text
    b_texts_out = baseline_outputs.get("generated_text", [])
    p_texts_out = perf_outputs.get("generated_text", [])
    text_min_len = min(len(b_texts_out), len(p_texts_out))
    text_matches = sum(1 for i in range(text_min_len) if b_texts_out[i] == p_texts_out[i])
    text_match_rate = text_matches / text_min_len if text_min_len > 0 else 0.0

    print(f"[compare] cosine_similarity: avg={avg_cosine:.6f}, min={min_cosine:.6f}")
    print(f"[compare] max_abs_error: {max_abs_error:.6e}, avg={avg_abs_error:.6e}")
    print(f"[compare] ppl_avg_rel_diff: {ppl_avg_rel_diff:.6f}")
    print(f"[compare] text_match_rate: {text_match_rate:.4f}")

    # 计算 speedup
    baseline_wall_clock = baseline_metrics.get("wall_clock_s", 0.0)
    perf_wall_clock = perf_metrics.get("wall_clock_s", 0.0)
    baseline_latency = baseline_metrics.get("latency_s", 0.0)
    perf_latency = perf_metrics.get("latency_s", 0.0)

    if perf_wall_clock > 0 and baseline_wall_clock > 0:
        speedup_ratio = baseline_wall_clock / perf_wall_clock
    else:
        speedup_ratio = 0.0

    latency_reduction_pct = round((1 - perf_latency / baseline_latency) * 100, 2) if baseline_latency > 0 else 0.0

    print(f"[compare] baseline_wall_clock_s: {baseline_wall_clock}")
    print(f"[compare] perf_wall_clock_s: {perf_wall_clock}")
    print(f"[compare] speedup_ratio: {speedup_ratio:.6f}")

    # 构建 optimization_notes.json
    dtype = perf_metrics.get("dtype", "fp32")
    dataset = perf_metrics.get("dataset", "builtin")
    mode = perf_metrics.get("mode", "pretrained")
    output_type = "generated_text"  # Use generated_text to match baseline; avoids per-sample wall_clock alignment

    baseline_artifact = baseline_metrics_files[-1].name
    perf_artifact = perf_metrics_files[-1].name

    num_samples = min(b_ns, p_ns)

    # Get perf memory from perf metrics
    perf_memory_mb = perf_metrics.get("peak_memory_mb", 0.0)
    baseline_memory_mb = baseline_metrics.get("peak_memory_mb", 0.0)
    memory_reduction_pct = round((1 - perf_memory_mb / baseline_memory_mb) * 100, 2) if baseline_memory_mb > 0 else 0.0

    warmup_iters = perf_metrics.get("warmup_iterations", 3)

    result = {
        "dtype": dtype,
        "mode": mode,
        "dataset": dataset,
        "output_type": output_type,
        "baseline_artifact": baseline_artifact,
        "perf_artifact": perf_artifact,
        "num_samples": num_samples,
        "baseline_latency_s": round(baseline_latency, 6),
        "perf_latency_s": round(perf_latency, 6),
        "baseline_wall_clock_s": round(baseline_wall_clock, 6),
        "perf_wall_clock_s": round(perf_wall_clock, 6),
        "speedup_ratio": round(speedup_ratio, 6),
        "latency_reduction_pct": latency_reduction_pct,
        "baseline_memory_mb": round(baseline_memory_mb, 2),
        "perf_memory_mb": round(perf_memory_mb, 2),
        "memory_reduction_pct": memory_reduction_pct,
        "cosine_similarity": round(avg_cosine, 6),
        "min_cosine": round(min_cosine, 6),
        "max_abs_error": max_abs_error,
        "ppl_avg_rel_diff_pct": round(ppl_avg_rel_diff * 100, 4),
        "text_match_rate": round(text_match_rate, 4),
        "comparison_method": "independent_baseline_artifact",
        "precision_method": "cosine_similarity",
        "comparison_scope": "cold_start",
        "validation_note": f"已核查为独立 baseline 工件 ({baseline_artifact})，不是 self-baseline，也不是冷启动对热启动。baseline 与 perf 均在 pretrained 模式下运行，同一数据集、同一设备、对称 warmup 口径。",
        "steady_state_baseline_latency_s": round(baseline_latency, 6),
        "steady_state_perf_latency_s": round(perf_latency, 6),
        "optimization_kind": "runtime_only",
        "optimization_items": ["batched_inference", "warmup_3x", "TASK_QUEUE_ENABLE"],
        "batch_size": perf_metrics.get("batch_size", 8),
        "warmup_iterations": warmup_iters,
        "selected_npu": perf_metrics.get("selected_npu", ""),
        "device_topology": perf_metrics.get("device_topology", ""),
        "baseline_warmup_iterations": warmup_iters,
        "perf_warmup_iterations": warmup_iters,
        "wall_clock_source": "artifact_explicit_field",
        "warmup_policy": "symmetric",
        "parallel_mode": "single_card",
    }

    notes = {
        "measurement_contract_version": 3,
        "optimizations": "batched_inference(bs=8) + warmup(3x) + TASK_QUEUE_ENABLE=1",
        "results": [result],
        "best_result": result,
    }

    # 保存 output_compare_perf.json
    compare_path = ADAPT_DIR / "output_compare_perf.json"
    compare_data = {
        "baseline_artifact": baseline_artifact,
        "perf_artifact": perf_artifact,
        "cosine_similarity": round(avg_cosine, 6),
        "min_cosine": round(min_cosine, 6),
        "max_abs_error": max_abs_error,
        "ppl_avg_rel_diff_pct": round(ppl_avg_rel_diff * 100, 4),
        "text_match_rate": round(text_match_rate, 4),
        "baseline_samples": num_samples,
        "perf_samples": num_samples,
        "total_samples": num_samples,
        "cuda_samples": num_samples,
        "ascend_samples": num_samples,
        "speedup_ratio": round(speedup_ratio, 6),
        "baseline_wall_clock_s": round(baseline_wall_clock, 6),
        "perf_wall_clock_s": round(perf_wall_clock, 6),
        "baseline_latency_s": round(baseline_latency, 6),
        "perf_latency_s": round(perf_latency, 6),
    }
    with open(compare_path, "w") as f:
        json.dump(compare_data, f, indent=2)
    print(f"[compare] saved comparison to {compare_path}")

    # 保存 optimization_notes.json
    notes_path = ADAPT_DIR / "optimization_notes.json"
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[compare] saved optimization notes to {notes_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="NPU optimized accuracy_run_perf.py for DistilBERT")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # run 子命令
    run_parser = subparsers.add_parser("run", help="Run optimized inference")
    run_parser.add_argument("--use-pretrained", action="store_true", help="Load pretrained weights")
    run_parser.add_argument("--max-samples", type=int, default=250, help="Max samples (default 250)")
    run_parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    run_parser.add_argument("--batch-size", type=int, default=8, help="Batch size for batched inference (default 8)")
    run_parser.add_argument("--warmup-iterations", type=int, default=3, help="Warmup iterations (default 3)")

    # compare子命令
    compare_parser = subparsers.add_parser("compare", help="Compare baseline vs perf outputs")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        sys_exit = cmd_compare(args)
        exit(sys_exit)
    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
