#!/usr/bin/env python3
"""Benchmark for cross-encoder/ms-marco-MiniLM-L6-v2 (reranker).

双塔相关度打分模型：输入 (query, passage) 对，输出单值相关性 logit。
画像：相关段落得分应高于无关段落（排序正确性）。
产出 outputs_*.pt / benchmark_metrics_*.json / trace_*.json。
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
DATASET_NAME = "builtin"

# 60 条 (query, relevant_passage, irrelevant_passage) 内置样本，满足 completed gate num_samples>=50
EVAL_PAIRS = [
    ("What is the capital of France?", "Paris is the capital and most populous city of France.", "Mix the flour and sugar in a large bowl."),
    ("How many planets are in the solar system?", "There are eight planets in the solar system.", "The cake should bake in the oven for about 45 minutes."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare.", "The stock market closed higher on Tuesday."),
    ("What is the chemical symbol for water?", "The chemical symbol for water is H2O.", "A bicycle has two wheels and a frame."),
    ("When did World War II end?", "World War II ended in 1945.", "Python is a popular programming language."),
    ("What is the largest mammal?", "The blue whale is the largest mammal on Earth.", "The library opens at nine in the morning."),
    ("What gas do plants absorb?", "Plants absorb carbon dioxide during photosynthesis.", "The train arrives at platform three."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci.", "Coffee is brewed from roasted beans."),
    ("What is the speed of light?", "The speed of light is approximately 300000 km per second.", "The hotel has a swimming pool on the roof."),
    ("What is the currency of Japan?", "The currency of Japan is the Japanese yen.", "Mount Everest is the tallest mountain."),
    ("What causes rainbows?", "Rainbows are caused by sunlight refracting through raindrops.", "The concert was postponed to next week."),
    ("Who developed the theory of relativity?", "Albert Einstein developed the theory of relativity.", "The recipe calls for two cups of rice."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius at sea level.", "The dog barked at the mailman."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth.", "She bought a new pair of shoes."),
    ("What is photosynthesis?", "Photosynthesis is the process by which plants make food from sunlight.", "The meeting was scheduled for Friday."),
    ("Who invented the telephone?", "Alexander Graham Bell invented the telephone.", "The river flows into the sea."),
    ("What is the capital of Japan?", "Tokyo is the capital of Japan.", "The garden is full of roses."),
    ("What is the tallest mountain?", "Mount Everest is the tallest mountain above sea level.", "The car needs an oil change."),
    ("What is the population of China?", "China has a population of over one billion people.", "The book is on the top shelf."),
    ("What is the main language of Brazil?", "Portuguese is the main language of Brazil.", "The clock on the wall is broken."),
    ("What metal is liquid at room temperature?", "Mercury is a metal that is liquid at room temperature.", "The children played in the park."),
    ("What is the smallest planet?", "Mercury is the smallest planet in the solar system.", "The store sells fresh vegetables."),
    ("What is the human body's largest organ?", "The skin is the largest organ of the human body.", "The plane took off on time."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au.", "The newspaper arrives every morning."),
    ("What is the capital of Italy?", "Rome is the capital of Italy.", "The bridge crosses the river."),
    ("Who wrote the Origin of Species?", "Charles Darwin wrote the Origin of Species.", "The museum is closed on Mondays."),
    ("What is the currency of the United Kingdom?", "The currency of the United Kingdom is the pound sterling.", "The forest is home to many animals."),
    ("What is the longest river in the world?", "The Nile is often considered the longest river in the world.", "The kitchen smells like cinnamon."),
    ("What is the main component of air?", "Nitrogen is the main component of air.", "The theater is showing a new film."),
    ("What is the capital of Germany?", "Berlin is the capital of Germany.", "The ladder leans against the wall."),
    ("Who painted the Sistine Chapel ceiling?", "Michelangelo painted the Sistine Chapel ceiling.", "The soup is too salty."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius at sea level.", "The baby is sleeping in the crib."),
    ("What is the largest desert?", "The Sahara is the largest hot desert in the world.", "The office is on the fifth floor."),
    ("What is the chemical symbol for oxygen?", "The chemical symbol for oxygen is O.", "The road is under construction."),
    ("What is the capital of Russia?", "Moscow is the capital of Russia.", "The flowers are blooming in spring."),
    ("Who wrote the play Hamlet?", "William Shakespeare wrote the play Hamlet.", "The computer needs a new battery."),
    ("What is the largest country by area?", "Russia is the largest country by area.", "The restaurant serves Italian food."),
    ("What is the speed of sound?", "The speed of sound is about 343 meters per second in air.", "The painting hangs in the hallway."),
    ("What is the capital of Canada?", "Ottawa is the capital of Canada.", "The shoes are too tight."),
    ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe.", "The moon is full tonight."),
    ("What is the tallest animal?", "The giraffe is the tallest living animal.", "The door is painted blue."),
    ("What is the capital of Australia?", "Canberra is the capital of Australia.", "The mug is filled with tea."),
    ("Who discovered penicillin?", "Alexander Fleming discovered penicillin.", "The rug is on the floor."),
    ("What is the largest planet?", "Jupiter is the largest planet in the solar system.", "The window is open."),
    ("What is the capital of India?", "New Delhi is the capital of India.", "The chair is made of wood."),
    ("What is the chemical symbol for silver?", "The chemical symbol for silver is Ag.", "The lamp is on the desk."),
    ("What is the smallest country?", "Vatican City is the smallest country in the world.", "The clock ticks loudly."),
    ("What is the capital of Egypt?", "Cairo is the capital of Egypt.", "The grass is green."),
    ("Who wrote the novel 1984?", "George Orwell wrote the novel 1984.", "The cup is empty."),
    ("What is the largest reptile?", "The saltwater crocodile is the largest living reptile.", "The sky is blue."),
    ("What is the capital of Brazil?", "Brasilia is the capital of Brazil.", "The book is heavy."),
    ("What is the chemical symbol for sodium?", "The chemical symbol for sodium is Na.", "The cat is sleeping."),
    ("What is the hottest planet?", "Venus is the hottest planet in the solar system.", "The table is round."),
    ("What is the capital of Mexico?", "Mexico City is the capital of Mexico.", "The pen is red."),
    ("Who composed the Ninth Symphony?", "Beethoven composed the Ninth Symphony.", "The shirt is cotton."),
    ("What is the largest bird?", "The ostrich is the largest living bird.", "The floor is wet."),
    ("What is the capital of Spain?", "Madrid is the capital of Spain.", "The bag is leather."),
    ("What is the chemical symbol for potassium?", "The chemical symbol for potassium is K.", "The vase is ceramic."),
    ("What is the deepest ocean trench?", "The Mariana Trench is the deepest ocean trench.", "The pillow is soft."),
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
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

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

    use_pretrained = args.use_pretrained
    max_samples = args.max_samples

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    if use_pretrained:
        print("[Setup] Loading pretrained weights...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    else:
        print("[Setup] DRY/config mode: random weights")
        model = AutoModelForSequenceClassification.from_config(config)
    model = model.to(device)
    model.eval()
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    mode_str = "pretrained" if use_pretrained else "config"
    dataset_name = DATASET_NAME
    print(f"[Setup] Model dtype: {dtype_str}, mode={mode_str}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    pairs = EVAL_PAIRS[:max_samples]
    n = len(pairs) * 2  # relevant + irrelevant
    print(f"[benchmark] {n} samples (relevant+irrelevant for {len(pairs)} queries)")

    import torch_npu  # noqa: F401  (ensure npu profiler import works)

    def score_batch(queries, passages):
        inputs = tokenizer(queries, passages, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.logits.squeeze(-1).float()

    # Step1: trace + metrics (single batch)
    queries0 = [pairs[0][0]] * 2
    passages0 = [pairs[0][1], pairs[0][2]]
    trace_path = Path(__file__).resolve().parent / f"trace_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    try:
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile
        with npu_profile(activities=[NPUActivity.CPU, NPUActivity.NPU]) as prof:
            t0 = time.time()
            _ = score_batch(queries0, passages0)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            step1_latency = time.time() - t0
        prof.export_trace(str(trace_path))
    except Exception as e:
        step1_latency = 0.0
        if not trace_path.exists():
            trace_path.write_text(json.dumps({"fallback": str(e)}))
    print(f"[benchmark] step1 latency: {step1_latency:.4f}s, trace: {trace_path.exists()}")

    # Step2: all samples
    all_scores = []
    all_logits = []
    correct = 0
    t_all = time.time()
    with torch.no_grad():
        for q, rel, irrel in pairs:
            sc = score_batch([q, q], [rel, irrel]).tolist()
            all_scores.append(sc)
            if sc[0] > sc[1]:
                correct += 1
    total_latency = time.time() - t_all
    peak_mem = 0
    try:
        if hasattr(torch, "npu"):
            peak_mem = torch.npu.max_memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    outputs = {
        "texts": [{"query": q, "relevant": r, "irrelevant": i} for q, r, i in pairs],
        "scores": all_scores,
        "ranking_correct": correct,
        "ranking_total": len(pairs),
    }
    out_name = f"outputs_npu_{dtype_str}_{mode_str}_{dataset_name}.pt"
    torch.save(outputs, Path(__file__).resolve().parent / out_name)
    print(f"[benchmark] outputs saved: {out_name} (ranking correct {correct}/{len(pairs)})")

    metric = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step1_forward_latency_s": round(step1_latency, 6),
        "latency_s": round(total_latency / max(n, 1), 6),
        "peak_memory_mb": round(peak_mem, 2),
        "num_samples": n,
        "device": device,
        "device_model": "Ascend910",
        "mode": "pretrained" if use_pretrained else "config",
        "output_type": "rerank_scores",
        "dataset": DATASET_NAME,
        "dtype": dtype_str,
        "ranking_accuracy": round(correct / len(pairs), 4),
        "packages": {"torch": torch.__version__, "torch_npu": getattr(__import__("torch_npu"), "__version__", "n/a")},
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    met_name = f"benchmark_metrics_npu_{dtype_str}_{mode_str}_{dataset_name}.json"
    json.dump(metric, open(Path(__file__).resolve().parent / met_name, "w"), indent=2, ensure_ascii=False)
    print(f"[benchmark] metrics saved: {met_name}")
    print(f"[benchmark] DONE: {n} samples, ranking {correct}/{len(pairs)}")


if __name__ == "__main__":
    main()
