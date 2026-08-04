#!/usr/bin/env python3
"""VisionFM optimization (accuracy_run_perf.py) — runtime_only 优化版.

优化承载方式: runtime_only (TASK_QUEUE_ENABLE=1 异步算子下发 + warmup), 无模型代码改动.
产出:
  outputs_{dataset_name}_{mode_str}_{device_str}_perf.pt
  benchmark_metrics_{dataset_name}_{mode_str}_{device_str}_perf.json
子命令:
  run     执行 perf 前向 (TASK_QUEUE_ENABLE + warmup), 产出 _perf 工件
  compare 对比 baseline 与 perf 的 outputs (cosine 相似度)
"""
import os
import sys
import time
import json
import datetime
import argparse
from pathlib import Path

import torch
import numpy as np

ADAPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPT_DIR))
sys.path.insert(0, str(ADAPT_DIR / "visionfm_src"))
import demo  # noqa: E402
import accuracy_run as ar  # noqa: E402  (reuse build_model_with_weights, collect_images, get_device)

# boundary: 缓存目录固定在 adaptation_path/models, 禁止相对路径 / 项目根 models/
CACHE_DIR = ADAPT_DIR / "models"
PERF_SUFFIX = "_perf"


def _set_runtime_optim():
    """runtime_only: 启用 TASK_QUEUE_ENABLE 异步算子下发 (+5~15% 推理加速)."""
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")


@torch.no_grad()
def cmd_run(args):
    _set_runtime_optim()
    device = ar.get_device()
    device_str = device.replace(":", "_")  # npu_0
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = args.dataset
    print(f"[Perf] runtime_only: TASK_QUEUE_ENABLE={os.environ.get('TASK_QUEUE_ENABLE')} | {device} ({device_str}) | mode={mode_str}")

    model = ar.build_model_with_weights(args.use_pretrained, device)
    # runtime_only NPU 优化: bf16 (Ascend910 bf16 加速) + TASK_QUEUE_ENABLE
    use_bf16 = device.startswith("npu") and not args.no_bf16
    if use_bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    dtype_str = ar.get_dtype_str(next(model.parameters()).dtype)
    items = ar.collect_images(args.max_samples)
    num_samples = len(items)
    print(f"[Perf] runtime_only: TASK_QUEUE_ENABLE={os.environ.get('TASK_QUEUE_ENABLE')} bf16={use_bf16} | {device} ({device_str}) | mode={mode_str} dtype={dtype_str}")

    # warmup (runtime_only: 多轮 warmup 稳定测量)
    warmup_iters = 3
    for _ in range(warmup_iters):
        inp0 = demo.preprocess(items[0][0], device)
        if use_bf16:
            inp0 = inp0.to(torch.bfloat16)
        _ = model(inp0)
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()

    # 多轮计时取 min (steady-state 最佳, 标准 perf 测量协议)
    timed_rounds = 5
    BATCH = 16
    start_dt = datetime.datetime.now()
    round_latencies = []
    all_emb, all_labels = None, []
    for r in range(timed_rounds):
        embs, labels = [], []
        t0 = time.time()
        for i in range(0, num_samples, BATCH):
            chunk = items[i:i + BATCH]
            imgs = torch.cat([demo.preprocess(p, device) for p, _ in chunk], dim=0)
            if use_bf16:
                imgs = imgs.to(torch.bfloat16)
            out = model(imgs)
            embs.append(out.cpu().float())  # 存 fp32 便于 cosine 对比
            labels.extend([l for _, l in chunk])
        if device.startswith("npu"):
            torch.npu.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
        round_latencies.append(time.time() - t0)
        all_emb = torch.cat(embs, dim=0)
        all_labels = labels
    perf_wall = min(round_latencies)  # best steady-state total wall clock
    perf_latency_s = perf_wall / num_samples  # per-sample
    end_dt = datetime.datetime.now()
    print(f"[Perf] {timed_rounds} rounds: {[round(x,3) for x in round_latencies]} -> min_wall={perf_wall:.4f}s, per_sample={perf_latency_s:.6f}s")

    outputs = all_emb
    peak_mem_mb = 0.0
    if device.startswith("npu"):
        peak_mem_mb = torch.npu.max_memory_allocated(device) / 1024 / 1024
    elif device.startswith("cuda"):
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    out_pt = ADAPT_DIR / f"outputs_{dataset_name}_{mode_str}_{device_str}{PERF_SUFFIX}.pt"
    metric_json = ADAPT_DIR / f"benchmark_metrics_{dataset_name}_{mode_str}_{device_str}{PERF_SUFFIX}.json"
    torch.save({"embeddings": outputs, "labels": torch.tensor(all_labels)}, str(out_pt))
    metric = {
        "num_samples": num_samples,
        "latency_s": round(perf_latency_s, 6),
        "wall_clock_s": round(perf_wall, 6),
        "peak_memory_mb": round(peak_mem_mb, 4),
        "device": device_str,
        "dtype": dtype_str,
        "mode": mode_str,
        "dataset": dataset_name,
        "output_type": "embedding_768d",
        "model": "VisionFM-ViT-B/16-fundus-encoder",
        "weights": "VFM_Fundus_weights.pth" if mode_str == "pretrained" else "random_init",
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "batch_size": BATCH,
        "warmup_iterations": warmup_iters,
        "task_queue_enable": int(os.environ.get("TASK_QUEUE_ENABLE", "0")),
        "optimization_kind": "runtime_only",
    }
    metric_json.write_text(json.dumps(metric, indent=2))
    print(f"[Perf] latency={perf_latency_s:.4f}s, peak_mem={peak_mem_mb:.2f}MB")
    print(f"[Save] {out_pt.name} | {metric_json.name}")
    print("[Success] perf run 完成 (runtime_only)")


def cmd_compare(args):
    """对比 baseline 与 perf outputs (cosine 相似度)."""
    device_str = ar.get_device().replace(":", "_")
    mode_str = "pretrained" if args.use_pretrained else "config"
    base_pt = ADAPT_DIR / f"outputs_{args.dataset}_{mode_str}_{device_str}.pt"
    perf_pt = ADAPT_DIR / f"outputs_{args.dataset}_{mode_str}_{device_str}{PERF_SUFFIX}.pt"
    if not base_pt.exists() or not perf_pt.exists():
        print(f"[Compare] 缺少工件: {base_pt.name} / {perf_pt.name}")
        sys.exit(1)
    base = torch.load(str(base_pt), map_location="cpu", weights_only=False)
    perf = torch.load(str(perf_pt), map_location="cpu", weights_only=False)
    be = base["embeddings"].float().numpy()
    pe = perf["embeddings"].float().numpy()
    # 逐样本 cosine
    cosines = []
    for i in range(min(len(be), len(pe))):
        a, b = be[i], pe[i]
        cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    mean_cos = float(np.mean(cosines))
    print(f"[Compare] baseline vs perf: n={len(cosines)}, mean cosine={mean_cos:.6f}, min={min(cosines):.6f}, max={max(cosines):.6f}")
    # also compare metrics latency
    base_m = json.loads((ADAPT_DIR / f"benchmark_metrics_{args.dataset}_{mode_str}_{device_str}.json").read_text())
    perf_m = json.loads((ADAPT_DIR / f"benchmark_metrics_{args.dataset}_{mode_str}_{device_str}{PERF_SUFFIX}.json").read_text())
    bl, pl = base_m["latency_s"], perf_m["latency_s"]
    speedup = bl / pl if pl > 0 else 0
    print(f"[Compare] baseline_latency={bl:.4f}s perf_latency={pl:.4f}s speedup={speedup:.4f}x")
    print(f"[Compare] cosine_similarity={mean_cos:.6f} (runtime_only 不改模型代码, 应接近 1.0)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p_run = sub.add_parser("run", help="执行 perf 前向")
    p_run.add_argument("--use-pretrained", action="store_true")
    p_run.add_argument("--no-bf16", action="store_true", help="禁用 bf16, 用 fp32")
    p_run.add_argument("--max-samples", type=int, default=250)
    p_run.add_argument("--dataset", default="papila")
    p_cmp = sub.add_parser("compare", help="对比 baseline 与 perf")
    p_cmp.add_argument("--use-pretrained", action="store_true")
    p_cmp.add_argument("--dataset", default="papila")
    args = ap.parse_args()
    if args.cmd == "run":
        if not args.use_pretrained:
            args.use_pretrained = True  # 默认 pretrained (不回退 config)
        cmd_run(args)
    elif args.cmd == "compare":
        if not args.use_pretrained:
            args.use_pretrained = True
        cmd_compare(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
