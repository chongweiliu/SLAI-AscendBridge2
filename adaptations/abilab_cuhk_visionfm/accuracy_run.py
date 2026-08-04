#!/usr/bin/env python3
"""VisionFM benchmark (accuracy_run.py) — PAPILA test set, pretrained encoder baseline.

产出:
  outputs_{dataset_name}_{mode_str}_{device_str}.pt          每图 768 维 CLS embedding + labels
  benchmark_metrics_{dataset_name}_{mode_str}_{device_str}.json  latency/memory/num_samples

口径: 加载 VFM_Fundus_weights.pth (iBOT 预训练 teacher encoder), 真实 PAPILA 眼底图前向.
mode_str=pretrained (加载预训练权重); --no-use-pretrained 走 from_config (随机初始化).
设备 NPU > CUDA > CPU, device_str = device.replace(':', '_') -> npu_0.
"""
import sys
import time
import json
import datetime
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

ADAPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPT_DIR))
sys.path.insert(0, str(ADAPT_DIR / "visionfm_src"))
import demo  # build_model, load_pretrained, preprocess, FUNDUS_MEAN/STD

# boundary: 缓存目录固定在 adaptation_path/models, 禁止相对路径 / 项目根 models/
CACHE_DIR = ADAPT_DIR / "models"
DATA_PATH = Path("/models/PAPILA/test")
CLASSES = ["anormal", "bsuspectglaucoma", "cglaucoma"]


def get_dtype_str(dt: torch.dtype) -> str:
    """模型实际 dtype -> 字符串, 禁止按设备推断 / 硬编码 fp32/fp16/bf16."""
    return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dt, str(dt))


def get_device():
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def collect_images(max_samples):
    items = []
    for lbl, cls_name in enumerate(CLASSES):
        d = DATA_PATH / cls_name
        for img_f in sorted(d.iterdir()):
            if img_f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((str(img_f), lbl))
    items = items[:max_samples]
    return items


def build_model_with_weights(use_pretrained, device):
    """Tier1: from_pretrained (加载 teacher 权重); Tier2: from_config (随机初始化)."""
    model = demo.build_model(device)  # vit_base, num_classes=0
    if use_pretrained:
        # from_pretrained 分支: 加载 iBOT 预训练 encoder
        model = demo.load_pretrained(model, CACHE_DIR / "VFM_Fundus_weights.pth", device, key="teacher")
    else:
        # from_config 分支: 随机初始化, 模型必须在 device 上推理 (禁止 model.cpu())
        model = model.to(device)
        model.eval()
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-pretrained", action="store_true",
                    help="加载预训练权重 (from_pretrained); 不加则 from_config 随机初始化")
    ap.add_argument("--max-samples", type=int, default=250)
    ap.add_argument("--dataset", default="papila")
    args = ap.parse_args()

    device = get_device()
    device_str = device.replace(":", "_")  # npu_0
    mode_str = "pretrained" if args.use_pretrained else "config"
    dataset_name = args.dataset
    print(f"[Device] {device} ({device_str}) | mode={mode_str} | dataset={dataset_name}")

    model = build_model_with_weights(args.use_pretrained, device)
    dtype_str = get_dtype_str(next(model.parameters()).dtype)
    print(f"[Model] ViT-B/16 encoder, dtype={dtype_str}")

    items = collect_images(args.max_samples)
    num_samples = len(items)
    print(f"[Data] {num_samples} images")
    if num_samples == 0:
        raise SystemExit("no images found")

    # warmup (与 perf 对称, 3 轮)
    for _ in range(3):
        inp0 = demo.preprocess(items[0][0], device)
        _ = model(inp0)
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()

    # 前向 + 计时 (与 perf 对称: 3 warmup + 5-round min)
    all_emb, all_labels = [], []
    start_dt = datetime.datetime.now()
    round_latencies = []
    BATCH = 16
    for r in range(5):
        embs, labels = [], []
        t0 = time.time()
        for i in range(0, num_samples, BATCH):
            chunk = items[i:i + BATCH]
            imgs = torch.cat([demo.preprocess(p, device) for p, _ in chunk], dim=0)
            out = model(imgs)  # [B, 768] CLS token
            embs.append(out.cpu().float())
            labels.extend([l for _, l in chunk])
        if device.startswith("npu"):
            torch.npu.synchronize()
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
        round_latencies.append(time.time() - t0)
        all_emb = torch.cat(embs, dim=0)
        all_labels = labels
    total_wall = min(round_latencies)
    latency_s = total_wall / num_samples  # per-sample 前向延迟
    end_dt = datetime.datetime.now()
    print(f"[Run] 5 rounds: {[round(x,3) for x in round_latencies]} -> min_wall={total_wall:.4f}s, per_sample={latency_s:.6f}s")

    outputs = all_emb  # [N, 768]
    labels = torch.tensor(all_labels)
    print(f"[Run] outputs shape={tuple(outputs.shape)}, latency={latency_s:.4f}s")

    # peak memory
    peak_mem_mb = 0.0
    if device.startswith("npu"):
        peak_mem_mb = torch.npu.max_memory_allocated(device) / 1024 / 1024
    elif device.startswith("cuda"):
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    out_pt = ADAPT_DIR / f"outputs_{dataset_name}_{mode_str}_{device_str}.pt"
    metric_json = ADAPT_DIR / f"benchmark_metrics_{dataset_name}_{mode_str}_{device_str}.json"
    torch.save({"embeddings": outputs, "labels": labels, "logits": outputs}, str(out_pt))
    metric = {
        "num_samples": num_samples,
        "latency_s": round(latency_s, 6),
        "wall_clock_s": round(total_wall, 6),
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
    }
    metric_json.write_text(json.dumps(metric, indent=2))
    print(f"[Save] {out_pt.name} | {metric_json.name}")
    print(f"[Metric] {json.dumps(metric)}")
    print("[Success] benchmark 完成")


if __name__ == "__main__":
    main()
