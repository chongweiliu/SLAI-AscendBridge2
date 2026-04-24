#!/usr/bin/env python3
"""
Benchmark Tool - 整合聚合、对比、Raw Profiling 与 Trace 分析功能。

子命令:
  aggregate  - 聚合 benchmark 数据，生成汇总报告
  compare    - 对比 CUDA/NPU 输出结果
  profiling  - 分析 torch_npu.profiler 原始目录
  trace      - 分析 trace 文件，统计 fallback 比例

Usage:
  uv run python benchmark/scripts/benchmark_tool.py aggregate
  uv run python benchmark/scripts/benchmark_tool.py compare --all
  uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1
  uv run python benchmark/scripts/benchmark_tool.py trace --all
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

# =============================================================================
# 路径常量
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", _SCRIPT_DIR.parent.parent))
DB_PATH = _PROJECT_ROOT / "board.db"
ADAPTATIONS_DIR = _PROJECT_ROOT / "adaptations"
BENCHMARK_DIR = _PROJECT_ROOT / "benchmark"
REPORTS_DIR = BENCHMARK_DIR / "reports"
FIGURES_DIR = BENCHMARK_DIR / "figures"


# =============================================================================
# 颜色常量
# =============================================================================

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


# =============================================================================
# 文件名解析
# =============================================================================

# 文件名格式: {type}_{device}_{precision}_{mode}_{dataset}.{ext}
FILENAME_PATTERN = re.compile(
    r"^(?P<file_type>outputs|logits|benchmark_metrics|trace)_"
    r"(?P<device>\w+)_"
    r"(?P<precision>\w+)_"
    r"(?P<mode>\w+)_"
    r"(?P<dataset>\w+)"
    r"\.(?P<ext>pt|json)$"
)


@dataclass(frozen=True)
class BenchmarkFilename:
    """解析后的 benchmark 文件名组件。"""

    file_type: str  # outputs | benchmark_metrics | trace
    device: str  # cuda | npu | cpu
    precision: str  # fp16 | fp32 | bf16 | int8
    mode: str  # config | pretrained | random
    dataset: str  # wikitext | cifar100 | imagenet | random | etc.
    extension: str  # pt | json

    @classmethod
    def parse(cls, filename: str) -> Optional["BenchmarkFilename"]:
        """解析文件名字符串。"""
        match = FILENAME_PATTERN.match(Path(filename).name)
        if not match:
            return None
        groups = match.groupdict()
        # 映射正则捕获组名到 dataclass 字段名
        return cls(
            file_type=groups["file_type"],
            device=groups["device"],
            precision=groups["precision"],
            mode=groups["mode"],
            dataset=groups["dataset"],
            extension=groups["ext"],
        )

    def to_filename(self) -> str:
        """生成文件名。"""
        return f"{self.file_type}_{self.device}_{self.precision}_{self.mode}_{self.dataset}.{self.extension}"


def get_adaptation_dir(model_id_or_path: str) -> Path:
    """解析适配目录路径。"""
    if model_id_or_path.startswith("adaptations/"):
        return _PROJECT_ROOT / model_id_or_path
    return ADAPTATIONS_DIR / model_id_or_path


def fmt_size(n: int) -> str:
    """格式化字节大小。"""
    for unit in ["G", "M", "K"]:
        factor = {"G": 1024**3, "M": 1024**2, "K": 1024}[unit]
        if n >= factor:
            return f"{n / factor:.2f}{unit}"
    return f"{n}B"


def _portable_output_path(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve().relative_to(_PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


# =============================================================================
# 输出类型检测
# =============================================================================


def detect_output_type(data: Any) -> str:
    """检测输出类型: generated_text / class_labels / cls_embeddings / logits / mixed / empty / unknown

    mixed 类型：dict 格式，同时包含 generated_text 和 logits
    """
    if data is None:
        return "unknown"
    if isinstance(data, dict):
        if "generated_text" in data and "logits" in data:
            return "mixed"
        if "embeddings" in data and _is_tensor_list(data["embeddings"]):
            return "cls_embeddings"
        if "cls_embeddings" in data and _is_tensor_list(data["cls_embeddings"]):
            return "cls_embeddings"
        if "image_embeddings" in data and _is_tensor_list(data["image_embeddings"]):
            return "cls_embeddings"
        if "mask_logits" in data and _is_tensor_list(data["mask_logits"]):
            return "logits"
        if "depth_maps" in data and _is_tensor_list(data["depth_maps"]):
            return "logits"
        if "output_images" in data and _is_tensor_list(data["output_images"]):
            return "logits"
        if "generated_text" in data or "generated_texts" in data:
            return "generated_text_ppl"  # 有文本，可能还有 perplexity
        # 扩散输出：对比 latent_mean / latent_std 统计量
        if "diffusion_outputs" in data and isinstance(data.get("diffusion_outputs"), list) and data["diffusion_outputs"]:
            first = data["diffusion_outputs"][0]
            if isinstance(first, dict) and ("latent_mean" in first or "latent_std" in first):
                return "diffusion_stats"
        # 检测结果：dict 顶层 detections（list of per-image results）可对比
        if "detections" in data and isinstance(data.get("detections"), list) and data["detections"]:
            first = data["detections"][0]
            if isinstance(first, dict) and ("detections" in first or "boxes" in first):
                return "detection_boxes"
        # 目标检测 dict 顶层 boxes（如 grounding-dino）：可能每图数量不同，也尝试 IoU 对比
        if "boxes" in data and isinstance(data.get("boxes"), list) and data["boxes"]:
            return "detection_boxes"
        if "scores" in data or "relevance_scores" in data:
            return "logits"
        if "predictions" in data and _is_tensor_list(data["predictions"]):
            return "logits"
        if "start_logits" in data or "end_logits" in data:
            return "qa_logits"
        if "class_logits" in data and _is_tensor_list(data["class_logits"]):
            return "logits"
        if "logits" in data or "tensor" in data:
            return "logits"
        if "forecasts" in data and _is_tensor_list(data.get("forecasts", [])):
            return "logits"
        if "answers" in data and isinstance(data.get("answers"), list) and data["answers"]:
            return "generated_text_list" if isinstance(data["answers"][0], dict) else "answers_text"
        return "logits"
    if isinstance(data, list):
        if not data:
            return "empty"
        first_item = data[0]
        if isinstance(first_item, str):
            return "class_labels" if len(first_item) < 50 and " " not in first_item else "generated_text"
        if isinstance(first_item, torch.Tensor):
            if first_item.dim() == 2 and first_item.shape[0] == 1:
                return "cls_embeddings"
            return "logits"
        if isinstance(first_item, (list, tuple)) and first_item:
            # NER/序列标注：list of list of int (label ids)
            if isinstance(first_item[0], (int, float)):
                return "class_labels"
        if isinstance(first_item, dict):
            if "generated_text" in first_item or "generated_texts" in first_item or "predicted_answer" in first_item:
                return "generated_text_list"
            if "boxes" in first_item and isinstance(first_item.get("boxes"), list):
                return "detection_boxes"
            if "detections" in first_item and isinstance(first_item.get("detections"), list):
                return "detection_boxes"
    if isinstance(data, torch.Tensor):
        return "logits"
    return "unknown"


def _is_tensor_list(data: Any) -> bool:
    """判断是否为张量列表。"""
    if not isinstance(data, list) or not data:
        return False
    return isinstance(data[0], torch.Tensor) or (isinstance(data[0], (list, tuple)) and len(data[0]) > 0 and isinstance(data[0][0], (int, float)))


# =============================================================================
# Aggregate 功能
# =============================================================================


def _parse_dt(s: str) -> Optional[datetime]:
    """解析日期时间字符串。"""
    if not s:
        return None
    s = s.strip()[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def get_processing_time_from_board() -> list[dict]:
    """从 board.db 读取 completed 任务，计算耗时。"""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT model_id, adaptation_path, adaptation_status, adaptation_started_at, adaptation_last_updated
        FROM models
        WHERE adaptation_status = 'completed'
        """
    )
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        started = _parse_dt(d.get("adaptation_started_at") or "")
        updated = _parse_dt(d.get("adaptation_last_updated") or "")
        if started and updated:
            d["time_cost_seconds"] = (updated - started).total_seconds()
        else:
            d["time_cost_seconds"] = None
        rows.append(d)
    conn.close()
    return rows


def collect_benchmark_metrics(
    device_filter: Optional[str] = None,
    mode_filter: Optional[str] = None,
) -> dict[str, list[dict]]:
    """扫描 adaptations/* 收集 benchmark_metrics_*.json。"""
    out: dict[str, list[dict]] = {}
    if not ADAPTATIONS_DIR.exists():
        return out

    for adir in ADAPTATIONS_DIR.iterdir():
        if not adir.is_dir():
            continue
        metrics_files = sorted(adir.glob("benchmark_metrics_*.json"))
        if not metrics_files:
            continue

        out[adir.name] = []
        for mf in metrics_files:
            parsed = BenchmarkFilename.parse(mf.name)
            if not parsed:
                continue
            if device_filter and parsed.device != device_filter:
                continue
            if mode_filter and parsed.mode != mode_filter:
                continue
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                data["_filename_context"] = {
                    "device": parsed.device,
                    "precision": parsed.precision,
                    "mode": parsed.mode,
                    "dataset": parsed.dataset,
                }
                out[adir.name].append(data)
            except Exception:
                pass
    return out


def cmd_aggregate(args):
    """聚合 benchmark 数据，生成汇总报告。"""
    print(f"{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}{BOLD}  Aggregate Benchmark Data{NC}")
    print(f"{BLUE}{'=' * 60}{NC}\n")

    device_filter = args.device
    mode_filter = args.mode

    board_rows = get_processing_time_from_board()
    metrics_by_adapt = collect_benchmark_metrics(device_filter, mode_filter)

    # 构建 safe_name 映射
    by_safe = {}
    for r in board_rows:
        path = (r.get("adaptation_path") or "").strip()
        if path.startswith("adaptations/"):
            safe = path.replace("adaptations/", "").strip("/")
        else:
            safe = (r.get("model_id") or "").replace("/", "_").replace("-", "_").lower()
        by_safe[safe] = r

    reports = []
    for safe, board in by_safe.items():
        rec = {
            "model_id": board.get("model_id"),
            "adaptation_path": board.get("adaptation_path") or f"adaptations/{safe}",
            "migration": {"time_cost_seconds": board.get("time_cost_seconds")},
            "performance": [],
        }
        if safe in metrics_by_adapt:
            for m in metrics_by_adapt[safe]:
                ctx = m.get("_filename_context", {})
                rec["performance"].append(
                    {
                        "latency_s": m.get("latency_s"),
                        "peak_memory_mb": m.get("peak_memory_mb"),
                        "device": m.get("device"),
                        "precision": ctx.get("precision"),
                        "mode": ctx.get("mode"),
                        "dataset": ctx.get("dataset"),
                        "output_type": m.get("output_type"),
                    }
                )
        reports.append(rec)

    # 输出
    out_dir = Path(args.reports_dir or str(REPORTS_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "filters": {"device": device_filter, "mode": mode_filter},
        "count": len(reports),
        "reports": reports,
    }

    if "json" in args.format:
        out_json = out_dir / "aggregate.json"
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{GREEN}✓{NC} Wrote {out_json} ({len(reports)} models)")

    if "csv" in args.format and reports:
        csv_path = out_dir / "aggregate.csv"
        keys = ["model_id", "time_cost_seconds", "latency_s", "peak_memory_mb", "device", "precision", "mode"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in reports:
                perf = r.get("performance") or [{}]
                for p in perf:
                    row = {
                        "model_id": r.get("model_id"),
                        "time_cost_seconds": (r.get("migration") or {}).get("time_cost_seconds"),
                        "latency_s": p.get("latency_s"),
                        "peak_memory_mb": p.get("peak_memory_mb"),
                        "device": p.get("device"),
                        "precision": p.get("precision"),
                        "mode": p.get("mode"),
                    }
                    w.writerow(row)
        print(f"{GREEN}✓{NC} Wrote {csv_path}")

    if "figures" in args.format:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = [r.get("migration", {}).get("time_cost_seconds") for r in reports if (r.get("migration") or {}).get("time_cost_seconds") is not None]
        if times:
            plt.figure()
            plt.hist(times, bins=min(30, max(1, len(times))))
            plt.xlabel("Processing time (s)")
            plt.ylabel("Count")
            plt.title("Benchmark: processing time distribution")
            plt.savefig(FIGURES_DIR / "time_cost_hist.png", dpi=120)
            plt.close()
            print(f"{GREEN}✓{NC} Wrote {FIGURES_DIR / 'time_cost_hist.png'}")

    print()


# =============================================================================
# Compare 功能
# =============================================================================


def compare_text_outputs(cuda_data: list, ascend_data: list) -> dict:
    """对比文本输出。"""
    cuda_data = cuda_data or []
    ascend_data = ascend_data or []
    if len(cuda_data) != len(ascend_data):
        raise ValueError(f"样本数不匹配: cuda {len(cuda_data)} vs ascend {len(ascend_data)}")

    total = len(cuda_data)
    matches = 0
    similarities = []

    for cuda_text, ascend_text in zip(cuda_data, ascend_data):
        if cuda_text == ascend_text:
            matches += 1
            similarities.append(1.0)
        else:
            sim = SequenceMatcher(None, cuda_text, ascend_text).ratio()
            similarities.append(sim)

    return {
        "match_rate": matches / total if total > 0 else 0.0,
        "avg_text_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
        "total_samples": total,
        "exact_matches": matches,
    }


def compare_class_labels(cuda_data: list, ascend_data: list) -> dict:
    """对比类别标签。"""
    if len(cuda_data) != len(ascend_data):
        raise ValueError(f"样本数不匹配: cuda {len(cuda_data)} vs ascend {len(ascend_data)}")

    total = len(cuda_data)
    matches = sum(1 for c, a in zip(cuda_data, ascend_data) if c == a)

    return {
        "match_rate": matches / total if total > 0 else 0.0,
        "total_samples": total,
        "exact_matches": matches,
    }


def compare_cls_embeddings(cuda_data: list, ascend_data: list) -> dict:
    """对比 [CLS] 嵌入向量。"""
    if len(cuda_data) != len(ascend_data):
        raise ValueError(f"样本数不匹配: cuda {len(cuda_data)} vs ascend {len(ascend_data)}")

    cosine_sims = []
    max_error = 0.0

    for cuda_emb, ascend_emb in zip(cuda_data, ascend_data):
        if not isinstance(cuda_emb, torch.Tensor):
            cuda_emb = torch.tensor(cuda_emb)
        if not isinstance(ascend_emb, torch.Tensor):
            ascend_emb = torch.tensor(ascend_emb)

        cuda_emb = cuda_emb.float().flatten()
        ascend_emb = ascend_emb.float().flatten()

        cos_sim = F.cosine_similarity(cuda_emb.unsqueeze(0), ascend_emb.unsqueeze(0), dim=1).item()
        cosine_sims.append(cos_sim)

        error = torch.max(torch.abs(cuda_emb - ascend_emb)).item()
        max_error = max(max_error, error)

    return {
        "avg_cosine_similarity": sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0,
        "min_cosine_similarity": min(cosine_sims) if cosine_sims else 0.0,
        "max_abs_error": max_error,
        "total_samples": len(cuda_data),
    }


def _extract_tensor_from_dict(d: dict) -> Optional[torch.Tensor]:
    """从 dict 中提取可对比的张量，优先 logits/tensor，避免取到字符串。"""
    for key in (
        "logits",
        "tensor",
        "start_logits",
        "end_logits",
        "class_logits",
        "predictions",
        "embeddings",
        "cls_embeddings",
        "image_embeddings",
        "mask_logits",
        "depth_maps",
        "scores",
        "relevance_scores",
        "output_images",
        "forecasts",
        "last_hidden_state",
        "pooler_output",
        "encoder_last_hidden_state",
    ):
        val = d.get(key)
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            return val
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, torch.Tensor):
                return torch.cat([x.float().flatten() for x in val])
            if isinstance(first, (int, float)):
                return torch.tensor(val, dtype=torch.float32)
    # 兜底：递归查找 dict 中任意张量（跳过字符串等）
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return v.float().flatten()
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            return torch.cat([x.float().flatten() for x in v])
    return None


def compare_embeddings_dict(cuda_data: dict, ascend_data: dict) -> dict:
    """对比 dict 中的 embeddings（如 texts + embeddings）。"""
    for key in ("embeddings", "cls_embeddings", "image_embeddings"):
        cuda_embs = cuda_data.get(key, [])
        ascend_embs = ascend_data.get(key, [])
        if cuda_embs and ascend_embs and _is_tensor_list(cuda_embs):
            return compare_cls_embeddings(cuda_embs, ascend_embs)
    raise ValueError("无法从 dict 中提取 embeddings")


def compare_generated_text_ppl(cuda_data: dict, ascend_data: dict) -> dict:
    """对比 generated_text + perplexity（无 logits 的生成模型）。"""
    cuda_texts = cuda_data.get("generated_text") or cuda_data.get("generated_texts") or []
    ascend_texts = ascend_data.get("generated_text") or ascend_data.get("generated_texts") or []
    text_result = compare_text_outputs(cuda_texts, ascend_texts)

    cuda_ppl = cuda_data.get("perplexity") or []
    ascend_ppl = ascend_data.get("perplexity") or []

    ppl_result = {}
    if cuda_ppl and ascend_ppl:
        ppl_diffs = [abs(c - a) / c for c, a in zip(cuda_ppl, ascend_ppl) if c != 0]
        ppl_result = {
            "ppl_avg_rel_diff": sum(ppl_diffs) / len(ppl_diffs) if ppl_diffs else 0.0,
            "ppl_max_rel_diff": max(ppl_diffs) if ppl_diffs else 0.0,
            "ppl_cuda_avg": sum(cuda_ppl) / len(cuda_ppl),
            "ppl_ascend_avg": sum(ascend_ppl) / len(ascend_ppl),
            "ppl_samples": len(cuda_ppl),
        }

    return {
        "text_match_rate": text_result["match_rate"],
        "text_exact_matches": text_result["exact_matches"],
        "text_total_samples": text_result["total_samples"],
        "avg_text_similarity": text_result.get("avg_text_similarity", 0.0),
        **ppl_result,
    }


def compare_qa_logits(cuda_data: dict, ascend_data: dict) -> dict:
    """对比 QA 模型的 start_logits + end_logits。"""

    def flatten_qa(d):
        sl = d.get("start_logits", [])
        el = d.get("end_logits", [])
        if isinstance(sl, torch.Tensor):
            sl = [sl]
        if isinstance(el, torch.Tensor):
            el = [el]
        parts = []
        for s, e in zip(sl, el):
            if isinstance(s, torch.Tensor) and isinstance(e, torch.Tensor):
                parts.extend([s.float().flatten(), e.float().flatten()])
        return torch.cat(parts) if parts else None

    cuda_flat = flatten_qa(cuda_data)
    ascend_flat = flatten_qa(ascend_data)
    if cuda_flat is None or ascend_flat is None:
        raise ValueError("无法提取 QA logits")
    if cuda_flat.shape != ascend_flat.shape:
        raise ValueError(f"QA 张量尺寸不匹配: {cuda_flat.shape} vs {ascend_flat.shape}")

    cos_sim = F.cosine_similarity(cuda_flat.unsqueeze(0), ascend_flat.unsqueeze(0), dim=1).item()
    max_error = torch.max(torch.abs(cuda_flat - ascend_flat)).item()
    return {"cosine_similarity": cos_sim, "max_abs_error": max_error}


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组 xyxy 格式框的 IoU 矩阵。boxes1: Nx4, boxes2: Mx4 -> NxM。"""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(0, 0, dtype=boxes1.dtype, device=boxes1.device)
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / torch.clamp(union, min=1e-6)


def _extract_boxes_tensor(sample: Any) -> torch.Tensor:
    """从单样本提取 Nx4 框张量 (xyxy)。"""
    if isinstance(sample, dict):
        if "boxes" in sample:
            boxes = sample["boxes"]
        elif "detections" in sample:
            boxes = [d.get("box", d.get("boxes", [])) for d in sample["detections"]]
        else:
            return torch.zeros(0, 4)
    else:
        return torch.zeros(0, 4)
    if not boxes:
        return torch.zeros(0, 4)
    if isinstance(boxes[0], (list, tuple)):
        arr = torch.tensor(boxes, dtype=torch.float32)
    else:
        arr = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)
    if arr.dim() == 1:
        arr = arr.unsqueeze(0)
    return arr.float()


def _match_boxes_greedy_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor, iou_thresh: float = 0.5) -> float:
    """贪心 IoU 匹配，返回匹配对的平均 IoU。"""
    if boxes_a.numel() == 0 and boxes_b.numel() == 0:
        return 1.0
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return 0.0
    iou_mat = _box_iou(boxes_a, boxes_b)
    used_b = set()
    matched_ious = []
    for i in range(iou_mat.shape[0]):
        best_j, best_iou = -1, -1.0
        for j in range(iou_mat.shape[1]):
            if j not in used_b and iou_mat[i, j].item() > best_iou:
                best_iou = iou_mat[i, j].item()
                best_j = j
        if best_j >= 0 and best_iou >= iou_thresh:
            used_b.add(best_j)
            matched_ious.append(best_iou)
    return sum(matched_ious) / len(matched_ious) if matched_ious else 0.0


def _get_detection_samples(data: Any) -> list:
    """统一提取检测样本列表。支持 list[dict] 或 dict.detections 或 dict.boxes。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "detections" in data:
        return data["detections"]
    if isinstance(data, dict) and "boxes" in data:
        boxes_list = data["boxes"]
        scores_list = data.get("scores", [None] * len(boxes_list))
        labels_list = data.get("labels", [None] * len(boxes_list))
        n = len(boxes_list)
        if len(scores_list) != n:
            scores_list = [None] * n
        if len(labels_list) != n:
            labels_list = [None] * n
        return [{"boxes": b, "scores": s, "classes": l} for b, s, l in zip(boxes_list, scores_list, labels_list)]
    return []


def compare_detection_boxes(cuda_data: Any, ascend_data: Any) -> dict:
    """对比检测框输出，使用 IoU 贪心匹配（业界常用）。"""
    cuda_samples = _get_detection_samples(cuda_data)
    ascend_samples = _get_detection_samples(ascend_data)
    if len(cuda_samples) != len(ascend_samples):
        raise ValueError(f"样本数不匹配: cuda {len(cuda_samples)} vs ascend {len(ascend_samples)}")

    ious = []
    count_diffs = []
    for c, a in zip(cuda_samples, ascend_samples):
        boxes_c = _extract_boxes_tensor(c)
        boxes_a = _extract_boxes_tensor(a)
        iou = _match_boxes_greedy_iou(boxes_c, boxes_a)
        ious.append(iou)
        count_diffs.append(abs(boxes_c.shape[0] - boxes_a.shape[0]))

    return {
        "avg_iou": sum(ious) / len(ious) if ious else 0.0,
        "min_iou": min(ious) if ious else 0.0,
        "total_samples": len(cuda_samples),
        "avg_count_diff": sum(count_diffs) / len(count_diffs) if count_diffs else 0.0,
    }


def compare_diffusion_stats(cuda_data: dict, ascend_data: dict) -> dict:
    """对比扩散模型输出的 latent_mean / latent_std 统计量。"""
    cuda_outputs = cuda_data.get("diffusion_outputs", [])
    ascend_outputs = ascend_data.get("diffusion_outputs", [])

    if len(cuda_outputs) != len(ascend_outputs):
        raise ValueError(f"样本数不匹配: cuda {len(cuda_outputs)} vs ascend {len(ascend_outputs)}")

    mean_diffs = []
    std_diffs = []
    for c, a in zip(cuda_outputs, ascend_outputs):
        cm = c.get("latent_mean")
        am = a.get("latent_mean")
        cs = c.get("latent_std")
        as_ = a.get("latent_std")
        if cm is not None and am is not None:
            mean_diffs.append(abs(cm - am))
        if cs is not None and as_ is not None:
            std_diffs.append(abs(cs - as_))

    return {
        "avg_mean_abs_error": sum(mean_diffs) / len(mean_diffs) if mean_diffs else 0.0,
        "max_mean_abs_error": max(mean_diffs) if mean_diffs else 0.0,
        "avg_std_abs_error": sum(std_diffs) / len(std_diffs) if std_diffs else 0.0,
        "max_std_abs_error": max(std_diffs) if std_diffs else 0.0,
        "total_samples": len(cuda_outputs),
    }


def compare_answers_text(cuda_data: dict, ascend_data: dict) -> dict:
    """对比 dict 中的 answers 列表（QA 答案文本）。"""
    cuda_answers = cuda_data.get("answers", [])
    ascend_answers = ascend_data.get("answers", [])
    return compare_text_outputs(cuda_answers, ascend_answers)


def compare_generated_text_list(cuda_data: list, ascend_data: list) -> dict:
    """对比 list[dict] 中每个 dict 的 generated_text / predicted_answer。"""

    def extract_texts(data):
        data = data or []
        return [d.get("generated_text") or d.get("generated_texts") or d.get("predicted_answer") or "" for d in data]

    cuda_texts = extract_texts(cuda_data)
    ascend_texts = extract_texts(ascend_data)
    return compare_text_outputs(cuda_texts, ascend_texts)


def compare_logits_legacy(cuda_data, ascend_data) -> dict:
    """兼容旧格式 logits 对比。"""
    if isinstance(cuda_data, dict):
        cuda_data = _extract_tensor_from_dict(cuda_data)
        if cuda_data is None:
            raise ValueError("无法从 dict 中提取可对比的张量")
    if isinstance(ascend_data, dict):
        ascend_data = _extract_tensor_from_dict(ascend_data)
        if ascend_data is None:
            raise ValueError("无法从 dict 中提取可对比的张量")

    if isinstance(cuda_data, list) and len(cuda_data) > 0:
        cuda_data = cuda_data[0]
    if isinstance(ascend_data, list) and len(ascend_data) > 0:
        ascend_data = ascend_data[0]

    if not isinstance(cuda_data, torch.Tensor):
        cuda_data = torch.tensor(cuda_data, dtype=torch.float32)
    if not isinstance(ascend_data, torch.Tensor):
        ascend_data = torch.tensor(ascend_data, dtype=torch.float32)

    cuda_flat = cuda_data.float().flatten()
    ascend_flat = ascend_data.float().flatten()

    if cuda_flat.shape != ascend_flat.shape:
        raise ValueError(f"张量尺寸不匹配: cuda {cuda_flat.shape} vs ascend {ascend_flat.shape}")

    cos_sim = F.cosine_similarity(cuda_flat.unsqueeze(0), ascend_flat.unsqueeze(0), dim=1).item()
    max_error = torch.max(torch.abs(cuda_flat - ascend_flat)).item()

    return {
        "cosine_similarity": cos_sim,
        "max_abs_error": max_error,
    }


def compare_mixed_outputs(cuda_data: dict, ascend_data: dict) -> dict:
    """对比混合格式输出（同时包含 generated_text、logits 和 perplexity）。"""
    cuda_texts = cuda_data.get("generated_text") or cuda_data.get("generated_texts") or []
    ascend_texts = ascend_data.get("generated_text") or ascend_data.get("generated_texts") or []
    if cuda_texts is None:
        cuda_texts = []
    if ascend_texts is None:
        ascend_texts = []
    text_result = compare_text_outputs(cuda_texts, ascend_texts)

    # 对比 logits（logits 可能为 None，如 prem_research_prem_1b_sql）
    cuda_logits = cuda_data.get("logits") or []
    ascend_logits = ascend_data.get("logits") or []

    cosine_sims = []
    max_errors = []
    for c_log, a_log in zip(cuda_logits, ascend_logits):
        if not isinstance(c_log, torch.Tensor):
            c_log = torch.tensor(c_log)
        if not isinstance(a_log, torch.Tensor):
            a_log = torch.tensor(a_log)

        c_flat = c_log.float().flatten()
        a_flat = a_log.float().flatten()

        cos_sim = F.cosine_similarity(c_flat.unsqueeze(0), a_flat.unsqueeze(0), dim=1).item()
        max_err = torch.max(torch.abs(c_flat - a_flat)).item()

        cosine_sims.append(cos_sim)
        max_errors.append(max_err)

    # 对比 perplexity
    cuda_ppl = cuda_data.get("perplexity") or []
    ascend_ppl = ascend_data.get("perplexity") or []

    ppl_result = {}
    if cuda_ppl and ascend_ppl:
        cuda_ppl_tensor = torch.tensor(cuda_ppl, dtype=torch.float32)
        ascend_ppl_tensor = torch.tensor(ascend_ppl, dtype=torch.float32)

        # 计算 PPL 的相对差异
        ppl_diffs = []
        for c, a in zip(cuda_ppl, ascend_ppl):
            if c != 0:
                rel_diff = abs(c - a) / c
                ppl_diffs.append(rel_diff)

        ppl_result = {
            "ppl_avg_rel_diff": sum(ppl_diffs) / len(ppl_diffs) if ppl_diffs else 0.0,
            "ppl_max_rel_diff": max(ppl_diffs) if ppl_diffs else 0.0,
            "ppl_cuda_avg": sum(cuda_ppl) / len(cuda_ppl),
            "ppl_ascend_avg": sum(ascend_ppl) / len(ascend_ppl),
            "ppl_samples": len(cuda_ppl),
        }

    result = {
        # 文本对比结果
        "text_match_rate": text_result["match_rate"],
        "text_exact_matches": text_result["exact_matches"],
        "text_total_samples": text_result["total_samples"],
        # Logits 对比结果
        "logits_avg_cosine_similarity": sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0,
        "logits_min_cosine_similarity": min(cosine_sims) if cosine_sims else 0.0,
        "logits_max_abs_error": max(max_errors) if max_errors else 0.0,
        "logits_samples": len(cuda_logits),
    }
    result.update(ppl_result)
    return result


def _sample_count(d) -> int:
    """计算输出样本数。"""
    if isinstance(d, dict):
        for k in ("generated_text", "generated_texts", "logits", "embeddings", "perplexity", "predictions", "diffusion_outputs", "forecasts", "answers"):
            v = d.get(k)
            if isinstance(v, list):
                return len(v)
    if isinstance(d, list):
        return len(d)
    return 1


def compare_outputs(cuda_pt_path: str, ascend_pt_path: str) -> dict:
    """加载两个 .pt 文件并自动检测类型进行对比。"""
    cuda_data = torch.load(cuda_pt_path, map_location="cpu", weights_only=False)
    ascend_data = torch.load(ascend_pt_path, map_location="cpu", weights_only=False)

    cuda_type = detect_output_type(cuda_data)
    ascend_type = detect_output_type(ascend_data)

    if cuda_type != ascend_type:
        # 尝试跨类型对比：若双方均为 dict 且能提取张量，则按 logits 对比
        if isinstance(cuda_data, dict) and isinstance(ascend_data, dict):
            cuda_t = _extract_tensor_from_dict(cuda_data)
            ascend_t = _extract_tensor_from_dict(ascend_data)
            if cuda_t is not None and ascend_t is not None and cuda_t.shape == ascend_t.shape:
                result = compare_logits_legacy(cuda_t, ascend_t)
                result["output_type"] = "logits"
                result["cuda_pt"] = _portable_output_path(cuda_pt_path)
                result["ascend_pt"] = _portable_output_path(ascend_pt_path)
                result["cuda_samples"] = _sample_count(cuda_data)
                result["ascend_samples"] = _sample_count(ascend_data)
                return result
        raise ValueError(f"输出类型不匹配: cuda={cuda_type} vs ascend={ascend_type}")

    output_type = cuda_type

    cuda_samples = _sample_count(cuda_data)
    ascend_samples = _sample_count(ascend_data)

    if output_type == "skip":
        raise ValueError("检测/扩散等复杂输出结构暂不支持对比")
    if output_type == "generated_text":
        result = compare_text_outputs(cuda_data, ascend_data)
    elif output_type == "class_labels":
        result = compare_class_labels(cuda_data, ascend_data)
    elif output_type == "cls_embeddings":
        if isinstance(cuda_data, dict):
            result = compare_embeddings_dict(cuda_data, ascend_data)
        else:
            result = compare_cls_embeddings(cuda_data, ascend_data)
    elif output_type == "generated_text_ppl":
        result = compare_generated_text_ppl(cuda_data, ascend_data)
    elif output_type == "qa_logits":
        result = compare_qa_logits(cuda_data, ascend_data)
    elif output_type == "generated_text_list":
        result = compare_generated_text_list(cuda_data, ascend_data)
    elif output_type == "answers_text":
        result = compare_answers_text(cuda_data, ascend_data)
    elif output_type == "detection_boxes":
        result = compare_detection_boxes(cuda_data, ascend_data)
    elif output_type == "diffusion_stats":
        result = compare_diffusion_stats(cuda_data, ascend_data)
    elif output_type == "logits":
        result = compare_logits_legacy(cuda_data, ascend_data)
    elif output_type == "mixed":
        result = compare_mixed_outputs(cuda_data, ascend_data)
    else:
        raise ValueError(f"未知的输出类型: {output_type}")

    result["output_type"] = output_type
    result["cuda_pt"] = _portable_output_path(cuda_pt_path)
    result["ascend_pt"] = _portable_output_path(ascend_pt_path)
    result["cuda_samples"] = cuda_samples
    result["ascend_samples"] = ascend_samples

    return result


def find_compare_pairs(adaptation_dir: Path) -> list[tuple[Path, Path]]:
    """在适配目录中查找 CUDA/NPU 输出对。"""
    pairs = []
    cuda_files = {}
    npu_files = {}

    for f in adaptation_dir.glob("outputs_*.pt"):
        parsed = BenchmarkFilename.parse(f.name)
        if not parsed:
            continue
        if parsed.device.startswith("cuda"):
            cuda_files[(parsed.precision, parsed.mode, parsed.dataset)] = f
        elif parsed.device.startswith("npu"):
            npu_files[(parsed.precision, parsed.mode, parsed.dataset)] = f

    for key in cuda_files:
        if key in npu_files:
            pairs.append((cuda_files[key], npu_files[key]))

    return pairs


def _compare_one_task(args: tuple) -> tuple:
    """多进程 worker：对比一对 CUDA/NPU 输出。返回 (result, adaptation, fname, error)。"""
    cuda_path, npu_path, adaptation_name = args
    try:
        result = compare_outputs(str(cuda_path), str(npu_path))
        result["_adaptation"] = adaptation_name
        return (result, adaptation_name, Path(cuda_path).name, None)
    except Exception as e:
        return (None, adaptation_name, Path(cuda_path).name, str(e))


def cmd_compare(args):
    """对比 CUDA/NPU 输出。"""
    print(f"{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}{BOLD}  Compare CUDA vs NPU Outputs{NC}")
    print(f"{BLUE}{'=' * 60}{NC}\n")

    results = []
    tasks: list[tuple[Path, Path, str]] = []

    if args.cuda_pt and args.ascend_pt:
        # 单文件对比（串行）
        cuda_path = Path(args.cuda_pt)
        ascend_path = Path(args.ascend_pt)
        if not cuda_path.exists():
            print(f"{RED}✗{NC} File not found: {cuda_path}")
            sys.exit(1)
        if not ascend_path.exists():
            print(f"{RED}✗{NC} File not found: {ascend_path}")
            sys.exit(1)

        try:
            result = compare_outputs(str(cuda_path), str(ascend_path))
            results.append(result)
        except Exception as e:
            print(f"{RED}✗{NC} Error: {e}")
            sys.exit(1)

    elif args.adaptation:
        adapt_dir = get_adaptation_dir(args.adaptation)
        if not adapt_dir.exists():
            print(f"{RED}✗{NC} Adaptation not found: {adapt_dir}")
            sys.exit(1)

        pairs = find_compare_pairs(adapt_dir)
        if not pairs:
            print(f"{YELLOW}⚠{NC} No CUDA/NPU pairs found in {adapt_dir}")
            return

        tasks = [(cuda_pt, npu_pt, adapt_dir.name) for cuda_pt, npu_pt in pairs]
        print(f"Found {len(tasks)} pairs to compare\n")

    elif args.all:
        if not ADAPTATIONS_DIR.exists():
            print(f"{YELLOW}⚠{NC} No adaptations directory")
            return

        for adir in sorted(ADAPTATIONS_DIR.iterdir()):
            if not adir.is_dir():
                continue
            pairs = find_compare_pairs(adir)
            for cuda_pt, npu_pt in pairs:
                tasks.append((cuda_pt, npu_pt, adir.name))

        if tasks:
            print(f"Found {len(tasks)} pairs to compare\n")

    else:
        print(f"{RED}✗{NC} 请指定 --cuda-pt 和 --ascend-pt，或使用 --adaptation / --all")
        sys.exit(1)

    # 多进程执行 tasks
    if tasks:
        n_jobs = args.jobs if args.jobs > 0 else 64
        use_parallel = n_jobs > 1 and len(tasks) > 1

        if use_parallel:
            print(f"Using {n_jobs} workers\n")
            with Pool(processes=n_jobs) as pool:
                for result, adapt_name, fname, err in pool.map(_compare_one_task, tasks):
                    if err is None:
                        results.append(result)
                        print(f"{GREEN}✓{NC} {adapt_name}/{fname}")
                    else:
                        print(f"{RED}✗{NC} {adapt_name}/{fname}: {err}")
        else:
            for cuda_pt, npu_pt, adapt_name in tasks:
                try:
                    result = compare_outputs(str(cuda_pt), str(npu_pt))
                    result["_adaptation"] = adapt_name
                    results.append(result)
                    print(f"{GREEN}✓{NC} {adapt_name}/{cuda_pt.name}")
                except Exception as e:
                    print(f"{RED}✗{NC} {adapt_name}/{cuda_pt.name}: {e}")

    # 输出结果
    if args.format == "table":
        print(f"\n{CYAN}{'-' * 120}{NC}")
        print(f"{BOLD}{'Adaptation':<35} {'Type':<10} {'Text Match':<10} {'Logits Cos':<12} {'PPL Diff':<10} {'Max Error'}{NC}")
        print(f"{CYAN}{'-' * 120}{NC}")

        for r in results:
            adapt = r.get("_adaptation", Path(r.get("cuda_pt", "")).parent.name)
            otype = r.get("output_type", "unknown")

            if otype == "mixed":
                # 新格式：同时显示文本匹配率、logits 余弦相似度和 PPL 差异
                text_match = f"{r.get('text_match_rate', 0):.1%}"
                logits_cos = f"{r.get('logits_avg_cosine_similarity', 0):.6f}"
                ppl_diff = r.get("ppl_avg_rel_diff", None)
                ppl_str = f"{ppl_diff:.2%}" if ppl_diff is not None else "-"
                err = f"{r.get('logits_max_abs_error', 0):.6f}"
            elif otype == "cls_embeddings":
                text_match = "-"
                logits_cos = f"{r.get('avg_cosine_similarity', 0):.6f}"
                ppl_str = "-"
                err = f"{r.get('max_abs_error', 0):.6f}"
            elif otype == "generated_text":
                text_match = f"{r.get('match_rate', 0):.1%}"
                logits_cos = "-"
                ppl_str = "-"
                err = "-"
            elif otype == "class_labels":
                text_match = f"{r.get('match_rate', 0):.1%}"
                logits_cos = "-"
                ppl_str = "-"
                err = "-"
            elif otype in ("generated_text_ppl", "generated_text_list", "answers_text"):
                text_match = f"{r.get('text_match_rate', r.get('match_rate', 0)):.1%}"
                logits_cos = "-"
                ppl_diff = r.get("ppl_avg_rel_diff")
                ppl_str = f"{ppl_diff:.2%}" if ppl_diff is not None else "-"
                err = "-"
            elif otype == "qa_logits":
                text_match = "-"
                logits_cos = f"{r.get('cosine_similarity', 0):.6f}"
                ppl_str = "-"
                err = f"{r.get('max_abs_error', 0):.6f}"
            elif otype == "detection_boxes":
                text_match = "-"
                logits_cos = f"{r.get('avg_iou', 0):.4f}"
                ppl_str = f"{r.get('avg_count_diff', 0):.1f}"
                err = f"IoU"
            elif otype == "diffusion_stats":
                text_match = "-"
                logits_cos = "-"
                ppl_str = f"{r.get('avg_mean_abs_error', 0):.6f}"
                err = f"{r.get('avg_std_abs_error', 0):.6f}"
            else:
                text_match = "-"
                logits_cos = f"{r.get('cosine_similarity', 0):.6f}"
                ppl_str = "-"
                err = f"{r.get('max_abs_error', 0):.6f}"

            print(f"{adapt:<35} {otype:<10} {text_match:<10} {logits_cos:<12} {ppl_str:<10} {err}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{GREEN}✓{NC} Report written to {args.output}")

    print()


# =============================================================================
# Raw Profiling 功能（解析 torch_npu.profiler 原始目录）
# =============================================================================

_PROFILING_CORE_FILES = ("api_statistic.csv", "step_trace_time.csv")
_PROFILING_AUX_FILES = ("op_statistic.csv", "kernel_details.csv", "trace_view.json")
_PROFILING_DB_FILES = ("cluster.db", "summary.db", "analysis.db", "ascend_pytorch_profiler.db")


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {_normalize_key(key): value for key, value in row.items() if key}


def _pick_value(row: dict[str, Any], *aliases: str) -> Any:
    normalized = _normalize_row(row)
    for alias in aliases:
        value = normalized.get(_normalize_key(alias))
        if value not in (None, ""):
            return value
    return None


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        pass
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def _to_int(value: Any) -> Optional[int]:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _duration_to_ms(row: dict[str, Any], alias_groups: list[tuple[list[str], float]]) -> Optional[float]:
    for aliases, factor in alias_groups:
        value = _pick_value(row, *aliases)
        parsed = _to_float(value)
        if parsed is not None:
            return round(parsed * factor, 4)
    return None


def _duration_from_count(total_ms: Optional[float], count: Optional[int]) -> Optional[float]:
    if total_ms is None or not count:
        return None
    return round(total_ms / count, 4)


def _summarize_named_rows(
    rows: list[dict[str, Any]],
    *,
    name_aliases: list[str],
    count_aliases: list[str],
    total_time_aliases: list[tuple[list[str], float]],
    avg_time_aliases: list[tuple[list[str], float]],
    top_k: int,
) -> dict[str, Any]:
    items = []
    total_calls = 0

    for row in rows:
        name = _pick_value(row, *name_aliases)
        if not name:
            continue
        count = _to_int(_pick_value(row, *count_aliases)) or 0
        total_ms = _duration_to_ms(row, total_time_aliases)
        avg_ms = _duration_to_ms(row, avg_time_aliases)
        if avg_ms is None:
            avg_ms = _duration_from_count(total_ms, count)
        items.append(
            {
                "name": str(name),
                "count": count,
                "total_time_ms": total_ms,
                "avg_time_ms": avg_ms,
            }
        )
        total_calls += count

    items.sort(key=lambda item: item.get("total_time_ms") or 0.0, reverse=True)
    return {
        "row_count": len(items),
        "total_api_calls": total_calls,
        "top_by_total_time": items[:top_k],
    }


def _summarize_step_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_time_ms = 0.0
    compute_time_ms = 0.0
    communication_time_ms = 0.0
    seen_total = False
    seen_compute = False
    seen_communication = False

    for row in rows:
        row_total = _duration_to_ms(
            row,
            [
                (["iteration time(us)", "iteration time", "step time(us)", "step time", "total time(us)", "total time", "stage(us)", "stage"], 0.001),
                (["iteration time(ms)", "step time(ms)", "total time(ms)", "stage(ms)"], 1.0),
            ],
        )
        row_compute = _duration_to_ms(
            row,
            [
                (["compute time(us)", "compute(us)", "computing time(us)", "computing(us)", "computing"], 0.001),
                (["compute time(ms)", "compute(ms)", "computing time(ms)", "computing(ms)"], 1.0),
            ],
        )
        row_communication = _duration_to_ms(
            row,
            [
                (["communication time(us)", "communication(us)", "comm time(us)", "communication(not overlapped)(us)", "communication(not overlapped)"], 0.001),
                (["communication time(ms)", "communication(ms)", "comm time(ms)", "communication(not overlapped)(ms)"], 1.0),
            ],
        )
        if row_total is not None:
            total_time_ms += row_total
            seen_total = True
        if row_compute is not None:
            compute_time_ms += row_compute
            seen_compute = True
        if row_communication is not None:
            communication_time_ms += row_communication
            seen_communication = True

    step_count = len(rows)
    summary: dict[str, Any] = {"step_count": step_count}
    if seen_total:
        summary["total_time_ms"] = round(total_time_ms, 4)
        summary["avg_time_ms"] = round(total_time_ms / step_count, 4) if step_count else None
    if seen_compute:
        summary["compute_time_ms"] = round(compute_time_ms, 4)
    if seen_communication:
        summary["communication_time_ms"] = round(communication_time_ms, 4)
    return summary


def _find_cluster_db(profiling_dir: Path) -> list[Path]:
    """Discover all profiling DB files in a profiling directory.

    Searches the directory itself and up to 3 levels of subdirectories.
    Returns all found DB paths (sorted by priority), or empty list.
    """
    found = []
    for name in _PROFILING_DB_FILES:
        # Direct child
        candidate = profiling_dir / name
        if candidate.is_file():
            found.append(candidate)
            continue
        # Recursive search up to 3 levels deep
        for match in profiling_dir.rglob(name):
            try:
                rel = match.relative_to(profiling_dir)
                if len(rel.parts) <= 3:
                    found.append(match)
            except ValueError:
                pass
    return found


def _parse_cluster_db(db_path: Path, top_k: int = 10) -> dict[str, Any]:
    """Parse cluster.db / summary.db to extract NPU operator statistics.

    The DB schema varies across CANN versions.  This function probes common
    table/column names so it works with multiple versions without hardcoding.

    Returns a dict with:
        db_path: str
        tables: list[str] — tables found in the DB (for debugging)
        npu_op_summary: {row_count, total_calls, top_by_total_time}
        warnings: list[str]
    """
    result: dict[str, Any] = {
        "db_path": str(db_path),
        "tables": [],
        "npu_op_summary": {"row_count": 0, "total_calls": 0, "top_by_total_time": []},
        "warnings": [],
    }

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. Discover all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row["name"] for row in cursor.fetchall()]
        result["tables"] = tables

        if not tables:
            result["warnings"].append(f"{db_path.name} contains no tables")
            conn.close()
            return result

        # 2. Detect DB schema version and choose query strategy
        #
        # CANN 8.x (ascend_pytorch_profiler.db / analysis.db):
        #   Tables: PYTORCH_API, CANN_API, TASK, STRING_IDS
        #   Names are INTEGER refs into STRING_IDS (id -> value)
        #   Timing: startNs, endNs (nanoseconds)
        #
        # CANN 7.x / older (cluster.db / summary.db):
        #   Tables: NpuOperator, npu_op_summary, cluster, Operator
        #   Names are direct TEXT columns
        #   Timing: total_duration, duration (mixed us/ms)

        has_string_ids = "STRING_IDS" in tables
        has_pytorch_api = "PYTORCH_API" in tables
        has_cann_api = "CANN_API" in tables

        items = []
        all_calls = 0
        source_table = None

        if has_string_ids and has_pytorch_api:
            # --- CANN 8.x schema: PYTORCH_API + STRING_IDS ---
            source_table = "PYTORCH_API"
            query = f"""
                SELECT s.value AS op_name, COUNT(*) AS total_calls,
                       SUM(p.endNs - p.startNs) AS total_ns
                FROM PYTORCH_API p
                JOIN STRING_IDS s ON p.name = s.id
                WHERE s.value IS NOT NULL AND s.value != ''
                GROUP BY p.name
                ORDER BY total_ns DESC
                LIMIT ?
            """
            cursor.execute(query, (top_k,))
            for row in cursor.fetchall():
                total_ms = (row["total_ns"] or 0) / 1e6
                calls = row["total_calls"] or 0
                items.append(
                    {
                        "name": str(row["op_name"]),
                        "count": calls,
                        "total_time_ms": round(total_ms, 4),
                        "avg_time_ms": round(total_ms / calls, 4) if calls > 0 else 0.0,
                    }
                )
                all_calls += calls

            # Also extract CANN_API (aclnn* operators) if available
            if has_cann_api:
                cann_query = f"""
                    SELECT s.value AS op_name, COUNT(*) AS total_calls,
                           SUM(c.endNs - c.startNs) AS total_ns
                    FROM CANN_API c
                    JOIN STRING_IDS s ON c.name = s.id
                    WHERE s.value IS NOT NULL AND s.value != ''
                    GROUP BY c.name
                    ORDER BY total_ns DESC
                    LIMIT ?
                """
                cursor.execute(cann_query, (top_k,))
                cann_items = []
                cann_calls = 0
                for row in cursor.fetchall():
                    total_ms = (row["total_ns"] or 0) / 1e6
                    calls = row["total_calls"] or 0
                    cann_items.append(
                        {
                            "name": str(row["op_name"]),
                            "count": calls,
                            "total_time_ms": round(total_ms, 4),
                            "avg_time_ms": round(total_ms / calls, 4) if calls > 0 else 0.0,
                        }
                    )
                    cann_calls += calls

                if cann_items:
                    result["cann_api_summary"] = {
                        "row_count": len(cann_items),
                        "total_calls": cann_calls,
                        "top_by_total_time": cann_items,
                        "source_table": "CANN_API",
                    }

        else:
            # --- Legacy schema: direct name/duration columns ---
            op_table_candidates = [
                "NpuOperator",
                "npu_op_summary",
                "op_summary",
                "NpuOperation",
                "Operator",
                "operator",
                "cluster",
            ]
            op_table = None
            for candidate in op_table_candidates:
                if candidate in tables:
                    op_table = candidate
                    break

            if op_table is None:
                # Fallback: try any table that has 'duration' or 'time' columns
                for tbl in tables:
                    cursor.execute(f'PRAGMA table_info("{tbl}")')
                    columns = [row["name"] for row in cursor.fetchall()]
                    col_lower = {c.lower() for c in columns}
                    if "duration" in col_lower or "total_time" in col_lower or "execution_time" in col_lower:
                        op_table = tbl
                        break

            if op_table is None:
                result["warnings"].append(f"Could not find operator table in {db_path.name}. Tables: {', '.join(tables)}")
                conn.close()
                return result

            source_table = op_table
            cursor.execute(f'PRAGMA table_info("{op_table}")')
            columns = [row["name"] for row in cursor.fetchall()]
            col_map = {c.lower(): c for c in columns}

            name_col = col_map.get("name") or col_map.get("op_name") or col_map.get("operator_name") or col_map.get("type")
            dur_col = col_map.get("total_duration") or col_map.get("duration") or col_map.get("total_time") or col_map.get("execution_time") or col_map.get("task_duration")
            count_col = col_map.get("count") or col_map.get("calls") or col_map.get("call_count") or col_map.get("execution_count")

            if not name_col or not dur_col:
                result["warnings"].append(f"Table '{op_table}' lacks name/duration columns. Columns: {', '.join(columns)}")
                conn.close()
                return result

            select_parts = [f'"{name_col}" AS op_name', f'SUM("{dur_col}") AS total_dur']
            group_parts = [f'"{name_col}"']
            if count_col:
                select_parts.append(f'SUM("{count_col}") AS total_calls')
                group_parts.append(f'"{count_col}"')

            query = f'SELECT {", ".join(select_parts)} FROM "{op_table}" GROUP BY {", ".join(group_parts)} ORDER BY total_dur DESC LIMIT ?'
            cursor.execute(query, (top_k,))
            for row in cursor.fetchall():
                item: dict[str, Any] = {"name": str(row["op_name"]) if row["op_name"] else "unknown"}
                total_dur_raw = row["total_dur"] if "total_dur" in row.keys() else 0
                if total_dur_raw > 0:
                    total_ms = total_dur_raw / 1000.0 if total_dur_raw > 1_000_000 else total_dur_raw
                else:
                    total_ms = 0.0
                item["total_time_ms"] = round(total_ms, 4)
                if "total_calls" in row.keys() and row["total_calls"] is not None:
                    calls = int(row["total_calls"])
                    item["count"] = calls
                    all_calls += calls
                    if calls > 0:
                        item["avg_time_ms"] = round(total_ms / calls, 4)
                items.append(item)

        result["npu_op_summary"] = {
            "row_count": len(items),
            "total_calls": all_calls,
            "top_by_total_time": items,
            "source_table": source_table,
        }

        conn.close()

    except sqlite3.Error as e:
        result["warnings"].append(f"SQLite error: {e}")
    except Exception as e:
        result["warnings"].append(f"Parse error: {e}")

    return result


def _empty_profiling_summary(profiling_dir: Path, warnings: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "source_dir": str(profiling_dir),
        "detected_files": [],
        "missing_inputs": list(_PROFILING_CORE_FILES),
        "step_summary": {"step_count": 0},
        "api_summary": {"row_count": 0, "total_api_calls": 0, "top_by_total_time": []},
        "op_summary": {"row_count": 0, "total_api_calls": 0, "top_by_total_time": []},
        "kernel_summary": {"row_count": 0, "top_by_total_time": []},
        "warnings": warnings or [],
    }


def analyze_profiling_dir(profiling_dir: Path, top_k: int = 10, deep: bool = False) -> dict[str, Any]:
    profiling_dir = Path(profiling_dir)
    if not profiling_dir.exists():
        raise FileNotFoundError(profiling_dir)
    if not profiling_dir.is_dir():
        raise ValueError(f"Not a directory: {profiling_dir}")

    warnings: list[str] = []
    detected_files = [name for name in (*_PROFILING_CORE_FILES, *_PROFILING_AUX_FILES) if (profiling_dir / name).exists()]
    # Also check for DB files (only noted, not counted as core)
    db_files = _find_cluster_db(profiling_dir)
    for dbf in db_files:
        detected_files.append(f"{dbf.relative_to(profiling_dir)}")

    missing_inputs = [name for name in _PROFILING_CORE_FILES if not (profiling_dir / name).exists()]
    if not detected_files:
        return _empty_profiling_summary(profiling_dir, warnings=["no recognized profiling files found"])

    api_summary = {"row_count": 0, "total_api_calls": 0, "top_by_total_time": []}
    step_summary: dict[str, Any] = {"step_count": 0}
    op_summary = {"row_count": 0, "total_api_calls": 0, "top_by_total_time": []}
    kernel_summary = {"row_count": 0, "top_by_total_time": []}

    api_path = profiling_dir / "api_statistic.csv"
    if api_path.exists():
        api_rows = _read_csv_rows(api_path)
        api_summary = _summarize_named_rows(
            api_rows,
            name_aliases=["api name", "name", "api"],
            count_aliases=["count", "calls", "call count"],
            total_time_aliases=[
                (["total time(us)", "total(us)", "total duration(us)", "total_time_us", "time(us)"], 0.001),
                (["total time(ms)", "total(ms)", "total_time_ms", "time(ms)"], 1.0),
            ],
            avg_time_aliases=[
                (["avg time(us)", "average time(us)", "avg(us)", "avg_time_us"], 0.001),
                (["avg time(ms)", "average time(ms)", "avg(ms)", "avg_time_ms"], 1.0),
            ],
            top_k=top_k,
        )
    else:
        warnings.append("api_statistic.csv missing")

    step_path = profiling_dir / "step_trace_time.csv"
    if step_path.exists():
        step_rows = _read_csv_rows(step_path)
        step_summary = _summarize_step_trace(step_rows)
    else:
        warnings.append("step_trace_time.csv missing")

    op_path = profiling_dir / "op_statistic.csv"
    if op_path.exists():
        op_rows = _read_csv_rows(op_path)
        op_summary = _summarize_named_rows(
            op_rows,
            name_aliases=["op name", "operator name", "operator", "name", "op type"],
            count_aliases=["count", "calls", "call count"],
            total_time_aliases=[
                (["total time(us)", "total(us)", "total duration(us)", "total_time_us", "time(us)"], 0.001),
                (["total time(ms)", "total(ms)", "total_time_ms", "time(ms)"], 1.0),
            ],
            avg_time_aliases=[
                (["avg time(us)", "average time(us)", "avg(us)", "avg_time_us", "avg time(us)"], 0.001),
                (["avg time(ms)", "average time(ms)", "avg(ms)", "avg_time_ms", "avg time(ms)"], 1.0),
            ],
            top_k=top_k,
        )

    kernel_path = profiling_dir / "kernel_details.csv"
    if kernel_path.exists():
        kernel_rows = _read_csv_rows(kernel_path)
        kernel_items = []
        for row in kernel_rows:
            name = _pick_value(row, "kernel name", "name", "op name", "operator name")
            if not name:
                continue
            total_ms = _duration_to_ms(
                row,
                [
                    (["duration(us)", "total time(us)", "duration", "task duration(us)"], 0.001),
                    (["duration(ms)", "total time(ms)", "task duration(ms)"], 1.0),
                ],
            )
            kernel_items.append(
                {
                    "name": str(name),
                    "step": _pick_value(row, "step id", "step", "iteration"),
                    "input_shape": _pick_value(row, "input shapes", "input shape", "shape"),
                    "total_time_ms": total_ms,
                }
            )
        kernel_items.sort(key=lambda item: item.get("total_time_ms") or 0.0, reverse=True)
        kernel_summary = {"row_count": len(kernel_items), "top_by_total_time": kernel_items[:top_k]}

    # Deep analysis: parse all DB files for NPU operator stats
    cluster_db_summary: dict[str, Any] = {}
    if deep and db_files:
        # Try all found DBs, pick the one with the most useful data
        for dbf in db_files:
            parsed = _parse_cluster_db(dbf, top_k=top_k)
            npu = parsed.get("npu_op_summary", {})
            # Prefer DBs that actually have operator data
            if npu.get("row_count", 0) > 0:
                cluster_db_summary = parsed
                break
            # Fallback: use last DB if none have operators
            cluster_db_summary = parsed
        warnings.extend(cluster_db_summary.get("warnings", []))

    return {
        "source_dir": str(profiling_dir),
        "detected_files": detected_files,
        "missing_inputs": missing_inputs,
        "step_summary": step_summary,
        "api_summary": api_summary,
        "op_summary": op_summary,
        "kernel_summary": kernel_summary,
        "cluster_db": cluster_db_summary,
        "warnings": warnings,
    }


def find_profiling_dirs(adaptation_dir: Path) -> list[Path]:
    """Discover profiling data directories within an adaptation directory.

    Searches for:
    1. Direct Profiling_* directories (legacy)
    2. profiling/*/ directories (new convention)
    3. Nested ASCEND_PROFILER_OUTPUT directories (CANN 8.x actual output location)

    Returns the innermost directories that contain profiling CSV files.
    """
    candidates: set[Path] = set()
    adaptation_dir = Path(adaptation_dir)

    # Pattern 1: Profiling_* at top level
    for path in adaptation_dir.glob("Profiling_*"):
        if path.is_dir():
            candidates.add(path)

    # Pattern 2: profiling/*/ at top level
    profiling_root = adaptation_dir / "profiling"
    if profiling_root.is_dir():
        for path in profiling_root.iterdir():
            if path.is_dir():
                candidates.add(path)

    # Pattern 3: Recursively find ASCEND_PROFILER_OUTPUT dirs (CANN 8.x actual data)
    for apo in adaptation_dir.rglob("ASCEND_PROFILER_OUTPUT"):
        if apo.is_dir():
            candidates.add(apo)

    # Deduplicate: if a candidate is a parent of another, keep only the leaf
    # BUT: also check if the leaf actually has profiling data (CSV files).
    # Prefer directories that contain actual CSV files.
    leaves: set[Path] = set()
    sorted_cands = sorted(candidates, key=str)
    for c in sorted_cands:
        is_parent_of_leaf = any(str(other).startswith(str(c) + "/") for other in sorted_cands if other != c)
        if not is_parent_of_leaf:
            leaves.add(c)

    # Among remaining, prefer dirs that actually have CSV files
    has_csv = [d for d in leaves if (d / "api_statistic.csv").is_file() or (d / "step_trace_time.csv").is_file()]
    no_csv = [d for d in leaves if d not in has_csv]

    # If no dir has CSVs, also return dirs that have DB files
    if not has_csv:
        has_db = [d for d in no_csv if (d / "analysis.db").is_file() or (d / "cluster.db").is_file()]
        return sorted(has_db + no_csv, key=lambda path: (0 if path.name.startswith("Profiling_") else 1, str(path)))

    return sorted(has_csv + no_csv, key=lambda path: (0 if path.name.startswith("Profiling_") else 1, str(path)))


def _enrich_profiling_result(result: dict[str, Any], profiling_dir: Path, adaptation_name: Optional[str] = None) -> None:
    result["_source_dir"] = str(profiling_dir)
    if adaptation_name is not None:
        result["_adaptation"] = adaptation_name


def print_profiling_report(result: dict[str, Any], top_k: int = 10) -> None:
    print("\n" + "=" * 60)
    print("Raw Profiling 分析报告")
    print("=" * 60)
    print(f"目录: {result.get('source_dir')}")
    print(f"识别文件: {', '.join(result.get('detected_files', [])) or '无'}")
    missing_inputs = result.get("missing_inputs", [])
    if missing_inputs:
        print(f"缺失核心文件: {', '.join(missing_inputs)}")

    step_summary = result.get("step_summary", {})
    print(f"Step 数: {step_summary.get('step_count', 0)}")
    if step_summary.get("total_time_ms") is not None:
        print(f"总耗时: {step_summary['total_time_ms']:.4f} ms")
    if step_summary.get("avg_time_ms") is not None:
        print(f"平均每步: {step_summary['avg_time_ms']:.4f} ms")

    api_summary = result.get("api_summary", {})
    print(f"API 总调用: {api_summary.get('total_api_calls', 0)}")
    top_api = api_summary.get("top_by_total_time", [])[:top_k]
    if top_api:
        print("\nTop APIs:")
        for item in top_api:
            total_ms = item.get("total_time_ms")
            total_str = f"{total_ms:.4f} ms" if total_ms is not None else "N/A"
            print(f"  - {item.get('name')}: count={item.get('count', 0)}, total={total_str}")

    top_ops = result.get("op_summary", {}).get("top_by_total_time", [])[:top_k]
    if top_ops:
        print("\nTop Operators:")
        for item in top_ops:
            total_ms = item.get("total_time_ms")
            total_str = f"{total_ms:.4f} ms" if total_ms is not None else "N/A"
            print(f"  - {item.get('name')}: count={item.get('count', 0)}, total={total_str}")

    top_kernels = result.get("kernel_summary", {}).get("top_by_total_time", [])[:top_k]
    if top_kernels:
        print("\nTop Kernels:")
        for item in top_kernels:
            total_ms = item.get("total_time_ms")
            total_str = f"{total_ms:.4f} ms" if total_ms is not None else "N/A"
            print(f"  - {item.get('name')}: total={total_str}")

    # cluster.db (deep) analysis
    cluster_db = result.get("cluster_db", {})
    if cluster_db and cluster_db.get("npu_op_summary", {}).get("row_count", 0) > 0:
        db_path = cluster_db.get("db_path", "unknown")
        source_table = cluster_db.get("npu_op_summary", {}).get("source_table", "unknown")
        print(f"\n── cluster.db Deep Analysis (table: {source_table}) ──")
        print(f"  DB: {db_path}")
        npu_ops = cluster_db["npu_op_summary"]
        print(f"  Operator types: {npu_ops['row_count']}, total calls: {npu_ops['total_calls']}")
        top_npu = npu_ops.get("top_by_total_time", [])[:top_k]
        if top_npu:
            print("\n  Top NPU Operators (from cluster.db):")
            for item in top_npu:
                total_ms = item.get("total_time_ms", 0)
                avg_ms = item.get("avg_time_ms")
                avg_str = f", avg={avg_ms:.4f} ms" if avg_ms is not None else ""
                print(f"    - {item['name']}: total={total_ms:.4f} ms, count={item.get('count', '?')}{avg_str}")

        # CANN_API summary (aclnn* operators from CANN 8.x)
        cann_summary = cluster_db.get("cann_api_summary", {})
        if cann_summary and cann_summary.get("row_count", 0) > 0:
            top_cann = cann_summary.get("top_by_total_time", [])[:top_k]
            print(f"\n  Top CANN API Operators (aclnn*):")
            for item in top_cann:
                total_ms = item.get("total_time_ms", 0)
                avg_ms = item.get("avg_time_ms")
                avg_str = f", avg={avg_ms:.4f} ms" if avg_ms is not None else ""
                print(f"    - {item['name']}: total={total_ms:.4f} ms, count={item.get('count', '?')}{avg_str}")

    warnings = result.get("warnings", [])
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("=" * 60)


def cmd_profiling(args):
    """分析 torch_npu.profiler 原始目录。"""
    if not args.json:
        print(f"{BLUE}{'=' * 60}{NC}")
        print(f"{BLUE}{BOLD}  Raw Profiling Analysis{NC}")
        print(f"{BLUE}{'=' * 60}{NC}\n")

    results = []
    errors = []

    def collect_one(profiling_dir: Path, adaptation_name: Optional[str] = None) -> None:
        result = analyze_profiling_dir(profiling_dir, top_k=args.top_k, deep=args.deep)
        _enrich_profiling_result(result, profiling_dir, adaptation_name)
        results.append(result)

    if args.profiling_dir:
        profiling_dir = Path(args.profiling_dir)
        if not profiling_dir.exists():
            print(f"{RED}✗{NC} File not found: {profiling_dir}")
            sys.exit(1)
        try:
            collect_one(profiling_dir)
        except Exception as e:
            print(f"{RED}✗{NC} Error: {e}")
            sys.exit(1)
        if args.json:
            out = results[0]
            if args.output:
                Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                print(json.dumps(out, indent=2, ensure_ascii=False))
            return
        print_profiling_report(results[0], top_k=args.top_k)
        if args.output:
            Path(args.output).write_text(json.dumps(results[0], indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\n{GREEN}✓{NC} Report written to {args.output}")
        print()
        return

    if args.adaptation:
        adapt_dir = get_adaptation_dir(args.adaptation)
        if not adapt_dir.exists():
            print(f"{RED}✗{NC} Adaptation not found: {adapt_dir}")
            sys.exit(1)
        profiling_dirs = find_profiling_dirs(adapt_dir)
        if not profiling_dirs:
            print(f"{YELLOW}⚠{NC} No profiling directories found in {adapt_dir}")
            return
        for profiling_dir in profiling_dirs:
            try:
                collect_one(profiling_dir, adapt_dir.name)
                if not args.json:
                    print(f"{GREEN}✓{NC} {profiling_dir.relative_to(adapt_dir)}")
            except Exception as e:
                errors.append((profiling_dir, str(e)))
                if not args.json:
                    print(f"{RED}✗{NC} {profiling_dir}: {e}")

    elif args.all:
        if not ADAPTATIONS_DIR.exists():
            print(f"{YELLOW}⚠{NC} No adaptations directory")
            return
        for adir in sorted(ADAPTATIONS_DIR.iterdir()):
            if not adir.is_dir():
                continue
            for profiling_dir in find_profiling_dirs(adir):
                try:
                    collect_one(profiling_dir, adir.name)
                    if not args.json:
                        print(f"{GREEN}✓{NC} {adir.name}/{profiling_dir.name}")
                except Exception as e:
                    errors.append((profiling_dir, str(e)))
                    print(f"{RED}✗{NC} {adir.name}/{profiling_dir.name}: {e}")

    else:
        print(f"{RED}✗{NC} 请指定 profiling 目录，或使用 --adaptation / --all")
        sys.exit(1)

    if not results:
        if errors:
            sys.exit(1)
        print()
        return

    if args.json:
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    for result in results:
        print_profiling_report(result, top_k=args.top_k)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{GREEN}✓{NC} Report written to {args.output}")
    print()


# =============================================================================
# Trace 功能（含 NPU Fallback 分析：D2H/H2D、算子分类）
# =============================================================================

# 关键计算算子、调度入口、轻量级操作（用于 fallback 分类）
_COMPUTE_OPS = {
    "matmul",
    "addmm",
    "bmm",
    "mm",
    "baddbmm",
    "mul",
    "add",
    "sub",
    "div",
    "neg",
    "pow",
    "embedding",
    "gather",
    "scatter",
    "index_select",
    "silu",
    "gelu",
    "relu",
    "tanh",
    "sigmoid",
    "softmax",
    "layernorm",
    "rmsnorm",
    "groupnorm",
    "batchnorm",
    "rsqrt",
    "sqrt",
    "sin",
    "cos",
    "exp",
    "log",
    "mean",
    "sum",
    "max",
    "min",
    "cat",
    "flashattention",
    "scaled_dot_product_attention",
}
_DISPATCH_OPS = {
    "linear",
    "conv2d",
    "conv1d",
    "layer_norm",
    "group_norm",
    "scaled_dot_product_attention",
}
# aten 算子名到规范名的映射（解决 native_batch_norm、_convolution 等解析问题）
_ATEN_OP_ALIASES = {
    "native_batch_norm": "batchnorm",
    "native_layer_norm": "layernorm",
    "batch_norm": "batchnorm",
    "layer_norm": "layernorm",
}
# NPU 算子命名可能与 aten 不同，用于 has_npu_impl 判断（子串匹配的备选）
_NPU_OP_ALTERNATES = {
    "bmm": ["bmm", "batch_matmul", "batchmatmul"],
    "addmm": ["addmm", "add_matmul"],
}

_LIGHTWEIGHT_OPS = {
    "view",
    "reshape",
    "transpose",
    "permute",
    "squeeze",
    "unsqueeze",
    "slice",
    "select",
    "narrow",
    "expand",
    "repeat",
    "to",
    "contiguous",
    "clone",
    "detach",
    "copy_",
    "empty",
    "zeros",
    "ones",
    "full",
    "arange",
    "item",
    "is_nonzero",
    "alias",
    "lift",
}


def analyze_trace_for_fallback(trace_path: Path, verbose: bool = False, quiet: bool = False) -> dict:
    """分析 trace_*.json 检测 NPU fallback 到 CPU 的情况（D2H/H2D、算子分类）。"""
    if not quiet:
        print(f"[analyze] 加载 Trace 文件: {trace_path}")
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    if not isinstance(events, list):
        return {
            "npu_ops": 0,
            "cpu_ops": 0,
            "fallback_ratio": 0.0,
            "fallback_invocations": 0,
            "top_cpu_ops": [],
            "total_events": 0,
            "cpu_op_types": 0,
            "npu_op_types": 0,
            "d2h_count": 0,
            "h2d_count": 0,
            "fallback_ops": [],
            "compute_on_npu": [],
            "dispatch_on_cpu": [],
            "lightweight_on_cpu": [],
            "has_fallback": False,
            "fallback_applicable": False,
            "has_data_transfer": False,
            "note": "trace format not recognized",
        }
    if not quiet:
        print(f"[analyze] 共 {len(events)} 个事件")

    copy_events = {"d2h": [], "h2d": []}
    for event in events:
        name = event.get("name", "").lower()
        if any(kw in name for kw in ["npu_to_cpu", "d2h", "tocpu", "to.device"]) and "cpu" in name:
            copy_events["d2h"].append(event)
        if any(kw in name for kw in ["cpu_to_npu", "h2d", "tonpu", "to.device"]) and "npu" in name:
            copy_events["h2d"].append(event)

    cpu_ops_counter: Counter = Counter()
    npu_ops_counter: Counter = Counter()
    for event in events:
        name = event.get("name", "")
        cat = event.get("cat", "")
        if cat == "cpu_op":
            if name.startswith("aten::"):
                raw = name.replace("aten::", "").lstrip("_")
                if raw:
                    op_name = _ATEN_OP_ALIASES.get(raw, raw)
                    cpu_ops_counter[op_name] += 1
            elif name.startswith("aclnn") or name.startswith("npu::"):
                npu_ops_counter[name] += 1
        elif name.startswith("aclnn") or name.startswith("npu::"):
            npu_ops_counter[name] += 1

    fallback_ops = []
    compute_on_npu = []
    dispatch_on_cpu = []
    lightweight_on_cpu = []
    for op, count in cpu_ops_counter.items():
        op_lower = op.lower()
        alternates = _NPU_OP_ALTERNATES.get(op_lower, [op_lower])
        has_npu_impl = any(any(alt in npu_op.lower() for alt in alternates) for npu_op in npu_ops_counter)
        if op_lower in _DISPATCH_OPS:
            dispatch_on_cpu.append(op)
        elif op_lower in _COMPUTE_OPS and not has_npu_impl:
            fallback_ops.append(op)
        elif op_lower in _COMPUTE_OPS:
            compute_on_npu.append(op)
        elif op_lower in _LIGHTWEIGHT_OPS:
            lightweight_on_cpu.append(op)

    if verbose and not quiet and copy_events["d2h"]:
        print(f"\n[analyze] 发现 {len(copy_events['d2h'])} 个 D2H 事件 (可能指示 Fallback)")

    # 仅 NPU trace 能判断「算子在该设备上不支持而退回到 CPU」；CUDA trace 格式不区分 GPU 算子，不分析
    parsed = BenchmarkFilename.parse(trace_path.name)
    is_npu_trace = parsed is not None and str(parsed.device).startswith("npu")
    if not is_npu_trace:
        fallback_ops = []
    has_fallback = len(fallback_ops) > 0
    cpu_ops_total = sum(cpu_ops_counter.values())
    npu_ops_total = sum(npu_ops_counter.values())
    fallback_invocations = sum(cpu_ops_counter.get(op, 0) for op in fallback_ops) if is_npu_trace else 0
    compute_total = (fallback_invocations + npu_ops_total) if is_npu_trace else 0
    fallback_ratio = (fallback_invocations / compute_total) if compute_total else 0.0
    top_cpu_ops = sorted(cpu_ops_counter.items(), key=lambda x: -x[1])[:10]

    return {
        "total_events": len(events),
        "cpu_op_types": len(cpu_ops_counter),
        "npu_op_types": len(npu_ops_counter),
        "d2h_count": len(copy_events["d2h"]),
        "h2d_count": len(copy_events["h2d"]),
        "fallback_ops": sorted(fallback_ops),
        "compute_on_npu": sorted(compute_on_npu),
        "dispatch_on_cpu": sorted(dispatch_on_cpu),
        "lightweight_on_cpu": sorted(lightweight_on_cpu),
        "has_fallback": has_fallback,
        "fallback_applicable": is_npu_trace,
        "has_data_transfer": len(copy_events["d2h"]) > 0 or len(copy_events["h2d"]) > 0,
        "npu_ops": npu_ops_total,
        "cpu_ops": cpu_ops_total,
        "fallback_ratio": round(fallback_ratio, 4),
        "fallback_invocations": fallback_invocations if is_npu_trace else 0,
        "top_cpu_ops": top_cpu_ops,
    }


def print_fallback_report(result: dict, verbose: bool = False) -> None:
    """打印 NPU Fallback 分析报告（人类可读）。"""
    print("\n" + "=" * 60)
    print("📊 NPU Fallback 分析报告")
    print("=" * 60)
    print(f"\n📈 总体统计:")
    print(f"  - 总事件数: {result.get('total_events', 0)}")
    print(f"  - CPU 算子类型: {result.get('cpu_op_types', 0)}")
    print(f"  - NPU 算子类型: {result.get('npu_op_types', 0)}")
    print(f"  - D2H 数据搬运: {result.get('d2h_count', 0)} 次")
    print(f"  - H2D 数据搬运: {result.get('h2d_count', 0)} 次")
    if result.get("has_fallback"):
        print(f"\n⚠️  检测到 Fallback (关键算子在 CPU 上执行):")
        for op in result.get("fallback_ops", []):
            print(f"  - {op}")
    else:
        print(f"\n✅ 未检测到关键算子 Fallback")
    if result.get("compute_on_npu"):
        print(f"\n🚀 在 NPU 上执行的计算算子 ({len(result.get('compute_on_npu', []))} 个):")
        for op in result.get("compute_on_npu", []):
            print(f"  - {op}")
    if result.get("dispatch_on_cpu"):
        print(f"\n📋 调度入口算子 (CPU 调度，NPU 计算):")
        for op in result.get("dispatch_on_cpu", []):
            print(f"  - {op} (调度入口，实际计算在子算子上)")
    if verbose and result.get("lightweight_on_cpu"):
        print(f"\n🔹 轻量级操作 (CPU 上正常执行):")
        lightweight = result.get("lightweight_on_cpu", [])
        for op in lightweight[:10]:
            print(f"  - {op}")
        if len(lightweight) > 10:
            print(f"  ... 共 {len(lightweight)} 个")
    print("\n" + "=" * 60)
    if result.get("has_fallback"):
        print("💡 建议:")
        print("  1. 检查 fallback 算子是否有 NPU 替代实现")
        print("  2. 考虑使用 torch_npu.extend_custom_op 注册自定义算子")
        print("  3. 使用 ASCEND_GLOBAL_LOG_LEVEL=1 查看详细 fallback 日志")
    elif result.get("d2h_count", 0) > 0:
        print("💡 提示:")
        print("  - 检测到数据搬运，但关键计算在 NPU 上")
        print("  - 这可能是正常的输入/输出处理")
    else:
        print("✅ 所有计算密集型算子都在 NPU 上执行，无明显性能问题")
    print("=" * 60)


def parse_trace_fallback(trace_path: str) -> dict:
    """解析 trace 文件，统计 NPU/CPU ops 和 fallback 比例（委托给 analyze_trace_for_fallback）。"""
    path = Path(trace_path)
    if not path.exists():
        raise FileNotFoundError(trace_path)
    return analyze_trace_for_fallback(path, verbose=False, quiet=True)


def _enrich_trace_result(result: dict, trace_path: Path, adaptation_name: Optional[str] = None) -> None:
    """给 result 写入 _file 与可选的 device/precision/mode/dataset、_adaptation。"""
    result["_file"] = str(trace_path)
    parsed = BenchmarkFilename.parse(trace_path.name)
    if parsed:
        result["device"] = parsed.device
        result["precision"] = parsed.precision
        result["mode"] = parsed.mode
        result["dataset"] = parsed.dataset
    if adaptation_name is not None:
        result["_adaptation"] = adaptation_name


def cmd_trace(args):
    """分析 trace 文件（含 NPU Fallback 详细分析）。"""
    if not args.json:
        print(f"{BLUE}{'=' * 60}{NC}")
        print(f"{BLUE}{BOLD}  Trace Analysis{NC}")
        print(f"{BLUE}{'=' * 60}{NC}\n")
    results = []

    if args.trace:
        trace_path = Path(args.trace)
        if not trace_path.exists():
            print(f"{RED}✗{NC} File not found: {trace_path}")
            sys.exit(1)
        try:
            result = analyze_trace_for_fallback(trace_path, verbose=args.verbose, quiet=args.json)
            _enrich_trace_result(result, trace_path)
            results.append(result)
        except Exception as e:
            print(f"{RED}✗{NC} Error: {e}")
            sys.exit(1)

        if args.json:
            out = results[0]
            if args.output:
                Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                print(json.dumps(out, indent=2, ensure_ascii=False))
            return
        if args.verbose:
            print_fallback_report(results[0], verbose=True)
        if args.top_ops and results[0].get("top_cpu_ops"):
            print(f"\n{CYAN}Top CPU ops for {trace_path.name}:{NC}")
            for op, count in results[0]["top_cpu_ops"]:
                print(f"  {op}: {count}")
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\n{GREEN}✓{NC} Report written to {args.output}")
        print()
        return

    if args.adaptation:
        adapt_dir = get_adaptation_dir(args.adaptation)
        if not adapt_dir.exists():
            print(f"{RED}✗{NC} Adaptation not found: {adapt_dir}")
            sys.exit(1)
        trace_files = sorted(adapt_dir.glob("trace_*.json"))
        if not trace_files:
            print(f"{YELLOW}⚠{NC} No trace files found in {adapt_dir}")
            return
        for tf in trace_files:
            try:
                result = parse_trace_fallback(str(tf))
                _enrich_trace_result(result, tf)
                results.append(result)
                _app = result.get("fallback_applicable", True)
                _fb = result.get("has_fallback", False)
                _ops = result.get("fallback_ops", [])
                if not _app:
                    print(f"{GREEN}✓{NC} {tf.name}: fallback=N/A（仅 NPU trace 可判断）")
                else:
                    print(f"{GREEN}✓{NC} {tf.name}: fallback={'有' if _fb else '无'}" + (f" ({', '.join(_ops)})" if _ops else ""))
                if args.verbose:
                    print_fallback_report(result, verbose=True)
            except Exception as e:
                print(f"{RED}✗{NC} {tf.name}: {e}")

    elif args.all:
        if not ADAPTATIONS_DIR.exists():
            print(f"{YELLOW}⚠{NC} No adaptations directory")
            return
        for adir in sorted(ADAPTATIONS_DIR.iterdir()):
            if not adir.is_dir():
                continue
            for tf in sorted(adir.glob("trace_*.json")):
                try:
                    result = parse_trace_fallback(str(tf))
                    _enrich_trace_result(result, tf, adir.name)
                    results.append(result)
                    _app = result.get("fallback_applicable", True)
                    _fb = result.get("has_fallback", False)
                    _ops = result.get("fallback_ops", [])
                    if not _app:
                        print(f"{GREEN}✓{NC} {adir.name}/{tf.name}: fallback=N/A（仅 NPU trace 可判断）")
                    else:
                        print(f"{GREEN}✓{NC} {adir.name}/{tf.name}: fallback={'有' if _fb else '无'}" + (f" ({', '.join(_ops)})" if _ops else ""))
                    if args.verbose:
                        print_fallback_report(result, verbose=True)
                except Exception as e:
                    print(f"{RED}✗{NC} {adir.name}/{tf.name}: {e}")

    else:
        print(f"{RED}✗{NC} 请指定 trace 文件，或使用 --adaptation / --all")
        sys.exit(1)

    if not results:
        print()
        return

    if args.json:
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    for r in results:
        if args.top_ops and r.get("top_cpu_ops"):
            print(f"\n{CYAN}Top CPU ops for {Path(r.get('_file', '')).name}:{NC}")
            for op, count in r["top_cpu_ops"]:
                print(f"  {op}: {count}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{GREEN}✓{NC} Report written to {args.output}")
    print()


# =============================================================================
# 主入口
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Tool - 整合聚合、对比、Raw Profiling 与 Trace 分析功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 聚合所有 benchmark 数据
  uv run python benchmark/scripts/benchmark_tool.py aggregate

  # 按设备/模式过滤聚合
  uv run python benchmark/scripts/benchmark_tool.py aggregate --device npu --mode pretrained

  # 对比单个文件对
  uv run python benchmark/scripts/benchmark_tool.py compare cuda.pt npu.pt

  # 批量对比指定适配目录
  uv run python benchmark/scripts/benchmark_tool.py compare --adaptation apple_mobilevit_small

  # 对比所有
  uv run python benchmark/scripts/benchmark_tool.py compare --all --format table

  # 分析单个 trace
  uv run python benchmark/scripts/benchmark_tool.py trace trace_npu_fp32_config_wikitext.json

  # 批量分析所有 trace
  uv run python benchmark/scripts/benchmark_tool.py trace --all --top-ops

  # 分析单个 raw profiling 目录
  uv run python benchmark/scripts/benchmark_tool.py profiling Profiling_L1
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    # aggregate 子命令
    agg_parser = subparsers.add_parser("aggregate", help="聚合 benchmark 数据")
    agg_parser.add_argument("--device", help="过滤设备 (cuda/npu)")
    agg_parser.add_argument("--mode", help="过滤模式 (config/pretrained)")
    agg_parser.add_argument("--format", default="json,csv", help="输出格式 (json,csv,figures)")
    agg_parser.add_argument("--reports-dir", help="输出目录")
    agg_parser.set_defaults(func=cmd_aggregate)

    # compare 子命令
    cmp_parser = subparsers.add_parser("compare", help="对比 CUDA/NPU 输出")
    cmp_parser.add_argument("cuda_pt", nargs="?", help="CUDA .pt 文件路径")
    cmp_parser.add_argument("ascend_pt", nargs="?", help="Ascend .pt 文件路径")
    cmp_parser.add_argument("--adaptation", help="指定适配目录名称")
    cmp_parser.add_argument("--all", action="store_true", help="对比所有适配目录")
    cmp_parser.add_argument("--format", choices=["json", "table"], default="table", help="输出格式")
    cmp_parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    cmp_parser.add_argument("-j", "--jobs", type=int, default=128, metavar="N", help="并行进程数 (默认64, 1=串行)")
    cmp_parser.set_defaults(func=cmd_compare)

    # profiling 子命令
    prof_parser = subparsers.add_parser("profiling", help="分析 torch_npu.profiler 原始目录")
    prof_parser.add_argument("profiling_dir", nargs="?", help="profiling 目录路径")
    prof_parser.add_argument("--adaptation", help="指定适配目录名称")
    prof_parser.add_argument("--all", action="store_true", help="分析所有适配目录中的 profiling 目录")
    prof_parser.add_argument("--deep", action="store_true", help="Deep analysis: parse cluster.db for NPU operator stats (slower but more detailed)")
    prof_parser.add_argument("--top-k", type=int, default=10, help="显示 Top K 项")
    prof_parser.add_argument("-j", "--json", action="store_true", help="仅输出 JSON（不打印报告）")
    prof_parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    prof_parser.set_defaults(func=cmd_profiling)

    # trace 子命令
    trc_parser = subparsers.add_parser("trace", help="分析 trace 文件（含 NPU Fallback）")
    trc_parser.add_argument("trace", nargs="?", help="trace .json 文件路径")
    trc_parser.add_argument("--adaptation", help="指定适配目录名称")
    trc_parser.add_argument("--all", action="store_true", help="分析所有 trace 文件")
    trc_parser.add_argument("--top-ops", action="store_true", help="显示 top CPU ops")
    trc_parser.add_argument("-v", "--verbose", action="store_true", help="详细 Fallback 报告与建议")
    trc_parser.add_argument("-j", "--json", action="store_true", help="仅输出 JSON（不打印报告）")
    trc_parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    trc_parser.set_defaults(func=cmd_trace)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
