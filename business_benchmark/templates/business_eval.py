#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import random
import re
import socket
import statistics
import string
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, cast

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None  # type: ignore[assignment]

try:
    from jiwer import wer as jiwer_wer
except Exception:  # pragma: no cover - optional runtime dependency
    jiwer_wer = None  # type: ignore[assignment]

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - optional runtime dependency
    rouge_scorer = None  # type: ignore[assignment]

try:
    from seqeval.metrics import accuracy_score as seqeval_accuracy_score
    from seqeval.metrics import f1_score as seqeval_f1_score
    from seqeval.metrics import precision_score as seqeval_precision_score
    from seqeval.metrics import recall_score as seqeval_recall_score
except Exception:  # pragma: no cover - optional runtime dependency
    seqeval_accuracy_score = None  # type: ignore[assignment]
    seqeval_f1_score = None  # type: ignore[assignment]
    seqeval_precision_score = None  # type: ignore[assignment]
    seqeval_recall_score = None  # type: ignore[assignment]

try:
    from sklearn.metrics import ndcg_score
except Exception:  # pragma: no cover - optional runtime dependency
    ndcg_score = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None  # type: ignore[assignment]

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional runtime dependency
    sf = None  # type: ignore[assignment]

try:
    import librosa
except Exception:  # pragma: no cover - optional runtime dependency
    librosa = None  # type: ignore[assignment]

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional runtime dependency
    PILImage = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - depends on runtime env
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not (PROJECT_ROOT / "scripts" / "download_datasets.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
ADAPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ADAPT_DIR / "business_benchmark_config.json"
DATASETS_DIR = PROJECT_ROOT / "datasets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _is_fleurs_dataset(dataset_key: str | None) -> bool:
    return str(dataset_key or "").strip().lower().startswith("fleurs_")


def _is_mcspeech_dataset(dataset_key: str | None) -> bool:
    return str(dataset_key or "").strip().lower().startswith("mcspeech_")


SCENARIO_SUFFIX = {
    "npu_baseline": "baseline",
    "npu_perf": "perf",
    "cuda_baseline": "baseline",
}
COCO_LABEL_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
FAIRFACE_AGE_LABEL_NAMES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
COLOUR_CHECKER_LABEL_NAMES = ["ColorCheckerClassic24"]


def load_config() -> dict:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("business_benchmark_config.json 必须是 JSON object")
    return data


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


_ASR_PUNCTUATION = set(string.punctuation) | {
    "。",
    "、",
    "，",
    "．",
    "！",
    "？",
    "：",
    "；",
    "「",
    "」",
    "『",
    "』",
    "（",
    "）",
    "［",
    "］",
    "【",
    "】",
    "〈",
    "〉",
    "《",
    "》",
    "“",
    "”",
    "‘",
    "’",
    "・",
    "…",
    "〜",
    "～",
}


def _contains_cjk_character(value: str) -> bool:
    for ch in str(value or ""):
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF or 0x20000 <= code <= 0x2A6DF:
            return True
    return False


def _normalize_asr_transcript(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    if not text:
        return ""
    cleaned_chars = []
    for ch in text:
        if ch in _ASR_PUNCTUATION:
            continue
        cleaned_chars.append(" " if ch.isspace() else ch)
    normalized = "".join(cleaned_chars)
    if _contains_cjk_character(normalized):
        return "".join(ch for ch in normalized if not ch.isspace())
    return " ".join(normalized.split())


def _clamp_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _qa_tokens(value: Any) -> list[str]:
    return [token for token in normalize_text(value).split(" ") if token]


def detect_device_from_scenario(scenario: str) -> str:
    return "cuda:0" if scenario == "cuda_baseline" else "npu:0"


def detect_device_model(device: str) -> str:
    if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "unknown_cuda"
    if device.startswith("npu") and torch is not None:
        try:
            import torch_npu  # noqa: F401

            npu_module = getattr(torch, "npu", None)
            if npu_module is not None and npu_module.is_available():
                return npu_module.get_device_name(0)
        except Exception:
            return "unknown_npu"
    return "cpu"


def detect_dtype() -> str:
    dtype = os.environ.get("BUSINESS_BENCHMARK_DTYPE", "").strip().lower()
    if dtype in {"fp32", "fp16", "bf16"}:
        return dtype
    return "fp32"


def _best_effort_host_ip() -> str:
    candidate_ips: list[str] = []
    try:
        host_infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET, type=socket.SOCK_DGRAM)
        for info in host_infos:
            host_ip = str(info[4][0]).strip()
            if host_ip and host_ip not in candidate_ips:
                candidate_ips.append(host_ip)
    except OSError:
        pass

    for probe_target in ("10.255.255.255", "8.8.8.8"):
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect((probe_target, 1))
            host_ip = str(sock.getsockname()[0]).strip()
            if host_ip and not host_ip.startswith("127."):
                return host_ip
            if host_ip and host_ip not in candidate_ips:
                candidate_ips.append(host_ip)
        except OSError:
            continue
        finally:
            if sock is not None:
                sock.close()

    for host_ip in candidate_ips:
        if host_ip and not host_ip.startswith("127."):
            return host_ip
    return candidate_ips[0] if candidate_ips else "127.0.0.1"


def collect_generation_metadata(*, tool: str | None = None) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(),
        "generated_by_tool": tool or Path(__file__).name,
        "generated_by_user": getpass.getuser(),
        "generated_by_hostname": socket.gethostname(),
        "generated_by_host_ip": _best_effort_host_ip(),
        "generated_by_pid": os.getpid(),
    }


def collect_package_versions() -> dict[str, str]:
    package_names = [
        "torch",
        "torch_npu",
        "transformers",
        "datasets",
        "numpy",
        "pillow",
        "timm",
        "accelerate",
        "tokenizers",
    ]
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _get_runtime_backend(device: str):
    if torch is None:
        return None
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda
    if device.startswith("npu"):
        npu_module = getattr(torch, "npu", None)
        if npu_module is not None:
            try:
                if npu_module.is_available():
                    return npu_module
            except Exception:
                return None
    return None


def reset_peak_memory(device: str) -> None:
    backend = _get_runtime_backend(device)
    if backend is None:
        return
    try:
        if hasattr(backend, "empty_cache"):
            backend.empty_cache()
        if hasattr(backend, "reset_peak_memory_stats"):
            backend.reset_peak_memory_stats()
    except Exception:
        return


def get_peak_memory_mb(device: str) -> float:
    backend = _get_runtime_backend(device)
    if backend is not None:
        try:
            if hasattr(backend, "synchronize"):
                backend.synchronize()
            if hasattr(backend, "max_memory_allocated"):
                peak_bytes = backend.max_memory_allocated()
                if isinstance(peak_bytes, (int, float)) and peak_bytes > 0:
                    return round(float(peak_bytes) / (1024 * 1024), 2)
            if hasattr(backend, "max_memory_reserved"):
                peak_bytes = backend.max_memory_reserved()
                if isinstance(peak_bytes, (int, float)) and peak_bytes > 0:
                    return round(float(peak_bytes) / (1024 * 1024), 2)
            if hasattr(backend, "memory_stats"):
                stats = backend.memory_stats()
                if isinstance(stats, dict):
                    for key in (
                        "allocated_bytes.all.peak",
                        "reserved_bytes.all.peak",
                        "active_bytes.all.peak",
                        "allocated_bytes.all.current",
                        "reserved_bytes.all.current",
                    ):
                        peak_bytes = stats.get(key)
                        if isinstance(peak_bytes, (int, float)) and peak_bytes > 0:
                            return round(float(peak_bytes) / (1024 * 1024), 2)
        except Exception:
            pass
    if psutil is not None:
        try:
            rss_bytes = psutil.Process().memory_info().rss
            if isinstance(rss_bytes, (int, float)) and rss_bytes > 0:
                return round(float(rss_bytes) / (1024 * 1024), 2)
        except Exception:
            pass
    return float(os.environ.get("BUSINESS_BENCHMARK_PEAK_MEMORY_MB", "1.0"))


def _require_optional_metric_dependency(dependency, package_name: str, metric_name: str):
    if dependency is None:
        raise ImportError(f"{metric_name} 需要可选依赖 `{package_name}`，请在当前环境安装后重试")
    return dependency


def metric_exact_match(predictions: list[str], references: list[str]) -> dict:
    scores = [1.0 if normalize_text(p) == normalize_text(r) else 0.0 for p, r in zip(predictions, references)]
    return {"exact_match": statistics.fmean(scores) if scores else 0.0, "match_rate": statistics.fmean(scores) if scores else 0.0}


def metric_qa_exact_match(predictions: list[str], references: list[str]) -> dict:
    exact_scores = []
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _qa_tokens(pred)
        ref_tokens = _qa_tokens(ref)
        exact_scores.append(1.0 if pred_tokens == ref_tokens else 0.0)
        if not pred_tokens and not ref_tokens:
            f1_scores.append(1.0)
            continue
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        pred_counts: dict[str, int] = {}
        ref_counts: dict[str, int] = {}
        for token in pred_tokens:
            pred_counts[token] = pred_counts.get(token, 0) + 1
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        common = sum(min(pred_counts.get(token, 0), ref_counts.get(token, 0)) for token in pred_counts)
        if common <= 0:
            f1_scores.append(0.0)
            continue
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        f1_scores.append((2 * precision * recall) / max(precision + recall, 1e-12))
    exact_match = statistics.fmean(exact_scores) if exact_scores else 0.0
    f1 = statistics.fmean(f1_scores) if f1_scores else 0.0
    return {"exact_match": exact_match, "f1": f1, "match_rate": exact_match}


def metric_accuracy(predictions: list[Any], references: list[Any]) -> dict:
    scores = [1.0 if str(p) == str(r) else 0.0 for p, r in zip(predictions, references)]
    return {"accuracy": statistics.fmean(scores) if scores else 0.0, "match_rate": statistics.fmean(scores) if scores else 0.0}


def _rouge_l_tokens(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token]


def _lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    dp = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        prev = 0
        for idx, token_b in enumerate(tokens_b, start=1):
            current = dp[idx]
            if token_a == token_b:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = current
    return dp[-1]


def _fallback_rouge_l_fmeasure(prediction: str, reference: str) -> float:
    pred_tokens = _rouge_l_tokens(prediction)
    ref_tokens = _rouge_l_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall <= 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _normalize_token_tag(label: Any, dataset_key: str | None) -> str:
    text = str(label or "").strip()
    if not text or text == "O":
        return "O"
    prefix = ""
    base = text
    if text.startswith(("B-", "I-")):
        prefix = text[:2]
        base = text[2:]
    normalized_base = base.replace("_", " ").strip().lower()
    if dataset_key == "ncbi_disease":
        if "disease" in normalized_base or "disorder" in normalized_base:
            return f"{prefix}Disease" if prefix else "Disease"
        return "O"
    return text


def metric_rouge_l(predictions: list[str], references: list[str]) -> dict:
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        values = [scorer.score(str(ref or ""), str(pred or ""))["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    else:
        values = [_fallback_rouge_l_fmeasure(str(pred or ""), str(ref or "")) for pred, ref in zip(predictions, references)]
    score = statistics.fmean(values) if values else 0.0
    return {"rougeL": score, "match_rate": score}


def metric_token_f1(predictions: list[list[str]], references: list[list[str]], metric_context: dict | None = None) -> dict:
    precision_fn = _require_optional_metric_dependency(seqeval_precision_score, "seqeval", "token_f1")
    recall_fn = _require_optional_metric_dependency(seqeval_recall_score, "seqeval", "token_f1")
    f1_fn = _require_optional_metric_dependency(seqeval_f1_score, "seqeval", "token_f1")
    accuracy_fn = _require_optional_metric_dependency(seqeval_accuracy_score, "seqeval", "token_f1")
    context = metric_context or {}
    dataset_key = str(context.get("dataset_key") or "").strip() or None
    pred_tags = [[_normalize_token_tag(token, dataset_key) for token in pred] for pred in predictions]
    ref_tags = [[_normalize_token_tag(token, dataset_key) for token in ref] for ref in references]
    return {
        "precision": float(cast(float, precision_fn(ref_tags, pred_tags))),
        "recall": float(cast(float, recall_fn(ref_tags, pred_tags))),
        "f1": float(cast(float, f1_fn(ref_tags, pred_tags))),
        "match_rate": float(cast(float, accuracy_fn(ref_tags, pred_tags))),
    }


def _levenshtein_distance(reference_tokens: list[str], prediction_tokens: list[str]) -> int:
    if not reference_tokens:
        return len(prediction_tokens)
    if not prediction_tokens:
        return len(reference_tokens)
    previous_row = list(range(len(prediction_tokens) + 1))
    for ref_index, ref_token in enumerate(reference_tokens, start=1):
        current_row = [ref_index]
        for pred_index, pred_token in enumerate(prediction_tokens, start=1):
            substitution_cost = 0 if ref_token == pred_token else 1
            current_row.append(
                min(
                    previous_row[pred_index] + 1,
                    current_row[pred_index - 1] + 1,
                    previous_row[pred_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _fallback_word_error_rate(predictions: list[str], references: list[str], *, tokenizer=None) -> float:
    if not references:
        return 1.0
    token_splitter = tokenizer or (lambda text: str(text or "").split())
    total_reference_words = 0
    total_distance = 0
    for prediction, reference in zip(predictions, references):
        reference_tokens = list(token_splitter(reference))
        prediction_tokens = list(token_splitter(prediction))
        total_reference_words += max(len(reference_tokens), 1)
        total_distance += _levenshtein_distance(reference_tokens, prediction_tokens)
    if total_reference_words <= 0:
        return 1.0
    return float(total_distance / total_reference_words)


def metric_wer(predictions: list[str], references: list[str]) -> dict:
    normalized_predictions = [_normalize_asr_transcript(pred) for pred in predictions]
    normalized_references = [_normalize_asr_transcript(ref) for ref in references]
    use_cjk_error_rate = any(_contains_cjk_character(text) for text in normalized_predictions + normalized_references)
    if use_cjk_error_rate:
        wer = _fallback_word_error_rate(normalized_predictions, normalized_references, tokenizer=list)
    elif jiwer_wer is not None:
        wer = float(jiwer_wer(normalized_references, normalized_predictions)) if normalized_references else 1.0
    else:
        wer = _fallback_word_error_rate(normalized_predictions, normalized_references)
    return {"wer": wer, "text_match_rate": max(0.0, 1.0 - wer)}


def metric_topk_accuracy(predictions: list[Any], references: list[Any], metric_context: dict | None = None) -> dict:
    context = metric_context or {}
    score_rows = context.get("prediction_scores")
    label_space = context.get("label_space")
    if not isinstance(score_rows, list) or not score_rows or not isinstance(label_space, list) or not label_space:
        top1_scores = [1.0 if str(pred) == str(ref) else 0.0 for pred, ref in zip(predictions, references)]
        top1 = statistics.fmean(top1_scores) if top1_scores else 0.0
        return {"top1_accuracy": top1, "top5_accuracy": top1, "match_rate": top1}

    label_strings = [str(label) for label in label_space]
    top1_hits = 0
    top5_hits = 0
    total = 0
    for ref, scores in zip(references, score_rows):
        if not isinstance(scores, list) or len(scores) != len(label_strings):
            continue
        ranking = sorted(zip(label_strings, scores), key=lambda item: float(item[1]), reverse=True)
        ranked_labels = [label for label, _ in ranking]
        ref_label = str(ref)
        total += 1
        if ranked_labels and ranked_labels[0] == ref_label:
            top1_hits += 1
        if ref_label in ranked_labels[:5]:
            top5_hits += 1
    if total == 0:
        return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "match_rate": 0.0}
    return {
        "top1_accuracy": top1_hits / total,
        "top5_accuracy": top5_hits / total,
        "match_rate": top1_hits / total,
    }


def _dcg_at_k(relevance: list[float], ranking: list[int], k: int) -> float:
    dcg = 0.0
    for rank_index, item_index in enumerate(ranking[:k], start=1):
        rel = float(relevance[item_index])
        if rel <= 0.0:
            continue
        dcg += (math.pow(2.0, rel) - 1.0) / math.log2(rank_index + 1.0)
    return dcg


def _fallback_ndcg_at_k(scores: list[Any], relevance: list[Any], k: int) -> float:
    if not scores or not relevance or len(scores) != len(relevance):
        return 0.0
    k = min(k, len(scores))
    score_ranking = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
    ideal_ranking = sorted(range(len(relevance)), key=lambda idx: float(relevance[idx]), reverse=True)
    dcg = _dcg_at_k([float(item) for item in relevance], score_ranking, k)
    idcg = _dcg_at_k([float(item) for item in relevance], ideal_ranking, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def metric_reranker_ndcg(predictions: list[Any], references: list[Any], metric_context: dict | None = None) -> dict:
    context = metric_context or {}
    score_rows = context.get("prediction_scores")
    relevance_rows = context.get("ranking_relevance")
    if not isinstance(score_rows, list) or not isinstance(relevance_rows, list) or len(score_rows) != len(relevance_rows):
        return {"ndcg_at_10": 0.0, "mrr": 0.0, "match_rate": 0.0}

    ndcg_values = []
    reciprocal_ranks = []
    top1_hits = []
    for scores, relevance in zip(score_rows, relevance_rows):
        if not isinstance(scores, list) or not isinstance(relevance, list) or len(scores) != len(relevance) or not scores:
            continue
        if ndcg_score is not None:
            ndcg_values.append(float(ndcg_score([relevance], [scores], k=min(10, len(scores)))))
        else:
            ndcg_values.append(_fallback_ndcg_at_k(scores, relevance, k=min(10, len(scores))))
        ranking = sorted(zip(scores, relevance), key=lambda item: float(item[0]), reverse=True)
        rr = 0.0
        for idx, (_, rel) in enumerate(ranking, start=1):
            if float(rel) > 0:
                rr = 1.0 / idx
                break
        reciprocal_ranks.append(rr)
        top1_hits.append(1.0 if ranking and float(ranking[0][1]) > 0 else 0.0)
    if not ndcg_values:
        return {"ndcg_at_10": 0.0, "mrr": 0.0, "match_rate": 0.0}
    return {
        "ndcg_at_10": statistics.fmean(ndcg_values),
        "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "match_rate": statistics.fmean(top1_hits) if top1_hits else 0.0,
    }


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _compute_average_precision(tp: list[int], fp: list[int], total_gt: int) -> float:
    if total_gt <= 0:
        return 0.0
    cum_tp = []
    cum_fp = []
    tp_sum = fp_sum = 0
    for t, f in zip(tp, fp):
        tp_sum += t
        fp_sum += f
        cum_tp.append(tp_sum)
        cum_fp.append(fp_sum)
    recalls = [val / total_gt for val in cum_tp]
    precisions = [cum_tp[i] / max(cum_tp[i] + cum_fp[i], 1) for i in range(len(cum_tp))]
    recalls = [0.0, *recalls, 1.0]
    precisions = [0.0, *precisions, 0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def _compute_detection_ap(predictions: list[dict], references: list[dict], iou_threshold: float) -> float:
    classes = sorted({int(label) for item in references + predictions for label in (item.get("labels") or []) if isinstance(label, (int, float))})
    per_class_ap = []
    for cls in classes:
        gt_by_image: dict[int, list[dict]] = {}
        total_gt = 0
        for image_idx, ref in enumerate(references):
            boxes = ref.get("boxes") or []
            labels = ref.get("labels") or []
            gt_boxes = []
            for box, label in zip(boxes, labels):
                if int(label) == cls:
                    gt_boxes.append({"box": [float(x) for x in box], "matched": False})
                    total_gt += 1
            gt_by_image[image_idx] = gt_boxes
        if total_gt == 0:
            continue

        pred_rows = []
        for image_idx, pred in enumerate(predictions):
            boxes = pred.get("boxes") or []
            labels = pred.get("labels") or []
            scores = pred.get("scores") or []
            for box, label, score in zip(boxes, labels, scores):
                if int(label) == cls:
                    pred_rows.append((float(score), image_idx, [float(x) for x in box]))
        pred_rows.sort(key=lambda item: item[0], reverse=True)

        tp = []
        fp = []
        for _, image_idx, pred_box in pred_rows:
            best_iou = 0.0
            best_gt = None
            for gt in gt_by_image.get(image_idx, []):
                if gt["matched"]:
                    continue
                iou = _bbox_iou(pred_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            if best_gt is not None and best_iou >= iou_threshold:
                best_gt["matched"] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        per_class_ap.append(_compute_average_precision(tp, fp, total_gt))
    return statistics.fmean(per_class_ap) if per_class_ap else 0.0


def metric_detection_map(predictions: list[Any], references: list[Any], metric_context: dict | None = None) -> dict:
    pred_items = [item for item in predictions if isinstance(item, dict)]
    ref_items = [item for item in references if isinstance(item, dict)]
    if not pred_items or len(pred_items) != len(ref_items):
        return {"mAP": 0.0, "map50": 0.0, "match_rate": 0.0}
    thresholds = [0.5 + 0.05 * idx for idx in range(10)]
    ap_values = [_compute_detection_ap(pred_items, ref_items, thr) for thr in thresholds]
    map50 = ap_values[0]
    return {
        "mAP": statistics.fmean(ap_values) if ap_values else 0.0,
        "map50": map50,
        "match_rate": map50,
    }


def _average_cosine_similarity(embeddings_a: list[list[float]], embeddings_b: list[list[float]]) -> float:
    if not embeddings_a or not embeddings_b or len(embeddings_a) != len(embeddings_b):
        return 0.0
    total = 0.0
    count = 0
    for vec_a, vec_b in zip(embeddings_a, embeddings_b):
        if len(vec_a) != len(vec_b) or not vec_a:
            continue
        dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
        norm_a = sum(float(a) * float(a) for a in vec_a) ** 0.5
        norm_b = sum(float(b) * float(b) for b in vec_b) ** 0.5
        if norm_a <= 0 or norm_b <= 0:
            continue
        total += _clamp_unit_interval(dot / (norm_a * norm_b))
        count += 1
    return _clamp_unit_interval(total / count) if count else 0.0


def metric_embedding_similarity(predictions: list[Any], references: list[Any], metric_context: dict | None = None) -> dict:
    context = metric_context or {}
    prediction_embeddings = context.get("prediction_embeddings")
    reference_embeddings = context.get("reference_embeddings")
    if not isinstance(prediction_embeddings, list) or not isinstance(reference_embeddings, list):
        return {"cosine_similarity": 0.0, "match_rate": 0.0}
    cosine = _average_cosine_similarity(prediction_embeddings, reference_embeddings)
    return {"cosine_similarity": cosine, "match_rate": cosine}


def _normalize_mmlu_choice(value: Any) -> str:
    text = str(value).strip().upper()
    if not text:
        return ""
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < 26:
            return chr(65 + idx)
        return text
    paren_match = re.search(r"\(([A-Z])\)", text)
    if paren_match:
        return paren_match.group(1)
    answer_match = re.search(r"\bANSWER\s*:\s*([A-Z])\b", text)
    if answer_match:
        return answer_match.group(1)
    letter_match = re.search(r"\b([A-Z])\b", text)
    if letter_match:
        return letter_match.group(1)
    return text


def metric_mmlu_accuracy(predictions: list[Any], references: list[Any], metric_context: dict | None = None) -> dict:
    scores = []
    for pred, ref in zip(predictions, references):
        pred_str = _normalize_mmlu_choice(pred)
        ref_str = _normalize_mmlu_choice(ref)
        scores.append(1.0 if pred_str == ref_str else 0.0)
    acc = statistics.fmean(scores) if scores else 0.0
    return {"accuracy": acc, "match_rate": acc}


PROFILE_METRICS = {
    "generation_exact_match": metric_exact_match,
    "qa_exact_match": metric_qa_exact_match,
    "classification_accuracy": metric_accuracy,
    "vlm_accuracy": metric_accuracy,
    "vision_topk_accuracy": metric_topk_accuracy,
    "reranker_ndcg": metric_reranker_ndcg,
    "summarization_rouge": metric_rouge_l,
    "token_classification_f1": metric_token_f1,
    "asr_wer": metric_wer,
    "detection_map": metric_detection_map,
    "embedding_similarity": metric_embedding_similarity,
    "audio_embedding_similarity": metric_embedding_similarity,
    "mmlu": metric_mmlu_accuracy,
}


def _maybe_disable_audio_decode(dataset_obj: Any, dataset_key: str | None):
    normalized_dataset_key = str(dataset_key or "").strip().lower()
    if normalized_dataset_key != "librispeech" and not _is_fleurs_dataset(normalized_dataset_key) and not _is_mcspeech_dataset(normalized_dataset_key):
        return dataset_obj
    if not hasattr(dataset_obj, "cast_column"):
        return dataset_obj
    features = getattr(dataset_obj, "features", None)
    if not hasattr(features, "__contains__") or "audio" not in features:
        return dataset_obj
    try:
        from datasets import Audio

        return dataset_obj.cast_column("audio", Audio(decode=False))
    except Exception:
        return dataset_obj


def _take_rows(dataset_obj: Any, max_samples: int, dataset_key: str | None = None) -> list[dict]:
    if hasattr(dataset_obj, "keys") and not hasattr(dataset_obj, "select"):
        preferred = ["test", "validation", "val", "train"]
        for key in preferred:
            if key in dataset_obj:
                dataset_obj = dataset_obj[key]
                break
        else:
            first_key = next(iter(dataset_obj))
            dataset_obj = dataset_obj[first_key]
    dataset_obj = _maybe_disable_audio_decode(dataset_obj, dataset_key)
    if hasattr(dataset_obj, "select"):
        # Some datasets like wikitext contain many blank rows near the head.
        # Over-sample the raw rows first, then filter to max_samples later.
        candidate_count = min(len(dataset_obj), max(max_samples, max_samples * 8))
        subset = dataset_obj.select(range(candidate_count))
        return [dict(subset[i]) for i in range(len(subset))]
    return [dict(row) for row in list(dataset_obj)[:max_samples]]


def _extract_feature_names(dataset_obj: Any, *keys: str) -> list[str]:
    features = getattr(dataset_obj, "features", None)
    if features is None:
        return []
    current = features
    for key in keys:
        if not hasattr(current, "__getitem__") or key not in current:
            return []
        current = current[key]
    names = getattr(current, "names", None)
    if isinstance(names, list):
        return [str(name) for name in names]
    feature = getattr(current, "feature", None)
    nested_names = getattr(feature, "names", None)
    if isinstance(nested_names, list):
        return [str(name) for name in nested_names]
    return []


def _coerce_classification_reference(raw_label: Any, label_names: list[str]) -> str:
    if isinstance(raw_label, bool) or raw_label is None:
        return ""
    if isinstance(raw_label, (int, float)):
        return str(int(raw_label))
    text = str(raw_label).strip()
    if not text:
        return ""
    lowered_text = normalize_text(text)
    for index, label_name in enumerate(label_names):
        if lowered_text == normalize_text(label_name):
            return str(index)
    return text


def _get_classification_label_name(raw_label: Any, label_names: list[str]) -> str:
    if isinstance(raw_label, bool) or raw_label is None:
        return ""
    if isinstance(raw_label, (int, float)):
        label_index = int(raw_label)
        if 0 <= label_index < len(label_names):
            return str(label_names[label_index])
        return str(label_index)
    text = str(raw_label).strip()
    if not text:
        return ""
    return text


def _append_classification_sample(samples: list[dict], text: Any, label: Any, label_names: list[str], *, input_pair: Any = None) -> None:
    input_text = str(text or "").strip()
    if not input_text:
        return
    sample: dict[str, Any] = {
        "input": input_text,
        "reference": _coerce_classification_reference(label, label_names),
        "reference_label_name": _get_classification_label_name(label, label_names),
    }
    if label_names:
        sample["dataset_label_names"] = [str(name) for name in label_names]
    pair_text = str(input_pair or "").strip()
    if pair_text:
        sample["input_pair"] = pair_text
    samples.append(sample)


def _append_vision_classification_sample(samples: list[dict], image: Any, label: Any, label_names: list[str]) -> None:
    if image is None:
        return
    sample: dict[str, Any] = {
        "input": image,
        "reference": _coerce_classification_reference(label, label_names),
        "reference_label_name": _get_classification_label_name(label, label_names),
    }
    if label_names:
        sample["dataset_label_names"] = [str(name) for name in label_names]
    samples.append(sample)


def _append_vision_embedding_sample(samples: list[dict], image: Any) -> None:
    if image is None:
        return
    samples.append({"input": image, "reference": image})


def _decode_image_bytes(image_bytes: Any, *, mode: str = "RGB"):
    if PILImage is None or not isinstance(image_bytes, (bytes, bytearray)):
        return None
    try:
        image = PILImage.open(io.BytesIO(bytes(image_bytes)))
        return image.convert(mode) if mode else image
    except Exception:
        return None


def _generate_synthetic_colour_checker_image(seed: int, size: int = 640):
    if PILImage is None or np is None:
        raise RuntimeError("synthetic_colour_checker 业务测评需要 Pillow 和 NumPy")

    from PIL import ImageDraw, ImageFilter

    rng = np.random.default_rng(int(seed))
    image = PILImage.new("RGB", (size, size), (228, 228, 228))
    draw = ImageDraw.Draw(image)
    chart_colors = [
        (115, 82, 68),
        (194, 150, 130),
        (98, 122, 157),
        (87, 108, 67),
        (133, 128, 177),
        (103, 189, 170),
        (214, 126, 44),
        (80, 91, 166),
        (193, 90, 99),
        (94, 60, 108),
        (157, 188, 64),
        (224, 163, 46),
        (56, 61, 150),
        (70, 148, 73),
        (175, 54, 60),
        (231, 199, 31),
        (187, 86, 149),
        (8, 133, 161),
        (243, 243, 242),
        (200, 200, 200),
        (160, 160, 160),
        (122, 122, 121),
        (85, 85, 85),
        (52, 52, 52),
    ]

    for _ in range(56):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        radius = int(rng.integers(18, 76))
        shade = int(rng.integers(176, 244))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(shade, shade, shade))

    chart_width = int(rng.integers(240, 360))
    chart_height = int(chart_width * 0.68)
    x0 = int(rng.integers(72, max(73, size - chart_width - 72)))
    y0 = int(rng.integers(72, max(73, size - chart_height - 72)))
    border = max(10, chart_width // 28)
    draw.rounded_rectangle(
        (x0 - border, y0 - border, x0 + chart_width + border, y0 + chart_height + border),
        radius=max(6, border // 2),
        fill=(26, 26, 26),
    )

    patch_width = chart_width / 6.0
    patch_height = chart_height / 4.0
    color_index = 0
    for row in range(4):
        for column in range(6):
            px0 = x0 + column * patch_width + 4
            py0 = y0 + row * patch_height + 4
            px1 = x0 + (column + 1) * patch_width - 4
            py1 = y0 + (row + 1) * patch_height - 4
            draw.rectangle((px0, py0, px1, py1), fill=chart_colors[color_index])
            color_index += 1

    reference_box = [float(x0 - border), float(y0 - border), float(x0 + chart_width + border), float(y0 + chart_height + border)]
    image = image.rotate(float(rng.uniform(-16.0, 16.0)), resample=PILImage.BICUBIC, fillcolor=(220, 220, 220), expand=False)
    blur_radius = float(rng.uniform(0.0, 1.0))
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    pixel_array = np.asarray(image).astype(np.int16)
    noise = rng.normal(0.0, 5.5, pixel_array.shape)
    pixel_array = np.clip(pixel_array + noise, 0, 255).astype(np.uint8)
    image = PILImage.fromarray(pixel_array, mode="RGB")
    return image, reference_box


def _iter_synthetic_ocr_font_candidates(preferred_font_path: str | None = None) -> list[str]:
    font_candidates: list[str] = []
    if preferred_font_path:
        normalized = str(preferred_font_path).strip()
        if normalized:
            font_candidates.append(normalized)
    font_candidates.extend(
        (
            "/usr/share/fonts/google-droid-fonts/DroidSansJapanese.ttf",
            "/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        )
    )
    deduped: list[str] = []
    for candidate in font_candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _load_synthetic_ocr_font(font_path: str | None, font_size: int):
    from PIL import ImageFont

    last_error: Exception | None = None
    for candidate in _iter_synthetic_ocr_font_candidates(font_path):
        if not Path(candidate).is_file():
            continue
        try:
            return ImageFont.truetype(candidate, size=font_size)
        except OSError as exc:
            last_error = exc
            continue
    if last_error is not None and font_path:
        print(f"[business-eval][synthetic_ocr] font fallback: {font_path} unavailable ({last_error})", file=sys.stderr)
    return ImageFont.load_default()


def _resolve_synthetic_ocr_font_path(preferred_font_path: str | None = None) -> str | None:
    for candidate in _iter_synthetic_ocr_font_candidates(preferred_font_path):
        if Path(candidate).is_file():
            return candidate
    return None


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _generate_synthetic_ocr_image(
    seed: int,
    text: str,
    *,
    width: int = 384,
    height: int = 384,
    style: str = "",
    font_path: str | None = None,
    font_size: int | None = None,
):
    if PILImage is None or np is None:
        raise RuntimeError("synthetic_ocr 业务测评需要 Pillow 和 NumPy")

    from PIL import ImageDraw, ImageFilter

    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("synthetic_ocr 文本不能为空")

    rng = np.random.default_rng(int(seed))
    background = int(rng.integers(236, 249))
    image = PILImage.new("RGB", (width, height), (background, background, background))
    draw = ImageDraw.Draw(image)
    normalized_style = str(style or "").strip().lower()
    resolved_font_path = _resolve_synthetic_ocr_font_path(font_path)
    resolved_font_size = font_size or (128 if normalized_style == "single_char_centered" else int(rng.integers(54, 72)))
    font = _load_synthetic_ocr_font(resolved_font_path, resolved_font_size)

    if normalized_style == "single_char_centered":
        for _ in range(6):
            x0 = int(rng.integers(0, width))
            y0 = int(rng.integers(0, height))
            x1 = int(rng.integers(0, width))
            y1 = int(rng.integers(0, height))
            shade = int(rng.integers(225, 240))
            draw.line((x0, y0, x1, y1), fill=(shade, shade, shade), width=1)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), normalized_text, font=font, stroke_width=0)
        except Exception:
            left, top, right, bottom = 0, 0, resolved_font_size, resolved_font_size
        text_width = max(1, right - left)
        text_height = max(1, bottom - top)
        anchor_x = (width - text_width) / 2.0 - left
        anchor_y = (height - text_height) / 2.0 - top
        text_color = int(rng.integers(18, 42))
        draw.text((anchor_x, anchor_y), normalized_text, fill=(text_color, text_color, text_color), font=font, stroke_width=0)
        tilt = float(rng.uniform(-2.5, 2.5))
        image = image.rotate(tilt, resample=PILImage.BICUBIC, fillcolor=(background, background, background), expand=False)
        blur_radius = float(rng.uniform(0.0, 0.2))
        noise_sigma = 1.25
    else:
        for _ in range(28):
            x0 = int(rng.integers(0, width))
            y0 = int(rng.integers(0, height))
            line_length = int(rng.integers(width // 8, width // 3))
            angle = float(rng.uniform(-0.9, 0.9))
            x1 = int(max(0, min(width - 1, x0 + line_length * math.cos(angle))))
            y1 = int(max(0, min(height - 1, y0 + line_length * math.sin(angle))))
            shade = int(rng.integers(210, 236))
            stroke = int(rng.integers(1, 3))
            draw.line((x0, y0, x1, y1), fill=(shade, shade, shade), width=stroke)
        anchor_x = int(rng.integers(24, 44))
        anchor_y = int(rng.integers(80, 132))
        text_color = int(rng.integers(22, 66))
        stroke_width = int(rng.integers(0, 2))
        draw.text((anchor_x, anchor_y), normalized_text, fill=(text_color, text_color, text_color), font=font, stroke_width=stroke_width)
        if rng.random() < 0.55:
            offset_x = int(rng.integers(-4, 5))
            offset_y = int(rng.integers(-3, 4))
            echo_shade = min(255, text_color + int(rng.integers(58, 92)))
            draw.text(
                (anchor_x + offset_x, anchor_y + offset_y),
                normalized_text,
                fill=(echo_shade, echo_shade, echo_shade),
                font=font,
                stroke_width=0,
            )
        if rng.random() < 0.65:
            padding_x = int(rng.integers(12, 24))
            padding_y = int(rng.integers(16, 30))
            try:
                left, top, right, bottom = draw.textbbox((anchor_x, anchor_y), normalized_text, font=font, stroke_width=stroke_width)
            except Exception:
                left, top, right, bottom = anchor_x, anchor_y, anchor_x + width // 2, anchor_y + height // 5
            box = (
                max(0, left - padding_x),
                max(0, top - padding_y),
                min(width - 1, right + padding_x),
                min(height - 1, bottom + padding_y),
            )
            draw.rounded_rectangle(box, radius=int(rng.integers(8, 16)), outline=(188, 188, 188), width=1)
        tilt = float(rng.uniform(-7.0, 7.0))
        image = image.rotate(tilt, resample=PILImage.BICUBIC, fillcolor=(background, background, background), expand=False)
        blur_radius = float(rng.uniform(0.0, 0.9))
        noise_sigma = 4.2

    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    pixel_array = np.asarray(image).astype(np.int16)
    noise = rng.normal(0.0, noise_sigma, pixel_array.shape)
    pixel_array = np.clip(pixel_array + noise, 0, 255).astype(np.uint8)
    return PILImage.fromarray(pixel_array, mode="RGB")


def _build_synthetic_ocr_entries(config: dict | None = None) -> list[dict[str, Any]]:
    config = config or {}
    raw_entries = config.get("synthetic_ocr_entries")
    parsed_entries: list[dict[str, Any]] = []
    if isinstance(raw_entries, list):
        for item in raw_entries:
            if isinstance(item, dict):
                render_text = str(item.get("render_text") or item.get("input") or item.get("text") or item.get("reference_text") or item.get("reference") or "").strip()
                reference_text = str(item.get("reference_text") or item.get("reference") or render_text).strip()
            else:
                render_text = str(item or "").strip()
                reference_text = render_text
            if not render_text or not reference_text:
                continue
            parsed_entries.append({"render_text": render_text, "reference_text": reference_text})
    if parsed_entries:
        return parsed_entries

    corpus = [
        "繁體中文",
        "手寫測試",
        "文字辨識",
        "模型驗證",
        "效能提升",
        "資料樣本",
        "影像輸入",
        "精度穩定",
        "台北101",
        "版本2號",
        "測試A1",
        "校驗B2",
        "今日晴朗",
        "春天到了",
        "研究報告",
        "部署完成",
        "請稍後",
        "確認成功",
        "讀取中",
        "觀察值9",
        "樣本12",
        "驗證集3",
        "推理完成",
        "速度1.2x",
    ]
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in corpus:
        normalized = str(item or "").strip()
        if normalized and normalized not in seen:
            deduped.append({"render_text": normalized, "reference_text": normalized})
            seen.add(normalized)
    return deduped


def _coerce_detection_label_id(raw_label: Any, label_names: list[str], dynamic_name_to_id: dict[str, int]) -> int | None:
    if isinstance(raw_label, bool):
        return None
    if isinstance(raw_label, (int, float)):
        return int(raw_label)
    text = str(raw_label or "").strip()
    if not text:
        return None
    if text in dynamic_name_to_id:
        return dynamic_name_to_id[text]
    if text in label_names:
        return label_names.index(text)
    dynamic_id = len(label_names) + len(dynamic_name_to_id)
    dynamic_name_to_id[text] = dynamic_id
    return dynamic_id


def _coerce_detection_label_name(raw_name: Any, label_id: int, label_names: list[str]) -> str:
    if isinstance(raw_name, bool):
        raw_name = ""
    elif isinstance(raw_name, (int, float)):
        raw_name = ""
    text = str(raw_name or "").strip()
    if text:
        return text
    if 0 <= label_id < len(label_names):
        name = str(label_names[label_id]).strip()
        if name:
            return name
    return f"class_{label_id}"


def _normalize_detection_label_name(value: Any) -> str:
    return normalize_text(str(value or "").replace("_", " "))


def _extract_gsm8k_answer(raw: Any) -> str:
    text = str(raw or "")
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return text.strip()


def _normalize_audio_array(audio: Any):
    if np is None:
        return audio
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim >= 2:
        channel_axis = 0 if array.shape[0] <= 8 and array.shape[-1] > 8 else -1
        array = array.mean(axis=channel_axis)
    return array.astype(np.float32, copy=False)


def _read_audio_source(source: Any) -> tuple[Any, int]:
    if sf is None and librosa is None:
        raise RuntimeError("当前环境缺少 soundfile/librosa，无法解码业务音频样本")
    if isinstance(source, (bytes, bytearray)):
        if sf is not None:
            decoded, sr = sf.read(io.BytesIO(source), dtype="float32", always_2d=False)
            return _normalize_audio_array(decoded), int(sr)
        decoded, sr = librosa.load(io.BytesIO(source), sr=None, mono=True)
        return _normalize_audio_array(decoded), int(sr)
    path = str(source or "").strip()
    if not path:
        raise RuntimeError("音频源为空，无法解码业务音频样本")
    if sf is not None:
        decoded, sr = sf.read(path, dtype="float32", always_2d=False)
        return _normalize_audio_array(decoded), int(sr)
    decoded, sr = librosa.load(path, sr=None, mono=True)
    return _normalize_audio_array(decoded), int(sr)


def _decode_audio_payload(audio: Any, *, fallback_path: Any = None) -> tuple[Any, int | None]:
    if not isinstance(audio, dict):
        return None, None
    audio_array = audio.get("array")
    sampling_rate = audio.get("sampling_rate")
    if audio_array is not None:
        return _normalize_audio_array(audio_array), int(sampling_rate) if sampling_rate else None

    raw_bytes = audio.get("bytes")
    if raw_bytes is not None:
        return _read_audio_source(raw_bytes)

    fallback_path_text = str(fallback_path or "").strip()
    if fallback_path_text and Path(fallback_path_text).is_file():
        return _read_audio_source(fallback_path_text)

    path = str(audio.get("path") or "").strip()
    if path and Path(path).is_file():
        return _read_audio_source(path)
    return None, int(sampling_rate) if sampling_rate else None


def _requires_image_latency_samples(config: dict | None) -> bool:
    config = config or {}
    model_type = str(config.get("model_type") or "").strip().lower()
    output_type_hint = str(config.get("output_type_hint") or "").strip().lower()
    model_backend = str(config.get("model_backend") or "").strip().lower()
    return (
        model_type
        in {
            "vision_classification",
            "vision_embedding",
            "vision_detection",
            "vision_keypoint_detection",
            "image_matting",
            "semantic_segmentation",
        }
        or output_type_hint
        in {
            "class_labels",
            "image_embeddings",
            "detection_boxes",
            "keypoints",
            "alpha_masks",
            "segmentation_logits",
            "image_tensors",
        }
        or model_backend in {"clipseg", "semantic_segmentation", "segment_anything_image_encoder"}
    )


def load_business_samples(dataset_key: str | None, dataset_path: str | None, max_samples: int, config: dict | None = None) -> list[dict]:
    config = config or {}
    model_backend = str(config.get("model_backend") or "").strip().lower()
    if model_backend == "openvla_action_prediction":
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("OpenVLA business benchmark 需要 Pillow 以构造内置图像样本") from exc
        prompt = str(config.get("openvla_prompt") or "In: What action should the robot take to pick up the object?\nOut:").strip()
        synthetic_samples: list[dict] = []
        for index in range(max(max_samples, 1)):
            # Small color/position perturbation keeps per-sample inputs non-identical
            # while staying fully synthetic for latency-only business measurement.
            color = (
                int((96 + index * 7) % 256),
                int((128 + index * 11) % 256),
                int((160 + index * 13) % 256),
            )
            image = Image.new("RGB", (224, 224), color=color)
            synthetic_samples.append(
                {
                    "input": prompt,
                    "image": image,
                    "reference": "",
                    "dataset_key": "latency_only",
                    "sample_id": index,
                    "seed": 20260418 + index,
                }
            )
        return synthetic_samples
    if dataset_key == "synthetic_colour_checker":
        synthetic_samples: list[dict] = []
        for index in range(max(max_samples, 1)):
            image, reference_box = _generate_synthetic_colour_checker_image(20260417 + index)
            synthetic_samples.append(
                {
                    "input": image,
                    "reference": {"boxes": [reference_box], "labels": [0]},
                    "query_texts": list(COLOUR_CHECKER_LABEL_NAMES),
                    "query_label_ids": [0],
                    "dataset_label_names": list(COLOUR_CHECKER_LABEL_NAMES),
                    "reference_label_name": COLOUR_CHECKER_LABEL_NAMES[0],
                    "dataset_key": "synthetic_colour_checker",
                    "sample_id": index,
                    "seed": 20260417 + index,
                }
            )
        return synthetic_samples
    if dataset_key == "synthetic_dna":
        rng = random.Random(20260418)
        dna_bases = ["A", "T", "G", "C", "N"]
        synthetic_samples: list[dict] = []
        for index in range(max(max_samples, 1)):
            seq_len = rng.randint(48, 192)
            sequence = "".join(rng.choices(dna_bases, k=seq_len + 1))
            prompt = sequence[:-1]
            target_base = sequence[-1]
            choices = list(dna_bases)
            rng.shuffle(choices)
            if target_base not in choices:
                choices[0] = target_base
            synthetic_samples.append(
                {
                    "input": prompt,
                    "reference": target_base,
                    "choices": choices,
                    "dataset_key": "synthetic_dna",
                    "reference_label_name": target_base,
                    "sample_id": index,
                    "seed": 20260418 + index,
                }
            )
        return synthetic_samples
    if dataset_key == "synthetic_ocr":
        config = config or {}
        ocr_entries = _build_synthetic_ocr_entries(config)
        ocr_style = str(config.get("synthetic_ocr_style") or "").strip().lower()
        ocr_font_path = str(config.get("synthetic_ocr_font_path") or "").strip() or None
        ocr_font_size = _coerce_positive_int(config.get("synthetic_ocr_font_size"))
        synthetic_samples: list[dict] = []
        for index in range(max(max_samples, 1)):
            entry = ocr_entries[index % len(ocr_entries)]
            render_text = str(entry.get("render_text") or "").strip()
            reference_text = str(entry.get("reference_text") or render_text).strip()
            seed = 20260417 + index
            image = _generate_synthetic_ocr_image(
                seed,
                render_text,
                style=ocr_style,
                font_path=ocr_font_path,
                font_size=ocr_font_size,
            )
            synthetic_samples.append(
                {
                    "input": image,
                    "reference": reference_text,
                    "render_text": render_text,
                    "dataset_key": "synthetic_ocr",
                    "sample_id": index,
                    "seed": seed,
                }
            )
        return synthetic_samples
    if dataset_key == "synthetic_timeseries":
        rng = random.Random(20260416)
        synthetic_samples: list[dict] = []
        for index in range(max(max_samples, 1)):
            synthetic_samples.append(
                {
                    "input": f"synthetic_timeseries_{index}",
                    "reference": None,
                    "dataset_key": "synthetic_timeseries",
                    "series_seed": rng.randint(0, 2**31 - 1),
                    "series_index": index,
                }
            )
        return synthetic_samples
    if dataset_key == "builtin_smiles":
        builtin_smiles = [
            "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CC(C)C", "CC(C)(C)C", "CCCCCCC",
            "C=C", "CC=C", "CCC=C", "C=CC=C", "CC=CC",
            "C#C", "CC#C", "C#CC",
            "c1ccccc1", "Cc1ccccc1", "c1ccc2ccccc2c1", "c1ccc(O)cc1", "c1ccnc(c1)",
            "CCO", "CCCO", "CCCCC(O)", "c1ccccc1O", "OCC(O)C(O)C(O)C(O)C",
            "CC=O", "CCC=O", "CC(=O)C", "CC(=O)CC", "c1ccc(C=O)cc1",
            "CC(=O)O", "CCC(=O)O", "OC(=O)c1ccccc1C(=O)O",
            "CCN", "CCCN", "NCCO", "c1ccc(N)cc1", "CC(N)C",
            "CCOC", "CCOCC", "c1ccOcc1", "COc1ccccc1OC",
            "CCCl", "CCBr", "CCI", "CCF",
            "CCS", "CCSCC", "CCCS",
            "C1CCCCC1", "C1=CC=CC=C1O", "CC(=O)NC", "CCOC(=O)C", "CCN(CC)CC", "COC",
            "CCOCN", "CC(Cl)Cl", "CNC", "CC(C)O", "CC(=O)N", "CCOC(=O)N", "c1ccccc1N",
        ]
        synthetic_samples: list[dict] = []
        limit = max(max_samples, 1)
        if limit <= len(builtin_smiles):
            selected_smiles = builtin_smiles[:limit]
        else:
            selected_smiles = [builtin_smiles[index % len(builtin_smiles)] for index in range(limit)]
        for index, smiles in enumerate(selected_smiles):
            synthetic_samples.append(
                {
                    "input": smiles,
                    "reference": smiles,
                    "dataset_key": "builtin_smiles",
                    "sample_id": index,
                    "seed": 20260423 + index,
                }
            )
        return synthetic_samples
    if dataset_key == "synthetic_3d":
        synthetic_samples: list[dict] = []
        for index in range(max(max_samples, 1)):
            synthetic_samples.append(
                {
                    "input": f"synthetic_3d_grid_{index}",
                    "reference": f"synthetic_3d_grid_{index}",
                    "dataset_key": "synthetic_3d",
                    "sample_id": index,
                    "seed": 20260423 + index,
                }
            )
        return synthetic_samples
    if dataset_key == "synthetic_triplets":
        rng = random.Random(20260416)
        num_entities = 10000
        num_relations = 100
        synthetic_samples: list[dict] = []
        for sample_id in range(max(max_samples, 1)):
            synthetic_samples.append(
                {
                    "input": {
                        "head": rng.randrange(num_entities),
                        "relation": rng.randrange(num_relations),
                        "tail": rng.randrange(num_entities),
                    },
                    "reference": sample_id,
                    "sample_id": sample_id,
                    "seed": 20260416,
                }
            )
        return synthetic_samples
    if dataset_key == "smartfracs_phase_field_aluminum_combined_bc_test":
        if not dataset_path:
            raise RuntimeError("smartFRACs phase-field 业务测评要求 dataset_local_path 指向本地 save_to_disk 数据集")
    if not dataset_key or not dataset_path:
        if _requires_image_latency_samples(config):
            synthetic_samples: list[dict] = []
            for index in range(max(max_samples, 1)):
                image, _reference_box = _generate_synthetic_colour_checker_image(20260419 + index, size=384)
                synthetic_samples.append(
                    {
                        "input": image,
                        "reference": "",
                        "dataset_key": "latency_only",
                        "sample_id": index,
                        "seed": 20260419 + index,
                    }
                )
            return synthetic_samples
        return [{"input": "latency-only business benchmark", "reference": ""} for _ in range(max(max_samples, 1))]
    from download_datasets import ensure_dataset, get_dataset_disk_path
    from datasets import load_from_disk

    path = Path(dataset_path)
    if not path.exists():
        fallback_path = get_dataset_disk_path(str(dataset_key))
        if fallback_path.exists():
            path = fallback_path
        else:
            path = ensure_dataset(str(dataset_key))
    if not path.exists():
        raise FileNotFoundError(f"业务数据集不存在: {path}")
    dataset_obj = load_from_disk(str(path))
    rows = _take_rows(dataset_obj, max_samples=max_samples, dataset_key=dataset_key)
    coco_label_names = _extract_feature_names(dataset_obj, "objects", "category") or _extract_feature_names(dataset_obj, "objects", "category_id") or list(COCO_LABEL_NAMES)
    cifar_label_names = _extract_feature_names(dataset_obj, "fine_label") or _extract_feature_names(dataset_obj, "label")
    fairface_age_label_names = _extract_feature_names(dataset_obj, "age") or list(FAIRFACE_AGE_LABEL_NAMES)
    classification_label_names = _extract_feature_names(dataset_obj, "label")
    conll_label_names = _extract_feature_names(dataset_obj, "ner")
    bionlp_label_names = _extract_feature_names(dataset_obj, "ner_tags")
    ncbi_label_names = list(bionlp_label_names)
    samples: list[dict] = []
    dynamic_coco_name_to_id: dict[str, int] = {}
    detection_target_labels = {_normalize_detection_label_name(item) for item in list(config.get("detection_target_labels") or []) if _normalize_detection_label_name(item)}
    is_vision_embedding = str(config.get("model_type") or "").strip().lower() == "vision_embedding" or str(config.get("output_type_hint") or "").strip().lower() == "image_embeddings"

    for row in rows:
        if dataset_key == "gsm8k":
            samples.append(
                {
                    "input": str(row.get("question") or ""),
                    "reference": _extract_gsm8k_answer(row.get("answer")),
                    "dataset_key": "gsm8k",
                }
            )
        elif dataset_key == "cnn_dailymail":
            samples.append({"input": str(row.get("article") or ""), "reference": str(row.get("highlights") or "")})
        elif dataset_key == "sst2":
            _append_classification_sample(samples, row.get("sentence"), row.get("label"), classification_label_names)
        elif dataset_key in {"tweet_eval_sentiment", "tweet_eval_emotion", "tweet_eval_offensive", "tweet_eval_hate"}:
            _append_classification_sample(samples, row.get("text"), row.get("label"), classification_label_names)
        elif dataset_key in {"imdb", "ag_news"}:
            _append_classification_sample(samples, row.get("text"), row.get("label"), classification_label_names)
        elif dataset_key == "glue_mnli":
            _append_classification_sample(samples, row.get("premise"), row.get("label"), classification_label_names, input_pair=row.get("hypothesis"))
        elif dataset_key == "glue_qnli":
            _append_classification_sample(samples, row.get("question"), row.get("label"), classification_label_names, input_pair=row.get("sentence"))
        elif dataset_key == "squad_v2":
            question = str(row.get("question") or "").strip()
            context = str(row.get("context") or "").strip()
            answers = row.get("answers") or {}
            answer_texts = []
            if isinstance(answers, dict):
                answer_texts = [str(text).strip() for text in (answers.get("text") or []) if str(text).strip()]
            reference = answer_texts[0] if answer_texts else ""
            if question and context:
                samples.append({"input": question, "context": context, "reference": reference})
        elif dataset_key == "conll2003":
            words = [str(word) for word in (row.get("words") or []) if str(word).strip()]
            raw_tags = list(row.get("ner") or [])
            if not words:
                continue
            reference_tags: list[str] = []
            for raw_tag in raw_tags[: len(words)]:
                if isinstance(raw_tag, bool) or raw_tag is None:
                    reference_tags.append("O")
                    continue
                if isinstance(raw_tag, (int, float)):
                    label_index = int(raw_tag)
                    if 0 <= label_index < len(conll_label_names):
                        reference_tags.append(str(conll_label_names[label_index]))
                    else:
                        reference_tags.append(str(label_index))
                    continue
                reference_tags.append(str(raw_tag))
            sample = {
                "input": " ".join(words),
                "reference": reference_tags,
            }
            if conll_label_names:
                sample["dataset_label_names"] = [str(name) for name in conll_label_names]
            samples.append(sample)
        elif dataset_key == "science_ie":
            text = str(row.get("text") or "").strip()
            raw_keyphrases = list(row.get("keyphrases") or [])
            if not text:
                continue
            label_names = ["MATERIAL", "PROCESS", "TASK"]
            keyphrases: list[dict[str, Any]] = []
            for raw_keyphrase in raw_keyphrases:
                if not isinstance(raw_keyphrase, dict):
                    continue
                try:
                    start = int(raw_keyphrase.get("start"))
                    end = int(raw_keyphrase.get("end"))
                except Exception:
                    continue
                if start < 0 or end <= start or end > len(text):
                    continue
                label_text = str(raw_keyphrase.get("type_") or "").strip().upper()
                if not label_text:
                    raw_type = raw_keyphrase.get("type")
                    if isinstance(raw_type, (int, float)) and not isinstance(raw_type, bool):
                        raw_index = int(raw_type)
                        if 0 <= raw_index < len(label_names):
                            label_text = label_names[raw_index]
                if not label_text:
                    continue
                keyphrase = {
                    "start": start,
                    "end": end,
                    "label": label_text,
                    "text": text[start:end],
                }
                keyphrase_id = raw_keyphrase.get("id")
                if keyphrase_id is not None:
                    keyphrase["id"] = keyphrase_id
                keyphrases.append(keyphrase)
            samples.append(
                {
                    "input": text,
                    "text": text,
                    "reference": keyphrases,
                    "keyphrases": keyphrases,
                    "dataset_label_names": label_names,
                }
            )
        elif dataset_key == "bionlp2004":
            tokens = [str(token) for token in (row.get("tokens") or []) if str(token).strip()]
            raw_tags = list(row.get("ner_tags") or [])
            if not tokens:
                continue
            reference_tags: list[str] = []
            for raw_tag in raw_tags[: len(tokens)]:
                if isinstance(raw_tag, bool) or raw_tag is None:
                    reference_tags.append("O")
                    continue
                if isinstance(raw_tag, (int, float)):
                    label_index = int(raw_tag)
                    if 0 <= label_index < len(bionlp_label_names):
                        reference_tags.append(str(bionlp_label_names[label_index]))
                    else:
                        reference_tags.append(str(label_index))
                    continue
                reference_tags.append(str(raw_tag))
            sample = {
                "input": " ".join(tokens),
                "reference": reference_tags,
            }
            if bionlp_label_names:
                sample["dataset_label_names"] = [str(name) for name in bionlp_label_names]
            samples.append(sample)
        elif dataset_key == "ncbi_disease":
            tokens = [str(token) for token in (row.get("tokens") or []) if str(token).strip()]
            raw_tags = list(row.get("ner_tags") or [])
            if not tokens:
                continue
            reference_tags: list[str] = []
            for raw_tag in raw_tags[: len(tokens)]:
                if isinstance(raw_tag, bool) or raw_tag is None:
                    reference_tags.append("O")
                    continue
                if isinstance(raw_tag, (int, float)):
                    label_index = int(raw_tag)
                    if 0 <= label_index < len(ncbi_label_names):
                        reference_tags.append(str(ncbi_label_names[label_index]))
                    else:
                        reference_tags.append(str(label_index))
                    continue
                reference_tags.append(str(raw_tag))
            sample = {
                "input": " ".join(tokens),
                "reference": reference_tags,
            }
            if ncbi_label_names:
                sample["dataset_label_names"] = [str(name) for name in ncbi_label_names]
            samples.append(sample)
        elif dataset_key == "wikitext":
            text = str(row.get("text") or "")
            if text.strip():
                samples.append({"input": text, "reference": text})
        elif dataset_key == "synthetic_protein":
            sequence = str(row.get("sequence") or row.get("text") or "").strip()
            reference_sequence = str(row.get("reference_sequence") or row.get("paired_sequence") or sequence).strip()
            if sequence:
                sample = {
                    "input": sequence,
                    "reference": reference_sequence or sequence,
                    "dataset_key": "synthetic_protein",
                }
                sample_id = row.get("sample_id")
                seed = row.get("seed")
                if sample_id is not None:
                    sample["sample_id"] = sample_id
                if seed is not None:
                    sample["seed"] = seed
                samples.append(sample)
        elif dataset_key == "mmlu":
            question = str(row.get("question") or "").strip()
            choices = row.get("choices") or []
            answer_idx = row.get("answer")
            if question and isinstance(choices, list) and isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                samples.append(
                    {
                        "input": question,
                        "choices": [str(c) for c in choices],
                        "reference": str(answer_idx),
                        "dataset_key": "mmlu",
                    }
                )
        elif dataset_key == "ms_marco":
            query = str(row.get("query") or "")
            passages = row.get("passages") or {}
            candidates = []
            relevance = []
            if isinstance(passages, dict):
                candidates = [str(item) for item in (passages.get("passage_text") or []) if str(item).strip()]
                raw_selected = passages.get("is_selected") or []
                relevance = [int(item) for item in raw_selected[: len(candidates)]]
            answers = row.get("answers") or []
            if not candidates and isinstance(answers, list) and answers:
                answer_text = str(answers[0]).strip()
                negative = f"Unrelated passage for query: {query}"
                candidates = [answer_text, negative]
                relevance = [1, 0]
            if candidates:
                if len(relevance) < len(candidates):
                    relevance.extend([0] * (len(candidates) - len(relevance)))
                samples.append({"input": query, "candidates": candidates, "reference": relevance[: len(candidates)]})
        elif dataset_key == "scienceqa":
            choices = row.get("choices") or []
            answer_idx = row.get("answer")
            answer = ""
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                answer = str(choices[answer_idx])
            # Include image for VLM evaluation
            image = row.get("image") or row.get("img")
            if str(config.get("model_type") or "").strip().lower() == "vlm" and image is None:
                continue
            sample = {"input": str(row.get("question") or ""), "reference": answer, "dataset_key": "scienceqa"}
            if isinstance(choices, list) and choices:
                sample["choices"] = [str(choice) for choice in choices]
            if image is not None:
                sample["image"] = image
            samples.append(sample)
        elif dataset_key == "pubmed_qa":
            model_type = str(config.get("model_type") or "").strip().lower()
            question = str(row.get("question") or "").strip()
            context_value = row.get("context")
            context_text = ""
            if isinstance(context_value, dict):
                raw_contexts = context_value.get("contexts") or context_value.get("context") or []
                if isinstance(raw_contexts, list):
                    context_text = " ".join(str(item).strip() for item in raw_contexts[:1] if str(item).strip())
                elif raw_contexts is not None:
                    context_text = str(raw_contexts).strip()
            elif context_value is not None:
                context_text = str(context_value).strip()
            long_answer = str(row.get("long_answer") or "").strip()
            biomedical_text = f"{question} {context_text}".strip() if question and context_text else (context_text or question or long_answer)
            if model_type == "masked_lm":
                if biomedical_text:
                    sample = {
                        "input": biomedical_text,
                        "dataset_key": "pubmed_qa",
                    }
                    if question:
                        sample["question"] = question
                    if context_text:
                        sample["context"] = context_text
                    if long_answer:
                        sample["long_answer"] = long_answer
                    samples.append(sample)
            elif model_type in {"embedding", "discriminator"}:
                if biomedical_text:
                    sample = {
                        "input": biomedical_text,
                        "reference": biomedical_text,
                        "dataset_key": "pubmed_qa",
                    }
                    if question:
                        sample["question"] = question
                    if context_text:
                        sample["context"] = context_text
                    if long_answer:
                        sample["long_answer"] = long_answer
                    samples.append(sample)
            else:
                samples.append(
                    {
                        "input": str(row.get("question") or ""),
                        "reference": str(row.get("final_decision") or ""),
                        "dataset_key": "pubmed_qa",
                        "choices": ["yes", "no", "maybe"],
                    }
                )
        elif dataset_key == "coco":
            image = row.get("image") or row.get("img")
            objects = row.get("objects") or {}
            ref_boxes = []
            ref_labels = []
            ref_label_names = []
            if isinstance(objects, dict):
                bbox_list = objects.get("bbox") or []
                raw_label_values = objects.get("category_id") or objects.get("label") or objects.get("category") or []
                raw_label_names = objects.get("category") or objects.get("label_name") or []
                for idx, (box, raw_label) in enumerate(zip(bbox_list, raw_label_values)):
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        label_id = _coerce_detection_label_id(raw_label, coco_label_names, dynamic_coco_name_to_id)
                        if label_id is None:
                            continue
                        raw_name = raw_label_names[idx] if idx < len(raw_label_names) else ""
                        label_name = _coerce_detection_label_name(raw_name, label_id, coco_label_names)
                        if detection_target_labels and _normalize_detection_label_name(label_name) not in detection_target_labels:
                            continue
                        x, y, w, h = [float(v) for v in box]
                        ref_boxes.append([x, y, x + w, y + h])
                        ref_labels.append(label_id)
                        ref_label_names.append(label_name)
            if image is not None and ref_boxes:
                query_pairs = []
                seen_query_labels = set()
                for label_id, label_name in zip(ref_labels, ref_label_names):
                    if label_id in seen_query_labels:
                        continue
                    seen_query_labels.add(label_id)
                    query_pairs.append((label_id, label_name))
                samples.append(
                    {
                        "input": image,
                        "reference": {"boxes": ref_boxes, "labels": ref_labels},
                        "query_texts": [label_name for _, label_name in query_pairs],
                        "query_label_ids": [label_id for label_id, _ in query_pairs],
                    }
                )
        elif dataset_key == "pubtables_detection_1500":
            image = row.get("image") or row.get("img")
            objects = row.get("objects") or {}
            ref_boxes = []
            ref_labels = []
            ref_label_names = []
            if isinstance(objects, dict):
                bbox_list = list(objects.get("bbox") or [])
                raw_categories = objects.get("categories") or objects.get("category") or objects.get("label") or []
                if isinstance(raw_categories, str):
                    raw_categories = [raw_categories] * len(bbox_list)
                elif not isinstance(raw_categories, list):
                    raw_categories = [raw_categories] * len(bbox_list)
                for idx, box in enumerate(bbox_list):
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        continue
                    raw_label = raw_categories[idx] if idx < len(raw_categories) else (raw_categories[0] if raw_categories else "table")
                    normalized_label = _normalize_detection_label_name(raw_label)
                    if normalized_label in {"", "table"}:
                        label_id = 0
                        label_name = "table"
                    elif normalized_label in {"tablerotated", "table_rotated"}:
                        label_id = 1
                        label_name = "table rotated"
                    else:
                        continue
                    if detection_target_labels and _normalize_detection_label_name(label_name) not in detection_target_labels:
                        continue
                    ref_boxes.append([float(v) for v in box])
                    ref_labels.append(label_id)
                    ref_label_names.append(label_name)
            if image is not None and ref_boxes:
                query_pairs = []
                seen_query_labels = set()
                for label_id, label_name in zip(ref_labels, ref_label_names):
                    if label_id in seen_query_labels:
                        continue
                    seen_query_labels.add(label_id)
                    query_pairs.append((label_id, label_name))
                samples.append(
                    {
                        "input": image,
                        "reference": {"boxes": ref_boxes, "labels": ref_labels},
                        "query_texts": [label_name for _, label_name in query_pairs],
                        "query_label_ids": [label_id for label_id, _ in query_pairs],
                    }
                )
        elif dataset_key == "librispeech" or _is_fleurs_dataset(dataset_key) or _is_mcspeech_dataset(dataset_key):
            audio = row.get("audio")
            audio_array = None
            sampling_rate = None
            if isinstance(audio, dict):
                audio_array, sampling_rate = _decode_audio_payload(audio, fallback_path=row.get("file") or row.get("path"))
            text = row.get("transcription") or row.get("raw_transcription") or row.get("text") or row.get("sentence") or ""
            if audio_array is not None:
                samples.append({"input": audio_array, "reference": text, "sampling_rate": sampling_rate})
        elif dataset_key == "imagenet":
            image_value = row.get("image") or row.get("img")
            if is_vision_embedding:
                _append_vision_embedding_sample(samples, image_value)
            else:
                _append_vision_classification_sample(samples, image_value, row.get("label"), classification_label_names)
        elif dataset_key == "cifar100":
            image_value = row.get("img") or row.get("image")
            if is_vision_embedding:
                _append_vision_embedding_sample(samples, image_value)
            else:
                _append_vision_classification_sample(
                    samples,
                    image_value,
                    row.get("fine_label") if "fine_label" in row else row.get("label"),
                    cifar_label_names,
                )
        elif dataset_key == "synthetic_keypoints":
            image_value = row.get("image") or row.get("img")
            if image_value is None:
                image_value = _decode_image_bytes(row.get("img_bytes"))
            if image_value is None:
                continue
            sample_id = row.get("sample_id")
            seed = row.get("seed")
            sample = {
                "input": image_value,
                "reference": sample_id if sample_id is not None else seed if seed is not None else 0,
            }
            if sample_id is not None:
                sample["sample_id"] = sample_id
            if seed is not None:
                sample["seed"] = seed
            samples.append(sample)
        elif dataset_key == "synthetic_matting":
            image_value = row.get("image") or row.get("img")
            if image_value is None:
                image_value = _decode_image_bytes(row.get("image_bytes") or row.get("img_bytes"), mode="RGB")
            trimap_value = row.get("trimap")
            if trimap_value is None:
                trimap_value = _decode_image_bytes(row.get("trimap_bytes"), mode="L")
            alpha_value = row.get("alpha")
            if alpha_value is None:
                alpha_value = _decode_image_bytes(row.get("alpha_bytes"), mode="L")
            if image_value is None or trimap_value is None or alpha_value is None:
                continue
            sample = {
                "input": image_value,
                "trimap": trimap_value,
                "reference": alpha_value,
            }
            sample_id = row.get("sample_id")
            seed = row.get("seed")
            if sample_id is not None:
                sample["sample_id"] = sample_id
            if seed is not None:
                sample["seed"] = seed
            samples.append(sample)
        elif dataset_key == "smartfracs_phase_field_aluminum_combined_bc_test":
            initial_field = row.get("input")
            reference_field = row.get("reference")
            if initial_field is None or reference_field is None:
                continue
            samples.append(
                {
                    "input": initial_field,
                    "reference": reference_field,
                    "sample_id": row.get("sample_id"),
                    "material": str(row.get("material") or "aluminum"),
                    "boundary_condition": str(row.get("boundary_condition") or "combined_bc"),
                    "prompt_text": str(row.get("prompt_text") or ""),
                    "break_time": row.get("break_time"),
                    "grid_height": row.get("grid_height"),
                    "grid_width": row.get("grid_width"),
                    "source_archive": str(row.get("source_archive") or ""),
                    "source_member": str(row.get("source_member") or ""),
                }
            )
        elif dataset_key == "fairface":
            image_value = row.get("image") or row.get("img")
            if image_value is None:
                image_value = _decode_image_bytes(row.get("img_bytes"))
            _append_vision_classification_sample(samples, image_value, row.get("age"), fairface_age_label_names)
        else:
            text = str(row.get("text") or row.get("sentence") or row.get("question") or "")
            ref = row.get("label") if "label" in row else text
            samples.append({"input": text, "reference": ref})

    filtered = []
    for sample in samples:
        input_value = sample.get("input")
        if input_value is None:
            continue
        if isinstance(input_value, str):
            if not input_value.strip():
                continue
        filtered.append(sample)
    return filtered[:max_samples]


def load_custom_handler(config: dict):
    relative = str(config.get("custom_evaluator") or "").strip()
    if not relative:
        return None
    module_path = ADAPT_DIR / relative
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("business_model_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 custom_evaluator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_business_eval"):
        raise RuntimeError(f"{module_path.name} 缺少 run_business_eval()")
    return module.run_business_eval


def placeholder_predictions(samples: list[dict], profile_name: str) -> tuple[list[Any], list[Any]]:
    references = [sample.get("reference") for sample in samples]
    if profile_name == "token_classification_f1":
        refs = [ref if isinstance(ref, list) else [str(ref)] for ref in references]
        return refs, refs
    preds = ["" if ref is None else ref for ref in references]
    return preds, references


def _lookup_numeric_metric(source: object, metric_name: object) -> float | None:
    metric_key = str(metric_name or "").strip()
    if not metric_key or not isinstance(source, dict):
        return None
    value = source.get(metric_key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    lowered_metric_key = metric_key.lower()
    for key, candidate in source.items():
        if str(key or "").strip().lower() == lowered_metric_key and isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def get_quality_metrics(
    profile_name: str,
    predictions: list[Any],
    references: list[Any],
    metric_context: dict | None = None,
    *,
    primary_metric: str = "",
    extra_metrics: dict | None = None,
) -> dict:
    if profile_name == "latency_only":
        return {}
    metric_fn = PROFILE_METRICS.get(profile_name)
    if metric_fn is None:
        fallback_value = _lookup_numeric_metric(extra_metrics, primary_metric)
        if fallback_value is None:
            fallback_value = _lookup_numeric_metric(metric_context, primary_metric)
        if primary_metric and fallback_value is not None:
            return {primary_metric: fallback_value}
        raise RuntimeError(f"未实现的业务评测画像: {profile_name}")
    if profile_name in {"vision_topk_accuracy", "reranker_ndcg", "detection_map", "embedding_similarity", "audio_embedding_similarity", "token_classification_f1"}:
        return cast(Any, metric_fn)(predictions, references, metric_context)
    return cast(Any, metric_fn)(predictions, references)


def _normalize_artifact_tag(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip().lower())
    return normalized.strip("_")


def build_artifact_name(device: str, dtype: str, dataset: str, scenario: str, *, artifact_tag: str = "") -> str:
    device_short = "cuda" if device.startswith("cuda") else "npu" if device.startswith("npu") else "cpu"
    suffix = SCENARIO_SUFFIX[scenario]
    normalized_tag = _normalize_artifact_tag(artifact_tag)
    tag_segment = f"_{normalized_tag}" if normalized_tag else ""
    return f"business_metrics_{device_short}_{dtype}_pretrained_{dataset}{tag_segment}_{suffix}.json"


def _preview_values(values: list[Any], limit: int = 8) -> list[Any]:
    preview: list[Any] = []
    for value in list(values)[: max(limit, 0)]:
        preview.append(json.loads(json.dumps(value, ensure_ascii=False, default=str)))
    return preview


def main() -> int:
    parser = argparse.ArgumentParser(description="Business benchmark evaluator harness")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_SUFFIX), required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--artifact-tag", default="", help="可选工件标签；小样本 smoke run 建议显式传入，避免覆盖正式工件")
    parser.add_argument("--allow-placeholder", action="store_true", help="缺少 custom_evaluator 时允许使用占位预测")
    args = parser.parse_args()

    config = load_config()
    dataset_key = str(config.get("dataset") or "").strip() or None
    dataset_path = str(os.environ.get("BUSINESS_BENCHMARK_DATASET_PATH") or "").strip() or str(config.get("dataset_local_path") or "").strip() or None
    benchmark_run_id = str(os.environ.get("BUSINESS_BENCHMARK_RUN_ID") or config.get("benchmark_run_id") or "").strip()
    profile_name = str(config.get("evaluation_profile") or "").strip()
    primary_metric = str(config.get("primary_metric") or "").strip()
    output_type_hint = str(config.get("output_type_hint") or "").strip()
    samples = load_business_samples(dataset_key, dataset_path, max(args.max_samples, 1), config=config)
    if not samples:
        raise RuntimeError("业务数据集为空，无法评测")

    device = detect_device_from_scenario(args.scenario)
    reset_peak_memory(device)
    handler = load_custom_handler(config)
    start_ts = time.perf_counter()
    start_time = datetime.now().isoformat()
    if handler is None:
        if not args.allow_placeholder:
            custom_path = ADAPT_DIR / str(config.get("custom_evaluator") or "business_model_eval.py")
            raise RuntimeError(f"缺少可执行业务评测代码: {custom_path}。请在 adaptation 目录实现 run_business_eval()，或显式传入 --allow-placeholder 仅做链路验证。")
        predictions, references = placeholder_predictions(samples, profile_name)
        extra_metrics: dict[str, Any] = {}
        metric_context: dict[str, Any] = {}
    else:
        result = handler(
            samples=samples,
            scenario=args.scenario,
            config=config,
            profile_name=profile_name,
            primary_metric=primary_metric,
            output_type_hint=output_type_hint,
            dataset_key=dataset_key,
        )
        if not isinstance(result, dict):
            raise RuntimeError("run_business_eval() 必须返回 dict")
        predictions = list(result.get("predictions") or [])
        references = list(result.get("references") or [sample.get("reference") for sample in samples])
        extra_metrics = dict(result.get("metrics") or {})
        metric_context = dict(result.get("metric_context") or {})
        if len(predictions) != len(references):
            raise RuntimeError("predictions 与 references 长度不一致")

    effective_num_samples = extra_metrics.get("effective_num_samples")
    if isinstance(effective_num_samples, (int, float)):
        effective_num_samples = int(effective_num_samples)
    else:
        effective_num_samples = len(predictions)
    if effective_num_samples <= 0:
        raise RuntimeError("业务评测有效样本数必须大于 0")

    total_latency_s = (time.perf_counter() - start_ts) / max(effective_num_samples, 1)
    measured_latency = extra_metrics.get("latency_s")
    latency_s = float(measured_latency) if isinstance(measured_latency, (int, float)) and measured_latency > 0 else total_latency_s
    measured_wall_clock = extra_metrics.get("wall_clock_s")
    if not isinstance(measured_wall_clock, (int, float)) or not math.isfinite(float(measured_wall_clock)) or float(measured_wall_clock) <= 0:
        measured_wall_clock = extra_metrics.get("total_duration_s")
    if isinstance(measured_wall_clock, (int, float)) and math.isfinite(float(measured_wall_clock)) and float(measured_wall_clock) > 0:
        measured_wall_clock_s = float(measured_wall_clock)
    else:
        measured_wall_clock_s = float(latency_s) * max(effective_num_samples, 1)
    quality_metrics = get_quality_metrics(
        profile_name,
        predictions,
        references,
        metric_context,
        primary_metric=primary_metric,
        extra_metrics=extra_metrics,
    )
    throughput_qps = effective_num_samples / max(measured_wall_clock_s, 1e-8)
    peak_memory_mb = get_peak_memory_mb(device)
    if not isinstance(peak_memory_mb, (int, float)) or float(peak_memory_mb) <= 0:
        if psutil is not None:
            try:
                rss_bytes = psutil.Process().memory_info().rss
                if isinstance(rss_bytes, (int, float)) and rss_bytes > 0:
                    peak_memory_mb = round(float(rss_bytes) / (1024 * 1024), 2)
            except Exception:
                peak_memory_mb = float(os.environ.get("BUSINESS_BENCHMARK_PEAK_MEMORY_MB", "1.0"))
        else:
            peak_memory_mb = float(os.environ.get("BUSINESS_BENCHMARK_PEAK_MEMORY_MB", "1.0"))
    device_model = detect_device_model(device)
    dtype = str(extra_metrics.get("dtype") or config.get("dtype") or config.get("torch_dtype") or detect_dtype()).strip().lower()
    if dtype not in {"fp32", "fp16", "bf16"}:
        dtype = detect_dtype()
    dataset_name = dataset_key or "latency_only"
    artifact_tag = str(args.artifact_tag or "").strip()
    if not artifact_tag and len(samples) <= 50:
        artifact_tag = f"smoke{len(samples)}"
    artifact_path = ADAPT_DIR / build_artifact_name(device, dtype, dataset_name, args.scenario, artifact_tag=artifact_tag)

    metric_start_time = str(
        extra_metrics.get("start_time")
        or metric_context.get("start_time")
        or start_time
    ).strip() or start_time
    metric_end_time = str(
        extra_metrics.get("end_time")
        or metric_context.get("end_time")
        or datetime.now().isoformat()
    ).strip() or datetime.now().isoformat()

    metric_payload = {
        "benchmark_run_id": benchmark_run_id,
        "measurement_contract_version": int(extra_metrics.get("measurement_contract_version") or os.environ.get("BUSINESS_BENCHMARK_MEASUREMENT_CONTRACT_VERSION") or config.get("measurement_contract_version") or 1),
        "start_time": metric_start_time,
        "end_time": metric_end_time,
        "latency_s": round(latency_s, 6),
        "wall_clock_s": round(measured_wall_clock_s, 6),
        "total_duration_s": round(measured_wall_clock_s, 6),
        "peak_memory_mb": peak_memory_mb,
        "num_samples": effective_num_samples,
        "device": device,
        "device_model": device_model,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "mode": "pretrained",
        "dtype": dtype,
        "dataset": dataset_name,
        "scenario": args.scenario,
        "scenario_command": str(os.environ.get("BUSINESS_BENCHMARK_SCENARIO_COMMAND") or "").strip(),
        "latency_measurement_scope": str(extra_metrics.get("latency_measurement_scope") or os.environ.get("BUSINESS_BENCHMARK_LATENCY_MEASUREMENT_SCOPE") or config.get("latency_measurement_scope") or "real_business"),
        "warmup_iterations": int(extra_metrics.get("warmup_iterations") or 0),
        "warmup_sample_count": int(extra_metrics.get("warmup_sample_count") or 0),
        "task_queue_enable": str(extra_metrics.get("task_queue_enable") or os.environ.get("TASK_QUEUE_ENABLE") or "0"),
        "ascend_rt_visible_devices": str(extra_metrics.get("ascend_rt_visible_devices") or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or ""),
        "cuda_visible_devices": str(extra_metrics.get("cuda_visible_devices") or os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        "selected_npus": list(extra_metrics.get("selected_npus") or config.get("selected_npus") or []),
        "parallel_mode": str(extra_metrics.get("parallel_mode") or config.get("parallel_mode") or ""),
        "device_topology": str(extra_metrics.get("device_topology") or config.get("device_topology") or ""),
        "loaded_from_model_files": bool(extra_metrics.get("loaded_from_model_files", False)),
        "model_source_effective": str(extra_metrics.get("model_source_effective") or ""),
        "model_source_kind": str(extra_metrics.get("model_source_kind") or "hub"),
        "tokenizer_source_effective": str(extra_metrics.get("tokenizer_source_effective") or ""),
        "tokenizer_source_kind": str(extra_metrics.get("tokenizer_source_kind") or "hub"),
        "patch_load_status": str(extra_metrics.get("patch_load_status") or "disabled"),
        "patch_modules": list(extra_metrics.get("patch_modules") or []),
        "patch_hooks": list(extra_metrics.get("patch_hooks") or []),
        "patch_errors": list(extra_metrics.get("patch_errors") or []),
        "output_type": output_type_hint or "generated_text",
        "evaluation_profile": profile_name,
        "primary_metric": primary_metric,
        "throughput_qps": round(throughput_qps, 6),
        "artifact_tag": _normalize_artifact_tag(artifact_tag),
        "package_versions": collect_package_versions(),
        "predictions_preview": _preview_values(predictions),
        "references_preview": _preview_values(references),
        **extra_metrics,
        **quality_metrics,
    }
    metric_payload["effective_num_samples"] = effective_num_samples
    inference_strategy = str(metric_payload.get("inference_strategy") or metric_context.get("inference_strategy") or "").strip()
    if inference_strategy:
        metric_payload["inference_strategy"] = inference_strategy
    fallback_reason = str(metric_payload.get("fallback_reason") or metric_context.get("fallback_reason") or "").strip()
    if fallback_reason:
        metric_payload["fallback_reason"] = fallback_reason
    if profile_name == "mmlu" or dataset_name == "mmlu":
        metric_payload["normalized_predictions_preview"] = [_normalize_mmlu_choice(value) for value in predictions[:8]]
        metric_payload["normalized_references_preview"] = [_normalize_mmlu_choice(value) for value in references[:8]]
    metric_payload.update(collect_generation_metadata(tool=Path(__file__).name))
    artifact_path.write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[business-eval] wrote {artifact_path.name} dataset={dataset_name} profile={profile_name} primary_metric={primary_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
