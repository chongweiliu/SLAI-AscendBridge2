"""
Ascend NPU adaptation demo for cross-encoder/ms-marco-MiniLM-L6-v2.

cross-encoder reranker：将 (query, passage) 拼接后一起送入 BERT 编码器，
输出单个相关性分数（num_labels=1 的 SequenceClassification logits），
用于搜索重排序。验证方式：两组对比样本（相关段落 vs 无关段落），
真实权重下要求相关段落分数高于无关段落。

- Supports --dry-run: random weights + shrunken layers, no weight download.
- Full run: loads pretrained weights onto the selected accelerator.
- Model cache is pinned to this adaptation's models/ directory.

保存输出: uv run python demo.py > output.txt 2>&1
"""

import argparse
import os
import time
from pathlib import Path

# 国内网络环境默认走 HF 镜像（外部已设置的环境变量优先）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


def shrink_config_for_dry_run(config):
    """保守缩小：仅将层数减为 2，其余不变，保证通用兼容。"""
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, key):
            old = getattr(config, key)
            if old is None:
                continue
            new = max(1, min(old, 2))
            setattr(config, key, new)
            if old != new:
                print(f"[Setup] Shrunk {key}: {old} -> {new}")


def select_idle_npu() -> int:
    """选择空闲 HBM 最多的 NPU 并设为当前设备。

    本机禁止设置 ASCEND_RT_VISIBLE_DEVICES（会导致 aclInit error 107001），
    因此通过 torch.npu.set_device() 在多卡中选定单卡。
    """
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


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def get_device():
    """Detects best device (NPU > CUDA > CPU). Returns device string."""
    # 1. Check for Ascend NPU
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            idx = select_idle_npu()
            print(f"[Device] Huawei Ascend NPU detected, selected npu:{idx}")
            return f"npu:{idx}"
    except ImportError:
        pass

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0"

    # 3. Fallback to CPU
    print("[Device] No accelerator detected, using CPU.")
    return "cpu"


# 两组对比样本：每组一个 query + (相关段落, 无关段落)
EVAL_SETS = [
    {
        "query": "What is the capital of France?",
        "relevant": "Paris is the capital and most populous city of France.",
        "irrelevant": "Mix the flour and sugar in a large bowl.",
    },
    {
        "query": "How many planets are in the solar system?",
        "relevant": "There are eight planets in the solar system.",
        "irrelevant": "The cake should bake in the oven for about 45 minutes.",
    },
]


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ascend Adaptation Demo (cross-encoder/ms-marco-MiniLM-L6-v2, reranker)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    args = parser.parse_args()

    # 缓存固定在 adaptation 目录内，避免污染项目根 models/
    cache_dir = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {cache_dir}")

    device = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # Step 1: Tokenizer
    print(f"[Setup] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)

    # Step 2: Model loading (Dry Run vs Real)
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    nlayers = getattr(config, "num_hidden_layers", "?")
    print(
        f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, "
        f"layers={nlayers}, num_labels={getattr(config, 'num_labels', '?')}"
    )

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers).")
        model = AutoModelForSequenceClassification.from_config(config)
        model = model.to(device)
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            cache_dir=cache_dir,
        )
        model = model.to(device)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")

    model.eval()

    # Step 3: Reranking scores for (query, passage) pairs
    queries, passages, meta = [], [], []
    for s in EVAL_SETS:
        for label in ("relevant", "irrelevant"):
            queries.append(s["query"])
            passages.append(s[label])
            meta.append(label)
    print(f"[Run] Input: {len(EVAL_SETS)} query sets x 2 passages (relevant/irrelevant)")

    inputs = tokenizer(queries, passages, padding=True, truncation=True, return_tensors="pt").to(device)

    print("[Run] Scoring (query, passage) pairs...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    assert logits.shape[0] == len(queries), f"Bad logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"

    # num_labels=1 的 reranker：单一 logit 即相关性分数
    scores = logits.squeeze(-1).tolist()
    score_time = time.time() - start_time

    all_ok = True
    for i, s in enumerate(EVAL_SETS):
        rel_score, irr_score = scores[2 * i], scores[2 * i + 1]
        print(f"[Run] Query: {s['query']}")
        print(f"[Run]   score(relevant)   = {rel_score:.4f} | {s['relevant']}")
        print(f"[Run]   score(irrelevant) = {irr_score:.4f} | {s['irrelevant']}")
        if rel_score > irr_score:
            print("[Run]   ranking: relevant > irrelevant (correct)")
        else:
            all_ok = False
            print("[Run]   ranking: relevant <= irrelevant (wrong)")

    print(f"[Run] Output: scores={ [round(x, 4) for x in scores] }")
    print(f"[Run] Scoring time: {score_time:.4f} seconds")

    if args.dry_run:
        # 随机权重下分数无意义，只验证拼接编码与前向路径可跑通
        print("[Run] Dry-run: random weights, scores are meaningless (path validation only).")
    else:
        assert all_ok, "Reranking order wrong with pretrained weights: relevant passage must score higher"
        print("[Run] Reranking correct: relevant passages scored higher in all sets.")

    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
