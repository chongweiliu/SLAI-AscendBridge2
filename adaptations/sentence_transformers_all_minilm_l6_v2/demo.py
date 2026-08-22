"""
Ascend NPU adaptation demo for sentence-transformers/all-MiniLM-L6-v2.

sentence-embedding 模型（BERT 系编码器，6 层，~22M 参数）：
用 transformers AutoModel + mean pooling 生成句向量，再以余弦相似度
对比"相近句 / 无关句"来验证编码质量，不依赖 sentence-transformers 库。

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
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


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


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """对 last_hidden_state 按 attention_mask 做 mean pooling，得到句向量。"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode_sentences(model, tokenizer, sentences, device):
    """批量编码句子为 L2 归一化句向量。"""
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    embeddings = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ascend Adaptation Demo (sentence-transformers/all-MiniLM-L6-v2, sentence embedding)"
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
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, layers={nlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers).")
        model = AutoModel.from_config(config)
        model = model.to(device)
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = AutoModel.from_pretrained(
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

    # Step 3: Sentence encoding + similarity check
    anchor = "The cat sits on the mat."
    similar = "A cat is sitting on a mat."
    unrelated = "The stock market crashed yesterday."
    sentences = [anchor, similar, unrelated]
    print(f"[Run] Input: {sentences}")

    print("[Run] Encoding sentences...")
    start_time = time.time()

    embeddings = encode_sentences(model, tokenizer, sentences, device)

    assert embeddings.shape == (len(sentences), config.hidden_size), f"Bad embedding shape: {embeddings.shape}"
    assert torch.isfinite(embeddings).all(), "Embeddings contain NaN/Inf"

    emb_anchor, emb_similar, emb_unrelated = embeddings.unbind(0)
    cos_similar = float(torch.dot(emb_anchor, emb_similar))
    cos_unrelated = float(torch.dot(emb_anchor, emb_unrelated))

    encode_time = time.time() - start_time
    print(f"[Run] Embedding dim: {embeddings.shape[1]}, dtype={embeddings.dtype}")
    print(f"[Run] Output: cos(anchor, similar)={cos_similar:.4f}, cos(anchor, unrelated)={cos_unrelated:.4f}")
    print(f"[Run] Encoding time: {encode_time:.4f} seconds")

    if args.dry_run:
        # 随机权重下相似度数值无意义，只验证前向与池化路径可跑通
        print("[Run] Dry-run: random weights, similarity values are meaningless (path validation only).")
    else:
        assert cos_similar > cos_unrelated, (
            f"Similarity ordering wrong with pretrained weights: "
            f"cos(similar)={cos_similar:.4f} <= cos(unrelated)={cos_unrelated:.4f}"
        )
        print("[Run] Similarity ordering correct: similar sentence scored higher than unrelated one.")

    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
