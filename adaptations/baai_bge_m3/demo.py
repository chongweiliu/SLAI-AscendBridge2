"""
Ascend NPU adaptation demo for BAAI/bge-m3 (multilingual retrieval embedding).

模型为 XLM-RoBERTa encoder（sentence-transformers 打包），用标准
transformers `AutoModel` 加载；验证方式：句子编码（mean pooling + L2 归一化）
+ 余弦相似度矩阵。

Host-specific adjustments:
- HF mirror defaults (direct huggingface.co is unreachable on this host)
- NPU card selection via torch.npu.set_device() (this host MUST NOT set
  ASCEND_RT_VISIBLE_DEVICES; it triggers aclInit error 107001)

Supports DRY RUN (random weights) and Full Run (real weights).
保存输出: uv run python demo.py --dry-run > output.txt 2>&1
"""

import argparse
import os
import time
from pathlib import Path

import torch

# HuggingFace mirror defaults for CN environments (overridable).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_ID = "BAAI/bge-m3"


def shrink_config_for_dry_run(config):
    """保守缩小：仅将层数减为 2，其余不变，保证通用兼容。"""
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, key):
            old = getattr(config, key)
            new = max(1, min(old, 2))
            setattr(config, key, new)
            if old != new:
                print(f"[Setup] Shrunk {key}: {old} -> {new}")


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def pick_npu_index(count):
    """选卡：优先 NPU_DEVICE_ID 环境变量；否则选空闲 HBM 最多的卡（避免写死 0 号卡）。"""
    env = os.environ.get("NPU_DEVICE_ID")
    if env is not None and env.strip() != "":
        idx = int(env)
        if 0 <= idx < count:
            print(f"[Device] NPU_DEVICE_ID={idx} requested via env")
            return idx
    best, best_free = 0, -1
    for i in range(count):
        try:
            free, total = torch.npu.mem_get_info(i)
        except Exception:
            free, total = 0, 0
        print(f"[Device] NPU {i}: free HBM {free / 1024**3:.1f} GiB / total {total / 1024**3:.1f} GiB")
        if free > best_free:
            best, best_free = i, free
    return best


def get_device():
    """Detects best device (NPU > CUDA > CPU). Returns (device_str, device_count)."""
    # 1. Check for Ascend NPU
    try:
        import torch_npu  # noqa: F401  # registers torch.npu

        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            print("[Device] Huawei Ascend NPU detected.")
            print(f"[Device] NPU count: {count}")
            idx = pick_npu_index(count)
            torch.npu.set_device(idx)
            print(f"[Device] Selected NPU device: npu:{idx}")
            return f"npu:{idx}", count
    except ImportError:
        pass

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0", torch.cuda.device_count()

    # 3. Fallback to CPU
    print("[Device] No accelerator detected, using CPU.")
    return "cpu", 0


def encode(model, tokenizer, sentences, device, max_length=512):
    """Mean-pooled, L2-normalized sentence embeddings."""
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # (B, L, H)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (BAAI/bge-m3, retrieval embedding)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    args = parser.parse_args()

    # Best practice: cache under this adaptation so downloads stay local and reproducible
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    from accelerate import dispatch_model, infer_auto_device_map
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    device, _count = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Tokenizer (Always needed)
    # ============================================================
    print(f"[Setup] Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # ============================================================
    # Step 2: Config + Model Loading (Dry Run vs Real)
    # ============================================================
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    nlayers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, layers={nlayers}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run (reduced layers/size).")
        model = AutoModel.from_config(config)
        device_map = infer_auto_device_map(model)
        # 多卡环境限制单卡：本机严禁 ASCEND_RT_VISIBLE_DEVICES，直接强制所有模块落在所选卡上
        device_map = {name: device for name in device_map}
        model = dispatch_model(model, device_map=device_map)
        print(f"[Setup] Model dispatched to single device: {device}")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        load_kwargs = dict(
            torch_dtype="auto",
            cache_dir=CACHE_DIR,
            device_map="auto",
        )
        if device.startswith("npu"):
            # 限制只使用所选单卡（其余 NPU 设 0GiB，避免 device_map="auto" 跨卡）
            idx = int(device.split(":")[1])
            count = torch.npu.device_count()
            max_memory = {i: ("0GiB" if i != idx else "60GiB") for i in range(count)}
            max_memory["cpu"] = "64GiB"
            load_kwargs["max_memory"] = max_memory
        model = AutoModel.from_pretrained(MODEL_ID, **load_kwargs)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")

    model.eval()

    # ============================================================
    # Step 3: Sentence Encoding + Cosine Similarity
    # ============================================================
    sentences = [
        "A cat is sitting on the mat",  # 0
        "A cat rests on a mat",  # 1 (与 0 语义相近)
        "The stock market crashed yesterday",  # 2 (与 0 无关)
        "一只猫坐在垫子上",  # 3 (0 的中文翻译，验证多语言)
    ]
    print(f"[Run] Encoding {len(sentences)} sentences...")
    for i, s in enumerate(sentences):
        print(f"[Run]   s{i}: {s}")

    start_time = time.time()
    embeddings = encode(model, tokenizer, sentences, first_device)
    encode_time = time.time() - start_time

    # Embedding 形状校验
    expected = (len(sentences), config.hidden_size)
    assert tuple(embeddings.shape) == expected, f"Unexpected embedding shape: {embeddings.shape} vs {expected}"
    print(f"[Run] Embedding shape OK: {tuple(embeddings.shape)}")

    sim = embeddings @ embeddings.t()
    print("[Run] Cosine similarity matrix:")
    for i in range(len(sentences)):
        row = " ".join(f"{sim[i][j].item():+.4f}" for j in range(len(sentences)))
        print(f"[Run]   s{i}: {row}")

    print(f"[Run] Output: sim(s0,s1)={sim[0][1].item():.4f}, sim(s0,s2)={sim[0][2].item():.4f}, sim(s0,s3)={sim[0][3].item():.4f}")
    print(f"[Run] Encoding time: {encode_time:.4f} seconds")

    if not args.dry_run:
        # 真实权重质量断言：相近句 > 无关句
        assert sim[0][1].item() > sim[0][2].item(), (
            f"Quality check failed: similar pair sim {sim[0][1].item():.4f} should exceed unrelated {sim[0][2].item():.4f}"
        )
        print("[Run] Quality check passed: similar pair similarity > unrelated pair similarity")

    print("[Success] Demo completed.")

    # torch_npu 已知问题：部分路径下解释器退出时设备管理线程不结束，进程挂死。
    # 成功后强制干净退出（先 flush，避免重定向输出被截断）。
    if device.startswith("npu"):
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
