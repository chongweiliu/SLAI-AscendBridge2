"""
Ascend NPU adaptation demo for laion/clap-htsat-fused (audio-text contrastive embeddings).

CLAP 音频-文本对比嵌入（transformers 原生 ClapModel，~400M）。
验证方式：合成正弦波音频 + 文本 -> 各自嵌入 -> 余弦相似度。
音频用 numpy 正弦波，不依赖真实音频文件，也不依赖（本机损坏的）torchaudio；
mel 谱由 transformers.audio_utils 的纯 numpy 实现计算，无需 librosa/scipy。

Supports DRY RUN (random weights, no download) and Full Run (real weights).
保存输出: uv run python demo.py --dry-run > output.txt 2>&1

Host notes (2x Ascend910):
- 本机严禁设置 ASCEND_RT_VISIBLE_DEVICES（会导致 aclInit error 107001），
  选卡一律使用 torch.npu.set_device()；默认自动选空闲 HBM 最多的卡。
- 结尾 flush + os._exit(0) 规避 torch_npu 解释器退出挂死（bge-m3 实案沉淀）。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# HuggingFace 镜像（国内直连 huggingface.co 不通）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import AutoConfig, AutoModel, AutoProcessor  # noqa: E402


def shrink_config_for_dry_run(config):
    """CLAP 复合配置保守缩小：
    - 文本分支层数 -> 2（不改 hidden_size，projection 维度不受影响）
    - 音频分支 (HTSAT, Swin-like)：仅将每阶段深度 -> 1，保持阶段数与空间结构不变
    """
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
        old = text_cfg.num_hidden_layers
        new = max(1, min(old, 2))
        text_cfg.num_hidden_layers = new
        if old != new:
            print(f"[Setup] Shrunk text_config.num_hidden_layers: {old} -> {new}")

    audio_cfg = getattr(config, "audio_config", None)
    if audio_cfg is not None:
        depths = getattr(audio_cfg, "depths", None)
        if depths:
            new_depths = [max(1, min(int(d), 1)) for d in depths]
            if new_depths != list(depths):
                audio_cfg.depths = new_depths
                print(f"[Setup] Shrunk audio_config.depths: {list(depths)} -> {new_depths}")


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def pick_npu_index(count: int, preferred: int | None = None) -> int:
    """选卡：优先 --npu-index / NPU_DEVICE_ID；否则选空闲 HBM 最多的卡（不写死 0 号卡）。"""
    if preferred is not None and 0 <= preferred < count:
        print(f"[Device] Requested NPU index {preferred}")
        return preferred
    env = os.environ.get("NPU_DEVICE_ID", "").strip()
    if env:
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


def get_device(preferred_index: int | None = None):
    """Detects best device. Returns device string like 'npu:0' / 'cuda:0' / 'cpu'."""
    # 1. Check for Ascend NPU
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            count = torch.npu.device_count()
            print("[Device] Huawei Ascend NPU detected.")
            print(f"[Device] NPU count: {count}")
            index = pick_npu_index(count, preferred_index)
            # 本机严禁 ASCEND_RT_VISIBLE_DEVICES，用 set_device 选卡
            torch.npu.set_device(index)
            print(f"[Device] Selected NPU device: npu:{index}")
            return f"npu:{index}"
    except ImportError:
        pass

    # 2. Check for CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0"

    # 3. Fallback to CPU
    print("[Device] No accelerator detected, using CPU.")
    return "cpu"


def make_sine_wave(freq_hz: float = 440.0, seconds: float = 1.0, sr: int = 48000) -> np.ndarray:
    """合成正弦波音频（替代真实音频文件，规避损坏的 torchaudio）。"""
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CLAP (htsat-fused) Ascend Adaptation Demo")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    parser.add_argument("--npu-index", type=int, default=None, help="Preferred NPU index (默认自动选空闲卡)")
    args = parser.parse_args()

    MODEL_ID = "laion/clap-htsat-fused"
    # 缓存固定在本 adaptation 内，避免污染项目根 models/
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    device = get_device(preferred_index=args.npu_index)
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Processor (tokenizer + mel feature extractor, numpy-only)
    # ============================================================
    print(f"[Setup] Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    sampling_rate = getattr(processor.feature_extractor, "sampling_rate", 48000)
    print(f"[Setup] Audio sampling_rate: {sampling_rate} Hz")

    # ============================================================
    # Step 2: Model Loading (Dry Run vs Real)
    # ============================================================
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    text_layers = getattr(getattr(config, "text_config", None), "num_hidden_layers", "?")
    audio_depths = getattr(getattr(config, "audio_config", None), "depths", "?")
    print(f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, text_layers={text_layers}, audio_depths={audio_depths}")

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        print("[Setup] Using minimal config for fast dry-run.")
        model = AutoModel.from_config(config)
        model = model.to(device)
        print(f"[Setup] Model placed on {device}")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
        )
        model = model.to(device)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Setup] Model params: {n_params:.1f}M")

    model.eval()

    # ============================================================
    # Step 3: Audio-text embeddings + cosine similarity
    # ============================================================
    texts = ["the sound of a bell ringing", "a pure tone beep", "a dog barking"]
    audio = make_sine_wave(freq_hz=440.0, seconds=1.0, sr=sampling_rate)
    print(f"[Run] Input texts: {texts}")
    print(f"[Run] Input audio: synthetic sine wave, {len(audio)} samples @ {sampling_rate} Hz")

    inputs = processor(text=texts, audio=[audio], sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    print("[Run] Forward pass (audio + text encoders)...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

    audio_embeds = outputs.audio_embeds
    text_embeds = outputs.text_embeds
    assert audio_embeds is not None and text_embeds is not None, "CLAP forward returned no embeddings"
    assert torch.isfinite(audio_embeds).all() and torch.isfinite(text_embeds).all(), "Embeddings contain NaN/Inf"
    fwd_time = time.time() - start_time
    print(f"[Run] audio_embeds: {tuple(audio_embeds.shape)}, text_embeds: {tuple(text_embeds.shape)}, all finite")

    # 余弦相似度: audio (1, D) vs texts (N, D)
    a = torch.nn.functional.normalize(audio_embeds, dim=-1)
    t = torch.nn.functional.normalize(text_embeds, dim=-1)
    sims = (a @ t.T).squeeze(0)
    top1 = int(sims.argmax())

    print(f"[Run] Cosine similarity per text: {[round(float(s), 4) for s in sims]}")
    print(f"[Run] Output: Top-1 matched text index={top1} -> '{texts[top1]}'")
    print(f"[Run] Forward time: {fwd_time:.4f} seconds")
    if args.dry_run:
        print("[Run] Note: dry-run uses random weights, similarities are meaningless by design.")
    print("[Success] Demo completed.")

    # 规避 torch_npu 解释器退出挂死：显式 flush 后直接退出
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
