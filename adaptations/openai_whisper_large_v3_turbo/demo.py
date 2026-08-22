"""
Ascend NPU adaptation demo for openai/whisper-large-v3-turbo.

SpeechSeq2Seq (automatic-speech-recognition): WhisperForConditionalGeneration + AutoProcessor.
验证音频不依赖任何音频文件 / torchaudio，直接用 numpy 合成正弦波 + 静音张量。
Supports DRY RUN (random weights, no weight download) and Full Run (real weights).

保存输出: uv run python demo.py --dry-run > output.txt 2>&1
"""

import argparse
import os
import time
from pathlib import Path

# 国内网络：HuggingFace 官方源直连不通，统一走镜像；禁用 Xet 协议（镜像不支持）。
# 使用 setdefault，外部显式设置的环境变量优先。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401  # 注册 torch.npu
except ImportError:
    pass

from transformers import AutoConfig, AutoProcessor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000


def shrink_config_for_dry_run(config):
    """保守缩小：encoder/decoder 层数均减为 2，其余不变，保证架构路径一致。"""
    for key in ("encoder_layers", "decoder_layers"):
        old = getattr(config, key, None)
        if old is not None:
            new = max(1, min(old, 2))
            if new != old:
                setattr(config, key, new)
                print(f"[Setup] Shrunk {key}: {old} -> {new}")


# ============================================================
# Device Selection Logic (NPU > CUDA > CPU)
# ============================================================
def get_device():
    """Detects best device. Returns (device_str, device_count).

    本机约定：严禁设置 ASCEND_RT_VISIBLE_DEVICES（会导致 aclInit error 107001），
    多卡时用 torch.npu.set_device() 选卡；挑当前空闲显存最多的卡，避免与其他任务抢卡。
    """
    # 1. Ascend NPU
    if hasattr(torch, "npu") and torch.npu.is_available():
        count = torch.npu.device_count()
        print("[Device] Huawei Ascend NPU detected.")
        print(f"[Device] NPU count: {count}")
        best, best_free = 0, -1
        for i in range(count):
            free, total = torch.npu.mem_get_info(i)
            print(f"[Device] npu:{i} free={free / 1e9:.1f}GB / total={total / 1e9:.1f}GB")
            if free > best_free:
                best, best_free = i, free
        torch.npu.set_device(best)
        return f"npu:{best}", count

    # 2. NVIDIA CUDA
    if torch.cuda.is_available():
        print("[Device] NVIDIA CUDA detected.")
        return "cuda:0", torch.cuda.device_count()

    # 3. CPU fallback（验收不允许，后续断言会拦截）
    print("[Device] No accelerator detected, using CPU.")
    return "cpu", 0


def patch_whisper_generation_config(model, processor):
    """transformers>=4.57 传 language=/task= 需要 generation_config 携带
    is_multilingual/lang_to_id/task_to_id；旧版 generation_config（或 dry-run 随机权重时
    根本没有该文件）缺少这些字段，运行时用 tokenizer 词表补齐。"""
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    gc = model.generation_config
    if hasattr(gc, "is_multilingual") and hasattr(gc, "lang_to_id") and hasattr(gc, "task_to_id"):
        return
    tok = processor.tokenizer
    gc.is_multilingual = True
    gc.task_to_id = {t: tok.convert_tokens_to_ids(f"<|{t}|>") for t in ("transcribe", "translate")}
    # 注意：transformers 4.57 要求 lang_to_id 的 key 是 token 形式（如 "<|en|>"）
    gc.lang_to_id = {f"<|{code}|>": tok.convert_tokens_to_ids(f"<|{code}|>") for code in LANGUAGES}
    if getattr(gc, "decoder_start_token_id", None) is None:
        gc.decoder_start_token_id = tok.convert_tokens_to_ids("<|startoftranscript|>")
    print("[Setup] Patched outdated generation_config (is_multilingual/lang_to_id/task_to_id)")


def make_synthetic_audio(seconds: float = 4.0) -> np.ndarray:
    """合成测试音频：静音 + 440Hz 正弦波 + 线性扫频 + 静音。

    不依赖任何外部音频文件或 torchaudio（本机 torchaudio 可能损坏）。
    """
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    wave = np.zeros(n, dtype=np.float32)
    seg = n // 4
    # 段1: 静音；段2: 440Hz 正弦；段3: 200->800Hz 扫频；段4: 静音
    wave[seg : 2 * seg] = 0.5 * np.sin(2 * np.pi * 440 * t[seg : 2 * seg])
    f = np.linspace(200.0, 800.0, n - 3 * seg, dtype=np.float32)
    wave[2 * seg : 3 * seg] = 0.5 * np.sin(2 * np.pi * f * t[2 * seg : 3 * seg])
    return wave


# ============================================================
# Main Logic
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ascend Adaptation Demo (Whisper ASR)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry run mode (random weights)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    MODEL_ID = "openai/whisper-large-v3-turbo"
    # 缓存固定在本 adaptation 的 models/ 下，避免写入项目根或其他目录
    CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
    print(f"[Setup] Model cache: {CACHE_DIR}")

    device, _ = get_device()
    assert device.startswith(("npu", "cuda")), f"Need NPU or CUDA, got device={device}"
    print(f"[Setup] Using device: {device}")

    # ============================================================
    # Step 1: Processor (WhisperFeatureExtractor + tokenizer)
    # ============================================================
    print(f"[Setup] Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # ============================================================
    # Step 2: Model Loading (Dry Run vs Real)
    # ============================================================
    config = AutoConfig.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print(
        f"[Setup] Config loaded: model_type={getattr(config, 'model_type', '?')}, "
        f"encoder_layers={getattr(config, 'encoder_layers', '?')}, "
        f"decoder_layers={getattr(config, 'decoder_layers', '?')}"
    )

    if args.dry_run:
        print("[Setup] DRY RUN MODE ENABLED: Initializing model with RANDOM weights.")
        shrink_config_for_dry_run(config)
        # transformers>=4.57 将公开的 from_config 改名为 _from_config，做兼容
        if hasattr(WhisperForConditionalGeneration, "from_config"):
            model = WhisperForConditionalGeneration.from_config(config, torch_dtype=torch.float16)
        else:
            model = WhisperForConditionalGeneration._from_config(config, torch_dtype=torch.float16)
        model.to(device)
        print("[Setup] Random-weight model created and moved to device.")
    else:
        print("[Setup] Attempting to load pre-trained weights...")
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
        )
        # 809M fp16 ~1.6GB，单卡足够；不用 device_map="auto"，避免跨卡与并行任务互相干扰
        model.to(device)
        print("[Setup] Pre-trained weights loaded")

    first_device = next(model.parameters()).device
    assert first_device.type in ("npu", "cuda"), f"Model must be on NPU or CUDA, got {first_device}"
    print(f"[Setup] Model on device: {first_device}")

    model.eval()

    # ============================================================
    # Step 3: ASR Inference Test (synthetic waveform -> text)
    # ============================================================
    waveform = make_synthetic_audio(4.0)
    print(f"[Run] Input: synthetic waveform {len(waveform)} samples @ {SAMPLE_RATE}Hz (silence + 440Hz tone + sweep + silence)")

    inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(first_device)
    # processor 输出 float32 特征，模型权重为 fp16，需对齐 dtype，否则 conv1d 报 dtype mismatch
    model_dtype = next(model.parameters()).dtype
    inputs["input_features"] = inputs["input_features"].to(dtype=model_dtype)
    print(f"[Run] Processed inputs: input_features={tuple(inputs.input_features.shape)} (dtype={model_dtype})")

    print("[Run] Generating (transcribing)...")
    patch_whisper_generation_config(model, processor)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            language="english",
            task="transcribe",
            return_timestamps=False,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_time = time.time() - start_time

    assert outputs is not None and outputs.shape[0] > 0, "generate() returned empty output"
    new_tokens = outputs[0]
    assert new_tokens.numel() > 0, "generate() produced no tokens"
    transcript = processor.batch_decode(new_tokens.unsqueeze(0), skip_special_tokens=True)[0].strip()

    print(f"[Run] Generated {new_tokens.numel()} tokens")
    print(f"[Run] Output: {transcript}")
    print(f"[Run] Generation time: {gen_time:.4f} seconds")
    print("[Success] Demo completed.")


if __name__ == "__main__":
    main()
