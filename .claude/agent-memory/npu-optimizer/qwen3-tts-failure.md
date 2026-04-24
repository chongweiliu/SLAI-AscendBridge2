# Qwen3-TTS 任务失败：qwen_tts/CUDA 依赖问题

## 失败时间
2026-03-24

## 任务
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice NPU 优化

## 失败原因
**环境不兼容**：`qwen_tts` pip 包有硬依赖 `torchaudio`，而 `torchaudio` 需要 CUDA 运行时库（`libcudart.so.13`）。当前环境是 NPU-only，没有 CUDA 运行时。

## 完整调用链
```
qwen_tts/__init__.py
  -> qwen_tts.inference.qwen3_tts_model
    -> qwen_tts.core.models
      -> qwen_tts.core.__init__
        -> tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1
          -> vq.speech_vq (line 24: import torchaudio.compliance.kaldi as kaldi)
```

## 错误信息
```
OSError: libcudart.so.13: cannot open shared object file
```

## 已完成的工作
1. `model_files/` 包含 npu_rms_norm + npu_swiglu 优化
2. `accuracy_run_perf.py` 已重写为使用正确的 2D 张量格式（修复了 step2 的 dimension error）

## 无法完成的工作
1. Pretrained 模式 benchmark（被 torchaudio/CUDA 依赖阻塞）
2. 有效的 benchmark_metrics（需要 pretrained 模式）
3. 有效的 optimization_notes.json（当前是 mode="config"，不满足完成标准）

## 结论
如果任务需要 pretrained 权重验证，但环境无法加载 qwen_tts 包，则任务无法完成。

## 可能的解决方向（未经测试）
1. 在有 CUDA 支持的环境中运行此任务
2. 卸载/屏蔽 torchaudio 并修改 qwen_tts 源码去除该依赖（不推荐修改环境）
3. 使用不依赖 qwen_tts 的方式直接加载模型（但 tokenizer 功能会受限）