# 特殊模型处理详解：MoE 大模型 / ESPnet ASR / TTS

（从 MEMORY.md 索引展开的细节文档）

## MoE / 大模型 benchmark 策略

**Qwen3-Coder-Next-Base (512 experts MoE)**:
- flash-linear-attention 未安装时退回纯 PyTorch 实现，极慢
- 单样本 forward ~112s (NPU x4, 41GB peak), generate ~18s/sample (max_new_tokens=8, truncation=128)
- 50 samples (含前10个PPL计算) 总耗时 ~10min (Step1 ~4min + Step2 ~7min)
- 优化: 使用 `generate(return_dict_in_generate=True, output_scores=True)` 合并 forward+generate
- 仅前10个样本做额外 forward 计算 PPL
- 使用 `local_files_only=True` 避免 HuggingFace 在线访问

## ESPnet ASR 模型特殊处理

**重要**: ESPnet ASR 模型（如 reazonspeech-espnet-v2）不使用 transformers 标准模型，
而是使用 ESPnet 库的 `Speech2Text` 类。

**特点**:
1. 模型加载需要从 HuggingFace 下载 config 和 checkpoint 文件
2. 使用 `Speech2Text` 类而非 `AutoModelForSpeechSeq2Seq`
3. 需要安装 `espnet` 和 `librosa` 依赖（通过 `pip install librosa==0.10.0`）
4. 与 Python 3.12 不兼容（llvmlite 需要 Python 3.9-3.11）

**解决方案**:
1. 添加 `--cpu` 参数支持强制 CPU 推理
2. 在 `load_benchmark_audio` 函数中使用 librosa 生成合成音频
3. 不使用 transformers 标准模型，而是使用 ESPnet 的 `Speech2Text` 类

**示例**:
```python
from espnet2.bin.asr_inference import Speech2Text

speech2text = Speech2Text(
    asr_train_config="https://huggingface.co/.../config.yaml",
    asr_model_file="https://huggingface.co/.../valid.acc.ave_10best.pth",
    device=str(torch_device),
)

result = speech2text(audio)  # 返回 List[Tuple]
transcription = result[0][0] if result else ""
```

**兼容性问题**:
- llvmlite==1.36.0 需要 Python 3.9-3.11，在 Python 3.12 上安装失败
- librosa 安装可能需要编译依赖

**建议**: 对 ESPnet 模型使用 `--cpu` 参数在 CPU 上运行评测

## TTS 模型特殊处理（Qwen3-TTS 等）

**重要**: TTS 模型使用 `qwen-tts` 库加载，不走标准 transformers 模板。

**特点**:
1. 使用 `Qwen3TTSModel.from_pretrained()` 加载预训练模型
2. 使用 `Qwen3TTSForConditionalGeneration(config)` 做架构验证（config mode）
3. 推理用 `model.generate_custom_voice(text=..., speaker=..., language=...)`
4. 无需外部数据集，用内置 TTS prompts 即可（dataset_name = "synthetic"）
5. `qwen-tts` 内部依赖 `torchaudio.compliance.kaldi`（需要 ONNX runtime）

**NPU 兼容性问题（已知）**:
- `qwen-tts` 的 `generate_custom_voice` 在 NPU 上因 `MultinomialWithReplacement`
  AICPU kernel 崩溃（errcode 0x2a）
- 设置 `do_sample=False` 不能解决，因为内部 `code_predictor.generate()`
  有独立的采样逻辑
- CPU 模式可用但极慢（单样本 >820s），ONNX tokenizer 开销巨大
- **建议**: TTS 模型在 NPU 上只能跑 config mode；pretrained 模式需要 CPU 或 CUDA

**torchaudio 在 NPU 环境的兼容性**:
- NPU 环境无 CUDA runtime，`torchaudio` 的 C 扩展加载失败（`libcudart.so.13` 缺失）
- 可通过 patch `torchaudio/_extension/__init__.py`，将 `_load_lib("_torchaudio")`
  包在 try/except OSError 中解决
- patch 后 `torchaudio._IS_TORCHAUDIO_EXT_AVAILABLE = False`，`torchaudio`
  可导入但 C 扩展不可用

**accuracy_run.py 编写要点**:
- 变量名用小写 `dataset_name`（不要用 `DATASET_NAME`），因为
  check_accuracy_run.py 的 regex 匹配 `{dataset_name}`
- output_type 用 `tts_audio_stats`（pretrained）或 `tts_architecture_test`（config）
- outputs_*.pt 格式：`{"tts_audio_stats": [dict, ...]}` 或
  `{"tts_architecture_outputs": [dict, ...]}`
- config 模式下做 talker forward pass 验证架构
