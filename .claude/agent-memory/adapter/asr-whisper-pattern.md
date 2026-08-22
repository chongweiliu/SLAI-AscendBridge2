---
name: asr-whisper-pattern
description: ASR/SpeechSeq2Seq（Whisper）适配要点：免 torchaudio 合成音频、transformers4.57 旧版 generation_config 补丁、fp16 dtype 对齐
metadata:
  type: project
---

Whisper 类 ASR 适配（2026-08-22 于 openai/whisper-large-v3-turbo 验证通过）。

- **不用 torchaudio**（本机可能损坏）：用 numpy 合成波形（静音+正弦+扫频），
  `WhisperFeatureExtractor`/`processor(waveform, sampling_rate=16000)` 直接吃 float32 数组。
- **dtype 对齐**：processor 输出 float32 mel 特征，fp16 模型在 NPU 上 `conv1d` 报
  `Input type (float) and bias type (Half)`；必须
  `inputs["input_features"] = inputs["input_features"].to(dtype=model_dtype)`。
- **transformers 4.57 三连环坑**（whisper generate）：
  1. 传 `language=/task=` 报 "generation config is outdated" —— 仓库的
     `generation_config.json` 缺 `is_multilingual/lang_to_id/task_to_id`；
  2. 旧写法 `forced_decoder_ids`（`processor.get_decoder_prompt_ids`）在 4.57 已被
     generate 拒绝（"not used by the model"），不能再走；
  3. 正解：运行时给 `model.generation_config` 补三字段，`lang_to_id` 的 key 必须是
     `<|en|>` 这种 token 形式：
     `{f"<|{code}|>": tok.convert_tokens_to_ids(f"<|{code}|>") for code in LANGUAGES}`
     （`from transformers.models.whisper.tokenization_whisper import LANGUAGES`），
     `task_to_id={"transcribe":..., "translate":...}`，必要时补
     `decoder_start_token_id=<|startoftranscript|>`；然后正常传
     `language="english", task="transcribe", return_timestamps=False`。
- **验证预期**：纯静音/正弦音频不含语音，真实权重下 large-v3-turbo no-speech 检测
  输出近空（`.`）属正确行为；随机权重 dry-run 输出乱码 token 即证明解码链路通。
- dry-run 缩层：`encoder_layers` 与 `decoder_layers` 均减为 2。

**Why:** Whisper 在 transformers 4.57 的 generate 兼容坑会连环报错，逐个试错很耗时。
**How to apply:** 适配任何 whisper 系 / multilingual seq2seq ASR 模型时先打
generation_config 补丁再 generate；见 [[vlm-demo-pattern]]、[[dependency-pinning]]。
