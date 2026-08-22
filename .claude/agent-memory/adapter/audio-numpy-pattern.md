---
name: audio-numpy-pattern
description: 音频类模型（CLAP 等）在本机适配时绕开损坏的 torchaudio：numpy 合成音频 + transformers 纯 numpy mel 特征
metadata:
  type: feedback
---

音频类模型适配不要引入 torchaudio/librosa/scipy/soundfile；用 numpy 合成音频 +
transformers 自带的纯 numpy 特征提取即可通过 dry run。

**Why:** 本机 torchaudio 损坏（系统环境记忆里已记录需打桩）；而
`transformers.audio_utils`（spectrogram / mel_filter_bank / window_function）是纯
numpy 实现，ClapProcessor 等可直接吃 `np.ndarray` 波形。laion/clap-htsat-fused
实案验证：440Hz 正弦波 1s @48kHz -> ClapProcessor(text=..., audio=[wave],
sampling_rate=sr, return_tensors="pt", padding=True) -> 模型输出
`outputs.audio_embeds` / `outputs.text_embeds` -> normalize 后 `a @ t.T` 余弦相似度。

**How to apply:**
- 复合配置（CLAP/CLIP 类）缩层：文本分支减 `num_hidden_layers` 到 2；音频分支
  若是 Swin/HTSAT 风格，把 `audio_config.depths` 每阶段降到 1（保持阶段数与空间
  结构，不动通道/分辨率，projection 维度不受影响）。
- 验证指标：嵌入形状 + `torch.isfinite` 断言 + 相似度/Top-1；真实权重下正弦波
  应明显匹配 "pure tone/beep" 类文本（实测 0.24 vs 0.04）。
- 相关：[[uv-torch-abi-trap]]（同目录 pyproject 模式）、[[npu-host-env-quirks]]。
