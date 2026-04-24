---
name: media_tek_breeze_asr25_failure
description: MediaTek-Research/Breeze-ASR-25 优化失败：ASR 音频数据集依赖 torchcodec（NVIDIA only）+ step1 profiling 导致无法验证 speedup
type: project
---

# MediaTek-Research/Breeze-ASR-25 优化失败记录

## 基本信息
- **model_id**: MediaTek-Research/Breeze-ASR-25
- **模型类型**: Whisper-based ASR (~800M params)
- **失败时间**: 2026-03-28
- **失败原因**: baseline_artifact_invalid（无法生成有效的独立 baseline 用于 speedup 对比）

## 失败根因

### 1. torchcodec 依赖（致命）
- librispeech 数据集需要 `torchcodec` 库来解码音频
- `torchcodec` 内部依赖 NVIDIA CUDA 运行时库 `libnvrtc.so.13`
- Ascend NPU 环境没有 NVIDIA CUDA 运行时，导致 `torchcodec` 无法加载
- 错误信息: `OSError: libnvrtc.so.13: cannot open shared object file`
- **影响**: 无法加载 librispeech 数据集（2620 样本），只能使用 builtin 合成音频（10 样本）

### 2. step1 profiling 导致 latency_s 失真
- `accuracy_run.py` 的 step1 使用 `torch.profiler.profile` 进行性能分析
- profiling 的 step1 latency 包含大量额外开销（~4.5s）
- 而真实推理延迟约为 0.56s（cold）
- step1 的 `latency_s=4.53` 和 step2 的 `TTFT avg=404.9ms` 口径不一致
- **影响**: 无法用 artifact 中的 latency_s 与其他 baseline/perf 进行公平对比

### 3. builtin 数据集只有 10 样本
- completion gate 要求 num_samples >= 50
- builtin 合成音频最多只能提供 10 样本
- **影响**: 无法满足 completion gate 的最低样本数要求

### 4. 无法生成有效的独立 baseline
尝试的解决方案及结果:
- **创建 symlink** `librispeech_asr___clean` → `librispeech_asr/clean`: 失败（格式不兼容 `load_from_disk`）
- **改用 openslr 数据集路径**: 失败（torchcodec 不可用）
- **安装 torchcodec**: 失败（libnvrtc.so.13 缺失）
- **不带 warmup 运行 baseline**: 生成 artifact 但 latency_s=4.53（profiling 过载），无法与 perf 的 0.198s 对比
- **带 warmup=3 运行 baseline**: warmup=3 不对称，无法用于独立 baseline 对比

## 技术细节

### Whisper 架构确认
```
Whisper (encoder-decoder):
- encoder: pre-norm + GELU + WhisperAttention (relative positional bias)
- decoder: pre-norm + GELU + CrossAttention (relative positional bias)
- positional: sinusoidal embeddings (不是 RoPE)
```

### 融合算子适用性
| 算子 | 适用性 | 原因 |
|------|--------|------|
| npu_rms_norm | N/A | 使用 nn.LayerNorm，不是 RMSNorm |
| npu_swiglu | N/A | 使用 GELU 激活，不是 SiLU |
| npu_rotary_mul | N/A | Whisper 使用 sinusoidal positional embeddings，不是 RoPE |
| npu_fusion_attention | N/A | fp32 精度退化；Whisper 使用自定义 attention + relative positional bias |
| npu_add_layer_norm | N/A | pre-norm 架构，LN 在 attn/FFN 之前，add 在之后 |
| npu_gelu | N/A | baseline NPU F.gelu 已经使用 tanh 近似 |

### 优化验证
- **warmup(3x) + TASK_QUEUE_ENABLE=1**: 有效（通过 self-baseline 确认）
- **self-baseline speedup**: cold=0.760s → warm=0.198s = **3.84x**
- **independent baseline speedup**: 无法验证（无法生成有效 baseline）

## 产出状态
| 产物 | 状态 | 说明 |
|------|------|------|
| model_files/ | N/A | 无需（所有 fusion ops N/A） |
| accuracy_run_perf.py | ✅ 存在 | 包含 warmup + TQE |
| benchmark_metrics_*_perf.json | ✅ 存在 | num_samples=1, self-baseline 3.84x |
| optimization_notes.json | ✅ 存在 | speedup_ratio=1.016, 但无法验证 |
| baseline artifact | ⚠️ 存在但无效 | num_samples=1, profiling 导致 latency_s 失真 |

## 教训
1. **ASR 模型先检查音频依赖**: 在 benchmark 阶段应确认 torchcodec 是否可用
2. **step1 profiling artifact 不能用于 speedup 对比**: step1 的 profiling overhead 使 latency_s 严重失真
3. **builtin 数据集样本数不足**: ASR 模型的 builtin 数据集通常只有 10 样本
4. **不要删除旧的 baseline artifact**: 在确认新 artifact 有效之前，不要删除旧的 artifact
5. **torchcodec 是 NVIDIA-only**: Ascend NPU 上无法使用 torchcodec

## 推荐解决方案
1. **选项 A（推荐）**: benchmark-runner 修复 `accuracy_run.py` 的音频加载逻辑，使用不依赖 torchcodec 的方式（如 librosa + soundfile）
2. **选项 B**: 提供预处理的 librispeech numpy 数组数据集（.npz 格式）
3. **选项 C**: 接受 self-baseline comparison 作为证据（warmup cold vs warm），标记为 runtime_only completed
