# openai/whisper-large-v3-turbo Ascend NPU Adaptation

## 模型信息

- **Model ID**: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- **架构**: `WhisperForConditionalGeneration`（`model_type=whisper`，标准 transformers 模型）
- **任务**: automatic-speech-recognition（SpeechSeq2Seq：音频 -> 文本，多语言 99 种）
- **规模**: ~809M（encoder 32 层 / decoder 4 层，d_model=1280）
- **许可**: MIT

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

本机（Ascend910 + CANN 25.5.5）已验证的 ascend 组合：

| 包 | 版本 |
|----|------|
| Python | 3.12 |
| torch | 2.8.0 |
| torch-npu | 2.8.0.post5 |
| transformers | 4.57.6 |

注意：`requires-python = ">=3.10,<3.13"`（torch_npu 最高支持 Python 3.12）。
**不依赖 torchaudio**（本机 torchaudio 可能损坏）：验证音频在 `demo.py` 内用
numpy 合成（静音 + 440Hz 正弦 + 线性扫频），`WhisperFeatureExtractor` 直接消费原始
float32 波形。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

- 不下载权重；`from_config` 随机初始化，encoder/decoder 层数均保守缩小为 2。
- 完整走 feature extraction -> encoder -> greedy generate 路径。

### Full Run（真实权重）

```bash
uv run python demo.py
```

- 加载预训练权重（单个 `model.safetensors`，~1.6GB），缓存到本目录 `models/`。
- 单卡推理（809M fp16 ~1.6GB）。

### 保存全部输出

```bash
uv run python demo.py --dry-run > output.txt 2>&1   # dry run 产物（output.txt 以此为准）
uv run python demo.py > output_full.txt 2>&1        # full run 产物
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` 与 `--max-new-tokens` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `README.md` | 本说明 |
| `models/` | 模型/分词器缓存目录（自动创建） |
| `output.txt` | dry-run 运行输出 |
| `output_full.txt` | full-run（真实权重）运行输出 |
| `.status.json` | 适配状态记录 |

## 适配要点与设备约定

- **设备检测**: NPU > CUDA，验收不允许 CPU 回退；`demo.py` 内置断言。
- **选卡**: 本机 **严禁** 设置 `ASCEND_RT_VISIBLE_DEVICES`（会导致 `aclInit error 107001`）；
  `demo.py` 运行时遍历 `torch.npu.mem_get_info()` 挑选当前空闲显存最多的卡，并用
  `torch.npu.set_device()` 绑定。
- **dtype 对齐**: processor 输出 float32 mel 特征，模型为 fp16；`demo.py` 显式把
  `input_features` 转成模型 dtype，否则 NPU 上 `conv1d` 报 dtype mismatch。
- **generation_config 旧版兼容（transformers 4.57）**: 该仓库的 `generation_config.json`
  缺少 `is_multilingual/lang_to_id/task_to_id`，直接传 `language=/task=` 会报
  "generation config is outdated"，且 4.57 已移除 `forced_decoder_ids`。`demo.py` 在运行时
  用分词器词表（`LANGUAGES` 常量 + `convert_tokens_to_ids`）补齐这三个字段
  （`lang_to_id` 的 key 必须是 `<|en|>` 这种 token 形式），随后正常传
  `language="english", task="transcribe", return_timestamps=False`。
- **generate**: `do_sample=False`（贪心），避免 NPU 上采样链路的 AiCpu 回退告警。
- **HF 镜像**: `demo.py` 内 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`
  与 `HF_HUB_DISABLE_XET=1`；外部显式设置的环境变量优先。
- **兼容性**: transformers>=4.57 将 `from_config` 改名为 `_from_config`，dry-run 分支做了双路径兼容。

## 验证结果说明

- 合成音频不含真实语音（静音 + 440Hz 正弦 + 扫频），full run 下 large-v3-turbo 的
  no-speech 检测正确工作，输出近空转录（`.`），属预期行为；dry run（随机权重）下
  输出 64 个乱码 token，证明完整解码链路可用。

## 适配状态

- **Dry Run**: 通过（npu:1，随机权重 generate 64 tokens，见 `output.txt`）
- **Full Run**: 通过（真实权重转录合成音频，见 `output_full.txt`）
- **设备**: Ascend NPU（单卡）
