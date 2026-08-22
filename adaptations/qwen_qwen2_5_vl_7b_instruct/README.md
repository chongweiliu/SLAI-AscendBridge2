# Qwen/Qwen2.5-VL-7B-Instruct Ascend NPU Adaptation

## 模型信息

- **Model ID**: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- **架构**: `Qwen2_5_VLForConditionalGeneration` (`model_type=qwen2_5_vl`，标准 transformers 模型，非自定义仓库)
- **任务**: image-text-to-text（VLM 多模态：图像 + 文本 -> 文本）
- **语言**: 多语言（中英为主）
- **许可**: Apache-2.0

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
| torchvision | 0.23.0（qwen_vl_utils 依赖） |
| transformers | 4.57.6（Qwen2.5-VL 需 >=4.49.0） |
| qwen-vl-utils | 0.0.14 |

注意：`requires-python = ">=3.10,<3.13"`，因为 torch_npu 目前最高支持 Python 3.12。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

- 不下载权重；`from_config` 随机初始化，文本塔层数与视觉塔深度均保守缩小为 2。
- 仍走完整多模态路径：`AutoProcessor` + `qwen_vl_utils.process_vision_info` 处理本地图像（448x448 渐变 + 红色方块，代码内生成，不依赖外部下载）+ `model.generate()`。

### Full Run（真实权重）

```bash
uv run python demo.py
```

- 加载预训练权重（5 个 safetensors 分片，~15.5GB），缓存到本目录 `models/`。
- 单卡推理（7B bf16 ~16GB，单卡 64GB 足够；未用 `device_map="auto"`，避免与其他并行任务跨卡干扰）。

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
- **VLM 输入路径**: `processor.apply_chat_template` 生成含视觉占位符的对话文本，
  `qwen_vl_utils.process_vision_info` 解析图像，`processor(text=..., images=..., videos=...)`
  统一编码后送入模型。
- **generate**: `do_sample=False`（贪心），避免 NPU 上采样链路的 AiCpu 回退告警。
- **HF 镜像**: `demo.py` 内 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`
  与 `HF_HUB_DISABLE_XET=1`（官方源直连不通 / 镜像不支持 Xet 协议）；外部显式设置的环境变量优先。
- **兼容性**: transformers>=4.57 将 `from_config` 改名为 `_from_config`，dry-run 分支做了双路径兼容；
  权重加载使用 `torch_dtype=torch.bfloat16`（4.57 提示未来改用 `dtype`，仅为弃用警告）。

## 适配状态

- **Dry Run**: 通过（npu:1，随机权重多模态 generate 32 tokens，见 `output.txt`）
- **Full Run**: 通过（真实权重图像描述生成，见 `output_full.txt`）
- **设备**: Ascend NPU（单卡）
