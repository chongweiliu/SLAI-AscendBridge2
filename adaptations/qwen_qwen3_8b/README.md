# Qwen/Qwen3-8B Ascend NPU Adaptation

## 模型信息

- **Model ID**: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **架构**: `qwen3`（`Qwen3ForCausalLM`，dense decoder-only，GQA：32 heads / 8 KV heads，head_dim=128）
- **任务**: text-generation（因果语言模型，非 instruct 基座）
- **语言**: 多语言（以中英文为主）
- **规模**: 约 8.2B 参数，36 层，hidden 4096，intermediate 12288，vocab 151936，bf16

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

关键版本约束：

- `transformers>=4.51,<5.0`（Qwen3 架构自 4.51 起原生支持；规避 5.x 兼容风险）
- `torch>=2.6.0,<2.9` + 匹配的 `torch-npu`（ascend extra）
- `requires-python >=3.10,<3.13`（torch-npu 发行版限制）

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

仅验证架构与代码路径，不下载权重；层数保守缩小到 2 以加速初始化。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（约 16GB，bf16），模型与 tokenizer 缓存到本目录 `models/`；
NPU 下通过 `max_memory` 限制只使用所选单卡。

## 本机（Ascend）注意事项

- **严禁设置 `ASCEND_RT_VISIBLE_DEVICES`**（本机会触发 `aclInit error 107001` /
  `torch.npu.is_available()=False`）；demo.py 用 `torch.npu.set_device()` 选卡。
- 选卡逻辑：优先环境变量 `NPU_DEVICE_ID`，否则自动选空闲 HBM 最多的卡（不写死 0 号卡）。
- HuggingFace 走镜像：默认 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`
  （demo.py 内 `os.environ.setdefault`，可被外部环境变量覆盖）。
- `generate()` 使用 `do_sample=False`，避免 NPU 上采样路径的 AiCpu 回退告警。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `models/` | 模型缓存目录（自动创建） |
| `output.txt` | Dry run 运行输出（重定向生成） |
| `.status.json` | 适配状态记录 |

## 适配状态

- **Dry Run**: 已验证（见 `.status.json` / `output.txt`）
- **Full Run**: 可选（需下载约 16GB 权重）
- **设备**: Ascend NPU / CUDA 双栈；NPU 多卡时固定单卡运行
