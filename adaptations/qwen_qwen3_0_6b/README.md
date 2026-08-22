# Qwen/Qwen3-0.6B Ascend NPU Adaptation

## 模型信息

- **Model ID**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **架构**: qwen3 (`Qwen3ForCausalLM`, GQA 16 heads / 8 KV heads, hidden 1024, 28 层)
- **任务**: 文本生成（causal LM，支持 thinking/no-thinking 对话模式）
- **语言**: 中英为主的多语言
- **许可**: Apache-2.0

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

Python 版本限制为 `>=3.10,<3.13`（torch-npu 兼容性要求）。

本机（aarch64, 2×Ascend910）注意事项：

- **严禁设置 `ASCEND_RT_VISIBLE_DEVICES`**（本机会触发 `aclInit error 107001` / `is_available()=False`）；选卡用 `--device-index`（内部走 `torch.npu.set_device()`），先用 `npu-smi info` 查看空闲卡。
- HF 下载走镜像：`HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`（demo.py 内已 setdefault）。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重；层数保守缩小至 2 以加速初始化。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（bf16，约 1.2GB），模型与 tokenizer 缓存到本目录 `models/`。

### 保存全部输出

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` 与 `--device-index` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `README.md` | 本文件 |
| `models/` | 模型缓存目录（自动创建，仅限本 adaptation 内） |
| `output.txt` | dry-run 运行输出（命令行重定向生成） |
| `.status.json` | 适配状态记录 |

## 适配状态

- **Dry Run**: 见 `.status.json` / `output.txt`
- **Full Run**: 见 `.status.json`
- **设备**: Ascend NPU（单卡；CUDA 亦兼容）

## 备注

- Qwen3 架构需 `transformers>=4.51`；此处 pin `transformers>=4.51,<5.0` 避免 5.x 破坏性变更。
- Qwen3 tokenizer 为 GPT2 风格 BPE（vocab.json + merges.txt），无需 sentencepiece/tiktoken。
- `generate()` 使用 `do_sample=False`，避免 NPU 上 ArgSort 触发 AiCpu 回退。
