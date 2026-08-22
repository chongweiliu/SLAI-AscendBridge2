# Qwen/Qwen2.5-1.5B-Instruct Ascend NPU Adaptation

## 模型信息

- **Model ID**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **架构**: qwen2 (Qwen2ForCausalLM)
- **任务**: 文本生成 / 对话 (text-generation, chat)
- **语言**: 多语言 (en, zh 等)
- **规模**: ~1.5B 参数, 28 层, hidden 1536, GQA (12 heads / 2 KV heads), bfloat16

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU (torch==2.8.0 + torch-npu==2.8.0.post4, 本机验证组合)
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

依赖源：默认走阿里云 PyPI 镜像，torch-npu 来自华为云 PyPI 仓库（见 `pyproject.toml`）。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重；层数保守缩小到 2 以加速初始化。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（bfloat16），模型与 tokenizer 缓存到本目录 `models/`。

### 保存全部输出

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

## 本机运行注意（2x Ascend910）

- **严禁设置 `ASCEND_RT_VISIBLE_DEVICES`**（本机一设就 `aclInit error 107001` / `is_available=False`）；
  选卡用 `--npu-index`（内部调用 `torch.npu.set_device()`），运行前先 `npu-smi info` 查看占用。
- HuggingFace 走镜像：`HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`（demo.py 已默认设置）。
- 模型缓存固定在本目录 `models/`，不写项目根 `models/`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` / `--npu-index` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `README.md` | 本文件 |
| `.status.json` | 适配状态记录 |
| `output.txt` | dry run 运行输出（命令行重定向生成） |
| `models/` | 模型缓存目录（自动创建） |

## 适配状态

- **Dry Run**: 通过（NPU）
- **Full Run**: 待验证
- **设备**: Ascend 910（单卡；逻辑卡号经 `torch.npu.set_device()` 选择）

## 备注

- Qwen2.5 系列为 transformers 原生支持架构（Qwen2ForCausalLM），无需 trust_remote_code 自定义代码，
  保留该参数仅为兼容。
- tokenizer 为 BPE（vocab.json + merges.txt），依赖 `tiktoken` 加速。
- 已知限制：dry run 使用随机权重，输出无意义属正常；输出质量验证属于 full run / benchmark 阶段。
