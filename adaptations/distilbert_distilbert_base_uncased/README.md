# distilbert/distilbert-base-uncased Ascend NPU Adaptation

## 模型信息

- **Model ID**: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- **架构**: distilbert (DistilBertForMaskedLM)
- **任务**: 完形填空 / 掩码语言模型 (fill-mask)，非生成式 encoder
- **语言**: en
- **规模**: ~66M 参数, 6 层, dim 768, 12 heads, float32

## 验证方式

非生成式模型，不使用 `generate()`；demo 通过**前向 logits + [MASK] 位置 argmax 填空**验证：

- 输入: `The capital of France is [MASK].`
- 校验: logits 形状 (batch, seq, vocab)、数值全为有限值
- 输出: 填充后的句子 + Top-5 预测 token

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU (torch==2.8.0 + torch-npu==2.8.0.post4, 本机验证组合)
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

依赖源：默认走阿里云 PyPI 镜像，torch-npu 来自华为云 PyPI 仓库（见 `pyproject.toml`）。
注意：不要给两个 extra 使用 `[tool.uv] conflicts` 分桶 + `torch>=`，
会让 `--extra ascend` 误装最新版 torch，与 torch-npu ABI 不匹配。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重；层数保守缩小到 2 以加速初始化（`n_layers: 6 -> 2`）。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（float32, ~268MB），模型与 tokenizer 缓存到本目录 `models/`。

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

- transformers 原生支持架构（DistilBertForMaskedLM），无自定义代码。
- tokenizer 为 WordPiece（vocab.txt），无需 sentencepiece/tiktoken。
- 已知限制：dry run 使用随机权重，[MASK] 预测无意义属正常；
  真实填空质量（如预测出 "paris"）属于 full run / benchmark 阶段验证。
