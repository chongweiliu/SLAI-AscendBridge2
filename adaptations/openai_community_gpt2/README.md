# openai-community/gpt2 Ascend NPU Adaptation

## 模型信息

- **Model ID**: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- **架构**: gpt2 (GPT-2, ~124M 参数, 12 层, n_layer=12)
- **任务**: text-generation (AutoModelForCausalLM)
- **语言**: en
- **License**: MIT

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

- Python `>=3.12,<3.13`（torch_npu 2.8.x 支持 3.12）
- ascend extra 固定为 `torch==2.8.0` + `torch-npu==2.8.0.post4`，与本机
  CANN 25.5.5 / Ascend910 验证过的系统组合保持一致

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重。层数保守缩小到 2 以加速初始化。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（~550MB），模型与 tokenizer 缓存到本目录 `models/`。

### 保存全部输出

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

## 设备选择

- NPU > CUDA > CPU 自动检测；必须运行在 NPU 或 CUDA 上（CPU 会断言失败）。
- 本机为 2×Ascend910，**不设置** `ASCEND_RT_VISIBLE_DEVICES`（本机一设置
  就会 `aclInit error 107001`）；demo.py 通过 `torch.npu.mem_get_info()`
  挑选空闲 HBM 最多的卡，并用 `torch.npu.set_device()` 绑定。

## 网络

- 默认 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`
  （demo.py 内 setdefault，外部已设置的环境变量优先）。
- 模型缓存固定为 `adaptations/openai_community_gpt2/models/`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `models/` | 模型缓存目录（自动创建） |
| `output.txt` | 运行输出（命令行重定向生成） |
| `.status.json` | 适配状态记录 |

## 适配状态

- **Dry Run**: 见 `.status.json` / `output.txt`
- **Full Run**: 待验证
- **设备**: Ascend NPU（单卡）/ CUDA
