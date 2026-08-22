# Qwen/Qwen2.5-7B-Instruct Ascend NPU Adaptation

## 模型信息

- **Model ID**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **架构**: qwen2 (`Qwen2ForCausalLM`，GQA 28 heads / 4 KV heads，hidden 3584，28 层)
- **任务**: text-generation（chat / instruct）
- **参数规模**: ~7.6B
- **语言**: 多语言（中英为主）
- **许可**: Apache-2.0（非 gated，无需授权）

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

版本锁定说明（ascend extra）：

- `torch==2.8.0` + `torch-npu==2.8.0.post4`：与本机 CANN 25.5.5（npu-smi 25.5.5）实测匹配的已知可用组合，aarch64 wheel 来自 `repo.huaweicloud.com` 镜像。
- `transformers>=4.45,<5.0`：Qwen2 架构在 transformers 4.x 树内原生支持；锁定 4.x 规避 5.x 破坏性变更。
- `tiktoken`：Qwen2 tokenizer 依赖。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重；层数保守缩小至 2，随机权重以 bf16 初始化。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（约 15.2GB，4 个 safetensors 分片），`device_map="auto"` 支持多卡；
模型与 tokenizer 缓存到本目录 `models/`。

### 保存全部输出

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

## 本机（2×Ascend910）注意事项

- **严禁设置 `ASCEND_RT_VISIBLE_DEVICES`**：本机一设就 `aclInit error 107001` / `torch.npu.is_available()=False`。
- demo.py 通过 `torch.npu.mem_get_info()` 选择空闲 HBM 最多的卡，并用 `torch.npu.set_device()` 选定（不写死 0 号卡）。
- HF 访问走镜像：`HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`（demo.py 内已 setdefault）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `README.md` | 本说明 |
| `.status.json` | 适配状态记录 |
| `models/` | 模型缓存目录（自动创建，仅限本 adaptation 内） |
| `output.txt` | dry run 运行输出 |

## 适配状态

- **Dry Run**: 通过（NPU，随机权重完整 generate）
- **Full Run**: 待验证（需下载 ~15.2GB 权重）
- **设备**: Ascend NPU（多卡时 full run 走 `device_map="auto"`）
