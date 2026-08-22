# Qwen/Qwen3.8-27B Ascend NPU Adaptation

## 模型信息

- **Model ID**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **架构**: qwen3_5 (`Qwen3_5ForConditionalGeneration`) — 多模态 (image-text-to-text)
  - 视觉编码器（ViT, depth 27）
  - 文本主干：混合注意力，64 层（48 linear_attention + 16 full_attention），hidden 5120，24 heads / 4 KV heads，head_dim 256，mrope
- **规模**: 约 27B 参数，bf16 权重约 54GB（18 个 safetensors 分片）
- **任务**: 图文多模态理解与文本生成；本 demo 验证文本生成路径（覆盖文本主干全部算子）
- **许可**: Apache-2.0

## 版本要求（重要）

Qwen3.8 属 `qwen3_5` 架构（官方标注 transformers 5.8.0.dev0 构建），需要
**transformers >= 5.15.1**（4.57.x 及更早版本没有 qwen3_5 模型定义）。
本 adaptation pin `transformers>=5.15.1,<6`。

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

Python 版本限制为 `>=3.10,<3.13`（torch-npu 兼容性要求）。

本机（aarch64, 2×Ascend910 64GB）注意事项：

- **严禁设置 `ASCEND_RT_VISIBLE_DEVICES`**（本机会触发 `aclInit error 107001` / `is_available()=False`）；选卡用 `--device-index`（内部走 `torch.npu.set_device()`），先用 `npu-smi info` 查看空闲卡。
- HF 下载走镜像：`HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`（demo.py 内已 setdefault）。
- 本 venv **不安装 torchaudio**（本机 torchaudio 损坏）；transformers 对 torchaudio 的导入有可用性守卫，文本路径不受影响。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重；文本主干层数保守缩小 64 -> 4（保留 1 个 full_attention 层），视觉塔深度缩小至 2，避免大模型 OOM。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（27B, bf16, 约 54GB）。单卡 64GB HBM 紧张，Full Run 使用
`device_map="auto"` 跨卡分片加载（本机 2×910 共 128GB）；模型与 tokenizer 缓存到本目录 `models/`（约 54GB 磁盘）。

### 保存全部输出

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` 与 `--device-index`；文本生成路径 |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `README.md` | 本文件 |
| `models/` | 模型缓存目录（自动创建，仅限本 adaptation 内，约 54GB） |
| `output.txt` | dry-run 运行输出（命令行重定向生成） |
| `.status.json` | 适配状态记录 |

## 适配状态

- **Dry Run**: 见 `.status.json` / `output.txt`
- **Full Run**: 见 `.status.json`
- **设备**: Ascend NPU（Full Run 跨卡分片；Dry Run 单卡；CUDA 亦兼容）

## 备注

- 视觉输入推理需 `AutoProcessor` + `pillow`（图像预处理在 CPU 侧）；本 demo 聚焦文本主干的 NPU 算子兼容性验证。
- `generate()` 使用 `do_sample=False`，避免 NPU 上 ArgSort 触发 AiCpu 回退。
- 混合注意力中的 linear_attention（Gated DeltaNet 风格）为纯 PyTorch 实现；若个别算子在 NPU 上走 AiCpu 回退，不影响正确性。
