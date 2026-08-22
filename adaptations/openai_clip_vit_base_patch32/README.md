# openai/clip-vit-base-patch32 Ascend NPU Adaptation

## 模型信息

- **Model ID**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **架构**: CLIP 双塔（ViT-B/32 视觉编码器 + 12 层文本编码器，~151M 参数）
- **任务**: zero-shot-image-classification / 图文匹配（CLIPModel + CLIPProcessor）
- **语言**: en
- **License**: 见模型页（MIT 系）

## 图文匹配验证说明

- 程序内用 PIL **合成**一张白色背景上的红色方块图像（224×224，无需外部
  数据下载），与候选文本逐一计算余弦相似度：
  1. `a solid red square`（应匹配）
  2. `a solid blue square`（颜色不匹配）
  3. `a photo of a mountain landscape`（语义无关）
- 真实权重下要求 `score(red) > score(blue)`；dry-run 随机权重只验证
  双塔前向/投影/相似度路径，分数本身无意义。
- 特征经 `get_image_features` / `get_text_features` 提取后 L2 归一化。

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
- 图像预处理依赖 `pillow`

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重。视觉/文本编码器层数均保守缩小到 2。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（~600MB，仓库仅提供 pytorch_model.bin），模型与
tokenizer/processor 缓存到本目录 `models/`。

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
- 模型缓存固定为 `adaptations/openai_clip_vit_base_patch32/models/`。

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
