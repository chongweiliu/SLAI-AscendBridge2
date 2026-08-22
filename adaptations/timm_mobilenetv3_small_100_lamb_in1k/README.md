# timm/mobilenetv3_small_100.lamb_in1k Ascend NPU Adaptation

## 模型信息

- **Model ID**: [timm/mobilenetv3_small_100.lamb_in1k](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **架构**: timm `mobilenetv3_small_100`（MobileNetV3-Small 1.0x，`.lamb_in1k` 为 LAMB 优化器 ImageNet-1k 预训练 tag）
- **任务**: 图像分类 (image-classification, ImageNet-1k, 1000 类)
- **输入**: 3x224x224，ImageNet mean/std 归一化
- **规模**: ~2.54M 参数

## 加载方式

非 transformers 模型，使用 **timm** 库：

- Dry Run: `timm.create_model("mobilenetv3_small_100", pretrained=False)` — 纯本地架构随机初始化，不下载
- Full Run: `timm.create_model("hf-hub:timm/mobilenetv3_small_100.lamb_in1k", pretrained=True)` — 从 HF hub 下载，
  缓存经 `HF_HOME`/`HF_HUB_CACHE` 固定到本目录 `models/`

## 验证方式

随机图像前向：输入 `torch.rand(1, 3, 224, 224)`（ImageNet 归一化）→ 输出 logits (1, 1000)，
校验数值全为有限值，取 Top-1 / Top-5 类别索引。

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU (torch==2.8.0 + torchvision==0.23.0 + torch-npu==2.8.0.post4, 本机验证组合)
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

依赖源：默认走阿里云 PyPI 镜像，torch-npu 来自华为云 PyPI 仓库（见 `pyproject.toml`）。
注意：`import timm` 依赖 torchvision；不要给两个 extra 使用 `[tool.uv] conflicts` 分桶 + `torch>=`，
会让 `--extra ascend` 误装最新版 torch，与 torch-npu ABI 不匹配。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run
```

仅验证架构与代码路径，不下载权重。模型本身很小（~2.54M 参数），无需缩层。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载 LAMB-ImageNet-1k 预训练权重（~22MB），缓存到本目录 `models/`。

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

- timm HF hub 模型（自定义模型库类加载器：timm 库），无需克隆外部代码仓库；
  `config.json` 中 `architecture=mobilenetv3_small_100`、`num_classes=1000`。
- 已知限制：dry run 使用随机权重，Top-1 类别索引无意义属正常；
  真实分类精度属于 full run / benchmark 阶段验证。
