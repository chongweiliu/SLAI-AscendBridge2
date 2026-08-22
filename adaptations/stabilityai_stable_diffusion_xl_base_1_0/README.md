# stabilityai/stable-diffusion-xl-base-1.0 Ascend NPU Adaptation

## 模型信息

- **Model ID**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- **类型**: diffusers 文生图 pipeline（`StableDiffusionXLPipeline`，标准多目录组件结构）
- **组件**: `unet` (UNet2DConditionModel, ~2.6B) + `text_encoder` (CLIP ViT-L) +
  `text_encoder_2` (OpenCLIP ViT-bigG, CLIPTextModelWithProjection) + `vae` (AutoencoderKL) +
  `scheduler` (EulerDiscreteScheduler) + tokenizer×2
- **权重体量**: ~6.9GB（fp16 variant）；单卡 64GB HBM 可整管线常驻，无需 offload
- **许可**: OpenRAIL++ / CreativeML Open RAIL++-M

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
| diffusers | 0.39.0 |
| transformers | 4.57.6 |

注意：`requires-python = ">=3.10,<3.13"`（torch_npu 最高支持 Python 3.12）。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run          # 或 DRY_RUN=1 uv run python demo.py
```

- 不下载权重；仅下载各组件 `config.json` 与 tokenizer 小文件，
  用 `from_config` 随机初始化 unet / vae / 双 text encoder 后手动组装
  `StableDiffusionXLPipeline`。
- 默认 2 步、64x64 走通完整链路（文本编码 → UNet 去噪 → VAE 解码 → 存图）。

### Full Run（真实权重）

```bash
uv run python demo.py
```

- `from_pretrained(..., variant="fp16", torch_dtype=torch.float16)`，约 6.9GB，
  缓存到本目录 `models/`；若镜像缺 fp16 variant 自动回退默认权重。
- 默认 1 步、512x512 小图验证（`--steps/--height/--width` 可调）。

### 保存全部输出

```bash
uv run python demo.py --dry-run > output.txt 2>&1   # dry run 产物（output.txt 以此为准）
uv run python demo.py > output_full.txt 2>&1        # full run 产物
```

生成图片：`sample_dry_run.png`（dry）/ `sample_full.png`（full）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本，支持 `--dry-run` / `DRY_RUN=1` / `--steps/--height/--width` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `README.md` | 本说明 |
| `models/` | 模型缓存目录（自动创建） |
| `output.txt` | dry-run 运行输出 |
| `output_full.txt` | full-run（真实权重）运行输出 |
| `sample_dry_run.png` / `sample_full.png` | 生成样例图 |
| `.status.json` | 适配状态记录 |

## 适配要点与设备约定

- **显存策略**: 整管线 `.to(device)`（SDXL fp16 全组件 ~7GB，单卡 64GB 足够），
  不用 offload / `device_map="balanced"`（Ascend 上兼容性差）。
- **选卡**: 本机 **严禁** 设置 `ASCEND_RT_VISIBLE_DEVICES`（会导致 `aclInit error 107001`）；
  `demo.py` 运行时遍历 `torch.npu.mem_get_info()` 挑空闲显存最多的卡，
  用 `torch.npu.set_device()` 绑定。
- **diffusers 版本坑（>=0.39）**: `Pipeline.to()` 的关键字参数是 `dtype=`/`device=`，
  `torch_dtype=` 会被静默忽略导致组件仍是 fp32，进而 `conv2d` 报
  `Input type (Half) and bias type (float)`。`demo.py` 统一用位置参数
  `pipe.to(device, dtype)`，并断言 UNet 参数 dtype 正确。
- **generator**: 固定使用 `torch.Generator(device="cpu")`（跨设备场景更稳，口径统一）。
- **HF 镜像**: `demo.py` 内 `os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")`
  与 `HF_HUB_DISABLE_XET=1`；外部显式设置的环境变量优先。

## 适配状态

- **Dry Run**: 通过（npu:1，随机权重 64x64 ×2 步，见 `output.txt`）
- **Full Run**: 通过（真实 fp16 权重 512x512 ×1 步，见 `output_full.txt`）
- **设备**: Ascend NPU（单卡，整管线常驻）
