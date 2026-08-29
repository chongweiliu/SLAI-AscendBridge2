# uv pyproject 模板（ascend extra）

> 复用 [[uv-env-setup]] 的规范。CANN 9.0.0 配 torch 2.10.0 + torch-npu 2.10.0。把以下存为工作目录的 `pyproject.toml`，`uv sync --extra ascend` 即可。

```toml
[project]
name = "<model-name>-lora-sft"
version = "0.1.0"
description = "LoRA SFT on Ascend NPU"
requires-python = ">=3.10, <3.13"
dependencies = [
    "transformers>=4.57.0",   # 须支持目标模型 model_type; qwen3_5 需 >=5.16.1
    "peft>=0.11.0",
    "accelerate>=1.0.0",
    "datasets>=2.14.0",
    "matplotlib>=3.8.0",
    "huggingface-hub>=0.20.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
]

[project.optional-dependencies]
# CANN 9.0.0 配套: torch 2.10.0 + torch-npu 2.10.0 (含 FlashAttention 内核)
# 8.3.RC1 及更早 CANN 无 FA 编译内核(headers 有 .o 无), F.scaled_dot_product_attention 会报错
ascend = ["torch==2.10.0", "torch-npu==2.10.0"]

[tool.uv]
index-strategy = "unsafe-best-match"
python-install-mirror = "https://ghfast.top/github.com/astral-sh/python-build-standalone/releases/download"

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pypi-tuna"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
explicit = false

[[tool.uv.index]]
name = "ascend-repo"
url = "https://repo.huaweicloud.com/repository/pypi/simple"
explicit = true

[tool.uv.sources]
# torch 走 PyTorch 官方完整 wheel(绝不要用 ascend-repo 的 torch, 是残缺 stub)
torch = [{ index = "pytorch-cpu", extra = "ascend" }]
# torch-npu 走华为 ascend-repo(torch_npu 官方发布渠道)
torch-npu = [{ index = "ascend-repo", extra = "ascend" }]
```

## CANN ↔ torch_npu ↔ PyTorch 版本映射（摘自 torch_npu wheel METADATA）

| CANN | torch-npu | PyTorch | 备注 |
|---|---|---|---|
| **9.0.0（推荐）** | **2.10.0** | **2.10.0** | FA 内核齐全；910C/910B/950 通用 |
| 8.5.0 | 2.10.0rc2 / 2.9.0 | 2.10.0 / 2.9.0 | 次新 |
| 8.3.RC1 | 2.8.0 / 2.7.1 / 2.6.0 | 同 | ⚠️ 无 FA 编译内核 |

> 950 系列需 torch_npu ≥ 2.12.0（torch 2.12.0）。详见 [[ascend-torch-cpt]] pitfalls #42-45。

## 安装提速

- `download.pytorch.org` 国内约 193KB/s（torch ~139MB 需 ~12min）。太慢则把 torch 的 index 换成 `pypi-tuna`（~450KB/s，镜像完整 torch wheel）。
- 跨文件系统（cache 在 `/root/.cache/uv`、venv 在 `/mnt/...`）hardlink 失败退化为 full copy，torch 解压 ~900MB 复制 5+ 分钟。设 `UV_LINK_MODE=copy` 抑制警告。耐心等 `.venv` 增长。
