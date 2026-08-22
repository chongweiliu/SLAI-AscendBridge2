---
name: cv-timm-pattern
description: "timm 图像分类模型（非 transformers）的 NPU 适配模板：import timm 依赖 torchvision，torch↔torchvision 需配套版本"
metadata:
  type: feedback
---

适配 timm 图像分类模型（如 `timm/mobilenetv3_small_100.lamb_in1k`）与
transformers 文本模型的差异点：

**Why:** 2026-08-22 适配 mobilenetv3_small_100 时确认；`import timm` 内部
`from .data import ...` 会 `import torchvision.transforms`，缺 torchvision 直接
ModuleNotFoundError；torch 与 torchvision 必须配套版本，否则 ABI/运行错。

**How to apply:**
- 加载：dry-run 用 `timm.create_model("<arch>", pretrained=False, num_classes=N)`
  （纯本地随机权重，零下载）；full-run 用 `timm.create_model("hf-hub:<org>/<name>", pretrained=True)`。
  `<arch>` 取 HF `config.json` 的 `architecture` 字段（tag 如 `.lamb_in1k` 只是预训练标记，建随机权重时去掉）。
- **依赖**：`timm` + `torchvision`，且版本配套：torch==2.8.0 ↔ torchvision==0.23.0。
  cuda/ascend 两个 extra 都要写 `torchvision==0.23.0`。
- 缓存：timm hf-hub 下载走 huggingface_hub，demo 顶部
  `os.environ.setdefault("HF_HOME"/"HF_HUB_CACHE", CACHE_DIR)` 固定到本目录 `models/`。
- 验证（随机图像前向）：`x = torch.rand(1,3,224,224)` → ImageNet mean/std 归一化 →
  `model(x)` → logits (1, num_classes)，断言全有限 + `torch.topk` 取 Top-1/Top-5。
  无 tokenizer、无 generate、无缩层（CV 小模型无需）。
- 本机 NPU 选卡/镜像规则同文本模型：见 [[npu-host-env-quirks]]、[[uv-torch-abi-trap]]。
