---
name: vlm-demo-pattern
description: VLM（image-text-to-text，如 Qwen2.5-VL）适配 demo.py 的输入路径与依赖要点；transformers 4.57 from_config 改名
metadata:
  type: project
---

VLM 适配（2026-08-22 于 Qwen2.5-VL-7B-Instruct 一次通过验证）。

- 依赖：`qwen-vl-utils` **隐式依赖 torchvision**（import 时硬导入），ascend extra 必须加
  `torchvision~=0.23.0`（与 torch 2.8.0 配套）；Qwen2.5-VL 视觉塔还需要 `einops`。
- Qwen2.5-VL (`qwen2_5_vl`) 需要 `transformers>=4.49.0`；本仓用 `>=4.49.0,<5.0`（4.57.6 验证通过）。
- **transformers>=4.57 把公开的 `from_config` 改名为 `_from_config`**，dry-run 随机权重分支要写
  `hasattr(cls, "from_config")` 双路径兼容（不属于禁止的"回退简单验证"，两条路径都完整建模型）。
- 输入路径：`processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` →
  `qwen_vl_utils.process_vision_info(messages)` → `processor(text=..., images=..., videos=..., return_tensors="pt")`。
  测试图用 PIL 代码内生成（渐变+色块），不下载外部图。
- dry-run 缩小：除 `num_hidden_layers->2` 外，VLM 还要缩 `config.vision_config.depth->2`。
- 7B bf16 单卡足够，用 `model.to(device)` 而非 `device_map="auto"`，避免与并行任务跨卡干扰；
  选卡用 `torch.npu.mem_get_info()` 挑空闲最大卡（见 [[device-selection]]）。

**Why:** VLM 与纯文本模板差异大（processor 替代 tokenizer、图像张量输入、vision 塔缩层），
且 qwen_vl_utils 的隐式 torchvision 依赖第一次跑就会 ModuleNotFoundError。
**How to apply:** 适配任何 image-text-to-text / VLM 模型时按本模式改 demo.py 与 pyproject。
