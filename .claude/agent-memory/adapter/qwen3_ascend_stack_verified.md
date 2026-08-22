---
name: qwen3-ascend-stack-verified
description: 本机（aarch64, CANN 25.5.5）已验证可一次跑通的 Qwen3/LLM 适配依赖栈与 uv 行为；dry-run 选卡模式
metadata:
  type: project
---

2026-08-22 适配 Qwen/Qwen3-8B（adaptations/qwen_qwen3_8b）时验证，dry-run 一次通过、check_adaptation 通过。

**已验证可解析且可运行的依赖栈（uv, Python 3.12）**：
- `torch>=2.6.0,<2.9` + `torch-npu>=2.6.0,<2.9`（ascend extra，ascend-repo 索引）→ 解析为 torch 2.8.0 + torch-npu 2.8.0.post5（与本机 CANN 25.5.5 兼容）
- `transformers>=4.51,<5.0`（Qwen3 需 ≥4.51；解析为 4.57.6，规避 5.x 风险）
- 同步时加 `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/` 提速。

**Why:** 版本组合在解析期就要一致，避免 torch/torch-npu 不匹配或 transformers 5.x 兼容问题，浪费 dry-run 轮次。
**How to apply:** 新的 LLM 适配直接沿用该约束区间；tokenizer 基于 tiktoken 的模型（Qwen 系）通常无需显式加 tiktoken（有 tokenizer.json 走 fast 路径，本次未加仍通过）。

**uv 行为（本机实测）**：`uv sync --extra ascend` 之后，裸 `uv run python ...`（不带 --extra）不会卸载 torch_npu，可直接用于 demo/checker 命令。

**dry-run 选卡模式（本机禁 ASCEND_RT_VISIBLE_DEVICES）**：demo.py 里用 `torch.npu.mem_get_info(i)` 选空闲 HBM 最多的卡 + `torch.npu.set_device(idx)`；dispatch/加载时用 `{name: device}` 或 `max_memory` 把模型限制在单卡。本次自动选中 npu:1，generate 0.43s 通过。

**check_adaptation.py 用法（三次踩坑，务必遵守）**：`--adapt` 只收 bare 名称（如 `qwen_qwen3_8b`），传 `adaptations/xxx` 路径会报目录不存在；**必须从项目根目录、用独立的 Bash 调用执行**——不要接在 `cd adaptations/...` 之后的复合命令里跑相对路径（复合命令内 cwd 不会重置，已连续三次报 "can't open file .../adaptation/scripts/check_adaptation.py"）。

**协作注意（2026-08-22, clap-htsat-fused）**：in-progress adaptation 的 pyproject/demo 可能被队友（如应用 [[uv-torch-abi-trap]] 的 agent）就地修订。收到 "file changed on disk" 提示时：不 revert，按其现状重新 `uv sync --extra ascend` 并复跑 dry/full run 重新验证即可。

**非生成式模型（2026-08-22, google-bert/bert-base-uncased）**：masked-LM 用 `AutoModelForMaskedLM` + [MASK] 填空 + logits shape 校验（同样适用 shrink 层数 + dispatch 单卡模式），一次通过。小模型（<1B，权重 <500MB）建议连 full run 一起跑（下载快），可直接以真实权重满足 DoD "uv run python demo.py 成功"；BERT full run 预测出 'paris' 即质量证据。`torch_dtype="auto"` 在 transformers 4.57 有 deprecation warning（建议 `dtype`），为兼容 >=4.51 全区间仍用 `torch_dtype`，warning 无害。
