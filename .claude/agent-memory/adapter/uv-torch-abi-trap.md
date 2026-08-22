---
name: uv-torch-abi-trap
description: "uv conflicts 分桶会让 --extra ascend 误装 cuda 桶的 torch 新版，导致 torch_npu undefined symbol；本机验证组合与修复法"
metadata:
  type: feedback
---

本机 (aarch64, 2x Ascend910, CANN 25.5.5, cp312) 适配 pyproject.toml 不要用
`[tool.uv] conflicts = [[{extra="cuda"},{extra="ascend"}]]` 加 `cuda = ["torch>=2.6.0"]`
的模板组合：`uv lock` 会把 cuda 桶解析到最新 torch（如 2.13.0+cu130），
`uv sync --extra ascend` 实际装进 venv 的也是这个 2.13，与
`torch-npu==2.8.0.post4` ABI 不匹配，报
`libtorch_npu.so: undefined symbol: ...is_contiguous_custom...`，
demo 在 `import torch` 阶段即崩（TORCH_DEVICE_BACKEND_AUTOLOAD 报错）。

**Why:** 2026-08-22 适配 Qwen/Qwen2.5-1.5B-Instruct 时实测踩中；
uv 的 conflict 分桶在单一 venv 安装时选了非 ascend 桶的 torch。

**How to apply:**
- 修复法：两个 extra 统一锁 `torch==2.8.0`，删除 `conflicts`；
  `uv lock && uv sync --extra ascend` 后用
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -c "import torch; print(torch.__version__)"`
  确认装的是 2.8.0，再跑 dry run。
- 本机验证组合：`torch==2.8.0`（aliyun/PyPI aarch64 即 2.8.0+cpu）+
  `torch-npu==2.8.0.post4`（源可用 aliyun 或 huaweicloud ascend-repo，两者都有 cp312 aarch64 wheel）。
- torch_npu 2.9.x 对应更高 CANN，别在本机用；`torch>=` 不锁版本会被解析到最新而崩。
- download.pytorch.org/whl/cu124 没有 aarch64 torch wheel，pyproject 里别把 torch 源
  固定到 pytorch-cu124（会导致 uv lock 在本机平台解析困难）。
- 参考成品：adaptations/qwen_qwen2_5_1_5b_instruct/pyproject.toml。

相关：[[npu-host-env-quirks]]（ASCEND_RT_VISIBLE_DEVICES 禁用、HF 镜像）。
