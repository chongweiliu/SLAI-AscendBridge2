---
name: dependency-pinning
description: 本机 aarch64 上 uv 解析 torch/torch_npu 的已验证组合与索引选择（cp312）
metadata:
  type: reference
---

本机（aarch64, CANN 25.5.5, 2×Ascend910）uv adaptation venv 已验证组合（2026-08-22，openai_community_gpt2）：

- `ascend = ["torch==2.8.0", "torch-npu==2.8.0.post4"]`，requires-python `>=3.12,<3.13`。
- aliyun/huaweicloud 镜像均有 `torch-2.8.0-cp312-manylinux_2_28_aarch64.whl` 与
  `torch_npu-2.8.0.post4-cp312-...aarch64.whl`；uv 实际解析出的 torch 是
  `2.8.0+cpu`（与系统环境一致），`torch.npu.is_available()=True`。
- pyproject 用虚拟项目（`[tool.uv] package = false`）可避免 hatchling 把
  models/ 大缓存打进 wheel；不影响 check_adaptation（其只查 pyproject/uv.lock 存在与内容）。
- aliyun 设为默认索引、ascend-repo(pytorch-cu124) 作 explicit 索引的模板写法可用。
- transformers 5.15.1 对 GPT-2 标准流程无兼容问题。
- 2026-08-22 qwen_qwen2_5_7b_instruct 一次通过：pin `transformers>=4.45,<5.0`（解析到 4.57.6）对 Qwen2 系同样安全；demo.py 顶部 `os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")` + `setdefault("HF_HUB_DISABLE_XET","1")` 可让下游 agent 免配环境。

**How to apply:** 新建 adaptation 时可直接沿用该 pin 组合与虚拟项目写法；
若未来 CANN 升级，先对齐系统 `python3 -m pip show torch_npu` 版本再改。
设备选择注意 [[npu-host-env-quirks]]：禁设 ASCEND_RT_VISIBLE_DEVICES，
用 `torch.npu.mem_get_info()` 挑空闲卡 + `torch.npu.set_device()`。
