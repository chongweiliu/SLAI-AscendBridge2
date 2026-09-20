---
name: venv-interpreter-loss-recovery
description: uv 托管 python 解释器目录跨会话丢失致多 venv 断链的原路径重装恢复法（site-packages 完好时零依赖重装）
metadata:
  type: project
---

# venv 解释器丢失的原路径重装恢复（2026-09-18 实战）

**现象**：跨会话（数天后）`lora-ws/*/.venv/bin/python`、`training-ws/.venv` 全部 "No such file or directory"——`/root/.local/share/uv/python/cpython-3.12-linux-aarch64-gnu/` 整目录消失（/root 下 uv 托管资产被清理），而 **/mnt 上的 venv site-packages 完好**（bin/python 是指向该目录的悬空符号链接）。

**Why**：uv 创建的 venv 解释器在 `~/.local/share/uv/python/`（root 盘），项目 venv 在 /mnt（共享盘）——两者生命周期不同步，/root 清理即断链。所有依赖该 venv 的批次工具链瞬间全瘫。

**恢复（零依赖重装，勿重建 venv）**：
```bash
/usr/bin/python3 -m pip install -q -U uv -i https://mirrors.aliyun.com/pypi/simple/
export UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"
uv python install 3.12   # 装回 /root/.local/share/uv/python/cpython-3.12-linux-aarch64-gnu/ 原路径
```
符号链接自动复原，torch/torch_npu/transformers/peft/diffusers/timm 全套已验证依赖原样恢复（比 uv sync 重建快两个数量级，且无 #39 拖 torch 升级风险）。恢复后必跑：版本核对 + NPU matmul 健康检查。

**How to apply**：venv python 报 No such file 时，先 `ls -la .venv/bin/python` 看符号链接指向 + `cat pyvenv.cfg` 的 home——若指向 uv python 目录且目录消失，走原路径重装；不要急于重建 venv/重装依赖。注意 pip 装的 uv 版本(0.12.15)可与原 pyvenv.cfg 的 uv 版本(0.12.6)不同，无碍。

关联：[[npu-lora-sft-pitfalls]] #7（venv mv 后 prefix 自动更新）、[[shared-storage-multi-session-interference]]。
