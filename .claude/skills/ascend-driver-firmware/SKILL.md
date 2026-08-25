---
name: ascend-driver-firmware
description: Inspect Ascend driver, firmware, CANN, and torch-npu compatibility for vLLM-Ascend deployments, and plan authorized repair. Use when NPU devices are missing, versions mismatch, or runtime initialization fails at the system layer.
---

# 昇腾驱动与固件

驱动、固件和 CANN 是节点级依赖。先只读确认，再提出修复计划；安装、升级、卸载、重启、
设备复位和修改 apt/yum 源都属于高风险变更，必须在显示版本、影响节点、回滚方式、维护窗口
和重启需求后取得用户明确授权。部署确认不自动包含节点级驱动变更授权。

## 只读检查

- `npu-smi info`：Product Name、Board ID、PCI Device ID、Subsystem Device ID、Chip Count、
  健康状态和可见设备。
- `npu-smi version`、CANN toolkit/runtime、driver/firmware、`torch_npu`、torch、Python 和
  vLLM/vLLM-Ascend 版本。
- `/usr/local/Ascend` 下的 toolkit 路径、`set_env.sh`、LD_LIBRARY_PATH 和容器内外映射。
- 多机所有节点的上述信息逐项比较；一台节点不一致就先停止 HCCL/PD 启动。

## 诊断与修复

先区分“设备不可见”“库加载失败”“版本不兼容”“设备健康异常”和“模型/拓扑问题”。输出
只读诊断报告与官方兼容版本矩阵；不要直接执行网上安装命令。获授权的维护流程读取
[references/repair-runbook.md](references/repair-runbook.md)。修复后按顺序验证设备可见、
最小 torch-npu op、单卡 vLLM 首次 forward、目标 TP/HCCL，再验证 PD/KV transfer。

本 Skill 不承诺自动安装驱动；它提供安全的检测、差异报告和授权后的复验路径。
