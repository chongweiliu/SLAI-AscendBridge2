---
name: cannbot-dav2201-viable
description: 本机 dav-2201 (Atlas A2/A3) 在 cannbot Ascend C 支持矩阵内，推翻 v1 旧记忆
metadata:
  type: project
---

# cannbot 在本机 dav-2201 可行（2026-06-30 核实）

**事实**：本机 IT22HMDA_2_S，8×Ascend910，NpuArch=**dav-2201**（Atlas A2/A3 代），CANN 8.5.0，torch_npu 2.9.0。adapter-1 用 cannbot `ascendc-env-check` skill 核实：**dav-2201 在 cannbot Ascend C 支持矩阵内**（arch22，`--npu-arch=dav-2201`）。

**Why**：v1 适配（某 3D 生成模型）记忆里写"cannbot BLOCKED by hardware: Ascend910=DAV_1001 (first-gen) 不在支持矩阵"。那是**旧机器**。本机换成了 dav-2201，cannbot 已可用。用户最初说"本机是 A3"是对的。

**How to apply**：
- 本机新建 adaptation 遇到 CUDA 算子缺口时，可以走 cannbot Ascend C 生成路径（Architect→Developer→Reviewer），不要再用"硬件不支持"为由跳过
- cannbot 调用参数带 `--npu-arch=dav-2201`
- 相关：（已过时，以本条为准） 的 cannbot 段落已过时，以本条为准
- 注意：纯 torch 实现若已 bit-exact 且性能可接受（如稀疏卷积 conv_none.py），仍优先纯 torch，不必非走 cannbot；cannbot 留给纯 torch 无法实现或性能太差的算子
