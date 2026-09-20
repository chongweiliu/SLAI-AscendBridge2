---
name: npu-poisoned-chip-hccl-hang
description: 昇腾 910 多卡 HCCL 训练挂死/AICPU RunAicpuKfcResInitV2 失败的根因排查法——某芯片被其他进程驻留/遗留泄漏显存后 AICPU 调度器被污染，排除该芯片的 die 即可恢复
metadata:
  type: project
---

**现象（2026-08-29 实测，Qwen3.6-35B-A3B 16卡 FSDP2 LoRA 训练）**：
1. 训练在 `dist.barrier()` / 首个集合通信处无限挂死，或 rank 报 `RuntimeError: ... HcclAllreduce`；
2. 底层错误链：`Aicpu kernel execute failed, soName=libccl_kernel.so, funcName=RunAicpuKfcResInitV2, errorCode=0x2a`（E39999，aicpu execute failed, errcode:16）；
3. npu-smi 各卡 HBM 膨胀到 ~61-65GB（正常 torch 峰值仅 ~10GB，大头是 HCCL 通信缓冲 + 挂死时反复重试累积），其中被污染芯片上的 worker 显存明显低于其他卡（分配卡住）。

**根因**：某个物理芯片（chipId N，含 die0/die1 = device 2N/2N+1）的 AICPU 调度器状态被破坏——通常是该芯片上驻留了其他用户/其他会话的进程，或此前强杀（SIGKILL）的训练进程在设备上留下泄漏显存与脏的 HCCL 资源。**任何一个芯片坏掉，全组集合通信都会挂死或失败**（HCCL 是全参与方）。

**排查法（按序）**：
1. `npu-smi info` 看进程表：有没有不属于自己的 PID 驻留某芯片；
2. 看各卡 HBM 是否异常不均（被挤占的卡显著低于其他卡 = 该卡 worker 分配卡住）；
3. 训练日志 grep `RunAicpuKfcResInitV2|aicpu execute failed`，错误消息里有 `device(chipId:X, dieId:Y)` 直接定位坏芯片；
4. 注意 device→chip 映射：**device N ↔ chipId N/2、dieId N%2**（npu-smi 的 "NPU X Chip Y" = chipId X die Y = device 2X+Y）。

**修复**：`ASCEND_RT_VISIBLE_DEVICES` 排除坏芯片的两个 die，world_size 相应减少（如 16→14）。FSDP2 对任意 world_size 均可。若坏芯片上有自己强杀残留的泄漏显存（npu-smi 显示占用但无进程），杀干净所有相关进程后大部分会释放。

**教训**：
- 多卡训练挂死且显存全员打满、AICore 0%、日志无 Python traceback 时，先怀疑个别芯片中毒，而不是脚本/沙箱问题；
- 诊断时别被"沙箱 PID namespace 不一致"（npu-smi 显示宿主机 PID，ps 显示沙箱 PID）带偏方向；
- 强杀 NPU 进程（kill -9）后先确认设备显存已释放再重启训练，否则可能把好芯片也搞出泄漏。
