---
name: cpt-950pr-32gb-cgroup
description: Ascend950PR CPT 容器 cgroup 内存限制仅 32GB，多模态 remap 必须在 NPU 上做不能在 CPU 堆积副本
metadata:
  type: project
---

Ascend950PR 单卡训练机（宿主 754GB RAM，单卡 128GB HBM）的容器 cgroup 内存限制实际只有 **32GB**（`/sys/fs/cgroup/memory.max = 34359738368`）。`free -h` 显示 754GB 是宿主值，不反映容器限制。

**Why**：2026-08-24 跑 Qwen3.5-4B CPT 时，多模态→文本头 remap 在 CPU 上做（model fp32 16.8GB + ckpt bf16 9.3GB + sd fp32 副本 16.8GB ≈ 43GB）→ SIGKILL(137)，进程连 `[remap]` 打印都没到就被杀。这正是 ascend-torch-cpt pitfalls #44。

**How to apply**：
- 该机器上任何大模型（≥4B）的权重加载/remap/转换，先查 `cat /sys/fs/cgroup/memory.max` 确认容器内存上限，不能信 `free`。
- remap 多模态 ckpt 到文本头：**整个搬 NPU**——`with torch.device('npu:0'): model=...` 在 NPU 构造，`load_file(path, device='npu:0')` 分片直加载到 NPU，逐分片 `load_state_dict(strict=False)` 累积写入并立即 `del + empty_cache`，不在 CPU 堆积 model+ckpt+sd 副本。128GB HBM 足够。
- eval 加载大 cpt state dict 也用 `map_location='npu:0'`，别 map 到 cpu。
- smoke 的 s/step 不可直接外推正式训练（首步含 NPU kernel 编译虚高）：Qwen3.5-4B smoke 显示 ~6.3s/step，实际稳态只有 ~2.34s/step。

相关：[[ascend-cpt-env-pitfalls]]，已沉淀至 ascend-torch-cpt skill pitfalls #44（容器 OOM）。
