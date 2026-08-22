---
name: torch-npu-exit-hang
description: torch_npu 2.8.0.post5 在部分加载路径下推理成功后进程于解释器退出阶段挂死（futex/线程不退出）；修法为成功末尾 flush+os._exit(0)
metadata:
  type: project
---

2026-08-22 适配 BAAI/bge-m3（adaptations/baai_bge_m3）时发现：
full run（`from_pretrained(..., device_map="auto", max_memory=...)`）打印 `[Success]` 之后进程挂死 3+ 分钟不退出：
`State: S (sleeping)`、`wchan=futex_wait_queue_me`、108 线程、HBM 不释放（npu-smi 仍占用）。
同目录同卡的 dry run（`from_config` + `dispatch_model`）却正常退出 —— 说明挂死与加载/分发路径相关，非必现。

**修法（已验证）**：在 demo.py 成功结尾追加（仅 NPU 分支）：
```python
if device.startswith("npu"):
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # 跳过解释器清理，避免设备管理线程 join 挂死
```
复跑 dry/full 均 exit 0。`os._exit` 前必须 flush，否则 `> output.txt` 重定向输出可能被截断。

**Why:** 挂死导致后台任务永不结束、拿不到 exit code、HBM 不释放影响并发选卡；等待无益（>3min 即可判定）。
**How to apply:** 诊断：`[Success]` 已写入输出但进程仍在、`cat /proc/<pid>/wchan` 为 futex_wait_queue_me、npu-smi 仍占 HBM。
处理：kill 进程 → 给 demo.py 加上述干净退出块 → 复跑。判定挂死前可先等 2-3 分钟（正常退出一般 <1 分钟）。
