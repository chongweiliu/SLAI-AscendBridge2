---
name: cannbot-mix-mode-pitfall
description: cannbot 融合 MIX 模式 (__mix__(1,2)) kernel 的 AIC 驱动陷阱与回退策略（block_attention_score 实战教训）
metadata:
  type: project
---

## 核心教训（2026-07-13 block_attention_score 实战）

**cannbot 融合 Cube+Vector kernel 用 MIX 模式 (`__mix__(1,2)`) 时，若算子有自定义 program 调度（batch×kv_head×q_tile 多核切分），AIC 不会被正确驱动，`mm.Iterate()` 挂起。**

**Why:** MIX 模式下 AIV 发起 Iterate 通知 AIC 做 Cube 计算，但 Matmul 内部的 AIC/AIV 调度机制与算子自定义的 `SetBlockDim`/`GetBlockIdx` program 调度冲突，AIC 不响应。3 轮调试（SetDim(1)→SetDim(2)、对照 matmul_leakyrelu、plog）确认 AIC 未执行，根因是机制不兼容而非配置 bug。

**How to apply:**
- **优先非融合双 kernel**：算子既有 Matmul 又有 Vector 逻辑时，不要首选 MIX 融合。拆成 Kernel A（纯 AIC `__global__ __aicore__` matmul → GM）+ Kernel B（纯 AIV `__global__ __vector__` 读 GM → vector → 写回）。单类型 kernel 无 AIC 驱动问题。
- **中间矩阵显存**：非融合会在 GM 存中间结果。block_attention_score 的 qk_score `[num_q_heads, total_q_len, compressed_k_len]` ~10GB，Ascend910 64GB HBM 可接受。设计时算好显存预算。
- **MIX 调试次序（若必须用）**：① plog (`ASCEND_SLOG_PRINT_TO_STDOUT=1` + `AscendC::PRINTF`) 确认 AIC 是否执行 ② `SetDim` 匹配 AIV 数（`__mix__(1,2)` → SetDim(2)）③ `numBlocks=1` ④ 对照 matmul_leakyrelu_advanced_api.asc 完整 tiling ⑤ `SetLocalWorkspace` 不要调用（184KB 占满 192KB UB 导致 InitBuffer 挂起）。

## 已确认的 MIX 模式正确配置（仍不够）
- `SetDim(2)` + `numBlocks=1` + `__mix__(1,2)`：不超时，但 Iterate 仍挂
- `SetLocalWorkspace` 必须移除（占满 UB）
- UB→GM 写回必须用 `TQue<VECOUT> + EnQue/DeQue`，`TBuf<VECOUT>` 直接 DataCopyPad 不工作
- `SetFixSplit` 的 baseN 要匹配 `SetDim(2)` 的 singleCoreN（baseN > singleCoreN 会 tiling 失败）

## block_attention_score 回退方案
非融合双 kernel：
- Kernel A（纯 AIC）：batched matmul QK → qk_score GM
- Kernel B（纯 AIV）：exp(qk-lse)→GQA求和→block聚合→TopK → block_topk
设计修订中。相关 memory: [[direct3d-s2-cannbot-required]]、[[direct3d-s2-setup-state]]
