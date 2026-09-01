# MoE 模型 LoRA 训练提速：稠密补丁与 grouped GEMM（数学等价双路线）

> 2026-08-29~09-01 在 Qwen3.6-35B-A3B（256 专家 top-8, 40 层）+ 14 卡 FSDP2 上实测闭环。
> 脚本入口：`scripts/lora_train_fsdp.py.tmpl` 的 `MOE_IMPL=eager|dense|gmm` 环境变量。

## 1. 根因：transformers 的 MoE eager 专家分发是 Python 循环

`qwen3_5_moe` 的 `Qwen3_5MoeExperts.forward` 逐专家 `for expert_idx in expert_hit:`，每次迭代
~10 个小算子（one_hot/where/gather/linear×2/act/mul/index_add）。短序列（~290 token × top-8
命中 ~全部专家）下 40 层 × 256 迭代 ≈ **每前向 10 万+ 微小内核**，反向翻倍——纯调度开销，
NPU 算力闲置（吞吐仅 ~15 tok/s/rank）。分相实测：fwd 40% + bwd(含 GC 重算) 58% + opt/comm 2%。

**先分相计时再优化**（fwd/bwd/opt/comm 各相单独计时 + 同步），别按直觉改：
实测反例——关 GC 直觉上省时间，实际更慢（全激活保留 → NPU 分配器碎片化，20.5→31.2s）。

## 2. 两条等价补丁路线与选型

| | **dense（稠密 bmm）** | **gmm（grouped GEMM）** |
|---|---|---|
| 原理 | 2 次批量 bmm 算**全部**专家再 gather top-k（多算的丢弃） | argsort 分发 → `npu_grouped_matmul` **只算被选中**的对 → 加权回收 |
| 计算浪费 | E/topK 倍（Qwen3.6: 32×） | 零 |
| 单层 fwd+bwd (T=290) | 27.1ms | 10.0ms |
| 单层 fwd+bwd (T=4096) | 178.6ms | **22.0ms** |
| 真实训练步时 (T~300, bs=1) | **13.5s（最快）** | 17.4s（分发小算子×40层×3遍在多进程下开销放大） |
| bs=4 步时 | 24.0s | 24.3s（打平） |
| 显存 (bs=4) | 18.1GB（中间结果 ∝ E×T） | **12.8GB（∝ topK×T）** |
| **选型** | **短序列 SFT（T≲500）** | **长序列（T≥1024）/ 大 batch / 显存受限** |

两条路线数学上都与逐专家循环等价（多算/只算的结果同权重同求和）。等价性验证协议见 §4。

## 3. 实现要点（tmpl 已内置，此处为适配其它 MoE 结构的指引）

**dense 补丁**（适配点：类名 + 权重布局）：
- qwen3_5_moe: `M.Qwen3_5MoeExperts.forward`，权重 `gate_up_proj (E,2I,H)` / `down_proj (E,H,I)`（3D 堆叠参数）
- 其它 MoE（如 Qwen3-MoE/Mixtral 用 nn.ModuleList[MLP]）：把循环换成按专家维堆叠的 bmm，或改用 gmm 路线更直接

**gmm 补丁**（适配点同上 + 三个坑）：
1. `npu_grouped_matmul([x],[w], group_list=offs, group_list_type=0, split_item=2, group_type=0)`
   —— x/w 必须包 List；**group_list 是各组行数的累计前缀和**（空组允许重复值）；
   group_type=0（M 轴分组）必须配 split_item=2 或 3
2. 权重传**转置视图**即可（前向反向都精确，无需物化连续副本——全量连续缓存需 64GB 放不下）
3. **FSDP2 下权重是 DTensor**：调用前 `.to_local()`（前向时已 all-gather 为 Replicate，零拷贝视图）

**反向**（LoRA 冻结专家权重，只需 dx）：`dx = npu_grouped_matmul([dy], [W_native], ...)` ——
W 的模型原生布局 (E,2I,H) 恰好就是 dx 所需方向，无需转置。自定义 ~15 行 autograd.Function 即可
（tmpl 已内置）。通用反向（含 dW）：`dW = grouped_matmul(xᵀ, dy, group_type=2)`，或照抄
torchtitan-npu 的 `aten::_grouped_mm` PrivateUse1 桥接让 PyTorch 核心公式代劳。

## 4. 等价性验证协议（改了 MoE 前向必须做）

1. **微观**：小规模构造（如 E=32/T=97/K=4，制造空组），新旧前向输出 diff 应在 bf16 舍入级；
   `x.requires_grad` 反向 dx/dw 与手写循环对照（gmm 路线实测逐位一致 0.000000）
   注意：`Qwen3_5MoeExperts.__init__` 用 `torch.empty` 未初始化权重，测试前需手动 `p.normal_()`
2. **训练级 step1 loss**：同种子同数据跑 1 步，新旧 loss 应几乎逐位一致（gmm 实测偏差 5e-6）
3. **多步轨迹**：4~200 步 loss 轨迹窗口均值偏差 ≤1~2%（**残余偏差来自 LoRA A 随机初始化不同**——
   lora_B 初始为 0 故 step1 必然一致，step2+ 每次重跑都有小幅随机偏差，勿误判为不等价）
4. **权重更新幅度**：Δ/W=‖B@A‖/‖W‖ 落在与原实现相同时长的预期插值区间（如 100 步 0.66% → 500 步 1.71%）

## 5. 已试错排除的方案（勿重复踩）

- ❌ 关 GC：更慢（分配器碎片化，见 §1）
- ❌ `reshard_after_forward=False`：40 层全聚合 35GB+ 激活 > 61GB 必 OOM
- ❌ transformers `experts_implementation="batched_mm"`：物化 S 份专家权重（~60GB）
- ❌ transformers `experts_implementation="grouped_mm"`：找 CUDA 的 torch._grouped_mm（NPU 桥接法见 npu-op-discovery.md，但 Python 回调链在多进程下无收益）
- ❌ 全量连续权重缓存：专家权重共 64GB（40 层×1.6GB），放不下
- ❌ 内置 `npu_moe_token_permute/unpermute` 替代分发链：前向偏差 0.000977 偏大 + autograd 未接通（grad=None）；但它们**层内单算子**的思路是后续优化方向（编译 ops-transformer 的 `moe_init_routing_v2`+grad 可行）
- ✅ `BATCH_SIZE=4`（右填充，pad label=-100）：吞吐 3.25×（25.3s/步处理 4 样本）；代价是有效 batch ×4

## 6. 实测数据存档（Qwen3.6-35B-A3B, 14 卡 FSDP2, CANN 9.0.0, bf16, GC on）

| 配置 | 步时 | s/样本 | 显存峰值 |
|---|---|---|---|
| eager bs=1 | 20.53s | 20.53 | 10.26GB |
| **dense bs=1** | **13.53s** | **13.53** | 11.95GB |
| gmm bs=1 | 17.38s | 17.38 | 10.25GB |
| dense bs=4 | 24.04s | 6.01 | 18.09GB |
| gmm bs=4 | 24.26s | 6.07 | **12.84GB** |

单层基准（fwd+bwd, E=256/K=8, ms/层）：T=290: eager 1622.7 / dense 27.1 / gmm 10.0；
T=2048: 1669.9 / 97.8 / 15.0；T=4096: 1715.7 / 178.6 / 22.0。
完整探针与验证脚本：`lora-ws/Qwen3.6-35B-A3B-lora/scripts/`（step_probe.py / gmm_moe_test.py / gmm_layer_bench.py）。
