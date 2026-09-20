---
name: npu-grouped-gemm-moe-ops
description: 昇腾 grouped GEMM/MoE 公开算子调研结论（2026-09-01 实测）——torch_npu 2.10 已内置 npu_grouped_matmul（前向精确、bf16、A2 训练系列），MoE permute/unpermute 反向算子也已内置；唯一缺 grouped_matmul 反向，可用前向组合法补
metadata:
  type: reference
---

# 昇腾 grouped GEMM / MoE 算子调研（gitcode.com/cann，2026-09-01 实测验证）

**问题背景**：35B MoE LoRA SFT 慢的根因是 transformers eager 专家分发 Python 循环（见 [[npu-lora-sft-pitfalls]]）。正解是 grouped GEMM（只算被选中的 token-专家对）。调研 CANN 生态是否已有公开算子。

## ⭐ 最终结论（2026-09-01 深度核实后更正）：训练可用，官方已给出桥接方案，本机实测跑通

**三层事实**：
1. **CANN 的 `npu_grouped_matmul` 本身无反向**（三层证据：实测 grad=None；本机 CANN 9.0.0 的 26 个 grouped 头文件中仅 grouped_bias_add_grad 一个 grad；libtorch_npu.so 所有 npu_grouped_matmul* schema 均纯前向无 autograd 注册）
2. **`cann/torchtitan-npu`（官方训练框架，gitcode.com/cann 下，组织页列表里看不到但存在）给出桥接**：`ops/_grouped_mm.py` 把 npu_grouped_matmul 注册为 PyTorch 标准算子 `aten::_grouped_mm` 的 PrivateUse1 实现 → PyTorch 2.10 核心自带 `_grouped_mm` 的梯度公式（libtorch_cpu.so 有 _grouped_mm_mat1/mat2_backward）→ **反向自动可用**
3. **本机实测（910B + CANN 9.0.0 + torch 2.10 + torch_npu 2.10）**：照抄 torchtitan-npu 的注册代码后，`torch._grouped_mm(x, w, offs)` 前向逐位一致（max_diff=0.0），**backward 的 dx/dw 也逐位一致（0.000000/0.000000）**

**可直接复制的注册代码**（~15 行）：
```python
@torch.library.impl("aten::_grouped_mm", "PrivateUse1")
def _grouped_mm_npu(self, mat2, offs, bias=None, out_dtype=None):
    split_along_k = self.ndim == 2 and mat2.ndim == 2
    return torch_npu.npu_grouped_matmul(
        [self], [mat2], group_list=offs.to(dtype=torch.int64),
        group_list_type=0, split_item=2,
        group_type=(2 if split_along_k else 0),
        bias=[bias] if bias is not None else None,
        output_dtype=out_dtype)[0]
# 调用: torch._grouped_mm(x(T,K), w(G,K,N), offs=累计偏移int32/int64)
# 三种 MoE 场景(torchtitan-npu 注释): output=x@w; dx=grad@w.T; dw=x.T@grad(group_type=2)
```

## 其他核实结果
- 本机 CANN 9.0.0 头文件全景：moe_*.h 36 个含 **10 个 grad**（token_permute/unpermute×8 + init_routing_v2_grad + finalize_routing_v2_grad），其中 4 个 grad 已在 torch_npu Python 层注册；grouped_*.h 26 个仅 1 个 grad（grouped_bias_add）
- **漏掉后补查的仓库**：`cann/torchtitan-npu`（官方 NPU 训练框架，Qwen3-30B-A3B MoE SFT 配方用 EP=8 全参训练；含 TileLang 写的 moe_reduce_fused_bwd 反向内核）、`cann/cann-recipes-train`（llm_sft/qwen3_30b_a3b、llm_pretrain/DeepSeekV3 等训练配方）、TileLang-Ascend（tile-ai/tilelang-ascend，MoE 内核生态）；hixl/cann-ops-competitions/ops-collections 无相关内容
- **直接调用 npu_grouped_matmul 的坑**（若不走 aten 桥接）：x/w 必须包 List[Tensor]；group_list=前缀和累计张量（非每组数量）；group_type=0 必须 split_item=2/3；返回 List 取 [0]
- transformers 的 `batched_mm` 专家实现会物化 S 份专家权重（200token×top8→~60GB）不可用；`grouped_mm` 需 CUDA torch._grouped_mm——**但现在通过 PrivateUse1 注册恰好把这条也打通了**

## 训练侧落地方案（2026-09-01 原型已完成，scripts/step_probe.py PROBE_GMM_MOE=1）
MoE 块：argsort 按专家排序 → npu_grouped_matmul(gate_up) → swiglu → npu_grouped_matmul(down) → ×topk 权重 index_add 回收。GEMM 用自定义 autograd.Function（冻结权重只 dx：dx=dy@W_native 原生布局直用）。**数值三层验证全过**：微观 dx/dw 逐位一致；训练 step1 loss 偏差 5e-6；4 步轨迹 ≤0.84%。

**性能实测结论（关键经验）**：
- 单层 fwd+bwd (E=256/K=8)：T=290 时 gmm 10ms vs dense 27ms vs eager 1623ms；T=4096 时 22 vs 179ms——**layer 级 GMM 全 T 碾压**
- 但**真实训练步（14 卡 FSDP2, 短序列 T~300）**：gmm 17.4s vs dense 13.5s——分发链 ~10 个小算子/层（argsort/bincount/cumsum/arange/gather×2/index_add，各 ~0.3ms）×40 层×3 遍（fwd+GC重算+bwd），多进程 CPU 争抢放大数倍，吃掉 layer 级优势
- bs=4 时打平（24.0 vs 24.3s）且 **GMM 显存省 30%**（12.8 vs 18.1GB：dense 中间结果 ∝ E×T，gmm ∝ K×T）
- **选型**：短序列 SFT（T≲500）用 dense；长序列 T≥1024/大 batch/显存受限用 GMM
- **已试错排除**：aten::_grouped_mm 桥接版（Python 回调链无收益，17.3s）；连续权重缓存（专家权重全量 64GB 放不下，OOM）；内置 npu_moe_token_permute/unpermute（前向偏差 0.000977 偏大 + autograd 未接通 grad=None）
- FSDP2 下权重是 DTensor：调 npu_grouped_matmul 前需 `.to_local()`（Replicate 时零拷贝视图）；转置视图权重可直接传给该算子（前向反向都精确）

### 1. `torch_npu.npu_grouped_matmul` —— 已内置，实测可用
**实测验证过的正确调用格式**（踩过 4 个坑才跑通）：
```python
out = torch_npu.npu_grouped_matmul(
    [x],                    # 必须包 List！直接传 Tensor 报 schema 错
    [w],                    # w: (G, K, N) 3D 堆叠权重, 也包 List
    group_list=cum_tensor,  # int64 前缀和累计张量 (如 [2,5,9,10])，不是每组的数量！
    group_type=0,           # 0=M轴分组(MoE), 2=K轴分组, -1=多组不同维度
    group_list_type=0,      # 0=group_list是累计值, 1=是每组数量
    split_item=2)           # group_type=0 时必须 2 或 3（默认 0 会 aclnn 报错 161002）
# 返回 List[Tensor]，out[0] 即 (T, N)
```
- **前向数值与手写循环逐位一致（max_diff=0.000000）**，bf16，Atlas A2 训练系列 ✓
- **反向不可用**：无 autograd 注册（backward 报 "autograd kernel not registered"，grad=None）→ 推理直用，训练需自己包 autograd.Function
- 最多 1024 组（够 256 专家）

### 2. MoE 全套算子族（本机 torch_npu 已注册的，utils/custom_ops.py）
- 前向：`npu_moe_gating_top_k` / `npu_moe_init_routing` / `npu_moe_token_permute(_with_routing_map)` / `npu_moe_token_unpermute(_with_routing_map)` / `npu_moe_finalize_routing` / `npu_grouped_matmul_finalize_routing` / `npu_grouped_matmul_swiglu_quant_v2`（GEMM+SwiGLU 融合）/ `npu_moe_compute_expert_tokens`
- **反向已内置**：`npu_moe_token_permute_grad` / `npu_moe_token_unpermute_grad` / `npu_moe_token_permute_with_routing_map_grad` / `npu_moe_token_unpermute_with_routing_map_grad`
- **仓库有但本机 torch_npu 未注册**：`moe_gating_top_k_backward`、`moe_init_routing_v2_grad`、`moe_finalize_routing_v2_grad`、`moe_token_permute_with_ep_grad` —— 编译 torch_extension (cann_ops_transformer wheel) 可补齐

### 3. 仓库坐标（gitcode.com/cann）
- **cann/ops-transformer**：transformer 大模型算子库（源码随 CANN 版本分支发布，CANN 9.0.0 对应 9.0.0 分支）。`gmm/` 有 9 个 grouped_matmul 变体（含 swiglu_quant/quant/dequant/add/finalize_routing），`moe/` 有全套路由+分发+回收算子（含 8 个 _grad），`mc2/` 有 mega_moe 和分布式 MoE（distribute_dispatch/combine）。硬件支持 A2/A3 训练系列、950PR/DT、Kirin
- **cann/catlass**：昇腾版 CUTLASS 模板库（grouped_matmul slice_m/slice_k/MoE per-token-dequant 模板齐全；v2.0.0 起有 Python DSL"CATLASS DSL"）。自研带反向的 grouped GEMM 用它写
- 其他相关仓：ops-math、ops-nn、cann-recipes-train/infer、hixl
- torch_extension 构建：`bash build.sh --torch_extension` 出 cann_ops_transformer wheel

### 4. 训练侧落地方案（grouped_matmul 反向的补法）
grouped GEMM 反向数学上可由前向组合（CUDA torch._grouped_mm 即此做法）：
- `dx = grouped_matmul(dy, W^T)`（同 group_list，M 轴分组）
- `dW = grouped_matmul(x^T, dy, group_type=2)`（K 轴分组，各组 k_i=该组 token 数）
- **LoRA SFT 特例更简单**：专家权重冻结 → 只需 dx，一次前向 GMM 即可
- 完整 MoE 块训练管线：token_permute → grouped_matmul(gate_up) → swiglu → grouped_matmul(down) → token_unpermute；permute/unpermute 的 grad 算子已内置，只有 GMM 需要自包 autograd.Function（~50 行）
- 预期收益：相比稠密补丁（32× 计算浪费）零浪费，MoE 部分有望 <1s/步，整步时间逼近 FSDP 通信+attention 下限
