# 超参自动择优

原则：**精度优先**。先保守保稳，再提效。所有选择写进 train_summary.json 与 README。

## 精度
- **fp32 主权重 + bf16 autocast 前向**（默认）。不要纯 bf16 前向（混合线性注意力数值崩，见 pitfalls #4）。
- 图模式时 fp32 master 同样，编译 bf16。

## 学习率 lr
| 场景 | lr |
|---|---|
| CPT 基线（单卡 bs≈32） | 1e-5 |
| DDP global batch↑（8×=256） | 2e-5（按 sqrt(global/32) 缩放，~2.8×，取保守 2×） |
| 短训练(100步)防发散 | 取保守值，不要激进 |
| 大模型(≥7B) CPT | 5e-6 ~ 1e-5 |
- warmup：~10% 步数线性；余弦衰减到 0。
- **基座自带"从零训练"调度不可照抄**（pitfalls #82 实证）：原仓库/论文的 Noam/大 warmup 大峰值调度按随机初始化设计，从已收敛 checkpoint 出发 CPT 会**发散**（train loss 不降反升）。照抄时峰值 lr 至少降 4–10×、warmup 缩到 CPT 总步数的 5–10%；判定：CPT 前几轮 train loss 持续上升 = lr 过大，降 factor/峰值重启，勿在发散点续训。

## 优化器
- `NpuFusedAdamW`（融合）。betas=(0.9,0.95)，eps=1e-8，weight_decay=0.01（embedding/norm/bias 的 wd=0），grad_clip=1.0。
- 不支持 set_to_none=True（见 pitfalls #6）。

## batch_size
- 按"激活显存预算"估上限：先取目标 bs，2 步 smoke 验不 OOM。
- **OOM 系统化回退阶梯（按顺序试，先低成本后高成本）**：
  1. 开梯度检查点（`use_reentrant=False`）——首选，保有效 bs，代价是 backward 重算。
  2. 降 per-rank bs（如 8→4→2→1）——保 global batch 靠 grad_accum 补。
  3. 加梯度累积 GRAD_ACCUM（per-rank bs 小时凑有效 batch：effective = bs×accum×world）。
  4. 降 seq_len（如 1024→512→320）——影响上下文长度，谨慎。
  5. FSDP2 + CPU offload（参数/优化器 offload 到 CPU 内存）——大模型最后手段，慢但能装下。
  6. 增加卡数（FSDP2 分片更多卡，每卡占更少）。
- DDP：per-rank bs × world = global batch。0.8B 单卡 65GB 可 bs=32；加融合优化器+DDP bucket 时降到 16 留空间。
- 经验：0.752B 单卡 bs=32 seq=320 grad-ckpt 通过；8卡DDP+NpuFusedAdamW bs=16/rank（global128）峰值~53GB。

## seq_len
- 用户指定优先。
- 否则据语料平均长度：avg<256→512；256–1024→1024；长上下文模型可 2048+（显存↑）。
- 打包：定长块，不足步数需求循环重采样补齐（记录 epoch 数）。

## 梯度累积（Grad Accumulation）
- per-rank bs 太小（装不下或 OOM）但想保有效 batch 时用：`effective_batch = per_rank_bs × GRAD_ACCUM × world_size`。
- 实现：`loss/GRAD_ACCUM.backward()`，累满 `GRAD_ACCUM` 步才 `optim.step()`（见 cpt_train/cpt_fsdp 模板）。
- lr schedule 仍按 optimizer step（不是 forward step），注意对齐。
- 经验：9B 单卡 bs=4 够（global 32）；30B+ 可能 bs=1 + accum 4 凑 effective 32。
- 注意：grad accum 不省激活显存（forward 激活照常），只省"凑 batch 不增显存"——OOM 时先开 grad-ckpt 再考虑 accum。

## 梯度检查点
- hybrid 线性注意力 fallback 激活显存大，默认开（`use_reentrant=False`，dynamo 兼容）。
- 标准模型显存充裕时可关（省重算）。

## attention
- Eager：默认 SDPA（→NPU fusion attention 自动路由），不要改。
- 图模式：full_attention 设 `attn_implementation="eager"`（绕开 npu_fusion_attention_v3 无 AscendIR，见 fusion-api.md）。

## 步数与图模式
- 训练步数 < ~150：**不**用图模式（编译~15min 摊销不了），走 Eager+融合优化器。
- ≥~150–250 或推理：可上图模式。

## 自动择优流程（脚本头实现）
1. 探测：卡数、单卡显存 free、CPU 核数、模型参数量 P、语料总 token T。
2. 算“单卡能否装下”：2P(bf16权)+8P(fp32状态)+激活预算 ≤ free×0.85。
3. 选并行：能装下→单卡/DDP；不能→FSDP2。
4. 选 bs：从 seq_len×bs 估激活，与显存预算比，定 per-rank bs（2步 smoke 确认）。
5. 选 lr：base 1e-5 × sqrt(global_batch/32) clamp 到 [5e-6, 5e-5]。
6. 选 precision：fp32+bf16 autocast；linear-attn 模型默认开 grad-ckpt。
7. 输出超参表 + 选择理由到 train_summary.json。

## 输出超参表（示例）
| 参数 | 值 | 选择理由 |
|---|---|---|
| precision | fp32 master + bf16 autocast | 数值稳 |
| parallel | DDP 8卡 | 模型<单卡容量，提速 |
| bs/rank | 16 | 融合优化器+DDP 显存预算 |
| seq_len | 320 | 用户指定/语料 |
| lr | 2e-5 | sqrt(128/32)≈2× |
| warmup | 10 | ~10% 步数 |
| optim | NpuFusedAdamW | 融合 |
| grad_clip | 1.0 | |
| grad_ckpt | True | linear-attn fallback |
