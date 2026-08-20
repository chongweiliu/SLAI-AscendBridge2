# 并行策略选型（单卡 / DDP / FSDP2）

## 决策树
```
单卡能装下整套(权重+优化器状态+激活)？
├─ 能 → 想多卡提速 且 步数>~150？
│      ├─ 是 → DDP（每卡持完整参数/梯度/优化器）
│      └─ 否 → 单卡 Eager + NpuFusedAdamW + SDPA
└─ 不能 → FSDP2（参数/梯度/优化器分片，fully_shard）
```

## 选型表（单张 ~65GB NPU，bf16 权重 + fp32 AdamW 状态，近似）

| 模型规模 | 权重(bf16) | +优化器状态(fp32) | 单卡? | 推荐 |
|---|---|---|---|---|
| ≤3B | ≤6GB | +~18GB | ✅ 装得下 | 单卡；提速走 DDP |
| 3–7B | 6–14GB | +~30–50GB | 勉强/临界 | DDP；若 OOM 转 FSDP2 |
| 7–14B | 14–28GB | +~50–90GB | 多数装不下 | FSDP2 |
| ≥30B | ≥60GB | >200GB | 装不下 | FSDP2 + CPU offload / 流水并行 |

> 估算式：权重≈2×P(GB,bf16)；AdamW 状态≈8×P(fp32 m+v+grad+master)；激活≈取决于 batch/seq/是否 grad-ckpt。
> 实测：0.752B → 单卡/8卡DDP 均可，DDP bs=16/rank（global 128）峰值 ~53GB。

## 何时用 DDP
- 模型单卡装得下，想多卡提速。
- 每卡持完整参数/梯度/优化器（复制 N 份）。
- 通信：反向一次 all-reduce（梯度平均），开销小。
- 启动：`torchrun --nproc_per_node=8 cpt_ddp.py`，hccl 后端。
- `find_unused_parameters=True`（有 tie/embedding 未用参数时安全，但有开销；确认无未用参数时可 False 提速）。

## 何时用 FSDP2
- 模型单卡装不下（权重+优化器超单卡显存）。
- `torch.distributed.fsdp.fully_shard` 逐 module 分片；每卡持 1/N 参数/梯度/优化器。
- 通信更重（前向 all-gather + 反向 all-gather+reduce-scatter）。
- 适合 7B+ 全参训练/长上下文。

## 单卡 Eager（本技能默认起步）
- 简单、无通信开销、易调试。
- 0.8B–3B 常够用。短训练(100步)首选。

## DDP 代码骨架（关键行）
```python
import torch, torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_npu.optim import NpuFusedAdamW

rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"]); lr=int(os.environ["LOCAL_RANK"])
dist.init_process_group("hccl", rank=rank, world_size=world)
torch.npu.set_device(lr)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, trust_remote_code=True).to(f"npu:{lr}")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = DDP(model, device_ids=[lr], find_unused_parameters=True, gradient_as_bucket_view=False)  # 融合优化器需后者False
optim = NpuFusedAdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.01)
# 训练循环：autocast(bf16) 前向 → loss.backward → clip_grad → optim.step
optim.zero_grad(set_to_none=False)   # NpuFusedAdamW 不支持 set_to_none=True
```
启动：`torchrun --nproc_per_node=8 --master_port=29512 cpt_ddp.py`，`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

## FSDP2 代码骨架（大模型，torch 2.7 实测）
```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy   # 注意：不是 FSDP1 的 MixedPrecision/ShardingStrategy
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, trust_remote_code=True)
# 逐层 fully_shard + 根
for layer in model.model.layers:            # 据架构取 transformer 层
    fully_shard(layer, mp_policy=mp, reshard_after_forward=True)
fully_shard(model, mp_policy=mp, reshard_after_forward=True)
optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.01)
```
关键（踩坑，见 pitfalls.md）：
- `fully_shard` 的参数是 `mp_policy=MixedPrecisionPolicy(...)` + `reshard_after_forward=True`，**没有** `mixed_precision`/`sharding_strategy`（那是 FSDP1 的）。
- **NpuFusedAdamW 与 FSDP2 不兼容**：FSDP2 用 meta/fake tensor，融合优化器要 `npu.get_npu_format` 无 fake impl → 报错。FSDP2 必须用 `torch.optim.AdamW`。
- 融合路径靠 SDPA→npu fusion attention + TASK_QUEUE（bf16 靠 MixedPrecisionPolicy）。
- 无 device_id 参数；用 `torch.npu.set_device(local_rank)` 绑定。
- **保存 ckpt**：`model.state_dict()` 返回 DTensor 分片，需 `v.full_tensor()` 聚合全量再 `torch.save`（见 pitfalls #20）。

## FSDP2 显存预算（决定卡数）
每卡 ≈ **参数×16 字节 / N卡** + 激活（fp32 master 4 + AdamW m 4 + v 4 + grad 4 = 16 字节/参数）。
| 模型 | 参数 | 单卡(65GB) | 8卡每卡 | 结论 |
|---|---|---|---|---|
| 0.8B | 0.75B | ~12GB 单卡可 | — | 单卡/DDP |
| 9B | 9B | ~144GB 装不下 | ~18GB | FSDP2 8卡 ✅（2卡 ~72GB OOM） |
| 30B | 30B | — | ~60GB | FSDP2 8卡临界 |

规则：`params × 16 / N ≤ 单卡 free × 0.85` 才够。

## 多卡选空闲卡
无 `npu-smi` 时，用 `torch.npu.mem_get_info(i)` 逐卡查 free 显存，选空闲卡。
单卡：`ASCEND_RT_VISIBLE_DEVICES=<idle_card>`；多卡：全可见，`torch.npu.set_device(local_rank)`。
