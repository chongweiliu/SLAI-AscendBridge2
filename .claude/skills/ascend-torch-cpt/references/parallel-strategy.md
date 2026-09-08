# 并行策略选型（单卡 / DDP / ZeRO-1 / FSDP2 / 模型并行）

## 决策树（**先定量实测卡间带宽**，见下「卡间互联探测」）
```
先跑 bench_hccl.py 实测 all-reduce 带宽（不要只凭 hccn_tool 推断，单机内走 HCCS 与 RoCE 无关）：
├─ <10GB/s（PCIe 级，互联慢）→ 大模型直接走【模型并行 device_map】(#24)，
│      不要用 FSDP2（其 all-gather/reduce-scatter 会通信-bound，实测慢 8×）
└─ ≥50GB/s（HCCS/RoCE 正常）→ 按下面常规决策树：
    单卡能装下整套(权重+优化器状态+激活)？
    ├─ 能 → 想多卡提速 且 步数>~150？
    │      ├─ 是 → DDP（每卡持完整参数/梯度/优化器）
    │      └─ 否 → 单卡 Eager + NpuFusedAdamW + SDPA
    └─ 不能 → 差距小(缺 5~15GB)？ → ZeRO-1（ZeroRedundancyOptimizer，改动最小）
             → 差距大/想扩到 4~8 卡 → FSDP2（参数/梯度/优化器分片，fully_shard）
```

## 卡间互联探测（选型前必做）
```bash
# 是否配了 RoCE IP（没配 = 只能走慢速 PCIe）
/usr/local/Ascend/driver/tools/hccn_tool -i <devid> -ip -g
# 报 "no ip was preset" → 无 RoCE（仅影响跨机；单机内走 HCCS）
```
- **判读红线（踩过坑）**：hccn_tool **空输出 ≠ "no ip was preset"**，两者含义不同；且注意路径 `/usr/bin/hccn_tool` 与 `/usr/local/Ascend/driver/tools/hccn_tool` 可能是不同工具。**RoCE 只决定跨机互联，单机多卡芯片间走 HCCS 高速总线（与 RoCE 配置无关）**——不要仅凭 hccn_tool 无 RoCE 输出就断定"单机互联慢"（实测误判案例：hccn_tool 空输出推断慢速，实为 HCCS 103GB/s）。
- **定量基准（首选，10 秒出结果）**：跑 `scripts/bench_hccl.py.tmpl` 实测 all-reduce 总线带宽：
  - `≥50GB/s`：互联正常，DDP/FSDP2 照常选
  - `<10GB/s`（PCIe 级）：互联慢，大模型优先模型并行
  ```bash
  ASCEND_RT_VISIBLE_DEVICES=0,1 BENCH_WORLD=2 BENCH_SIZE_MB=1024 BENCH_ITERS=20 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29533 $PYTHON bench_hccl.py
  # 示例输出: bus_bw=103.2GB/s (Atlas 910 单机 2 卡, HCCS)
  ```
- 训练级确认（存疑时）：跑 2 步训练 + torch_npu.profiler，看 `operator_details.csv` 里 `HcclAllGather`+`HcclReduceScatter` 占比。>50% 说明通信-bound，改模型并行。

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

## 何时用 ZeRO-1（朴素 DDP 差一点装不下的中间档）
`torch.optim.ZeroRedundancyOptimizer`（PyTorch 原生，纯 Python，NPU 可用）= ZeRO-1：DDP 数据并行不变，
只把 **AdamW 的 m/v 优化器状态**按 rank 分片（省 8 字节/参数 ÷ N）。适用于"朴素 DDP 差 5~15GB 装不下、
又不想上 FSDP2 全套改造"的场景（如 3–7B 模型 2 卡）。

- 每卡显存 ≈ 权重 4 + 梯度 4 + AdamW 8/N 字节/参数：4.2B/2卡 ≈ 52GB ✅、/4卡 ≈ 34GB ✅
- 与 DDP 组合即可（`fully_shard` 不需要）；ZeRO-2/3 没有一对一原生物，直接用 FSDP2
- DeepSpeed 原版 ZeRO 不在本技能范围（Ascend 需专用适配分支，本技能定位为 PyTorch 原生）
- 代码改动极小（在 DDP 骨架上只换优化器一行）：
```python
optim = torch.optim.ZeroRedundancyOptimizer(
    trainable, optimizer_class=torch.optim.AdamW, lr=LR, betas=(0.9, 0.95),
    weight_decay=0.01, foreach=False)
# 注意: ZeRO-1 的 ckpt 保存需在各 rank 收集 optimizer state 分片 (param_groups 同 rank 对齐),
# 短训练可接受 rank0 只存 model state + 提示 optimizer 分片位置; 长训练用 FSDP2 更规范
```

## 何时用 FSDP2

> ⚠️ FSDP2 与 `expandable_segments:True` 不兼容（all-gather buffer 累积致假性 OOM，曾误判为"fully_shard 不分片"）——cpt_fsdp.py.tmpl 已内置守卫自动禁用；本文中 FSDP2 的卡数阈值/吞吐结论（"9B 需 8 卡""12B/4卡 50s/step 通信-bound"）可能部分被该混杂变量污染，expandable_segments 关闭后需重新标定。见 pitfalls #77。
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

## 模型并行（device_map="auto"，卡间互联慢时首选）
当卡间 RoCE 未配、HCCL 走慢速 PCIe 时，FSDP2 的 all-gather/reduce-scatter 会通信-bound（实测 12B/4卡 50s/step）。模型并行把层拆到多卡、参数各归其卡、无 all-gather，优化器放 NPU，实测降到 ~5.5s/step（8.3×）。

**代码骨架（单进程即可，无需 torchrun）**
```python
import torch, torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM

# 关键：必须 fp32（bf16 权重 + 小 lr 会被精度吞掉，见 pitfalls #25）
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, trust_remote_code=True,
    low_cpu_mem_usage=True, device_map="auto")   # 自动拆到可见的所有 NPU 卡
model.train(); model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WD)  # 默认 foreach=False
# 训练循环：输入放 model.device（embed 所在卡），其余同单卡
batch = x[bidx].to(model.device); attn = torch.ones_like(batch)
loss = model(input_ids=batch, attention_mask=attn, labels=batch).loss
loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optim.step(); optim.zero_grad(set_to_none=False)
# 保存：model.named_parameters() 已含各卡权重，直接 .cpu().to(bf16) 存
```

关键：
- `.to(device)` 可微分，autograd 自动跨卡回传梯度，无需手写 P2P/`dist.send`。
- 显存预算：`params × 16 / N`（fp32 参数 4 + m 4 + v 4 + grad 4，共 16 字节/参数）。12B/4卡 ≈ 48GB/卡，装得下。
- 模型拆分配由 device_map 自动按字节均衡；embed（首卡）与 lm_head（末卡）tied 时，多模态模型需注意（见 pitfalls #30）。
- 评估时用 bf16 单卡加载（fp32 会 OOM，见 pitfalls #31）。

## 多卡选空闲卡
无 `npu-smi` 时，用 `torch.npu.mem_get_info(i)` 逐卡查 free 显存，选空闲卡。
单卡：`ASCEND_RT_VISIBLE_DEVICES=<idle_card>`；多卡：全可见，`torch.npu.set_device(local_rank)`。

## 附录: device_map 单进程模型上叠加 DDP 的显存压法（实测 4.5B/65GB 卡）
当朴素 DDP 全训练态超卡（如 4.2B×16B=67GB>65GB）又想先试 DDP 时，按优先级压显存（实测案例）：
1. **冻结非关键参数**：冻结 embed_tokens(+tied lm_head) 与视觉塔——可训练 4.2B→3.57B（省 ~10GB）。
   CPT 语义变化（embedding/lm_head 不更新），只建议短测试用，正式训练走 FSDP2。
2. **冻结部分保持 bf16**：`from_pretrained(torch_dtype=bf16)` 后仅对可训练参数 `p.data=p.data.float()`
   上转 fp32 master（冻结部分省一半，又省 ~2GB）。
3. **`gradient_as_bucket_view=True`**：DDP 梯度桶直接视图化 `.grad`，省一份梯度大小的桶内存
   （4.2B fp32 梯度桶 ≈ 14GB！默认 False 会双倍占）——但因此**不能用 NpuFusedAdamW**（融合优化器
   与桶视图冲突），须用 `AdamW(foreach=False)`。
- 结论：压到极限后 4.5B 在 2×65GB 上也只勉强贴线（静态 ~59GB + 激活/logits 峰值 2~4GB），
  大词表模型（24.8 万 vocab 的 CE logits fp32 峰值可达 ~7GB@bs2）随 batch 线性放大，极易 OOM。
  **教学价值大于实用价值：预算不够时正路是 FSDP2 / ZeRO-1，而不是给 DDP 挤显存。**

## 实测基准脚本
`scripts/bench_hccl.py.tmpl`：torchrun-free 的 HCCL all-reduce 带宽基准（torch.multiprocessing.spawn），
选型前必跑，10 秒判定互联档位。实测参考：Atlas 910 单机 2 卡 HCCS = 103GB/s（fast 档）。
