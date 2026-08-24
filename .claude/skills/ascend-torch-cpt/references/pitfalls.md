# 踩坑清单（必读，避免重犯）

按“症状→根因→解法”记录本类任务在昇腾 NPU 上反复出现的问题。脚本模板已规避多数，但新模型/新环境仍可能复现。

## 1. torch_npu 导入即崩 / autoload 报错
- **症状**：`import torch_npu` 后 RuntimeError: Failed to load the backend extension: torch_npu。
- **根因**：torch 在 init 时自动 autoload 设备后端，torch_npu 的 autoload 路径与某些 CANN/torch 版本组合下失败。
- **解法**：设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，再 `import torch_npu` 手动注册。所有脚本/env 入口都加。

## 2. 多模态 checkpoint 键前缀不匹配
- **症状**：`Qwen3_5ForCausalLM`（文本头）load_state_dict 全 miss；ckpt 键是 `model.language_model.*` + `model.visual.*` + `mtp.*`。
- **根因**：多模态模型 ckpt 含 vision/mtp，文本头只需 `model.*`。
- **解法**：strip `model.language_model.` 前缀→`model.`；丢弃 `visual`/`mtp` 键；`strict=False`；`tie_word_embeddings=True` 时 lm_head 缺失正常，调 `model.tie_weights()`。详见 multimodal-remap.md。

## 3. 线性注意力 torch fallback 激活显存爆炸（58GB OOM）
- **症状**：小模型(0.8B) bs=32 seq=320 单卡 OOM 到 58GB。
- **根因**：hybrid 线性注意力(Qwen3.5/GatedDeltaNet)缺 flash-linear-attention/causal-conv1d 快速核，torch fallback 把中间注意力张量完整物化。
- **解法**：开梯度检查点（`use_reentrant=False`）压激活峰值；不要靠降 batch_size（保有效 batch）。

## 4. 纯 bf16 前向数值崩（NLL 为负/NaN）
- **症状**：模型 `.to(bfloat16)` 后纯前向，PPL=0.01、NLL=-4.5（不可能）或 NaN。
- **根因**：混合线性注意力内部 `.float()` cast 与 bf16 权重相互作用产生异常。
- **解法**：模型 fp32 主权重 + `torch.autocast("npu", dtype=bfloat16)` 前向（与训练一致）。**不要** `model.to(bfloat16)` 后裸跑。

## 5. NLL 公式漏取负
- **症状**：评估 NLL 为负值（如 -4.5），PPL<1。
- **根因**：`log_softmax` 输出 ≤0 是 log-prob；NLL 需取负。
- **解法**：`nll = float(-tgt_lp.mean())`；PPL=`exp(min(nll,20))`。

## 6. NpuFusedAdamW 不支持 set_to_none
- **症状**：`zero_grad(set_to_none=True)` 抛 `ValueError: set_to_none is not supported in fused optimizers`。
- **解法**：`optim.zero_grad(set_to_none=False)`。

## 7. NpuFusedAdamW + DDP gradient_as_bucket_view 冲突
- **症状**：优化器 step 抛 `AclNN_Parameter_Error: X and Y cannot broadcast`。
- **根因**：梯度是 DDP bucket 视图，融合优化器 fuse 时 shape 不匹配。
- **解法**：`DDP(..., gradient_as_bucket_view=False)`（让每参数独立 .grad）。

## 8. 融合优化器显存紧张 OOM（aclnnInplaceAdd alloc failed）
- **症状**：`optim.step()` 时 OOM 207001（融合缓冲+DDP bucket 顶到上限）。
- **解法**：降 per-rank batch（如 32→16）+ `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 减碎片。

## 9. torchair 图模式：缺 GE converter 长尾
- **症状**：`NotImplementedError: torch.ops.aten.X ge_converter is not implemented!`。
- **根因**：torchair 对部分 aten op 是空 stub。Qwen3.5 线性注意力实际缺：`softplus`、`eye`、`softplus_backward`（+ dynamic=True 时 `constant_pad_nd` 的 tensor-pad 分支）。
- **解法**：用 `ge.Softplus`/`ge.Eye` 原生算子 + `grad_output*ge.Sigmoid(self)` 注册 converter（见 fusion-api.md）。`register_fx_node_ge_converter` 覆盖 stub。`dynamic=False` 可规避 constant_pad_nd 的 tensor-pad 分支。

## 10. torchair 图模式：full_attention SDPA→npu_fusion_attention_v3 无 AscendIR
- **症状**：`Failed to converter npu.npu_fusion_attention_v3.default to AscendIR`。
- **解法**：图模式时把 full_attention 设 `attn_implementation="eager"`（手动 bmm+softmax，全部 op 有 converter）；线性注意力层不受影响。Eager 训练不需要改（SDPA 自动路由 NPU fusion）。

## 11. torchair 依赖缺失致 GE 初始化失败
- **症状**：`Failed to initialize GE ... No module named 'scipy'/'pkg_resources'`。
- **根因**：GE Python 侧依赖未补齐；setuptools≥82 移除 pkg_resources。
- **解法**：装 protobuf/scipy/attrs/decorator/cloudpickle/ml_dtypes/tornado；`setuptools<82`。

## 12. huggingface.co 不可达 / pip 慢
- **症状**：下载/安装超时。
- **解法**：模型用 modelscope 或 hf-mirror.com（git clone+git-lfs）；pip 用 `mirrors.aliyun.com/pypi/simple` 或 `pypi.tuna.tsinghua.edu.cn/simple`；大包(scipy)后台装。

## 13. 单 CPU 核环境
- **症状**：`nproc`=1，数据加载/编译慢。
- **解法**：不用 DataLoader 多 worker；语料小直接整表载入；图模式编译会很长（单核 GE 构建），短训练别用图模式。

## 14. 0x0.st / catbox.moe 公网上传常不可用
- **症状**：上传 FAILED。
- **解法**：依次试 catbox.moe → 0x0.st → uguu.se（`curl -F files[]=@png https://uguu.se/upload.php`，返回 JSON url）。uguu.se 通常可用。

## 15. 用户给的模型/语料路径不存在
- **症状**：`/mnt/host-model/X`、`/mnt/share/...` 等 ls 报 No such file。
- **解法**：`find` 搜本机可用副本，**显式告知**用户改用了哪个路径。不要静默改路径。

## 17. FSDP2 `fully_shard` 参数名不同
- **症状**：`fully_shard() got an unexpected keyword argument 'mixed_precision'`。
- **根因**：`mixed_precision`/`sharding_strategy` 是 FSDP1 的 `FullyShardedDataParallel` 参数；FSDP2 的 `fully_shard` 用 `mp_policy=MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16)` + `reshard_after_forward=True`。
- **解法**：`from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy`（从 fsdp 导入，非 .wrap）。

## 18. NpuFusedAdamW 与 FSDP2 不兼容
- **症状**：`optim.step()` 报 `UnsupportedOperatorException: npu.get_npu_format.default`（无 fake impl）。
- **根因**：FSDP2 内部用 meta/fake tensor，融合优化器要 `npu.get_npu_format` 真实算子，无 fake impl。
- **解法**：FSDP2 下用 `torch.optim.AdamW`。融合路径靠 SDPA→npu fusion attention + TASK_QUEUE（bf16 靠 MixedPrecisionPolicy），不要强用融合优化器。

## 19. FSDP2 卡数不足会 OOM
- **症状**：9B 模型 2 卡 `optim.step()` OOM（~72GB/卡）。
- **根因**：每卡 = 参数×16字节/N（fp32 master+m+v+grad）。9B/2卡=4.5B×16=72GB > 65GB。
- **解法**：先算 `params×16/N ≤ free×0.85` 定卡数。9B 需 ≥8 卡（~18GB/卡）。smoke 别用 2 卡测 9B。

## 20. FSDP2 保存 ckpt 得到的是分片不是全量
- **症状**：`torch.save(model.state_dict())` 存出的 9B ckpt 只有 2.1GB（本应 ~17GB），后续加载 eval 报缺键/形状不对。
- **根因**：FSDP2（fully_shard）的 `model.state_dict()` 返回的是 **DTensor 分片**（每 rank 只含 1/N 本地 shard，但 `.shape` 仍显示全局形状，具迷惑性）。直接 save 只存了 rank0 的 1/8。
- **解法**：每个 DTensor 参数调 `.full_tensor()` 聚合全量（collective，需所有 rank 一起调）再存：
```python
from torch.distributed._tensor import DTensor
sd = model.state_dict()
full = {}
for k, v in sd.items():
    ft = v.full_tensor() if isinstance(v, DTensor) else v
    if r == 0:
        full[k] = ft.detach().cpu().to(torch.bfloat16)   # 转 bf16 减半(9B 36GB→18GB)
    del ft
if r == 0:
    torch.save(full, ckpt_path)
```
- 备选：用 `torch.distributed.checkpoint`（DCP）做分片 checkpoint，但要单卡 eval 时仍需聚合，故 full_tensor 更省事。

## 21. torch.load 默认 weights_only=True 拒载 NPU optimizer state
- **症状**：resume 时 `torch.load(ckpt_path)` 报 `_pickle.UnpicklingError: Weights only load failed... Unsupported global: torch_npu.utils.storage._rebuild_npu_tensor`。
- **根因**：torch 2.6+ 把 `torch.load` 默认 `weights_only` 从 False 改成 True；resume ckpt 的 optimizer state 含 NPU tensor（`_rebuild_npu_tensor`），不在默认 allowlist，被拒载。
- **解法**：resume 分支用 `torch.load(ck_path, map_location="cpu", weights_only=False)`。本地自存的可信 ckpt，安全。**只对 model 权重 ckpt（`cpt_model_state.pt`，纯 CPU tensor）则不受影响**——只有含 NPU optimizer state 的 resume ckpt 需要。
- 已在 cpt_train.py.tmpl / cpt_fsdp.py.tmpl resume 分支 + resume.md 修复（smoke 验证：resume 从 step3 续到 step5 正常）。

## 16. Qwen3.5 是 hybrid linear/full attention
- **症状**：比标准模型慢、编译图大。
- **根因**：chunked delta rule 有 Python 嵌套循环，Dynamo 展开成巨图；torch fallback 慢。
- **解法**：接受 Eager 较慢；图模式仅长训练/推理用；真正解法是等上游 NPU 原生线性注意力算子。

## 22. `nproc` 误报单核（OMP_NUM_THREADS=1）
- **症状**：`nproc` 返回 1，误以为单核，怪 CPU。
- **根因**：环境里 `OMP_NUM_THREADS=1`，coreutils 的 `nproc` 会受 `OMP_NUM_THREADS`/`OMP_THREAD_LIMIT` 影响取最小值。
- **解法**：用 `nproc --all` / `lscpu` / `os.cpu_count()` / `os.sched_getaffinity(0)` 查真实核数。实战：机器实际 640 核（8 NUMA × 80），CPU 并非瓶颈，别被 `nproc`=1 误导。

## 23. HCCL 卡间互联慢（RoCE 未配置）→ FSDP2 通信成为主瓶颈
- **症状**：FSDP2 训练 50s/step，但 torch_npu.profiler 显示 HcclAllGather+HcclReduceScatter 占 ~97%，计算只占 ~3%。
- **根因**：卡间 RoCE IP 未配置（`hccn_tool -i <devid> -ip -g` 报 "no ip was preset"），HCCL 走慢速 PCIe（实测 ~2.16GB/s；正常 HCCS/RoCE 应 25-100GB/s）。
- **解法**：**选并行策略前先探测卡间互联**。`hccn_tool -i <devid> -ip -g` 看是否配了 RoCE IP；若通信带宽 ~GB/s 级（远低于 25GB/s），FSDP2 会通信-bound，改走模型并行（#24）。

## 24. 卡间互联慢时用朴素模型并行（device_map="auto"）替代 FSDP2
- **症状**：FSDP2 通信-bound 慢（见 #23）。
- **根因**：FSDP2 每步 all-gather 参数 + reduce-scatter 梯度，搬运 ~108GB 数据；互联慢时通信成为瓶颈。而模型并行拆层后参数各归其卡，无 all-gather。
- **解法**：`AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32, device_map="auto")` 自动把层拆到多卡，优化器直接放 NPU。实测 12B/4 卡从 50s/step 降到 ~5.5s/step（8.3×）。**单进程即可**：`.to(device)` 可微分，autograd 自动跨卡回传梯度，无需手写 P2P。显存 = 参数×16/N（12B/4卡 ≈ 48GB/卡，fp32 参数 12GB + m/v 24GB + 激活）。optimizer 用默认 `torch.optim.AdamW`（`foreach=True` 对多 device 参数可能不兼容，用默认 False）。见 references/parallel-strategy.md「模型并行」节。

## 25. bf16 权重 + 小 lr → 更新被精度吞掉（训练不前进）
- **症状**：用 bf16 权重 + lr 5e-6，loss 几乎不动。
- **根因**：bf16 相对精度 ~0.4%（~4e-3），lr 5e-6 的更新量远小于权重分辨率，加不到权重上。fp16 同理（~5e-4 仍不够）。
- **解法**：CPT 小 lr（≤1e-5）必须 **fp32 主权重**。device_map 用 `torch_dtype=torch.float32`；单卡用 fp32 主权重 + bf16 autocast。

## 26. 单卡 + CPU-offload 优化器反而更慢（别这么干）
- **症状**：把 fp32 主权重 + AdamW 放 CPU 省显存，结果 ~60s/step 比 FSDP2 还慢。
- **根因**：CPU AdamW over 12B fp32 要 ~42s/step（CPU 内存带宽 ~7GB/s 慢，`foreach=True` 也只降到 ~32s）；copy-back `master[n].to(device, bf16)` 若先传 fp32 再 cast 会传 48GB 而非 24GB（~10s）。
- **解法**：12B 全参训练别用 CPU-offload 优化器，用模型并行（#24）把优化器放 NPU。若必须 CPU-offload：`foreach=True` + 先在 CPU cast 成 bf16 再传（`master[n].to(torch.bfloat16).to(device)`）。

## 27. torch_npu.profiler 无 key_averages，用 operator_details.csv 解析
- **症状**：`prof.key_averages()` 报 `'profile' object has no attribute 'key_averages'`。
- **根因**：torch_npu 的 profiler 实现与 CUDA 不同。
- **解法**：`on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(logdir)` 落盘，解析 `<rank>_ascend_pt/ASCEND_PROFILER_OUTPUT/operator_details.csv`（列 Name / Device Self Duration(us) / Host Total Duration 等），按 Device Self Duration 聚合找 top 算子。

## 28. transformers 5.x apply_chat_template 返回 BatchEncoding 不是 list
- **症状**：`len(tok.apply_chat_template(msgs, tokenize=True))` 返回 2（以为是 2 个 token）。
- **根因**：transformers 5.x 返回 `BatchEncoding`（dict 含 input_ids/attention_mask），`len()` 是键数。
- **解法**：取 `ids["input_ids"]` 再 `tolist()`（模板已处理，新代码注意）。

## 29. mem_get_info 返回值语义混乱（负 free 是假象）
- **症状**：`torch_npu.npu.mem_get_info(i)` 返回的 free 可能为负，误判卡满了。
- **根因**：返回值语义与 CUDA 不同/含 reserved 内存，与真实可分配量不符。
- **解法**：以**实际分配测试**为准（`torch.empty(N, device='npu:i')` 逐级试 10/30/50GB），不要依赖 mem_get_info 的 free 判断卡是否可用。

## 30. 多模态模型（Gemma4Unified 等）文本 CPT 的注意点
- **症状**：`AutoModelForCausalLM` 加载多模态模型，vision/audio 塔无梯度；FSDP2 逐层分片找错层路径。
- **根因**：Gemma4Unified 文本塔在 `model.model.language_model`（48 层 `layers`），vision/audio 塔（`embed_vision`/`embed_audio`）很小且文本 CPT 无梯度；`lm_head.weight` 与 `embed_tokens.weight` 是 tied（同一 tensor）。
- **解法**：文本 CPT 全参训练即可（vision/audio 塔 grad=None，AdamW 自动跳过），或 freeze `embed_vision`/`embed_audio`。FSDP2/模型并行取层路径是 `model.model.language_model.layers`，不是 `model.model.layers`。

## 31. eval 阶段 fp32 加载 12B 单卡 OOM → 用 bf16 加载评估
- **症状**：eval 时 `from_pretrained(torch_dtype=float32)` + `.to('npu')` OOM。
- **根因**：fp32 权重 48GB + 激活超单卡 61GB。
- **解法**：eval 用 bf16 加载（24GB）。标准 attention 模型（Gemma4 sliding/full）无纯 bf16 数值问题（pitfalls #4 仅针对线性注意力）。

## 32. kill -9 后 NPU 显存残留
- **症状**：kill -9 训练进程后立刻重跑，`model.to(device)` OOM（"2.5GB reserved, 76MB free"），但 ps 无残留进程。
- **根因**：SIGKILL 不让进程清理 NPU 显存，driver 异步回收有延迟。
- **解法**：kill 后 `sleep` 几秒 + `torch.npu.empty_cache()` 再重跑；或改用 SIGTERM 优雅退出。**正常退出的多卡训练也复现**：8 卡 FSDP2 `rc=0` 退出后数秒内跑单卡评估仍 OOM（card 仅剩几百 MB free，无残留进程），driver 回收延迟同样存在——多卡训练后跑单卡评估前先 `npu-smi` 确认卡空闲（或 `torch.empty(40GB, device='npu:0')` 实测）再开评估。

## 33. prepare_data 多卡未设 WORLD_SIZE → 训练内循环重复采样
- **症状**：DDP/FSDP2 N 卡训练，`prepare_data` 只产出 `NUM_STEPS×BATCH_SIZE×1` 块（远少于 `×N`），DistributedSampler 把少量块分到 N 卡、每卡仅 1/N 块，`step%nb` 在 200 步内循环 N× 重复采样同一批。
- **根因**：`prepare_data.py` 的 `WORLD_SIZE` 默认 1；`run_env.sh` 历史上不导出它。多卡场景 need 算少。
- **解法**：多卡训练前 `export WORLD_SIZE=N` 再跑 `prepare_data`（`run_env.sh.tmpl` 已加 `WORLD_SIZE=${WORLD_SIZE:-1}` 旋钮）。脚本在 `WORLD_SIZE==1` 时打印 hint 提醒。单卡不受影响。

## 34. FSDP2 日志名 `step_loss_fsdp.jsonl` 与 plot 模板不匹配
- **症状**：`plot_loss.py` 报 `step_loss.jsonl`/`step_loss_ddp.jsonl` No such file，因为 FSDP2 训练写 `step_loss_fsdp.jsonl`，而 plot 模板只认无后缀或 `_ddp`。
- **解法**：`plot_loss.py.tmpl` 已改为自动探测（依次试 `_ddp`/`_fsdp`/无后缀 + glob 兜底），无需手动 cp/symlink。注意：若手动建符号链接，`ln -sf` 目标相对路径以**链接所在目录**解析，写成 `ln -sf logs/step_loss_fsdp.jsonl logs/step_loss.jsonl` 会指向 `logs/logs/...`（dangling）；要么用 `cp`，要么目标写文件名 `ln -sf step_loss_fsdp.jsonl logs/step_loss.jsonl`。

## 35. smoke 的 s/step 严重高估正式训练用时
- **症状**：2 步 smoke 显示 26s/step，据此估 200 步需 ~90min；实际正式跑稳态 3s/step，200 步仅 10min。
- **根因**：`s/step = 累计耗时/(step+1)` 把首次 import(~90s)+多卡加载模型+FSDP2/DDP 初始化全摊进前几步，步数少时 s/step 虚高；步数多了初始化被摊薄。
- **解法**：smoke 的 s/step **不可直接外推**正式训练用时。预估取正式 run 稳态步（累计平均越往后越准），或 `总耗时/NUM_STEPS`；用时表阶段 7 预估等正式 run 跑出稳态后再回填校准。

## 36. 多模态 tie=False 模型重映射须保留 lm_head.weight
- **症状**：照搬 tie=True 模型（如 0.8B）的重映射逻辑（依赖 `tie_weights()` 补 lm_head），到 tie=False 模型（如 9B）时 lm_head 真缺失 → 输出层随机，loss/评估异常。
- **根因**：同族不同规格模型 tie 配置不同；tie=False 时 lm_head 是 ckpt 里独立的顶层张量。
- **解法**：重映射前 `cat config.json` 看 `tie_word_embeddings`。tie=False 时**保留** ckpt 顶层 `lm_head.weight`（不 strip、不 drop），验证 `miss_lm_head=0`；tie=True 时 lm_head 缺失正常，调 `model.tie_weights()`。多模态 remap 见 references/multimodal-remap.md。

## 37. eval 载 `cpt_model_state.pt` 也需 `weights_only=False`（扩展 #21）
- **症状**：eval 阶段 `torch.load("cpt_model_state.pt")` 报 `UnpicklingError: Unsupported global: getattr was not an allowed global`。
- **根因**：#21 只说了 resume ckpt（含 NPU optimizer state）需 `weights_only=False`，并断言"model 权重 ckpt（`cpt_model_state.pt`，纯 CPU tensor）不受影响"——**该断言过强**。部分架构（如 Qwen3.5 文本头）的 `model.state_dict()` 含非纯-tensor 全局（`getattr` 等），torch 2.6+ 默认 `weights_only=True` 同样拒载。
- **解法**：eval 载 **任何本地自存的 cpt ckpt**（含 `cpt_model_state.pt`）一律 `torch.load(..., map_location='cpu', weights_only=False)`。本地自存可信，安全。`eval_cpt.py.tmpl` 已改。#21 的"纯 CPU tensor 不受影响"仅在 state_dict 确为纯 tensor 时成立，不能假设。

## 38. hf-mirror 大文件 Xet 401 → 用 ModelScope 或禁 Xet
- **症状**：用 `hf`（新版 huggingface_hub）/ `huggingface-cli` 从 `hf-mirror.com` 拉大文件（safetensors/分片），报 `RuntimeError: CAS Client Error: HTTP 401 Unauthorized, domain: cas-server.xethub.hf.co`；小文件正常。
- **根因**：新版 `hf` 默认走 Xet 后端做大文件重组，经 `cas-server.xethub.hf.co` 鉴权，hf-mirror 不代理该 Xet 鉴权域 → 401。ModelScope 的 file resolve 是普通 HTTP 直链，无此问题且国内更快。
- **解法**（按优先）：① 大权重文件优先 **ModelScope** `https://www.modelscope.cn/models/<id>/resolve/master/<file>`（`wget -c` 即可，国内 ~20MB/s，比 hf-mirror 美区 CDN ~0.8MB/s 快 ~25×）；② 必须用 hf-mirror 时设 `HF_HUB_DISABLE_XET=1`（或 `huggingface_hub<0.30`）回退普通 HTTP LFS；③ `git clone` + `git-lfs` 从 hf-mirror 整仓克隆也绕开单文件 Xet。校验：下完比对 `model.safetensors.index.json` 元数据字节数。

## 39. 系统 torchaudio .so 符号错配，链式挡住 transformers 多模态文本头导入
- **症状**：`from transformers.models.<mm>_xxx import XxxForCausalLM`（多模态模型的文本头类）报 `OSError: torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl`；而 `from transformers import AutoModelForCausalLM` 正常。
- **根因**：共享镜像里系统 torchaudio 按另一个 torch 版本编译，符号不匹配；transformers 多模态模型模块链式触发 `import torchaudio` 即崩（grep 源码未必直接引用，是间接加载）。纯文本 LM 不触发。
- **解法（首选正式版，默认）**：装与当前 torch ABI 匹配的正式版 torchaudio（见 #43 对应表与命令），`import` 即正常，**无需打桩**。系统 site-packages 只读时 `pip install --user torchaudio==<匹配版>`，用户 site 优先级高于系统，直接覆盖坏包。装完 `python -c "import torchaudio,torchvision; from transformers.models.qwen3_5 import Qwen3_5ForCausalLM"` 验证导入链不崩。
- **解法（fallback 打桩，最后手段）**：**仅当 ≥3 次尝试安装匹配正式版仍失败**（torch 太新无配套 release、离线无 mirror、aarch64 无 wheel 等）才用桩。模板**默认不打桩**——`cpt_train.py.tmpl`/`eval_cpt.py.tmpl` 顶部默认 `import torchaudio, torchvision` 走正式版；只有显式 `export STUB_MM_FALLBACK=1` 时才装桩 finder（并打印 WARN 提示优先装正式版）。桩代码见下（同时拦 torchaudio 和 torchvision，torchvision 需带 `__getattr__` sentinel 的 `_StubModule`）：
```python
import sys, types, importlib.abc, importlib.machinery
class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torchaudio' or fullname.startswith('torchaudio.'):
            return importlib.machinery.ModuleSpec(fullname, self)
    def create_module(self, spec): return types.ModuleType(spec.name)
    def exec_module(self, module): module.__version__ = '0.0.0'; module.__file__ = '<stub>'
sys.meta_path.insert(0, _StubFinder())
```
注意：桩模块必须带 `ModuleSpec`（仅设 `sys.modules['torchaudio']=ModuleType(...)` 会因 `__spec__ is None` 再报 `ValueError`）；torchvision 需更强的桩——`from torchvision.io import ImageReadMode` 要取属性，空 `ModuleType` 会 `ImportError`，需 `__getattr__` 返回 sentinel 的 `_StubModule`（完整代码见 #43）。模板 `cpt_train.py.tmpl`/`eval_cpt.py.tmpl` 顶部默认 `import torchaudio, torchvision` 走正式版；**仅 `export STUB_MM_FALLBACK=1`（≥3 次装正式版失败后）才装桩并打 WARN**。同模式适用于 soundfile/librosa 等其它坏掉的音频可选依赖。

## 40. torch 是 `+cpu` build 不代表没 NPU
- **症状**：`torch.__version__ == '2.8.0+cpu'`，误以为该机器无 NPU 支持或装错 torch。
- **根因**：昇腾镜像常用 `torch==2.8.0+cpu`（无 CUDA 扩展）配 `torch_npu`——NPU 后端由 `torch_npu` 在 import 时注册，不依赖 torch 的 CUDA build。`+cpu` 仅表示无 CUDA kernels，对 NPU 训练无影响。
- **解法**：以 `torch.npu.is_available()` / `torch.npu.device_count()` 为准判断 NPU 可用性，别被 `+cpu` 后缀误导。版本匹配只需 `torch_npu` 版本与 `torch` 主次版本对齐（如 torch 2.8.0 ↔ torch_npu 2.8.0.post4）。

## 41. `set_env.sh` 路径可能不存在 / CANN env 已注入 base
- **症状**：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 报 `No such file or directory`；或该机器 `LD_LIBRARY_PATH`/`ASCEND_TOOLKIT_HOME` 已在 base profile 注入，根本不需要 source。
- **根因**：不同镜像 CANN 安装路径不同（有的是 `ascend-toolkit/set_env.sh`，有的只有 `driver/`，有的 env 已在 `/etc/profile` 注入）。
- **解法**：`run_env.sh` 里对 `set_env.sh` 做 `[ -f ]` 条件 source，不存在则跳过（env_probe 阶段先 `python -c "import torch_npu; print(torch.npu.is_available())"` 验证 env 是否已就绪）。不要硬 source。

## 42. torch_npu 不支持新芯片 soc（Ascend950 系列：950PR/950DT）→ 升级 torch+torch_npu
- **症状**：`torch.npu.set_device(0)` / `torch.npu._lazy_init()` 报 `RuntimeError: Unsupported soc version: Ascend950PR 9579` 或 `Ascend950DT xxxx`（950 系列两款之一，或其它新型号）；CANN 版本被判 `"X is invalid or not supported yet"`。`device_count()`/`get_device_name()` 可能能返回（不触发 lazy_init），误导以为 NPU 可用，但真正初始化/算子执行即崩。
- **根因**：镜像预装的 `torch_npu` 版本太旧，C++ 层 soc 版本映射表里没有 950 系列芯片。Ascend950 系列（**950PR 与 950DT 两款**，2026 年新芯片，CANN 9.0.0-beta.2 已支持，`platform_config` 下有 `Ascend950PR_*.ini` 与 `Ascend950DT_*.ini`）在 `torch_npu 2.7.1.post2.dev` 里都不识别——两款同根因、同解法。
- **解法**：升级到支持 950 系列的 torch_npu 新版本（与 torch 配套）。**950PR 已实测可行组合**：`pip install torch==2.12.0 torch_npu==2.12.0`（torch_npu 2.12 识别 950PR，matmul 正常，HBM free 131.8GB）。**950DT** 同属 950 系列、同 CANN 9.0.0-beta，预期同方案（torch2.12+torch_npu2.12）适用，但**需在 950DT 实机验证** `set_device`+一次 `x@x` matmul 后才算确认。注意 torch_npu wheel 对 torch 精确 pin（`torch_npu==2.12.0` 要求 `torch==2.12.0`），pip 会一并升级 torch（默认拉 CUDA 版 nvidia 依赖数 GB；纯 NPU 机器想省可加 `--index-url https://download.pytorch.org/whl/cpu` 装 torch CPU build，`==2.12.0` 仍匹配 `2.12.0+cpu`）。升级后务必 `python -c "import torch,torch_npu; torch.npu.set_device(0); x=torch.randn(8,8,device='npu:0'); print((x@x).sum().item())"` 实测算子可执行（别只看 device_count）。
- **判定要点**：`device_count()` 返回 >0 ≠ NPU 可用；必须 `set_device` + 真实算子能跑通才算。旧 torch_npu + 950 系列芯片（950PR 或 950DT）必中此坑。env_probe 记 `torch_npu.__version__` + `npu-smi info -t board -i 0`（Chip Name，匹配 `Ascend950(PR|DT)` 即 950 系列，都需检查 soc 支持）。

## 43. torchvision 也与 torch ABI 错配（扩展 #39，不只 torchaudio）
- **症状**：升级 torch（如 2.12）后，`from transformers.models.<mm>_xxx import XxxForCausalLM` 报 `RuntimeError: operator torchvision::nms does not exist` 或 torchvision `.so undefined symbol`；而 `from transformers import AutoTokenizer` 正常。导入链：多模态文本头 → `modeling_utils` → `loss_utils` → `image_transforms` → `image_utils` → `from torchvision.io import ...` 即崩。
- **根因**：同 #39，系统 torchvision 按旧 torch 编译，符号/算子注册不匹配；transformers 多模态模型模块间接 `import torchvision`。#39 只覆盖 torchaudio，**torchvision 是同一类坑但常被漏**。
- **解法（首选正式库，默认）**：装匹配当前 torch 的正式版 torchvision/torchaudio，import 即正常，无需打桩。**torch 2.8.0+cpu 实测可行组合（2026-08-24，aarch64 Ascend910）**：`pip install --user torchaudio==2.8.0`（华为云 mirror 有 cp311 manylinux aarch64 wheel，自动 pin `torch==2.8.0`，与 `2.8.0+cpu` ABI 对齐），torchvision 0.23.0 已预装匹配。原坏包是 `torchaudio 2.11.0+cpu`（太新，引用 torch 2.11 符号 `torch_library_impl`，torch 2.8 无此符号 → undefined symbol）——典型"装新了"而非"装旧了"。torch 2.12.x 实测组合：`pip install torchvision==0.27.1`（自动带 +cu130，匹配 torch 2.12.1+cu130）+ `torchaudio==2.11.0`（华为 mirror 最新；虽版本号对应 torch 2.11，实测与 torch 2.12.1 import 兼容）。torchvision 版本对应表：torch 2.8↔torchvision 0.23.x、torch 2.12↔0.27.x、torch 2.11↔0.26.x、torch 2.10↔0.25.x（torchaudio 版本号则与 torch 主版本对齐：torchaudio 2.8.0↔torch 2.8、2.12.0↔torch 2.12）。装完 `python -c "import torch,torchvision,torchaudio; from transformers.models.qwen3_5 import Qwen3_5ForCausalLM"` 验证导入链不崩。注意 torch_npu 与 torch patch 版本：torchvision 0.27.1 会拉 torch 到 2.12.1，torch_npu 2.12.0（pin torch==2.12.0）实测仍兼容 2.12.1（950PR matmul 正常），但若 torch_npu 报 ABI 错则固定 torch==2.12.0 + 装匹配 0.27.0。**判断"装新了 vs 装旧了"**：报 `undefined symbol: <某 torch 符号>` 多半是 torchaudio/torchvision 比当前 torch 新（引用了新 torch 才有的符号）；装匹配主版本号即可。
- **解法（fallback 打桩）**：**仅当 ≥3 次尝试安装匹配正式版仍失败**（torch 太新还没配套 release、离线无 mirror、aarch64 无 wheel 等）才用桩。模板 `cpt_train.py.tmpl`/`eval_cpt.py.tmpl` 顶部**默认 `import torchaudio, torchvision` 走正式版（不打桩）**；只有显式 `export STUB_MM_FALLBACK=1` 时才装下面的 stub finder（同时拦 torchaudio **和** torchvision），并打印 `[WARN] STUB_MM_FALLBACK=1` 提示优先装正式版。torchvision 需更强的桩——`from torchvision.io import ImageReadMode` 要取属性，空 `ModuleType` 会 `ImportError`，需 `__getattr__` 返回 sentinel 的 `_StubModule`：
```python
class _Any:
    def __call__(self, *a, **k): return _Any()
    def __getattr__(self, n): return _Any()
_SENT = _Any()
class _StubModule(types.ModuleType):
    def __getattr__(self, name): return _SENT
class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torchaudio' or fullname.startswith('torchaudio.') \
           or fullname == 'torchvision' or fullname.startswith('torchvision.'):
            return importlib.machinery.ModuleSpec(fullname, self)
    def create_module(self, spec): return _StubModule(spec.name)
    def exec_module(self, module): module.__version__ = '0.0.0'; module.__file__ = '<stub>'
sys.meta_path.insert(0, _StubFinder())
```
模板 `cpt_train.py.tmpl`/`eval_cpt.py.tmpl` 顶部默认 `import torchaudio, torchvision` 走正式版；`STUB_MM_FALLBACK=1`（≥3 次装正式版失败后）才装上述桩 finder 并打 WARN。**默认优先正式库，stub 仅兜底且必须显式开启。**

## 44. 容器内存限制 + torch.load 大 ckpt 到 CPU → OOM SIGKILL(137)（静默死）
- **症状**：eval/加载阶段 `torch.load(ckpt, map_location='cpu')` 一个大 state（如 4B fp32 = 16.8GB），叠加已建的 fp32 模型（16GB），进程被 `SIGKILL`（exit 137），**无任何 traceback/错误输出**（日志突然断在加载那行）。系统 `free -g` 显示 hundreds of GB available，看似不该 OOM。
- **根因**：容器（K8s/Docker）环境的 cgroup 内存限制远小于宿主机视图（`free -g` 的 total 是宿主机，容器实际 limit 可能仅 24-32GB）；且 `/sys/fs/cgroup/memory.max` 可能不在标准路径或无权限读，查不到真实 limit。CPU 峰值 = 模型(16GB) + torch.load 的 state(16.8GB) + load_state_dict 复制 ≈ 32-49GB，超 cgroup limit → OOM killer 发 SIGKILL，python 来不及打印异常。
- **解法**：别把大 state 一次性 load 到 CPU。改用 **`torch.load(ckpt, map_location='npu:0', weights_only=False)`** 把张量直接加载到 NPU（HBM 通常 32-128GB free，够），再 `load_state_dict` 跨设备 in-place copy 到 CPU 模型（逐参数 copy，CPU 只持模型本身 16GB，峰值≈16GB < limit）。即：`model = XxxForCausalLM(tc)`(CPU fp32) → `sd = torch.load(CKPT, map_location='npu:0')` → `model.load_state_dict(sd, strict=False)`(NPU→CPU copy) → `model.to(device)`。NPU 侧 16GB，CPU 侧 16GB，互不叠加。
- **判定要点**：exit 137 + 无 traceback + 大 ckpt 加载 = 强烈暗示容器 OOM。查 cgroup limit：`cat /sys/fs/cgroup/memory.max`(v2) 或 `memory.limit_in_bytes`(v1)，不在标准路径则 `free -g` 不可信。训练时 `torch.save` 大 ckpt 不受影响（流式写盘，内存峰值低）；只有 `torch.load` 反序列化会峰值翻倍。

## 45. meta 模型 + assign + to_empty 路线的两个陷阱（#44 的错误尝试）
- **症状 A**：为省 CPU 内存用 `with torch.device('meta'): model=XxxForCausalLM(tc)` + `load_state_dict(sd, assign=True)` + `model.to_empty(device=device)`，结果模型输出全是随机（NLL=ln(vocab_size)、ppl=vocab_size、acc=0），等于没加载权重。
- **根因 A**：`to_empty(device)` 会把**所有**未初始化 tensor（含 `assign` 刚赋入的真实权重）重新 materialize 成未初始化的 empty，覆盖了真实数据。assign 后**不能**再调 to_empty。
- **症状 B**：去掉 to_empty 后，前向崩 `NotImplementedError: Cannot copy out of meta tensor; no data!`（在 RoPE `self.inv_freq`）。
- **根因 B**：`inv_freq` 等 non-persistent buffer（`persistent=False`，不在 `state_dict`）在 meta 建模时被跳过初始化，`assign` 只覆盖 state_dict 里的键，这些 buffer 仍是 meta tensor，前向取值即崩。
- **解法**：大模型 eval 加载**不要走 meta 路线**，直接用 #44 的"正常建 fp32 CPU 模型 + `map_location='npu'` 加载 + 跨设备 copy"——inv_freq 正常初始化、权重正确、CPU 峰值低。meta+assign 仅在确信无 non-persistent buffer 且不用 to_empty 时才可，风险高不推荐。

## 46. 多模态 remap 在 CPU 三份叠加 → OOM 137（#44 的 remap 变体，最隐蔽）
- **症状**：多模态→文本头 remap（`REMAP=1` 路径）一启动即 `SIGKILL`（exit 137），日志只停在 import/`LD_PRELOAD` 那行，**连 `[remap]` 打印都没到**就被杀，无 traceback。`free -h` 显示 754GB available，NPU HBM 也几乎全空——两边都"看起来不该 OOM"，极具迷惑性。
- **根因**：remap 路径在 CPU 上同时持有**三份**大对象（比 #44 的两份更狠）：
  1. 文本头模型 `XxxForCausalLM(tc).to(torch.float32)`：4.2B × 4B = **16.8GB**
  2. `load_file(shard, device='cpu')` 全分片加载到 CPU（bf16）：**9.3GB**
  3. remap 时 `v.to(torch.float32)` 把命中的权重再转一份 fp32 到 `sd` 字典：**16.8GB**
  - 峰值 ≈ 43GB。容器（K8s/Docker）cgroup 限制常远小于宿主视图（本机 `free` 显示 754GB，但 `cat /sys/fs/cgroup/memory.max` = **32GB**），43GB > 32GB → OOM killer 发 SIGKILL，python 来不及打印。
- **与 #44 的区别**：#44 是 `torch.load` 单个 ckpt + CPU 模型（两份）；本条是 **remap 路径的三份叠加**（模型 + 多分片 ckpt + dtype 转换副本），且常发生在"我以为 remap 只是复制几个 tensor"时，三份峰值更易被忽略。`free` 误导性在此条最严重——宿主 754GB 让人完全不设防。
- **解法**：整个 remap 搬到 **NPU** 上做（HBM 128GB 绰绰有余），CPU 不留任何大副本：
  ```python
  with torch.device(device):              # 模型直接在 NPU 上构造，不占 CPU
      model = XxxForCausalLM(tc)
  model = model.to(torch.float32)        # NPU 上转 fp32
  for f in shard_files:
      shard = load_file(path/f, device=device)   # 分片直加载到 NPU
      sd = {nk: v.to(torch.float32) for ...命中的重映射键...}
      model.load_state_dict(sd, strict=False)     # partial：仅本片 key 写入
      loaded_keys.update(sd.keys())
      del shard, sd; torch.npu.empty_cache()
  model.tie_weights()  # tie=True 时绑 lm_head
  ```
  关键三点：①`with torch.device(device)` 让模型构造落在 NPU（不是 CPU 建完再 `.to(device)`，那一步 CPU 仍持 16.8GB）；②`load_file(device=device)` 分片直加载 NPU，不在 CPU 暂存全量 ckpt；③逐分片 `load_state_dict(strict=False)` 累积写入 + 立即 `del + empty_cache`，NPU 峰值≈模型(16.8GB)+单分片(~5GB)，CPU 峰值≈0。
- **eval 同理**：加载 `cpt_model_state.pt`（16.8GB fp32 全量 state_dict）用 `torch.load(CKPT, map_location='npu:0', weights_only=False)`，别 `map_location='cpu'`；NPU 上建模型 + NPU 上 load，再 `load_state_dict`。
- **判定要点**：exit 137 + 无 traceback + remap/load_model 阶段即死 + `free` 显示充足 = **第一时间查 cgroup**：`cat /sys/fs/cgroup/memory.max`(v2) / `memory.limit_in_bytes`(v1)。**永远先 `cat /sys/fs/cgroup/memory.max`，不要信 `free`。** 950PR 机器实测：cgroup=32GB，宿主=754GB，Qwen3.5-4B remap 三份 43GB 必中此坑。本条已验证解法（搬 NPU 后 smoke 一次通过，200 步训练正常）。

## 47. MLX(Apple Silicon) 格式扩散模型 → NPU 无法加载，换同源 PyTorch 基座
- **症状**：用户给 `FastVideo/FastMetal-1.3B-QAD` 等模型，`ls` 看到 `mlx_dit.json`/`mlx_dit.safetensors`（而非 `config.json`+标准 safetensors），README/library_name 标 `mlx`，tags 含 `int8/quantization/apple-silicon`。直接当 diffusers/transformers 加载失败或加载到错误架构。
- **根因**：MLX 是 Apple Silicon 专有格式（`format_version`+`_class_name` 的 json + MLX 约定 safetensors），torch_npu/torch 无法用标准 `from_pretrained` 加载；且常带 int8 量化，权重无法直接塞回 fp16/fp32 PyTorch 模型。
- **解法**：找**同源 PyTorch 基座**。FastMetal 是 `FastWan2.1-T2V-1.3B-Diffusers` 的 MLX 量化版 → 改用该 PyTorch 基座的 `transformer/`（`WanTransformer3DModel` 标准 diffusers 加载）。判定基座：看模型的 `BaseModel`/`base_model` 字段或 README "based on"。**FastMetal 特定 finetune 权重(MLX)无法继承**，须在 README 注明用同源 PyTorch 基座替换。
- **判定要点**：出现 `mlx_*.json/safetensors` 或 `library_name=mlx` → MLX。阶段 0 范式判定时必查此点（核心原则 10），否则后续全错。

## 48. hf_hub snapshot_download 在 flaky 连接上对大文件死亡螺旋 → 改 wget -c
- **症状**：`snapshot_download` 下一个 >2GB 文件，`.incomplete` 长到 ~4.98GB 后**突然回到 52MB 从头重下**，反复如此，永远下不完；日志 `Fetching N files` 进度卡在某个文件，ETA 飙到 1 小时+。
- **根因**：hf_hub 下载到 `.incomplete`，flaky 连接中断后**校验失败即丢弃整个文件从头重下**（不复用已下字节）。大文件 + flaky = 死亡螺旋。与 #38(Xet 401)不同：这是普通 LFS 连接中断重下。
- **解法**：停 hf_hub，改 **`wget -c --tries=0 --timeout=30 --retry-connrefused --waitretry=3 -q <url> -O <path>`**——`-c` 断点续传，连接断从断点续，不重头；`--tries=0` 无限重试。大文件分片可并行多 wget（各分片独立 `-c`）。`wget` 通常已装（无 aria2c 时用 wget）。URL 用 `https://hf-mirror.com/<repo>/resolve/main/<path>`。
- **判定要点**：`.incomplete` 大小忽大忽小（从 GB 回到 MB）= 确认死亡螺旋，立即换 wget -c。

## 49. 扩散 VAE 编码输入 layout [B,C,T,H,W]，不是 [B,T,C,H,W]
- **症状**：`vae.encode(video)` 报 `RuntimeError: expected input to have 3 channels, but got 21 channels`（21=T 被当成 channels）。
- **根因**：decord `get_batch` 返回 `[T,H,W,3]`，转 tensor 后常见误成 `[B,T,C,H,W]`；Wan/diffusers VAE 的 conv3d 权重是 `[out,C,kt,kh,kw]`，要求输入 dim1 = C。
- **解法**：`[T,3,H,W]` → `permute(1,0,2,3).unsqueeze(0)` 得 `[1,3,T,H,W]`（视频）。图片同理 `[1,3,H,W]`。VAE 编码全在 NPU 上做（CPU 撞 cgroup 32GB，#44/#46）。
- **判定要点**："got T channels, expected 3" → layout 错，permute 成 [B,C,T,H,W]。

## 50. text_encoder 巨大(11-23GB) → 预计算缓存 embedding 后释放 / 零嵌入兜底
- **症状**：扩散模型的 text_encoder（UMT5-XXL 等）下载 11GB+ 耗时极长（flaky 下几十分钟到 1 小时+），阻塞训练启动；或加载占满显存。
- **根因**：T2V/T2I 的条件编码器本身是另一个大模型（UMT5-XXL ~5.7B），与 DiT 同量级，下载/加载成本高。
- **解法（三选一，按优先级）**：
  1. **预计算+缓存+释放**（首选）：编码完全部 caption → cache emb 到磁盘 → `del text_encoder; empty_cache()` → 训练只用缓存 emb，省 11GB 显存、不再依赖 TE。
  2. **换同族 bf16 版**：基座 text_encoder 常是 fp32(23GB)，同族量化版(如 FastMetal 的 text_encoder bf16 ~12GB)省一半，下载/加载都快。
  3. **零嵌入近似无条件兜底**（TE 完全不可得时）：`torch.zeros(1,L,D)` 作 text emb，DiT 流匹配 velocity 预测仍正常学习视频分布，loss 正常下降；代价是无文本条件对齐，**须 README 注明**。TE 可后台继续下载，就绪后重跑带真实 caption 版。
- **判定要点**：TE 下载 ETA > 训练+评估总时长 → 用方案 2/3 不阻塞；TE 分片不全(`len(shards)<5`)→ 自动退零嵌入（`prepare_generative_data.py.tmpl` 已内置此判定）。

## 51. 扩散 DiT forward 返回 tensor|dict 兼容 + 组件分离 smoke 顺序
- **症状A**：`out = transformer(...)` 后 `out.sample` 报错（tensor 无 .sample）或 `out[0]` 把 `[1,16,...]` 索引成 `[16,...]` 丢 batch 维 → 后续 loss/shape 全错。
- **根因A**：diffusers 不同版本/不同 Transformer 类的 forward 返回类型不一致——可能直接是 tensor，可能是 dict(`out['sample']`)，可能是 dataclass(`out.sample`)。
- **解法A**：`out = dit(hidden_states=, timestep=, encoder_hidden_states=, return_dict=True); if isinstance(out,dict): pred=out['sample']; elif hasattr(out,'sample'): pred=out.sample; else: pred=out`。**绝不**无脑 `out[0]`。
- **症状B**：直接合练 2 步报 NPU 算子错（3D conv / 某 attention 算子不支持/fallback），不知是 DiT 还是 VAE 的问题。
- **根因B**：扩散 pipeline 有 VAE(3D conv 重)+DiT(attention)+text_encoder 三个独立大组件，NPU 算子支持问题可能出在任一处，合练时定位难。
- **解法B**：**先单组件 dummy smoke**：①DiT 前向反向用 `dummy latent [1,C,Tl,Hl,Wl]` + `dummy emb [1,L,D]`；②VAE encode 用 `dummy [1,3,T,H,W]`。各自通过再合练。早抓算子问题（如 Wan DiT attention + VAE 3D conv 在 950PR 实测都通过）。
- **判定要点**：扩散 CPT 的 smoke 必须组件级，不能跳到合练。

## 52. 扩散 VAE latent 归一化符号：`(raw-mean)/latents_std` 除，不是乘（910 回归发现）
- **症状**：流匹配训练 loss 虚高 ~10×（如 910 上 loss 21 vs 同 convention 950PR 的 1.63），但 loss 仍下降、CPT 似"有效"。
- **根因**：`AutoencoderKLWan.encode().latent_dist.sample()` 返回 **raw latent（无自动归一化）**。diffusers Wan pipeline 的 decode 是 `raw = stored * config.latents_std + config.latents_mean`，故 DiT 原生 latent 空间 = `stored = (raw - latents_mean) / latents_std`（**除**，单位方差）。若误写成 `*(latents_std)`（乘），latent 放大 std² 倍 → velocity target 同比放大 → MSE 虚高 ~std⁴（std~2 → ~16×）。raw 不归一化虽能跑（loss ~var(lat)+1），但与预训练 DiT 期望 scale 不对齐、CPT ckpt 不能直接进原 pipeline 生成。
- **解法**：prepare 阶段 `lat = vae.encode(v).latent_dist.sample(); lat = (lat - latents_mean) / latents_std`（**除**），`latents_mean/std/z_dim` 读自 `vae.config`。eval 生成时 `vae.decode(lat)` 前先逆归一化 `lat = lat*latents_std + latents_mean`。`prepare_generative_data.py.tmpl` 默认 `LATENT_NORM=1` 开启；`=0` 存 raw（原 950PR 方式）。
- **判定要点**：同 convention 下 loss 量级比正常高一个数量级 → 查归一化符号（乘 vs 除）。

## 53. UMT5 必须用 UMT5EncoderModel，不要用 T5EncoderModel（910 回归发现）
- **症状**：用 `T5EncoderModel.from_pretrained(text_encoder_dir)` 加载 UMT5，`LOAD REPORT` 报 `encoder.block.{1..23}.layer.0.SelfAttention.relative_attention_bias.weight | UNEXPECTED`，embedding 静默退化（loss 仍跑但文本条件不对齐）。
- **根因**：UMT5 在**每层** block 都有 `relative_attention_bias`（universal 多语言机制），而 T5 只在 block 0 有；用 T5EncoderModel 加载会把 block1-23 的偏置当 UNEXPECTED 丢弃。
- **解法**：`from transformers import UMT5EncoderModel; te = UMT5EncoderModel.from_pretrained(...)`（每层都有偏置，正确加载，无 UNEXPECTED）。CLIP 走 CLIPTextModel。`prepare_generative_data.py.tmpl` 已改用 UMT5EncoderModel。
- **判定要点**：UMT5 加载时 `LOAD REPORT` 出现 `block.{1..23}.relative_attention_bias UNEXPECTED` → 用错了类。

## 54. transformers 5.x + `_keep_in_fp32_modules` 的 DiT，`from_pretrained` 崩（910 回归发现）
- **症状**：`WanTransformer3DModel.from_pretrained(dir, torch_dtype=float32)` 报 `ValueError: low_cpu_mem_usage cannot be False when keep_in_fp32_modules is True`；加 `low_cpu_mem_usage=True` 也救不回。
- **根因**：WanTransformer3DModel 有 `_keep_in_fp32_modules = ["rope","time_embedder","scale_shift_table","norm1","norm2","norm3"]`；transformers 5.15.1 的检查要求 `low_cpu_mem_usage=True` 才能启用 keep_in_fp32，但该 kwarg 被前置逻辑吞掉/不生效。950PR（torch 2.12 + 另一版 transformers）未触发此检查。
- **解法**：回退到 `from_config` + 手动 `load_state_dict` 绕开该机制：
  ```python
  cfg = json.load(open(f"{BK}/config.json"))
  dit = WanTransformer3DModel.from_config(cfg, torch_dtype=torch.float32)
  sd = load_file(f"{BK}/diffusion_pytorch_model.safetensors")
  dit.load_state_dict(sd, strict=False)  # miss=0 正常(unexp 是非持久 buffer/rope freqs)
  ```
  `cpt_diffusion.py.tmpl`/`eval_diffusion.py.tmpl` 的 `load_backbone` 已内置 `try from_pretrained; except → from_config` 回退。
- **判定要点**：报错信息含 `keep_in_fp32_modules`/`low_cpu_mem_usage` → 走 from_config 回退。跨 910/910C/950 都须带此 fallback。

> 注：WanTransformer3DModel 不支持 `gradient_checkpointing_enable`（小 latent 如 `[1,16,4,32,56]` 在 64GB+ 卡上无需）；模板 `load_backbone` 已 `try/except` 仅防崩，每次跑打一行 `[gc]` 噪声可忽略。

## 55. libgomp TLS 冲突：sklearn + torch_npu（音频-LLM/processor 间接 import sklearn）
- **症状**：`import AutoProcessor`/`Qwen2AudioForConditionalGeneration`（或任何触发 transformers generation 模块）时报 `ImportError: .../scikit_learn.libs/libgomp-*.so: cannot allocate memory in static TLS block`。
- **根因**：transformers 5.x 的 `generation/candidate_generator` `from sklearn.metrics import roc_curve` 间接 import sklearn；sklearn 的 libgomp 在 torch_npu 的 OMP 已填满静态 TLS 后 dlopen 失败。librosa 也走同 libgomp（同症）。
- **解法**：脚本顶部**先 `import sklearn`**（让 libgomp 早入静态 TLS）再 `import torch, torch_npu`。无需 LD_PRELOAD。`librosa` 非必需时跳过（音频-LLM 用 soundfile 读 flac→numpy→processor 算 mel 即可，mel 用 numpy/scipy，MLS/fleurs 16k 无需重采样）。
- **判定要点**：报错栈含 `candidate_generator`/`sklearn`/`libgomp`/`static TLS` → import 顺序问题。

## 56. Qwen2-Audio processor kwarg 是 `audio=`（单数），返回 `input_features`/`feature_attention_mask`
- **症状**：`proc(text=, audios=[arr], ...)` 打印 `Keyword argument 'audios' is not a valid argument for this processor and will be ignored`，返回只有 `input_ids`+`attention_mask`（无 mel/audio_features）→ 后续 forward 报 audio 维度错。
- **根因**：Qwen2AudioProcessor.__call__ 的参数名是 **`audio=`（单数）**，不是 `audios=`；返回 key 是 `input_features`(mel `[B,128,~3000]`) + `feature_attention_mask`，**不是 `audio_features`**。
- **解法**：`proc(text=[full,...], audio=[arr,...], padding=True, return_tensors="pt", sampling_rate=16000)`；取 `inp.input_features` + `inp.feature_attention_mask`。
- **判定要点**：processor 返回无 mel / "audios ignored" warning → kwarg 名错。

## 57. Qwen2-Audio model forward 须传 `input_features` + `feature_attention_mask`
- **症状**：`model(input_ids=, attention_mask=, input_features=af, labels=)` 报 `AttributeError: 'NoneType' object has no attribute 'to'`（在 modeling_qwen2_audio forward 的 `feature_attention_mask.to(target_device)`）。
- **根因**：Qwen2AudioModel.forward 需要 `input_features` **和** `feature_attention_mask`（音频 mel 的 valid mask，audio_tower 用它忽略 padding 帧）；只传 input_features → feature_attention_mask=None。
- **解法**：`model(input_ids=ids, attention_mask=am, input_features=af, feature_attention_mask=fam, labels=labels)`。
- **判定要点**：forward 报 NoneType.to 在 `feature_attention_mask` 行 → 漏传。

## 58. audio 特殊 token 必须 mask in labels（漏则 loss 虚高 ~8×）
- **症状**：bs=1 loss ~1.2 正常，bs≥2(批量+padding) loss 突涨到 ~10（同模型同数据）。
- **根因**：`<|audio_bos|>`/`<|AUDIO|>`(重复)/`<|audio_eos|>` 是**条件输入 token**（被 audio embedding 替换），不该作为预测目标。批量 padding 时这些 audio token 可能落到 prompt-mask 区之外（pinp 与 inp 的 audio token 展开/padding 位置错位），漏进 labels → loss 计算在"预测 audio token"上 → 虚高。
- **解法**：显式 mask：`tok=proc.tokenizer; aids={tok.convert_tokens_to_ids(t) for t in ["<|audio_bos|>","<|AUDIO|>","<|audio_eos|>"]}; for a in aids: if a is not None and a>=0: labels[ids==a]=-100`。叠加 prompt 段 mask + pad mask。
- **判定要点**：bs1 正常、bs2 loss 飙升一个数量级 → 查 audio token 是否在 labels。

## 59. 模型并行 device_map="auto" 不自动分卡 + 2卡 visible + freeze 勿用类名"Audio"
- **症状A**：`device_map="auto"` 加载后 `first param device=npu:0, device_map=npu:0`（全在单卡），训练时全量 CPT 单卡 OOM。
- **根因A**：device_map="auto" 只在模型**单卡装不下**时才分卡；7B bf16=16.6GB<64GB → auto 全放 card0，无切分。
- **解法A**：`max_memory={0:"18GB",1:"42GB"}`（cap card0 模型量）强制切分到 npu:0,npu:1，模型+optim+grad 随切分摊两卡。
- **症状B**：`ASCEND_RT_VISIBLE_DEVICES=0,1` 没生效，只看到 card0（"Device 1 not available"）。
- **根因B**：run_env.sh 默认 `ASCEND_RT_VISIBLE_DEVICES=0`（单卡）；脚本内 `os.environ.setdefault` 不覆盖已设值；CANN 在进程启动时读 env。
- **解法B**：shell 里 `source run_env.sh` 后 `export ASCEND_RT_VISIBLE_DEVICES=0,1` 再启动 python（覆盖，且 CANN 启动时读到）。
- **症状C**：freeze 后 `trainable=0`（全冻）。
- **根因C**：用 `"Audio" in type(m).__name__` 匹配，误中顶层 `Qwen2AudioForConditionalGeneration`/`Qwen2AudioModel`（都含 "Audio"）→ 冻结全部。
- **解法C**：按**子模块路径名**匹配 `"audio_tower" in n or "multi_modal_projector" in n`（只冻真正的音频塔+投影器）。
- **判定要点**：device_map 只列 npu:0 / Device 1 not available / trainable=0 → 分别查 max_memory / VISIBLE_DEVICES / freeze 匹配方式。

## 60. 7B 全量 CPT 2×64GB: DDP bf16 OOM, fp32 master OOM; 模型并行 bf16 fits
- **症状**：7B(Qwen2-Audio) 全量 CPT 在 2×910C(64GB) 上：DDP bf16 master `optim.step` 处 OOM（59.5/61GB）；fp32 master model-parallel 也 OOM。
- **根因**：7.755B 可训 LLM 全量：bf16 master(权重16.6+AdamW m,v 31+grad 15.5)=63GB/card(DDP 每卡复制)>61GB 可用→OOM ~2GB；fp32 master(权重33+AdamW 62+grad 31)=126GB total>2×61=122→OOM ~4GB。
- **解法**：**模型并行 device_map+max_memory 强制 2 卡切分 + bf16 master**——模型+optim+grad 随切分摊两卡 ~31GB/card fits。bf16 master 有精度吞小更新风险(#25)，但 7B + lr2e-5 + 标准attention 实测有效(held-out CE 降82%)。要 fp32 master 需 ≥3 卡或 FSDP2 分片。
- **判定要点**：7B 全量 CPT 在 2×64GB DDP OOM → 改模型并行 device_map+max_memory(bf16)；仍要 fp32 master → 加卡或 FSDP2。

> 注：音频-LLM 全流程见 `references/audio-llm-cpt.md`。GGUF/MLX 音频模型(如 OmniAudio)换同源 PyTorch 基座(#47)。

## 61. 训练 loss 趋近 0 = 过拟合红旗（数据量不足多 epoch 循环）（MOSS CPT 实证）
- **症状**：CPT 训练 loss 降到 ~0.01-0.05（near-0），但 held-out CE/loss 几乎不改善（-3% 甚至正 delta）。
- **根因**：数据量太少 + 步数多 → epoch 过多 → 模型**记忆训练样本**（train_loss→0）但未学到可泛化模式。训练 loss 大降 ≠ 训练有效。
- **实证**：MOSS 0.9B + AISHELL-4，8 会议 × 200 步(25 epoch)→train_loss 0.012(过拟合)、held-out -3%；50 会议 × 300 步(6 epoch)→train_loss 0.110(健康)、held-out -87%。**扩数据(8→50)比加步数更有效**——数据量是泛化的关键杠杆，非步数。
- **解法**：①train_loss < 0.05 且 held-out 不改善 → 停止加步，**扩数据**；②train_loss 降到 0.1-0.5 + held-out 持续改善 = 健康收敛；③优先扩数据再加步（小数据多步=白烧算力）。
- **判定要点**：train_loss near-0 + held-out delta 小/正 → 过拟合，扩数据。
- **caveat**：上述 MOSS 对比同时改了数据量(8→50)和步数(200→300)，非严格受控实验。推断"数据为主因"的依据是 train_loss 从 0.012→0.110 **升高**（纯加步只会降 train_loss，不会升），故数据扩容是主因——但严格结论需固定步数变数据量的受控实验。

## 62. held-out 评估须用独立 test split，非同源不同段（MOSS CPT 实证）
- **症状**：用"同源数据不同段"（如同一会议 120-180s 段）作 held-out → 评估结果不干净，低估/高估泛化。
- **根因**：同源不同段仍共享说话人/声学/主题分布 → 非真正独立；模型过拟合到该源后，"held-out" 段也被记忆或不被学到，指标失真。
- **解法**：held-out 用**独立 test split**（完全不同会议/样本），训练完全未见过。MOSS 实证：同源不同段 held-out -3%（误导），test 独立会议 held-out -87%（真实泛化）。
- **判定要点**：held-out delta 与训练 loss 不匹配（train→0 但 held-out 不动）→ 查 held-out 是否独立；换 test split 重评。

## 63. 音频-LLM forward API 未标准化——各模型 forward kwarg + processor 返回键不同（MOSS vs Qwen2-Audio 实证）
- **症状**：用 Qwen2-Audio 的 forward kwarg（`input_features` + `feature_attention_mask`）跑 MOSS → 报 NoneType.to 或维度错。
- **根因**：音频-LLM 的 forward 签名 + processor 返回键**未跨模型标准化**：Qwen2-Audio forward 用 `input_features`+`feature_attention_mask`；MOSS 用 `input_features`+`audio_feature_lengths`（无 `feature_attention_mask`）。audio token 名也不同（Qwen2-Audio `<|audio_bos|>/<|AUDIO|>/<|audio_eos|>` vs MOSS `<|audio_start|>/<|audio_pad|>/<|audio_end|>`）。
- **解法**：①audio token 发现**可通用**（`discover_audio_token_ids` 按名含 "audio" 抓所有模型 ✓）；②forward kwarg **自动探测通用**——`cpt_audio_llm.py.tmpl` 用 `inspect.signature(model.forward)` 探测 forward 参数，从 `("feature_attention_mask","audio_feature_lengths")` 中取 forward 接受 + processor 返回的那个，通过 `**audio_kwargs` 动态传（不再写死 Qwen2-Audio 的 `feature_attention_mask`）。换音频-LLM 无需改模板 forward 调用。③若 forward 接受其它 mask kwarg 名，扩展 `("feature_attention_mask","audio_feature_lengths", ...)` 列表。
- **判定要点**：跨音频-LLM 报 forward kwarg / NoneType.to → 查该模型 forward 签名 + processor 返回键。

## 64. 长音频数据集分段——CPT 取前 N 秒段，非整会长音频（AISHELL-4 实证）
- **症状**：会议/长音频数据集（AISHELL-4 15-32min/会议）整条喂训练 → mel 巨大 + context 极长 → 慢/OOM。
- **解法**：取每条前 N 秒段（60-120s）做 CPT 样本（`soundfile.read(fl, start=0, frames=N*sr)`）；标注同步截到该段（`parse_textgrid(tg, max_sec=N)` 只取 <N 秒的 interval）。MOSS 等支持长音频的模型 CPT 用短段即可学转写模式，不必整条。
- **判定要点**：长音频（>5min）数据集 → 分段，非整条。

## 65. 多通道音频取 ch0（AISHELL-4 等 mic array）
- **症状**：`soundfile.read` 返回 `shape=(N, 8)`（多通道），直接传 processor 报 channel 维错或 mel 算错。
- **解法**：`arr,sr=sf.read(fl, ...); if arr.ndim>1: arr=arr[:,0]`（取第 0 通道）。AISHELL-4 是 8 通道 mic array，取 ch0 = 近场主麦克风，与 16k 单声道训练期望一致。
- **判定要点**：`arr.ndim>1` → 取 `[:,0]`。

## 66. 标注格式转换——TextGrid/RTTM/JSON → 模型输出格式 transcript（AISHELL-4 实证）
- **症状**：数据集标注不是纯文本 transcript（AISHELL-4 用 Praat TextGrid IntervalTier + RTTM），无法直接当训练 target。
- **解法**：解析标注 → 转模型输出格式。TextGrid：解析 IntervalTier（每 speaker tier 的 `[xmin,xmax,text]`），按 xmin 排序，speaker 映射 `[S01]/[S02]`（首见序），格式化 `[Sxx] HH:MM:SS,mmm text`，去 `<sil>` 标记。RTTM（`SPEAKER file 1 onset dur ... spk`）类似。目标格式对齐模型输出（如 MOSS 输出 `[S01] 时间戳 文本`）。
- **判定要点**：数据标注非纯文本 → 写转换器（TextGrid/RTTM/JSON → model-output-format）。

> 注：#61/#62 是**通用**规律（适用于文本 LM / 扩散 / 音频-LLM 所有 CPT 范式），#63-66 是音频-LLM 特有。数据规模 vs 泛化规律详见 `references/audio-llm-cpt.md`「数据规模 vs 泛化」节。
