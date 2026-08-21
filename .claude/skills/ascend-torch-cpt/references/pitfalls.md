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
- **解法**：文本/视觉 CPT 用不到音频，把 torchaudio 拦截成干净桩模块（带合法 `ModuleSpec`，可选依赖即跳过）。`sys.meta_path` 插 finder：
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
注意：桩模块必须带 `ModuleSpec`（仅设 `sys.modules['torchaudio']=ModuleType(...)` 会因 `__spec__ is None` 再报 `ValueError`）。模板 `cpt_train.py.tmpl`/`eval_cpt.py.tmpl` 顶部已加**带探测的防御版**（先 `try: import torchaudio`，坏了才桩，健康时 no-op）。同模式适用于 soundfile/librosa 等其它坏掉的音频可选依赖。

## 40. torch 是 `+cpu` build 不代表没 NPU
- **症状**：`torch.__version__ == '2.8.0+cpu'`，误以为该机器无 NPU 支持或装错 torch。
- **根因**：昇腾镜像常用 `torch==2.8.0+cpu`（无 CUDA 扩展）配 `torch_npu`——NPU 后端由 `torch_npu` 在 import 时注册，不依赖 torch 的 CUDA build。`+cpu` 仅表示无 CUDA kernels，对 NPU 训练无影响。
- **解法**：以 `torch.npu.is_available()` / `torch.npu.device_count()` 为准判断 NPU 可用性，别被 `+cpu` 后缀误导。版本匹配只需 `torch_npu` 版本与 `torch` 主次版本对齐（如 torch 2.8.0 ↔ torch_npu 2.8.0.post4）。

## 41. `set_env.sh` 路径可能不存在 / CANN env 已注入 base
- **症状**：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` 报 `No such file or directory`；或该机器 `LD_LIBRARY_PATH`/`ASCEND_TOOLKIT_HOME` 已在 base profile 注入，根本不需要 source。
- **根因**：不同镜像 CANN 安装路径不同（有的是 `ascend-toolkit/set_env.sh`，有的只有 `driver/`，有的 env 已在 `/etc/profile` 注入）。
- **解法**：`run_env.sh` 里对 `set_env.sh` 做 `[ -f ]` 条件 source，不存在则跳过（env_probe 阶段先 `python -c "import torch_npu; print(torch.npu.is_available())"` 验证 env 是否已就绪）。不要硬 source。
