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
