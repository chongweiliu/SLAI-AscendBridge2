# LoRA SFT 踩坑全集（昇腾 NPU）

> 2026-08-28 在 Ascend910 + CANN 9.0.0 + torch 2.10 + torch_npu 2.10 + transformers 5.16.1 + peft 0.20，对 Qwen3.5-0.8B（混合 linear+full attention 多模态架构）做 LoRA SFT 实战踩中。每条都会在同类模型/同类环境重复出现，务必逐条规避。

## #1 CANN 默认 `latest` 软链常指向 8.3.RC1（无 FlashAttention 编译内核）

**症状**：attention 模型训练/推理报 `Cannot find binary for op FlashAttentionScore` / `FlashAttention` 找不到 .o。
**根因**：`/usr/local/Ascend/ascend-toolkit/latest` 常指向 8.3.RC1；其 OPP 里 FlashAttention 只有头文件、无编译好的 .o 内核二进制。对所有 head 配置都缺。
**解决**：
```bash
# 优先 source 9.0.0 (含 FA 内核); 不要无脑 source latest
if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi
```
枚举已装 CANN：`ls /usr/local/Ascend/ascend-toolkit/ /usr/local/Ascend/`。
**绕过**：`attn_implementation="eager"`（纯 torch 算子 softmax+matmul，无 FA 内核依赖，NPU 可用，稍慢）。CANN 8.3.RC1 也能靠 eager 跑通。

## #2 `device_count` 返回值不算数，必须实测 set_device + matmul

**症状**：`torch.npu.device_count()` 返回 N，但 `set_device` 或算子执行失败。
**根因**：新芯片（Ascend950 系列）torch_npu 太旧不识别；或 CANN 环境未 source。
**解决**：env_probe 必实测：
```python
import torch, torch_npu
torch.npu.set_device(0)
print(float((torch.randn(4,4,device='npu:0') @ torch.randn(4,4,device='npu:0')).sum()))
```
以 `torch.npu.is_available()` 为 NPU 可用判据（不是 device_count）。950 系列需 torch_npu ≥ 2.12.0。

## #3 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` + 手动 `import torch_npu`

**症状**：不设此变量，import torch 后 `torch.npu` 不存在或崩。
**根因**：torch_npu autoload 在某些环境崩溃。
**解决**：环境变量（import torch 前设）`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，然后代码里显式 `import torch_npu` 注册后端。

## #4 `apply_chat_template(tokenize=True)` 返回 BatchEncoding（dict），`len()` 是键数不是 token 数

**症状**：用 `len(tok.apply_chat_template(msgs, tokenize=True))` 拿"token 数"，结果恒为 2，导致 assistant label 区间全错（span [2,2) 无效），全部样本被 skip。
**根因**：`tokenize=True`（无 return_dict）在较新 transformers 返回 `BatchEncoding`（含 `input_ids`/`attention_mask` 两键），`len()` = 键数 2。
**解决**：取 token 数用 `seg['input_ids']`（再 `len`）；或直接用 `tokenize=False` 拿字符串再 `tok(...)`，规避歧义。字符偏移法见 label-masking.md。

## #5 Qwen3.5 的 `return_assistant_tokens_mask=True` 返回 assistant_masks 全 0

**症状**：用标准 `return_assistant_tokens_mask=True` 做 SFT label 掩码，结果 mask 全 0，label token = 0，全部样本 skip（同 #4 似的静默失败）。
**根因**：Qwen3.5 的 chat template（jinja）未正确实现 assistant mask 分支，返回全 0。其它模型（Llama-3 等）通常正常，但不能假设。
**解决**：**先试 `return_assistant_tokens_mask`，若 `int(am.sum()) == 0` 则 fallback 字符偏移法**（见 label-masking.md）。务必自检 `n_label > 0` 否则训练前就发现。

## #6 `NpuFusedAdamW.zero_grad(set_to_none=True)` 报错

**症状**：`ValueError: set_to_none is not supported in fused optimizers`。
**解决**：`opt.zero_grad(set_to_none=False)`。融合优化器内部用 TypedStorage 管理，不支持置 None。会有 `TypedStorage is deprecated` 的 UserWarning，无害。

## #7 uv sync 在跨文件系统时 hardlink 失败退化为 full copy，torch 复制要 5 分钟

**症状**：`uv sync` 卡在 `Resolved ... in 1ms` 后长时间无新输出；`.venv` 不增长；看似进程挂死。
**根因**：uv cache 在 `/root/.cache/uv`（fs A），venv 在 `/mnt/model/...`（fs B），跨 fs hardlink 不支持，退化为 full copy。torch 解压后 ~900MB，复制 5+ 分钟，期间无输出。
**解决**：`export UV_LINK_MODE=copy` 抑制警告并接受；或把 venv 与 cache 放同 fs。耐心等，看 `.venv` 大小增长判断存活。**好消息**：venv 目录 `mv` 到同 fs 另一位置后，python `sys.prefix` 会自动更新，venv 仍可用（无需重建）。

## #8 增量前缀法（encode(msgs[:i])）在部分 chat template 上边界不对齐

**症状**：用"编码前缀取长度差"求 assistant 区间，结果 label 里混入 user token（如 `<|im_start|>user\n嗯`）。
**根因**：某些 chat template（含 Qwen3.5）在"末轮是 assistant（终止态）"和"末轮是 user（中间态）"时，对尾部 `<|im_end|>\n` 的处理不一致，前缀 token 序列不是完整序列的真前缀，边界漂移。
**解决**：**改用字符偏移法**（渲染全字符串 + token offset 映射 + find assistant 块），见 label-masking.md。此法不依赖前缀一致性，最稳健。

## #9 transformers 版本不够 → `qwen3_5` 等 model_type 不被识别

**症状**：`KeyError: qwen3_5` 或 `AutoModelForCausalLM` 找不到对应类。
**根因**：新模型架构需对应 transformers 版本。Qwen3.5（qwen3_5）需 ≥ 5.16.1（其 config 标 `transformers_version: 4.57.0.dev0`，但正式支持在 5.x）。
**解决**：装足够新的 transformers（`transformers>=5.16.1` 或 git HEAD）。加载前 `AutoConfig.from_pretrained` 看 `model_type` 是否被 `_mapping` 识别。

## #10 多模态模型用 `AutoModelForCausalLM` 加载即得文本头（纯文本 SFT 无需图像）

**现象**：Qwen3.5-VL / Qwen2-VL 等多模态模型，`AutoModelForCausalLM.from_pretrained` 加载后 top children = `model`/`lm_head`，无 vision tower——直接当文本 LM 训即可，纯文本对话 SFT 不需要 image input。
**注意**：`dtype=torch.bfloat16`（transformers 5.x `torch_dtype` 已 deprecated，用 `dtype`）。`attn_implementation="eager"` 兼容性最好。

## #11 ASCEND_RT_VISIBLE_DEVICES 不要默认写死 0 号

**症状**：0 号卡被别的进程占满，训练 OOM 或排队。
**解决**：先 `npu-smi info` 看 HBM 占用，把 `ASCEND_RT_VISIBLE_DEVICES` 设到空闲/低占用卡。visible device 对应芯片序号。

## #12 chat template 要求至少一个 user（前缀无 user 会报 "No user query"）

**症状**：增量编码 `msgs[:1]`（仅 system）时 `TemplateError: No user query found`。
**解决**：不要编码 system-only 前缀。字符偏移法只编码完整对话一次，天然规避。

## #13 大模型(>=13B)单卡 64GB 放不下训练，须多 die 切分

**症状**：27B（bf16 权重 54GB）在单张 64GB HBM 卡上 OOM（`Tried to allocate ... 60.45 GiB already allocated; 61.27 GiB total`）。即使梯度检查点 + seq 1024 也 OOM——权重+LoRA 优化器状态+激活就 >61GB。
**根因**：单卡 64GB 减去 54GB 权重仅剩 ~7GB，LoRA AdamW fp32 状态（120M×12B≈1.4GB）+ 激活（seq 1024 grad-ckpt ~3GB）+ 碎片即超。
**解决**：`DEVICE_MAP=auto` + `MAX_MEMORY="0:16GiB;1:16GiB;2:16GiB;3:16GiB"` 强制跨 4 die 切分（每 die ~13.5GB 权重，留 ~50GB 激活空间）。`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`。**注意 device_map="auto" 默认不切**——模型能塞进单卡时它全塞 die0，大模型必须配 MAX_MEMORY 强制分散。单卡放得下的小模型不用这套。

## #14 device_map 多 die 训练触发 GE 图引擎，需 CANN 依赖 decorator + scipy

**症状**：device_map 多 die 加载/前向报 `Environment_Error_Import_Python_Module_Failed(EC0010): No module named 'decorator'` / `No module named 'scipy'`，级联 `GEInitializeV2 failed` / `InitTbeFunc failed`。
**根因**：device_map 走 accelerate 的 pipeline hooks，触发 torch_npu 图引擎(GE)/TBE 编译器初始化，它们 `import decorator`、`import scipy`。单卡 eager 不触发 GE（故小模型没事）。
**解决**：`uv pip install decorator scipy psutil attrs`（venv 里装）。这些是 CANN GE 的隐式 python 依赖，torch_npu 不声明。

## #15 多 die device_map 时必须用 AdamW，不能用 NpuFusedAdamW

**症状**：device_map 切分后 LoRA 参数跨多 die，NpuFusedAdamW（融合优化器）只支持单设备参数，报错。
**解决**：device_map 模式用 `torch.optim.AdamW`（非融合，稍慢但跨设备可用）。模板已自动：`if DEVICE_MAP: opt=AdamW else: NpuFusedAdamW`。

## #16 device_map + peft 训练须 `enable_input_require_grads()`

**症状**：device_map + gradient_checkpointing 下 backward 报输入无 grad / 不更新。
**解决**：`get_peft_model` 后调 `model.enable_input_require_grads()`（grad ckpt 冻结了输入 grad，peft 需要重开）。`gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})`。

## #17 OOM 后设备 507033 "Failed to start device"：先杀残留进程

**症状**：OOM 后重跑报 `SetDevice error 507033 / Failed to start device`。
**根因**：上一个 OOM 进程没释放干净，设备处于不健康态。
**解决**：`pkill -9 -f scripts/lora_train.py`，`npu-smi info` 确认 HBM 回到 ~3GB/die 再重跑。

## #18 极长输出数据集（如 CoT，中位 6585 token）验证时 max_new 与 GT 长度差大，绝对 ROUGE 必低

**症状**：hint-tuning-1k 数据 GT 中位 6585 token，validate 用 max_new=200 生成，base/LoRA 的绝对 ROUGE-L 都 <0.12。
**根因**：生成只 ~200 token 却与 6000+ token GT 比，重叠天然极小。不代表微调无效。
**判据**：看**相对 Δ + 样例格式**。27B 例：base ROUGE 0.092→LoRA 0.114（+25%），且 LoRA 学会了 `timent`/`intuition` 推理标记格式（base 用英文 meta-preamble），格式学习信号明确。长输出版本微调该看格式/风格匹配，不是字面 ROUGE 绝对值。

## #19 FSDP2 大模型 LoRA 训练 OOM 的两个真根因（expandable_segments + 缺 model.train()）

> 2026-08-29 27B/Qwen3.5 实测定位。曾误判为"torch_npu fully_shard 不实现真分片"——**错误**，
> 严格逐阶段显存测量推翻了它：fully_shard 分片完全正常（848 层参数全变 DTensor，4-die 每 rank shard 11.34GB，前向 peak 17.8GB）。真根因是下面两个，均可修。

**根因①（环境混杂变量）：`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 破坏 FSDP2 all-gather buffer 复用。**
- 症状：逐层 fully_shard 后前向 OOM，每卡 resident ~60GB（≈全模型 54GB），看起来像"没分片"。
- 根因：expandable_segments 分配器使 FSDP2 逐层 all-gather 的输出 buffer 无法跨层复用回收，64 层的 buffer 累积 ≈ 全模型大小。
- 解决：**FSDP2 训练禁用 expandable_segments**，用系统默认或 `max_split_size_mb:256`：
```python
if "expandable_segments" in os.environ.get("PYTORCH_NPU_ALLOC_CONF", ""):
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
```
（注意 expandable_segments 对单卡/device_map 路径无害且有益，只与 FSDP2 的 buffer 复用冲突。）

**根因②：没调 `model.train()` → transformers 的 gradient_checkpointing 被跳过。**
- 症状：修掉①后静态显存正常（16.28GB=11.3 分片+5 顶层），但首个前向仍 OOM（涨到 ~59GB）。
- 根因：transformers 各模型的 GC 只在 `self.training=True` 时生效；`from_pretrained` 默认 eval 模式。
  无 GC 时 seq2048 全层激活 ~43GB——**FSDP2 是数据并行，每卡过全模型 → 43GB 全压单卡**；
  device_map 流水线版没炸是因为激活被流水线分摊到各卡（43/4≈11GB/卡），同样的 bug 被 topology 掩盖了。
- 解决：训练循环前 `model.train()`（同时验证 `gradient_checkpointing` flag 生效）。

**FSDP2 可用配置（27B 实测跑通）**：
- 加载：bf16 CPU `low_cpu_mem_usage=True`（**不** `.to(dev)`；init_empty_weights/meta-init 反而不行——Qwen3.5 VL 的 state_dict 键与 meta 模型键零重叠灌不进）
- 分片：**只逐层 `fully_shard(layer)`，不切顶层**（顶层会在前向一次性 all-gather 整模型 54GB buffer）；顶层 embed/lm_head/norm(~5GB) 手动 `.to(dev)`
- `mp_policy=MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16)`，`reshard_after_forward=True`
- 优化器 `torch.optim.AdamW`（NpuFusedAdamW 与 FSDP2 不兼容）；`model.enable_input_require_grads()`（peft+GC 必需）
- **`model.train()`**；环境不带 expandable_segments
- 保存 adapter：只取 `requires_grad` 参数，DTensor 用 `.full_tensor()` 聚合（**集合通信，所有 rank 都要执行**，放 `if is_main` 里会挂死）；peft 的 save_pretrained 对 DTensor state 会炸（"invalid python storage"），手动存
- 启动：`python -m torch.distributed.run --nproc_per_node=N`（torchrun 脚本的 shebang 在 venv 被移动后失效，用 `python -m` 稳）

**实测（27B, seq2048, grad-ckpt, 150 步）**：4-die FSDP2 = 12.59s/step（有效 batch=4）→ **3.15 s/样本**；4-die device_map = 6.87s/step（batch=1）→ 6.87 s/样本。**FSDP2 每样本提速 2.18×**，且同 wall-clock 看 4× 样本。

## #20 device_map 流水线也要 model.train()；多卡两路线选型

**device_map 流水线同样缺 `model.train()`**（GC 没生效被流水线分摊掩盖）。补上 train() 后 GC 生效，可支持更大 micro-batch（此前 bs=2 OOM 的边缘问题部分源于此）。

**多卡路线选型（27B 实测）**：
| 路线 | 每样本耗时 | 特点 |
|---|---|---|
| device_map 流水线（4-die） | 6.87s | 简单稳，激活分摊各卡；mb=1 时有流水线气泡，mb≥2 提速 27% |
| **FSDP2 数据并行（4-die）** | **3.15s（2.18×）** | 每卡过全模型但 GC 后显存 ~25GB 富余；每层 all-gather/reduce-scatter 通信被 HCCS 掩盖 |
| FSDP2（8-die） | 实测见运行记录 | shard 6.75GB/卡，有效 batch=8 |

两路线都可用；大模型吞吐优先 FSDP2（按 pitfalls #19 配置），稳妥/简单优先 device_map。

## #21 MoE 模型 LoRA 实测（Qwen3.6-35B-A3B, 2026-08-29）：融合路由专家挂不上 LoRA

**实测结论**（16 卡 FSDP2, 2 步 probe + 完整探针）：
- **兼容性 ✅**：`qwen3_5_moe` 在 transformers 5.16.1 可加载（Qwen3_5MoeConfig）；FSDP2 逐层分片正常（40 层 1310 DTensor 参数）；训练跑通、loss 正常下降（1.93→1.79）。
- **关键限制：融合实现的 256 路由专家不是 `nn.Linear`，peft 的 target_modules 按名匹配挂不上**。实测 adapter 只覆盖 `linear_attn`(300 张量) + `self_attn`(80) + `mlp`=共享专家(240)，共 **21.2M 可训练参数**（若能挂上路由专家应 ~250M）。即 LoRA 只适配注意力+共享专家，路由专家不动——对风格/格式类 SFT 通常够用（共享专家承担了 FFN 适配），知识注入任务需知悉此限制。
- **显存**：权重按总参数（35.95B→71.9GB，所有专家驻留）；激活按 top-k 有效中间维（256 专家 top-8 × 512 + shared 512 = 4608）。实测 9.25GB/rank(16卡) vs 公式预测 14.8GB（公式偏保守 37%，安全向）。
- **步时陷阱（probe 稳态修正的由来）**：首步含 NPU 算子编译（38~210s），稳态仅 18~21s/step——**ETA 必须用末步增量（稳态），不能用平均值**（平均法曾把 45min 高估成 285min）。probe 已改为稳态口径。
- **瞬时不稳定**：16-rank 大模型运行后紧接重跑可能报 driver 级 OOM（`aclnnMm` 207001, "Failed to apply for memory"），卡显存却基本干净——等 30s 重试即可（同 #17 变体）。
