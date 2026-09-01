---
name: ascend-torch-lora
description: 在华为昇腾 NPU（Ascend 910/910B/910C/950 等）上，用 PyTorch + torch_npu + peft 对任意 HuggingFace CausalLM（含多模态模型的文本头，如 Qwen3/3.5、Qwen2、Llama、GLM、Mistral 等）做 LoRA 监督微调（SFT / instruction tuning）的端到端技能。用户只需给出【模型权重路径】+【对话/指令数据集路径】即可启动：自动探测环境/依赖、数据自动转 chat 格式并对 assistant 轮做 loss 掩掩、自动发现 LoRA 目标模块、**自动选路（route_select：按模型大小+空闲卡自动决策单卡/FSDP2 数据并行/device_map 流水线，超参按模型尺寸/步数自动择优，精度优先性能自动最优）**、bf16 autocast + grad-ckpt + cosine 调度、**MoE 模型训练提速（MOE_IMPL=dense/gmm 数学等价补丁，短序列 -34% 步时 / 长序列 3-8×）**、报"昇腾不支持某算子"前的三步算子发现法（本机 CANN 接口 → gitcode.com/cann → 文档案例）、默认产出 loss 曲线图+公网可访问直链（catbox.moe/0x0.st/uguu.se 顺序尝试，外网全不通则降级表格）、概要总结、base vs LoRA 验证对比（ROUGE-L/字符重叠/生成长度）。只要用户要在昇腾上"LoRA 微调/SFT/指令微调/对话微调"某模型（含 MoE 模型如 Qwen3.6-A3B/DeepSeek 系），或提到 NPU + LoRA + 对话数据 + loss 曲线 + 验证，或 MoE 训练慢要提速，就应使用本技能——即使用户没明说"skill"。本技能基于 PyTorch + torch_npu + peft（不是 CUDA/deepspeed/unsloth）。与 [[ascend-torch-cpt]]（继续预训练，喂原始语料）区分：本技能是 SFT，喂的是 chat/指令对话数据并只对 assistant 回复算 loss。
---

# Ascend NPU LoRA 监督微调（SFT）技能

## 这是什么

把一个 HF 模型 + 一份对话/指令数据集，在昇腾 NPU 上端到端跑一遍 LoRA SFT，并产出：训练脚本、loss 曲线（含公网直链）、LoRA adapter 权重、概要总结、base vs LoRA 验证对比。

**适用模型类（一类，不是某一个）**：HuggingFace CausalLM 架构的文本生成模型。包括：
- 纯文本 decoder-only LM（Qwen2/Qwen3、Llama、GLM、Mistral、DeepSeek、Yi 等）
- 多模态模型的**文本语言头**（如 Qwen3.5-VL 的 `Qwen3_5ForConditionalGeneration` 用 `AutoModelForCausalLM` 加载即得文本 LM；Qwen2-VL/Qwen2.5-VL 同理）。纯文本 SFT 时不需要图像输入，加载文本头即可。

**与 CPT 的区别（务必分清，选错技能会白跑）**：
| 维度 | ascend-torch-cpt（继续预训练） | ascend-torch-lora（本技能，SFT） |
|---|---|---|
| 数据 | 原始语料（纯文本/文档） | 对话/指令（chat messages：user/assistant 多轮） |
| loss | 全 token 算 next-token CE | **只对 assistant 回复 token 算 loss**，user/system 掩 -100 |
| 训练对象 | backbone 全量或 LoRA | **LoRA 适配器**（冻结 base） |
| 评估 | PPL/acc/F1 | base vs LoRA 生成对比（ROUGE-L/字符重叠/长度） |
| 典型场景 | 喂领域语料再训一轮 | 教会模型按某风格/任务对话 |

用户给的是"对话数据"且要"LoRA"→ 用本技能；给的是"原始语料"要"继续预训练"→ 用 [[ascend-torch-cpt]]。

## 输入（用户给路径即可启动）
- **模型权重路径**（必填）：HF 目录（含 config.json/safetensors/tokenizer）。如 `/mnt/model/Qwen3.5-0.8B`。本地没有则从 modelscope / hf-mirror 下载（设 `HF_ENDPOINT=https://hf-mirror.com`）。
- **对话数据集路径**（必填）：jsonl/json/parquet/csv 均可。每条需含多轮对话字段。`prepare_data.py.tmpl` 默认认 `{"dialogue":[{"student":...,"teacher":...}]}` 或 `{"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}` 两种格式自动识别；其它格式按 `references/label-masking.md` 的占位约定改 2 个解析函数即可。
- 可选：步数/轮次（epochs 与 steps 二选一，未给默认 150 步）、seq_len、batch、lr/lora_r 等超参（未给则由 `route_select.py` 按模型大小/步数/数据量自动择优，见 references/hyperparam-selection.md）、抽样比例（如取 50% 多轮）、用卡上限（--max-cards）、小模型是否多卡扩吞吐（--scale-up）。**单卡还是多卡、FSDP2 还是流水线由选路器按模型大小+空闲卡自动决定**，用户无需指定。
- 工作目录：统一建在 **SLAI-AscendBridge2 仓库根目录下的 `lora-ws/` 内**（无则新建）。每个模型一个子目录 **`lora-ws/<模型名>-lora/`**（模型名取权重路径最后一段，如 `/mnt/model/Qwen3.5-0.8B` → `lora-ws/Qwen3.5-0.8B-lora/`）。所有脚本与产物统一归档于此，不散落到仓库根或其他位置。

## 默认产出
- **loss 曲线 png + 公网直链**：上传顺序 catbox.moe → 0x0.st → uguu.se（见 `scripts/plot_loss.py.tmpl`）；外网全不通则降级表格展示（逐 step / 每 10 步 + 首末 10 步均值 + min）。本地 png 与 `loss.jsonl`/`loss_raw.json` 始终保存。临时直链会过期，长期以本地为准。
- **LoRA adapter 权重**：`adapter_model.safetensors` + `adapter_config.json` + tokenizer，可直接 `PeftModel.from_pretrained` 加载推理。
- **验证报告**：`validation_results.json` + `validation_report.md`（base vs LoRA 对比）。
- **概要总结**：`SUMMARY.md`（配置/超参/loss 表/结论）。

## 端到端流程（7 步）

### 1. 环境探测与搭建
- NPU/CANN：`npu-smi info`、`source /usr/local/Ascend/cann-9.0.0/set_env.sh`（**不要用 latest，它常指向 8.3.RC1 无 FA 内核**，详见 references/pitfalls.md #1）。
- torch/torch_npu/transformers/peft 版本：CANN 9.0.0 配 torch 2.10.0 + torch_npu 2.10.0；transformers 版本须支持目标模型的 `model_type`（如 qwen3_5 需 ≥5.16.1）。
- 建独立 venv：`uv sync --extra ascend`，pyproject 见 `references/env-pyproject.md` 或直接复用 [[uv-env-setup]] 模板。
- 必设环境变量：`TORCH_DEVICE_BACKEND_AUTOLOAD=0`（规避 autoload 崩，手动 import torch_npu）、`TASK_QUEUE_ENABLE=1`、`ASCEND_RT_VISIBLE_DEVICES=<空闲卡>`（由第 2 步选路器给出，不默认写死 0 号）。`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（单卡/流水线用；**FSDP2 路线脚本会自动禁用**，它破坏 all-gather buffer 复用，pitfalls #19）。
- 健康检查：`torch.npu.is_available()` + `torch.npu.set_device(0)` + 一次 `x@x` matmul（device_count 不算数，pitfalls #2）。

### 2. 自动选路与超参（`route_select.py.tmpl`）——精度优先，性能自动最优
```bash
python scripts/route_select.py --model-dir <模型> --seq-len 2048 [--steps N | --epochs E --data train.jsonl] [--max-cards K] [--scale-up] [--probe --ws-dir <ws>]
```
选路器读**模型参数量/层数/hidden/vocab（safetensors+config）+ npu-smi 空闲卡 + seq/steps/数据量**，输出：
- **路线决策**（按"精度安全前提下性能最优"排序）：
  ① 单卡放得下（权重+LoRA优化器+激活 ≤ 0.85×HBM）→ **single_card**（零通信最快迭代；`--scale-up` 可改多卡扩吞吐）
  ② 单卡放不下 → **FSDP2 数据并行，用满空闲卡**（HCCS 下实测 device_map 的 2.2-4.7×/样本；`--max-cards` 限卡数）
  ③ FSDP2 激活也放不下（超大 seq）→ **device_map 流水线**（激活也分摊，兜底）
- **显存估算**（27B/seq2048 实测对照校准过）
- **自动超参**（模型尺寸/步数/数据量感知，规则见 references/hyperparam-selection.md）：LR（<1B→2e-4, 1-30B→1e-4, >30B→5e-5）、r=16/alpha=32、warmup=10%×步数、grad_accum 凑有效 batch≈8、epochs↔steps 换算
- **可直接复制的 export 启动块**（含 ASCEND_RT_VISIBLE_DEVICES 与对应启动命令）
- **`--probe`（强烈推荐）**：自动跑 2 步试训（输出隔离到 ws/probe_run/ 不污染正式产物），一次性完成：
  ① **兼容性实测**（含 MoE——架构能否跑通、loss 是否合理，替代"理论支持"）
  ② **显存校准**：实测峰值 vs 公式预测，偏差落盘遥测账本 `outputs/route_calibration.jsonl`（积累越多尺寸越准）
  ③ **ETA 外推**：用**末步增量（稳态步时）**×总步数——首步含 NPU 算子编译（可达数倍步时），平均法会高估 ETA 数倍（实测 45min 被均法报成 285min）
**精度优先的保证**：三条路线的训练数值配置完全一致（bf16 autocast + grad-ckpt + eager + cosine），路线只影响并行方式不影响精度；训练前标签自检 + 训练后 base vs LoRA 验证门为强制步骤（见核心原则）。

### 3. 数据准备（`prepare_data.py.tmpl`）
- 抽样（如取 50% 多轮）：固定 seed 可复现。
- 转 chat messages：`[{"role":"system","content":...}, {"role":"user","content":student}, {"role":"assistant","content":teacher}, ...]`。
- **loss 掩掩（核心，详见 references/label-masking.md）**：只对 assistant 回复 token 算 loss，user/system 标 -100。用**字符偏移映射法**（稳健，不依赖 chat template 实现）：① `apply_chat_template(tokenize=False)` 渲染全字符串；② `tok(rendered, return_offsets_mapping=True, add_special_tokens=False)` 拿 token→字符映射；③ 在字符串中 find assistant 块的起止特殊标记（如 `<|im_start|>assistant\n` ... `<|im_end|>\n`）得字符区间；④ token 的 offset 落在任一 assistant 区间即为 label token。**优先试 `return_assistant_tokens_mask=True`，但若返回全 0（如 Qwen3.5 的 bug）则用字符偏移法兜底。**

### 4. 训练脚本（`lora_train.py.tmpl` / `lora_train_fsdp.py.tmpl`，由第 2 步选定）
- 加载：`AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, attn_implementation="eager")`。eager attention 在 NPU 纯 torch 算子可用，无 FA 内核依赖（CANN 8.3.RC1 也能跑，只是慢）。若 CANN 9.0.0 + 模型支持可试 sdpa。
- **MoE 模型提速开关（fsdp 模板）**：`MOE_IMPL=eager|dense|gmm`（默认 eager）。dense=稠密 bmm 补丁（短序列 T≲500 最快，实测 -34% 步时）；gmm=npu_grouped_matmul 分发（长序列 T≥1024 单层 3-8×、显存省 30%）。两者数学等价，选型与等价性验证协议见 references/moe-optimization.md。
- **LoRA 目标自动发现**：扫 `model.named_modules()`，取所有 `nn.Linear` 的末名，与候选 `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj, ...]` 交集作为 target_modules。混合注意力架构（如 Qwen3.5 的 gated delta network）还会命中 `in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/out_proj` 等——自动发现可覆盖，无需手填。
- 优化器：**`NpuFusedAdamW`**（昇腾融合优化器），`zero_grad(set_to_none=False)`（True 会报错，pitfalls #5）。
- 精度：bf16 autocast（`torch.autocast("npu", dtype=torch.bfloat16)`）。base 冻结，只 LoRA 参数 `requires_grad`。
- 调度：cosine + warmup。逐 step 记录 loss 到 `loss.jsonl`。
- `use_cache=False`（训练时）。
- **大模型(≥13B)多卡训练，两条路线都已实测跑通（27B/Qwen3.5）**：
  - **路线A：FSDP2 数据并行（吞吐优先，实测 2.18× 提速/样本）**——用 `scripts/lora_train_fsdp.py.tmpl`，`python -m torch.distributed.run --nproc_per_node=N` 启动。**必须**：① 禁用 `expandable_segments`（它破坏 FSDP2 all-gather buffer 复用致假 OOM，pitfalls #19 根因①）；② 训练前 `model.train()`（否则 GC 被跳过、全层激活压单卡，pitfalls #19 根因②）；③ 只逐层 `fully_shard` 不切顶层 + 顶层参数手动上 NPU；④ `torch.optim.AdamW`；⑤ adapter 保存用 `requires_grad` 参数 + `DTensor.full_tensor()`（所有 rank 都要执行，集合通信）。27B 4-die 实测：3.15s/样本。
  - **路线B：device_map 流水线（稳妥简单）**——`DEVICE_MAP=auto` + `MAX_MEMORY="0:16GiB;1:16GiB;2:16GiB;3:16GiB"`（强制切分，auto 默认不切）+ `GRADIENT_CHECKPOINTING=1` + **`model.train()`**。AdamW（pitfalls #15）+ 装 CANN GE 依赖 `decorator scipy`（pitfalls #14）。mb≥2 填流水线再提速 27%（pitfalls #20）。27B 4-die 实测：6.87s/样本。
  - 选型：吞吐优先路线A（HCCS 快互联下通信被计算掩盖）；求稳/不想管分布式细节用路线B。详见 pitfalls #13-20。

### 5. 绘制 loss 曲线（`plot_loss.py.tmpl`）
- matplotlib 画 loss + smoothed + LR 双轴。
- 上传 catbox.moe → 0x0.st → uguu.se 拿公网直链。

### 6. 验证（`validate.py.tmpl`，base vs LoRA 对比）
- 从**未参与训练**的数据中选 N 条（复现训练抽样 seed 求补集，再另取 seed 选 N）。
- **teacher-forced 多轮**：逐 assistant 轮，用 GT 的 student/assistant 历史 + 当前 student 作上下文，`add_generation_prompt=True` 生成 teacher，与 GT 对比。
- 指标：char-level ROUGE-L F1（LCS）、字符重叠（Jaccard multiset）、生成长度。三者互补（详见 references/eval-metrics.md）。
- base 与 LoRA 各跑一遍，算 Δ。

### 7. 概要总结
- `SUMMARY.md`：环境/模型/数据/LoRA 配置/超参/loss 表/结论。

## 核心原则（务必内化）

1. **先 Eager 基线，再谈融合。** NpuFusedAdamW 等融合路径只在 Eager 单卡 loss 正常后才上。不要没基线就归因到融合层。
2. **只对 assistant token 算 loss。** SFT 的本质——user/system 不参与梯度。mask 做错等于在做 CPT，loss 看似下降但学错东西。**用字符偏移法验证 label token 数 > 0 且不含 user token**（见 references/label-masking.md 的自检片段）。
3. **CANN 必须用 9.0.0+（或 8.5.0+）。** latest 软链常指向 8.3.RC1（无 FA 编译内核），attention 模型会报 "Cannot find binary for op FlashAttentionScore"。eager 可绕过但慢。
4. **`torch_npu` 必须 `import` 才注册 NPU 后端**（autoload=0 时）。以 `torch.npu.is_available()` 为准，不要看 device_count。
5. **NpuFusedAdamW 的 `zero_grad(set_to_none=False)`**；否则报 "set_to_none is not supported in fused optimizers"。
6. **`apply_chat_template(tokenize=True)` 返回 BatchEncoding（dict），`len()` 是键数（通常 2）不是 token 数。** 取 token 数用 `seg['input_ids']`（pitfalls #4）。
7. **精度优先于速度。** 若怀疑融合/量化改变数值，先用验证集 ROUGE-L/生成样例校验 base 一致性再接受加速。
8. **MoE 模型训练慢先查专家分发，优化前必须分相实测。** transformers 的 MoE eager 实现是 Python 逐专家循环（短序列下纯调度开销，每前向 7万+ 微小内核）。两条数学等价补丁：`MOE_IMPL=dense`（短序列 -34% 步时）/ `MOE_IMPL=gmm`（grouped GEMM，长序列 3-8×/省显存 30%），见 references/moe-optimization.md。优化决策靠分相计时探针（fwd/bwd/opt/comm），不靠直觉——GC 直觉上该关、实测该开。
9. **报"昇腾不支持 X"之前，先走三步算子发现法**（见 references/npu-op-discovery.md）：① 盘点本机 CANN 注册接口（aclnnop 头文件 / torch_npu custom_ops / 二进制 schema）→ ② 搜 https://gitcode.com/cann 官方仓（ops-transformer / torchtitan-npu / catlass / cann-recipes-train）→ ③ 结合文档与参考案例写最小用例实测。PyTorch/CUDA 的函数在昇腾上几乎都有对应物；"NPU 无此内核"绝大多数时候只是"没找到入口"（grouped GEMM 曾被误判，次日即推翻）。结论必须写明排查范围与实测数据。
10. **改动训练数值路径后必须做等价性验证**（三层）：微观 dx/dw 对照 → 同种子 step1 loss 近逐位 → 多步轨迹在噪声带内重合。注意 LoRA A 随机初始化导致 step2+ 天然有 ~0.5-2% 重跑偏差，勿误判（pitfalls #25）。

## 通用性矩阵（适用范围）

| 模型/场景 | 支持度 | 说明 |
|---|---|---|
| HF CausalLM 纯文本（Qwen2/3/3.5、Llama-2/3、GLM、Mistral、DeepSeek、Yi 等） | ✅ 完整验证 | 0.8B（单卡）~27B（FSDP2 8-die）两端实测 |
| 多模态模型文本头（Qwen3.5-VL / Qwen2-VL 系） | ✅ 验证 | `AutoModelForCausalLM` 加载即文本头，纯文本 SFT 无需图像（pitfalls #10） |
| 混合注意力（gated delta / linear attention） | ✅ 验证 | LoRA 目标自动发现含 in_proj_* 系列 |
| MoE 模型 | ✅ 实测（Qwen3.6-35B-A3B, 14卡FSDP2） | 跑通且 loss 正常；**限制**：融合路由专家非 nn.Linear 挂不上 LoRA，仅注意力+共享专家被适配（21.2M 参数），见 pitfalls #21；**训练提速**：`MOE_IMPL=dense/gmm` 数学等价补丁（-34% 步时 / 长序列 3-8×），见 moe-optimization.md |
| 数据格式：messages / dialogue(student-teacher) / Alpaca | ✅ 验证 | prepare_data 三格式自动识别 |
| chat 模板：ChatML / Llama-3 / Llama-2 / Mistral / DeepSeek | ✅ | 定界符表见 label-masking.md，可 env 覆盖 |
| QLoRA / 量化基座 | ❌ 不支持 | 后续可扩展（NPU 量化路线另议） |
| 非 CausalLM（T5 类 encoder-decoder、扩散模型） | ❌ 超范围 | 扩散见 [[ascend-torch-cpt]] |

## 何时用本技能 vs 其它

- **CPT（原始语料继续预训练）** → [[ascend-torch-cpt]]
- **全量微调（不冻结 base，显存够、要彻底改行为）** → 改 `lora_train.py.tmpl` 去掉 peft、放开 `requires_grad`（注意 64GB 单卡全量训 7B+ 仍吃力，优先 DDP/FSDP2，见 [[ascend-torch-cpt]] 的并行策略）
- **RL 后训练（GRPO/RLHF）** → [[ascend-torch-cpt]] 的 `references/text-lm-rl.md`（RL 范式与 SFT 不同，先有 SFT/全量基线再上 RL）
- **推理优化（不训练，只加速推理）** → [[torch-npu-optimization]]

## 文件清单
- `scripts/route_select.py.tmpl` — 自动选路器：模型大小+空闲卡+seq/steps → 单卡/FSDP2/device_map + 显存估算 + 自动超参 + export 启动块（纯标准库，无需 torch）
- `scripts/prepare_data.py.tmpl` — 数据抽样 + 转 chat + loss 掩掩（字符偏移法）
- `scripts/lora_train.py.tmpl` — LoRA SFT 训练（自动发现目标、NpuFusedAdamW、bf16、loss 记录）
- `scripts/lora_train_fsdp.py.tmpl` — 大模型 FSDP2 数据并行训练（27B 实测 2.18×/样本；含 expandable_segments 禁用与 train() 两个关键修复；MoE 可选 dense/gmm 等价提速补丁 + batch>1 右填充）
- `scripts/plot_loss.py.tmpl` — loss 曲线 + 公网上传
- `scripts/validate.py.tmpl` — base vs LoRA teacher-forced 多轮验证
- `scripts/run_env.sh.tmpl` — 环境变量与 CANN source
- `references/pitfalls.md` — 踩坑全集（25 条，避免重复浪费时间）
- `references/moe-optimization.md` — MoE 训练提速双路线（dense/gmm 补丁代码、选型表、等价性验证协议、已试错清单）
- `references/npu-op-discovery.md` — 三步算子发现法（本机 CANN 接口盘点 → gitcode.com/cann 搜索 → 文档案例落地；报"不支持"前的强制排查流程）
- `references/label-masking.md` — loss 掩掩三种方法 + 自检
- `references/hyperparam-selection.md` — LoRA 超参自动选择规则（LR 按模型尺寸/warmup=10%步数/有效batch≈8/显存估算公式）
- `references/eval-metrics.md` — ROUGE-L/字符重叠/长度 三指标意义与选择
- `references/env-pyproject.md` — uv pyproject 模板（ascend extra）
