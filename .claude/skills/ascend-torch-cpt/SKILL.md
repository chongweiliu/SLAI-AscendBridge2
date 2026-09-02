---
name: ascend-torch-cpt
description: 在华为昇腾 NPU（Ascend 910/910B/910C/950 等）上，用 PyTorch + torch_npu 对任意 HuggingFace 模型做继续预训练（Continued Pre-Training, CPT）的端到端技能。用户只需给出【模型权重路径】+【训练数据集路径】即可启动：自动探测环境/依赖、（按需下载模型与语料）、语料格式自动转换、训练脚本自动生成、单卡/DDP/FSDP2 自动选型、torch_npu 融合路径（NpuFusedAdamW、SDPA→npu fusion attention、TASK_QUEUE）、超参自动择优、默认产出 loss 曲线图+公网可访问直链（catbox.moe/0x0.st/uguu.se 顺序尝试，外网全不通则降级表格展示）、概要总结、训练前后 PPL/acc/F1 评估。只要用户要在昇腾上“继续预训练/二次预训练/CPT/再喂点领域语料训一轮 某模型”，或提到 NPU 训练 + 模型 + 语料 + DDP/FSDP/融合算子/loss 曲线/评估，就应使用本技能——即使用户没明说“skill”。也用于“评估继续预训练效果”“NPU 上 DDP 训练”“昇腾 CPT 踩坑”等。本技能基于 PyTorch 与 torch_npu（不是 CUDA/megatron/deepspeed）。
---

# Ascend NPU 继续预训练（CPT）技能

## 这是什么

把一个 HF 模型 + 一个语料，在昇腾 NPU 上端到端跑一遍继续预训练，并产出：训练脚本、loss 曲线（含公网直链）、概要总结、训练前后域内评估。全程在保证精度与正确性前提下尽量提效降时。

**这是通用 CPT 技能，不止文本 LM**——按模型类型自动选范式（文本 LM / diffusers 生成式 / 音频-LLM / Keras→PyTorch 权重迁移 / MLIP 力场，见核心原则 10 与「自动选型速查」）。核心方法论通用：①判定模型类型与格式 → ②拆出可训练 backbone + 冻结编解码器/条件编码器 → ③按范式构造数据流水线 → ④用原生损失训练 backbone → ⑤按范式选评估指标。

## 维护与治理（本 skill 的自我纪律）

- **分层加载**：本文每次触发全量加载（**红线 ≤350 行**）；细节只活在 `references/`（按需读）与 `scripts/`（按需复制，从不进上下文）。**加坑 = pitfalls.md 加完整条目 + 本文至多一行"触发特征一句话 + #编号"**，禁止把细节内联进本文。
- **检索约定**：遇到症状**按关键词 grep `references/pitfalls.md` 取相关 1–2 条**，勿通读（94+ 条通读既慢又稀释注意力）。
- **坑集生命周期**：同类合并（废弃条目标"并入 #N"）；版本敏感条目标注适用版本（CANN/torch 升级后归档）；**编号永不复用**（防引用漂移）；≥120 条触发一次整备。
- **瘦身触发**：本文 >350 行或字符数 >28K → 执行一轮"内联细节压回 references"瘦身（回归验证：骨架完整 + 编号引用有效 + 被删细节在 references 有家）。

## 输入（用户给路径即可启动）

- **模型权重路径**（必填）+ **训练数据集路径**（必填，jsonl/json/parquet/csv/xyz 均可）——缺失则按需从 modelscope / hf-mirror / Zenodo / GitHub 下载（阶段 2 源探测）。
- 可选：seq_len、batch_size、步数、并行方式（单卡/DDP/FSDP2）、语料比例、是否评估。未指定者由技能自动择优。
- 工作目录：`training-ws/<模型名>-cpt/`（模型名取权重路径最后一段），所有产物归档于此，不散落。

## 默认产出（loss 曲线 + 公网链接 为默认方式）

- loss 曲线 png + 逐 step 日志；上传顺序 catbox → 0x0 → uguu（校验 HTTP 200 后告知直链；外网全不通降级表格展示；uguu 临时链会过期，本地 png 为准）。
- 训练前后域内评估、`train_summary.json`、README 总结、`cpt_model_state.pt`（评估用）+ `ckpt_latest.pt`（含优化器，供续训）。

## 核心原则（务必内化）

1. **先 Eager 基线，再谈融合/图模式。** 任何融合路径只在 Eager 单卡能跑通、loss 正常后才上；没基线时不要归因到融合层。
2. **精度优先于速度。** 融合算子/图模式若改变数值，先用域内指标校验一致性再接受加速比（纯 bf16 前向会数值崩，见 pitfalls #4）。
3. **不装 CUDA 专属核。** flash-linear-attention / causal-conv1d 等装不上也无用；走 torch_npu 自己的融合路径（references/fusion-api.md）。
4. **每个产物本地预检再交付。** 脚本过 `ast.parse`；真跑前先 `--dry-run` 或 2 步 smoke。
5. **训练结束必须保存 ckpt**（`cpt_model_state.pt`），否则前后对比评估无 ckpt 可用。DDP 用 `model.module.state_dict()`；FSDP2 用 `DTensor.full_tensor()` 聚合（#20，否则只存 1/N 分片）。
6. **必须用昇腾 NPU 训练，禁止静默回退 CPU。** 因算子缺失/环境异常确实必须回退时，先向用户说明并征得确认。
7. **后台长跑要有心跳输出。** 训练/评估/下载最长 2–3 分钟必须打一次进度（按时间折算间隔，`print(..., flush=True)`）。
8. **所有产物统一归档到 `training-ws/<模型名>-cpt/` 子目录。**
9. **全程实时用时表（执行红线，不可省）**。用 `scripts/timing_table.py.tmpl` 维护（读写 `outputs/timing.json`），纪律是 **T1–T6 六个硬性刷新触发点**——每个触发点把整表（或长跑期间的 `⏳doing` 行+ETA）重印到对话，用户任何时刻看屏幕都知道"现在做到哪、还要多久"：
   - **T1 开工基线**：阶段 0–1 勘察完，给全程总预估 + 9 阶段各自 est，打印第一张完整表。
   - **T2 进入即标**：每进入一个阶段（含长跑子任务：下载/权重迁移/脚本开发/smoke）立即 `--doing` 置 ⏳ 并重印——**严禁实际在做而表上挂 pending**。
   - **T3 完成即结**：每完成一个阶段立即 `--set <id> actual <s>` 置 ✅ 并重印。
   - **T4 长跑心跳**：任何预计 >3min 的阶段（下载/权重迁移/训练/评估，不只训练！）每 1–2min 轮询并重印进度（已耗/百分比/ETA）。**"挂后台就埋头干活"是明确违规**；训练脚本自身另每 ~30s 调 `timing_table.py --doing 7 ...` 刷训练日志。
   - **T5 评估切换**：阶段 8 每切换一个模型重印整表。
   - **T6 收尾汇总（不可省）**：**最终答复与 README 都必须含完整用时表**（9 阶段预计/实际/说明 + 合计 + 预估偏差一句话解释）。只写 README 不上屏算未完成。
10. **按模型类型选训练范式（第一步，不可跳过）**，判定结果写进 `env_probe.json` 的 `model_type`/`train_paradigm`：
    - `ForCausalLM`/config+safetensors → **文本 LM**（CE loss，`cpt_train.py.tmpl`）；多模态文本头同路（references/multimodal-remap.md）。
    - `model_index.json`+transformer/vae/text_encoder/scheduler → **diffusers 生成式**（流匹配 loss，references/generative-diffusion-cpt.md）。
    - `Qwen2Audio*`/audio-text-to-text → **音频-LLM**（转写 CE，references/audio-llm-cpt.md）。
    - MLX 格式 → 须换同源 PyTorch 基座（#47）。
    - Keras/TF `.pkl`/`.h5` → **PyTorch 复刻+权重迁移**（#84–#86）。
    - **MLIP 力场（能量+力输出，如 MatterSim/EquiformerV2/MACE）** → **能量+力联合回归**（力=-∂E/∂x autograd 或 direct force head；评估禁 no_grad #92、shift-only scaling #93、后端静默降级 patch #91、科研包兼容链 #99、能量 per-element 参考口径 #100）。
11. **确定性 NPU 崩溃用"插桩→单批复现→二分"定位，变长 batch 必开 expandable_segments**（EE9999/507035 无 Python 堆栈；完整四步法 #79，多区域 checkpoint 反传 bug #78；s/step 渐进劣化特征 #80）。

## 工作流（9 阶段，每阶段都要在屏幕实时更新用时表）

### 阶段 0 · 意图确认与路径核对
- 核对模型/语料路径是否真实存在（`ls`）；不存在则搜本机可用副本并**显式告知**用户用了哪个、改了什么。
- 确认：模型 id/路径、语料、seq_len、batch_size、步数、并行方式、是否评估（未指定者自动择优）。
- **【范式判定，决定全局分支】** 按核心原则 10 判定并写入 `env_probe.json`；判定错则全盘错。

### 阶段 1 · 环境/依赖勘察（写进 `env_probe.json`）
- NPU：卡数、每卡显存（以实际分配测试为准，`mem_get_info` 的 free 可能为负 #29）、CANN 版本、`npu-smi`；**实测 `set_device`+一次 matmul**（+cpu build 不代表无 NPU #40；Ascend950 系列需新 torch_npu #42）。
- **容器 CPU 内存限制强制查**（cgroup，别信 free——remap/load 三份叠加可撞 32GB 被 SIGKILL #44/#46）；限制 < 加载峰值 → remap 搬 NPU 做。
- 卡间互联：`hccn_tool -i <id> -ip -g`（"no ip preset"=只能慢速 PCIe #23，影响并行选型）；真实核数用 `nproc --all`（#22）。
- torch/torch_npu/transformers 版本与解释器路径；torchvision/torchaudio 须与 torch ABI 匹配（对应版本表 #39/#43）；CANN env 条件 source 后以 `torch.npu.is_available()` 验证（#41）。
- 必设环境变量：`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`TASK_QUEUE_ENABLE=1`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（FSDP2 例外，见 #77，模板已内置守卫）。torchair 依赖清单见 references/fusion-api.md。
- 用 `scripts/run_env.sh.tmpl` 生成统一入口。

> ⏱ **T1**：阶段 0–1 完成后给全程总预估 + 9 阶段 est，写入 `outputs/timing.json` 并打印第一张完整表。

### 阶段 2 · 模型与数据集获取
- **第一步：源可达性矩阵探测（~30s，必做）**：`bash robust_download.sh probe`（从 `scripts/robust_download.sh.tmpl` 复制）探测**当日**可用性（hf-mirror/huggingface/modelscope/github/raw/codeload/zenodo），先建"当日可用源清单"再定路线，**不按历史经验盲试**（源逐日漂移 #89）。权重降级链：本地 → ModelScope → Zenodo（API 拿清单+md5）→ hf-mirror（禁 Xet #38）→ GitHub（三级降级+截断抢救 #90）。
- 大权重文件优先 ModelScope `resolve/master/<file>` 直链；**官方示例数据/权重可能只在某一平台镜像目录**（GitHub ≠ ModelScope 镜像；全网按名搜索失败 ≠ 不存在，递归列全平台目录树+官方 config 相对路径线索，#94）。
- **大文件下载与开发并行**：下载挂后台后立刻用小子集推进阶段 3-6，勿干等；大文件用 `robust_download.sh get`（多路 Range+分块断点续传+size/md5 终检，`sha256:` 前缀支持），小文件用 `fetch`；GB 级 tarball 下完必须流级校验（`gzip -t`，#68）。**下载是长跑阶段**：按 T4 每 1–2min 刷进度。
- 非 PyTorch 原生权重（Keras `.pkl`/`.h5`）：pickle 纯 numpy 元组可直接解包（无需 TF），逐层复刻 PyTorch 架构后做**形状严格校验+同形交换消融+语义 sanity**三重验证（#84–#86）。
- 数据集只有 train 分割时用 seed 重建 held-out（references/eval-metrics.md）。

### 阶段 3 · 语料格式转换与打包（按范式分支）
- **A. 文本 LM**：读语料（jsonl/json/parquet/csv）→ 判定格式（chat `{"messages"}` 用 `apply_chat_template`；`{"text"}` 直接 tokenize；其它取可读字段）→ 打包 `seq_len` 定长块（不足步数则循环重采样，记录 epoch 数）→ `scripts/prepare_data.py.tmpl`，**务必打印**总样本/子集样本/总 token/块数/epoch 估计。数据边界（去重/分块/混合/packing vs padding）见 references/data-prep.md。**多卡必设 `WORLD_SIZE=N`**（否则 need 算少→内循环重复采样）。
- **B. diffusers 生成式**：解码→resize 到原生分辨率→VAE 编码（输入 `[B,C,T,H,W]` #48，编码上 NPU 规避 cgroup OOM #44/#46）→ 缓存 latent+text_emb（预计算-后训练模式）；text_encoder 巨大时可缓存 embedding 或退零嵌入兜底（#49）。用 `prepare_generative_data.py.tmpl`，全流程见 references/generative-diffusion-cpt.md。
- **C. 音频-LLM**：soundfile 读 16k → `AutoProcessor(text=, audio=)`（单数 kwarg #56）→ labels 掩 pad+prompt+audio 特殊 token（漏 mask 致 loss 虚高 ~8× #58）；forward 须传 input_features**和** feature_attention_mask（#57）；冻 audio_tower+projector 训 language_model。见 references/audio-llm-cpt.md + `cpt_audio_llm.py.tmpl`。
- **D. MLIP 力场**：ase 读 EXTXYZ（能量+力标签）→ 官方 GraphConverter 构图（cutoff/threebody_cutoff）；能量基准差对齐见阶段 8-D。

### 阶段 4 · 训练方式自动选型（references/parallel-strategy.md）
```
卡间互联慢（RoCE 未配 / PCIe ~GB/s）？
├─ 是 → 大模型 → 模型并行 device_map="auto"（无 all-gather，实测 8× 加速）
└─ 否 → 单卡装得下(权重+优化器+激活)？
        ├─ 能 → 想多卡提速且步数>~150 → DDP(hccl)；否则单卡 Eager
        └─ 不能 → FSDP2（fully_shard）
```
经验阈值：权重占单卡 ≤~40% → 单卡；0.5–3B 常单卡/DDP；≥30B → FSDP2/模型并行；步数 <~150 不上图模式（torchair 首图编译 ~15min 摊销不了）；互联慢时大模型勿用 FSDP2（通信-bound，实测 97% 时间在通信）。

### 阶段 5 · 超参自动择优（references/hyperparam-selection.md）
- **precision**：fp32 主权重 + bf16 autocast（**不要**纯 bf16 前向 #4）；**lr**：CPT 基线 1e-5（全局 batch 大按 sqrt 上调；短训练保守值；照抄基座从零训练的调度会发散 #82）；**warmup** ~10% 步数 cosine 到 0；**optimizer**：`NpuFusedAdamW` + betas(0.9,0.95) wd=0.01 eps=1e-8 clip=1.0——**例外**：分阶段重建优化器（freeze/unfreeze 切换）用 plain AdamW（#87 跨阶段 saved-tensor 崩溃），小模型(<100M)融合无收益；**batch_size** 按"激活显存预算"估上限+2 步 smoke 验不 OOM（优先开梯度检查点而非降 bs，`use_reentrant=False`）；**attention**：full_attention 走 SDPA（→NPU fusion 自动路由）。

### 阶段 6 · 生成训练脚本并 smoke
- 按范式选模板：文本 `cpt_train.py.tmpl`（单卡+DDP 自动检测）/`cpt_fsdp.py.tmpl`/`cpt_mp.py.tmpl`；扩散 `cpt_diffusion.py.tmpl`；音频 `cpt_audio_llm.py.tmpl`。模板已支持断点续训（`RESUME=1` 默认关）与梯度累积（references/resume.md）。
- 模板通用化：文本 `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=float32)`；多模态走 remap；组件分离加载见各范式 reference。
- **smoke**：2 步确认前向+反向+优化器 step 全通过、loss 合理再上正式；扩散先对 backbone 与 VAE 分别单组件前向 smoke（抓 NPU 算子问题 #50）。**smoke 的 s/step 不可外推正式用时**（首 import ~90s 摊进前几步虚高，取稳态步）。
- 踩坑先 grep references/pitfalls.md（模板已规避多数）。
- **官方训练栈首次死锁/OOM → 立即 MINREPRO**（模型+collater+单批显存复现，~20 行）定位是模型需求还是栈问题，**禁止盲调 batch/换卡试错**（#95）；栈级不可用则拆组件自管轻量循环。

> **范围声明**：本 skill 覆盖单机多卡（1–8 卡）；多机需 `torchrun --nnodes` + RDMA/HCCL 跨机配置，不在本 skill 范围。

### 阶段 7 · 正式训练 + loss 曲线 + 公网直链
- 逐 step 记 `step, loss, lr, elapsed, s/step, tok/s` 到 `logs/step_loss.jsonl` + stdout（心跳按时间折算，屏幕 ≤2–3 分钟必有输出）；每 ~30s 外推剩余 ETA 并刷用时表（T4）。
- **中期评估成本预计算**（10 帧计时×次数，>总时长 20% 先调间隔/帧数；base 评估缓存复用 #96）；**三份 ckpt 语义**（best 非 EMA 评估默认 / final EMA / latest 续训；短程训练慎用 EMA #97）；长实验 `timeout 2×预估` 且监控方 2× 预期无输出即杀（#98）。
- 训完出 `train_summary.json`；保存 `cpt_model_state.pt`（评估用）+ `ckpt_latest.pt`（续训用，按时间基准周期保存 ≤5 次、间隔 ≥15min，训练结束总存一次，见 references/resume.md）。
- 画 loss 曲线（`plot_loss.py.tmpl`，EMA 平滑）并尝试上传公网直链（catbox→0x0→uguu）；外网全不通降级表格展示。产物全部存 `<模型名>-cpt/`。

### 阶段 8 · 训练前后域内评估（按范式分支；对比三原则见 references/eval-metrics.md）
**通用**：base vs CPT ①严格同条件（同种子一切随机源+确定性自检）②协议锚定（base 关键指标与论文/官方数字同量级才算协议对）③持平可能是正确结论（短程 CPT 域内收敛基座预期 ±1%）；**过拟合检查**：held-out 须独立 split；train loss 趋 0 + held-out 不改善=红旗；**数据量是泛化关键杠杆**（非步数）。多卡训练后别立刻单卡评估（显存异步回收 #32）；评估载入前 `torch.npu.empty_cache()`。
- **A. 文本 LM**（`eval_cpt.py.tmpl`）：PPL/NLL（公式务必取负 #5）/next-token acc；chat 数据加末轮生成 F1；域内语料用域内指标，勿强行套 MMLU。
- **B. diffusers**（`eval_diffusion.py.tmpl`）：固定 σ 算 velocity MSE（Δ<0 训练有效）+ 采样生成定性；勿套 PPL。
- **C. 音频-LLM**：held-out 转写 CE loss（base 全新 vs CPT `strict=False`，Δ<0 有效）；勿套文本 PPL/velocity MSE。
- **D. MLIP 力场**（#91–#93）：能量 MAE(eV 与 meV/atom)+力 MAE/RMSE(eV/Å)+力方向余弦；**评估循环禁 @torch.no_grad()**（力=-∂E/∂x 需 autograd 图 #92）；能量基准差用 **shift-only scaling** 对齐（shift 拟合自训练集、scale 保留原值保力基线 #93）；性能用单结构前向+力微分延迟。
- 出 `eval_results.json` + 结论表。ckpt 转 HF 目录可选（`from_pretrained` 直接加载）。

### 阶段 9 · 概要总结报告
输出（Markdown）含：任务/模型/语料、并行策略、融合 API、超参表、loss 收敛(first5→last5, min)、**完整用时表(预计/实际+合计+偏差解释)**、公网直链、关键修正记录、复跑命令。归档到 `${WS_DIR}/`。**最终答复必须同样携带完整用时表**（T6）。

## 自动选型速查

| 模型类型 | 范式/损失 | 脚本 | 评估 |
|---|---|---|---|
| 文本 LM / 多模态文本头 | next-token CE | `cpt_train.py.tmpl` | PPL/acc/F1 |
| diffusers 生成式 | 流匹配 on VAE latent | `cpt_diffusion.py.tmpl` | velocity MSE+采样 |
| 音频-LLM | 转写 CE（mask audio token） | `cpt_audio_llm.py.tmpl` | 转写 CE loss/WER |
| MLIP 力场 | 能量(per-atom)+力（力=-∂E/∂x） | 按官方库复用+patch | E/F MAE（禁 no_grad） |
| Keras/TF 权重 | 先复刻迁移（#84–86）再按原生范式 | 按范式 | 按范式 |

并行：小模型(<3B)单卡 Eager+NpuFusedAdamW；中模型+步数>150 → DDP；单卡装不下优化器 → FSDP2；互联慢+大模型 → 模型并行；短训练(<150步)勿图模式。

## 产物目录约定

```
<REPO_ROOT>/training-ws/<模型名>-cpt/
├── run_env.sh / env_probe.json / timing_table.py / README.md
├── prepare_data.py / cpt_train.py / eval_cpt.py / plot_loss.py
├── robust_download.sh        # 源探测/可靠下载（probe/fetch/get）
├── logs/{step_loss.jsonl, *_stdout.log}
└── outputs/{timing.json, input_ids.pt, losses/step jsonl,
            train_summary.json, loss_curve.png, public_links.json,
            cpt_*_state.pt, eval_results.json}
```

## 用时表（全程实时刷新）

工具 `scripts/timing_table.py.tmpl`（读 `outputs/timing.json`），触发点=核心原则 9 的 T1–T6。要求：整体预估（T1 后写入 `overall_estimate_s`）；进入即 `--doing`、完成即 `--set actual`；长跑阶段按速率外推 ETA 每 1–2min 刷新；**收尾汇总（T6）最终答复与 README 各带一份完整表**。格式（列：阶段/预计/实际/状态/说明）：

```
| 阶段 | 预计 | 实际 | 状态 | 说明 |
|---|---|---|---|---|
| 0 意图确认与路径核对 | 1.0min | 0.5min | ✅done | 路径已核对 |
| 7 正式训练+曲线 | 6.0min | 2.9min | ⏳doing | 50/100步 ETA1.5min |
| **合计** | **~22min** | **10.5min** | **进行中** | 剩余 ~11min |
```

长跑期间可只追加 `⏳doing` 行（带已耗/百分比/ETA）；合计行始终显示"已实际/整体预估/剩余"。

## references（按需读）
- `references/pitfalls.md` — 踩坑清单（**按症状 grep 取相关条目，勿通读**）
- `references/fusion-api.md` — torch_npu 融合 API + torchair 图模式
- `references/parallel-strategy.md` — 单卡/DDP/FSDP2/模型并行选型与骨架 + 互联探测
- `references/hyperparam-selection.md` — 超参择优 + OOM 回退阶梯 + 梯度累积
- `references/data-prep.md` / `references/resume.md` / `references/eval-metrics.md` — 数据边界 / 断点续训 / 评估指标与对比三原则
- `references/multimodal-remap.md` — 多模态 checkpoint 文本头重映射
- `references/generative-diffusion-cpt.md` — 扩散 DiT CPT 全流程（+可选 RL 后训练）
- `references/audio-llm-cpt.md` — 音频-LLM CPT
- `references/text-lm-rl.md` — 文本 LM/音频 LLM RL 后训练（RLHF/GRPO，可选）

## scripts（标准模板）
见 `scripts/*.tmpl`，复制到 `${WS_DIR}/` 按当前模型/语料替换占位。模板已规避多数踩坑（set_to_none=False、gradient_as_bucket_view=False、expandable_segments、autocast、grad-ckpt）。`run_env.sh.tmpl` 的 `WS_DIR` 已自动指向 `training-ws/<模型名>-cpt/`；`timing_table.py.tmpl` 用于用时表（核心原则 9）；`robust_download.sh.tmpl` 用于阶段 2 源探测/可靠下载（#88–#90）。

按范式选模板：文本 LM → `prepare_data.py.tmpl`+`cpt_train.py.tmpl`(+fsdp/mp)；扩散 → `prepare_generative_data.py.tmpl`+`cpt_diffusion.py.tmpl`；音频 → `cpt_audio_llm.py.tmpl`。
