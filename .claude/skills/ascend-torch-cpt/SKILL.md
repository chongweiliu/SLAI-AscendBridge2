---
name: ascend-torch-cpt
description: 在华为昇腾 NPU（Ascend 910/910B/910C/950 等）上，用 PyTorch + torch_npu 对任意 HuggingFace 模型做继续预训练（Continued Pre-Training, CPT）的端到端技能。用户只需给出【模型权重路径】+【训练数据集路径】即可启动：自动探测环境/依赖、（按需下载模型与语料）、语料格式自动转换、训练脚本自动生成、单卡/DDP/FSDP2 自动选型、torch_npu 融合路径（NpuFusedAdamW、SDPA→npu fusion attention、TASK_QUEUE）、超参自动择优、默认产出 loss 曲线图+公网可访问直链（catbox.moe/0x0.st/uguu.se 顺序尝试，外网全不通则降级表格展示）、概要总结、训练前后 PPL/acc/F1 评估。只要用户要在昇腾上“继续预训练/二次预训练/CPT/再喂点领域语料训一轮 某模型”，或提到 NPU 训练 + 模型 + 语料 + DDP/FSDP/融合算子/loss 曲线/评估，就应使用本技能——即使用户没明说“skill”。也用于“评估继续预训练效果”“NPU 上 DDP 训练”“昇腾 CPT 踩坑”等。本技能基于 PyTorch 与 torch_npu（不是 CUDA/megatron/deepspeed）。
---

# Ascend NPU 继续预训练（CPT）技能

## 这是什么

把一个 HF 模型 + 一个语料，在昇腾 NPU 上端到端跑一遍继续预训练，并产出：训练脚本、loss 曲线（含公网直链）、概要总结、训练前后域内评估（PPL / next-token acc / 生成 F1·Recall·EM）。全程在保证精度与正确性前提下尽量提效降时。

适用：任意 decoder-only LM（CausalLM）。多模态模型的“文本头”也支持（见 references/multimodal-remap.md）。

## 输入（用户给路径即可启动）
- **模型权重路径**（必填）：HF 目录（含 config.json/safetensors/tokenizer）。可在命令里直接指定，如
  `ascend-torch-cpt --model-dir /mnt/model/Qwen3-0.8B --data-file /path/train.jsonl`，
  或由 `run_env.sh` 的 `MODEL_DIR` 注入。本地没有则按需从 modelscope / hf-mirror 下载。
- **训练数据集路径**（必填）：jsonl/json/parquet/csv 均可；`{"messages":[...]}` chat 格式或 `{"text":...}` 纯文本均自动识别转换。由 `DATA_FILE` 指定，缺失则按需下载。
- 可选：seq_len、batch_size、步数、并行方式（单卡/DDP/FSDP2）、语料比例（如取 30%/60%）、是否评估。未指定者由技能自动择优。
- 工作目录：统一建在 **SLAI-AscendBridge2 仓库根目录下的 `training-ws/` 内**（`training-ws/` 无则新建，**位于仓库内部而非与仓库平行**），每个模型一个子目录 **`training-ws/<模型名>-cpt/`**。模型名取权重路径最后一段（如 `/mnt/model/gemma-4-12B-it` → `training-ws/gemma-4-12B-it-cpt/`）。所有脚本与产物统一归档于此，不散落到仓库根目录或其他位置。`run_env.sh` 的 `WS_DIR` 已自动指向该路径并 `mkdir -p`。

## 默认产出（loss 曲线 + 公网链接 为默认方式）
每次训练**默认**产出 loss 曲线图（png）并尝试上传为**外部公网可访问直链**：
- 上传顺序：`catbox.moe` → `0x0.st` → `uguu.se`（按可达性依次尝试，见 scripts/plot_loss.py.tmpl）。
- 校验链接 HTTP 200 后告知用户直链。
- **若外网全不通**（catbox/0x0/uguu 均不可用）：自动降级为**表格展示 loss 收敛趋势**（逐 step / 每 10 步 + first5→last5 + min）。
- 不论是否拿到公网链接，本地 png 与 `losses.json`/`step_loss.jsonl` 始终保存。
- **链接时效**：uguu.se 等临时直链会过期；永久留本地 png + `losses.json`，长期复看以本地为准。
- 除曲线外另产出：训练前后 PPL/acc/F1 评估、`train_summary.json`、概要总结报告（README.md）、**resume ckpt**（`ckpt_latest.pt`，含模型+优化器+step，供续训）。

## 核心原则（务必内化）

1. **先 Eager 基线，再谈融合/图模式。** 任何融合路径（torchair 图模式、NpuFusedAdamW 等）只在 Eager 单卡能跑通、loss 正常后才上。不要在没基线时归因到融合层。
2. **精度优先于速度。** 融合算子/图模式若改变数值（如 bf16 纯前向 vs fp32 主权重+autocast），先用域内 PPL/acc 校验一致性再接受加速比。
3. **不装 CUDA 专属核。** 昇腾上 flash-linear-attention / causal-conv1d 等 CUDA 库装不上也无用；走 torch_npu 自己的融合路径（SDPA→npu fusion、NpuFusedAdamW）或 torchair 图模式，参考 references/fusion-api.md。
4. **每个产物都做本地预检再写库/交付。** 脚本能 `python -c "import ast; ast.parse(open(...).read())"` 过；真跑前先 `--dry-run` 或 2 步 smoke。
5. **训练结束必须保存 ckpt。** CPT 的验收标准之一就是"训练前后对比评估"，因此训练脚本末尾必须落盘模型权重（`cpt_model_state.pt`），否则后续 PPL/acc/F1 评估无 ckpt 可用。DDP 用 `model.module.state_dict()`（每卡持全量，rank0 存即可）；FSDP2 用 `DTensor.full_tensor()` 聚合（见 references/pitfalls.md #20，否则只存到 1/N 分片）。
6. **必须用昇腾 NPU 训练，禁止回退 CPU。** 继续预训练默认且只能跑在 Ascend NPU 上（`torch_npu`），不允许静默回退到 CPU 训练（如 `device='cpu'` 或 `torch.device('cpu')`）。若因算子缺失/环境异常确实必须回退 CPU，**必须先向用户说明原因并征得确认**后再进行。
7. **后台长跑要有心跳输出。** 后台运行的脚本（训练/评估/下载）最长 **2–3 分钟**必须向 stdout 打印一次进度信息（step/loss/用时或"仍在运行"心跳），`print(..., flush=True)`，让人感知任务还在跑。不要在步数很少时才打印——打印间隔要按**时间**折算（如 50s/step 时每 2–3 步打一次），保证屏幕 ≤2–3 分钟必有输出。
8. **所有产物统一归档到 `training-ws/<模型名>-cpt/` 子目录。** 训练/评估全过程的脚本、代码、README、ckpt 权重、loss 曲线、日志、summary 等全部放进 SLAI-AscendBridge2 仓库根目录下 `training-ws/` 内以"模型名-cpt"命名的子目录（见「产物目录约定」），**不得**散落到仓库根目录、与仓库平行的目录、或其他位置。

9. **全程实时用时表（执行红线，不可省）**。整个 CPT 过程**必须**在屏幕上维护一张格式化用时表（见「用时表」节），让用户随时知道：每阶段预计/实际用时、整体总预估、以及剩余预估。**长跑阶段（7）必须周期性刷新，不能只在阶段边界打一次就不管**：
   - **训练脚本自动刷新**：`cpt_train.py`/`cpt_fsdp.py` 每 ~30s 自动调 `timing_table.py --doing 7 <已耗时> "X/N步 ETA Y loss Z"`，重印整张表到训练日志（不依赖 agent 记得轮询）。
   - **agent 主动轮询**：训练后台跑期间，agent 每 1–2min `tail` 训练日志并把最新用时表/心跳行回显到对话——**不得"启动后台就长 sleep 只查一次"**。
   - 每完成一阶段、评估每模型切换时也重印整张表（Markdown，`print(..., flush=True)`）。阶段 0–1 勘察完即给全程总预估；阶段 7 据已耗 step×s/step 外推剩余并周期刷新。

## 工作流（9 阶段，每阶段都要在屏幕实时更新用时表）

### 阶段 0 · 意图确认与路径核对
- 核对用户给的**模型路径**与**语料路径**是否真实存在（`ls`）。本类任务里用户常给理想化/不存在路径；不存在则搜索本机可用副本并**显式告知**用户用了哪个、改了什么。
- 确认：模型 id/路径、语料路径、seq_len、batch_size、步数、是否要 DDP/FSDP2、是否要评估。用户未指定的超参由本技能自动择优（见下）。

### 阶段 1 · 环境/依赖勘察
探测并记录（写进 `${WS_DIR}/env_probe.json`，即 `training-ws/<模型名>-cpt/env_probe.json`）：
- NPU：卡数、每卡显存（**以实际分配测试为准**，`mem_get_info` 的 free 可能为负，见 pitfalls #29）、CANN 版本、`npu-smi`/`/dev/davinci*`。
- **卡间互联**：`hccn_tool -i <devid> -ip -g` 看是否配了 RoCE IP（报 "no ip was preset" = 只能走慢速 PCIe，见 pitfalls #23）；**真实 CPU 核数**用 `nproc --all`/`os.cpu_count()`（`nproc` 会受 OMP_NUM_THREADS 误导，见 pitfalls #22）。
- torch / torch_npu / transformers 版本；解释器路径。**注意 `torch==X.Y.Z+cpu` 不代表无 NPU**——昇腾镜像常用 +cpu build 配 torch_npu，NPU 后端由 torch_npu 注册，以 `torch.npu.is_available()` 为准（pitfalls #40）。
- **新芯片 soc 不支持（Ascend950 系列：950PR/950DT）**：若 `torch.npu.set_device(0)` 报 `Unsupported soc version: Ascend950PR` 或 `Ascend950DT`（device_count 能返回但 lazy_init/算子执行失败），是 torch_npu 太旧不识别 950 系列芯片——升级 torch+torch_npu 到支持 950 系列的版本（**950PR 实测可行组合** `torch==2.12.0 + torch_npu==2.12.0`，pip 华为 mirror；**950DT** 同系列同 CANN 9.0.0-beta，预期同方案，需实机验证 set_device+matmul）。env_probe 必实测 `set_device` + 一次 `x@x` matmul，**不能只看 device_count 就认定 NPU 可用**（pitfalls #42）。`npu-smi info -t board -i 0` 的 Chip Name 匹配 `Ascend950(PR|DT)` 即 950 系列，都需检查 soc 支持。
- **torchvision/torchaudio 与 torch ABI 错配**：升级 torch 后必须同步装匹配的**正式版** torchvision/torchaudio（torch 2.12↔torchvision 0.27.1+cu130、torchaudio 2.11+；pip 华为 mirror），否则旧版 import 崩会挡住 transformers 多模态文本头导入链（`torchvision::nms does not exist` / `torchaudio undefined symbol`）。装正式版后**无需打桩**；仅当匹配版不可得（torch 太新无配套 release）才用 stub fallback（pitfalls #43）。env_probe 验证 `from transformers.models.<mm> import XxxForCausalLM` 不崩才算依赖就绪。
- CANN env：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`（**部分镜像该路径不存在或 env 已在 base 注入，需条件 source + `import torch_npu; torch.npu.is_available()` 验证**，pitfalls #41）。
- **torchaudio 探测**（多模态模型才需要）：若 `import transformers.models.<mm>_xxx`（多模态文本头类）报 torchaudio `.so` 符号错配，用 `sys.meta_path` 桩掉（文本/视觉 CPT 不需音频，pitfalls #39）。模板已带防御版。
- 必设环境变量：`TORCH_DEVICE_BACKEND_AUTOLOAD=0`（规避 torch_npu autoload 崩，手动 import）、`TASK_QUEUE_ENABLE=1`（异步算子下发）、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（减碎片，给融合优化器留空间）。
- torchair 依赖（仅在走图模式时需要，见 references/fusion-api.md）：protobuf/scipy/attrs/decorator/cloudpickle/ml_dtypes/tornado，`setuptools<82`（≥82 移除 pkg_resources 破坏 GE init）。

用 `scripts/run_env.sh.tmpl` 生成统一入口（`WS_DIR` 已指向 `training-ws/<模型名>-cpt/` 并自动 `mkdir`）。

> ⏱ **用时表初始化**：阶段 0–1 勘察完后，据模型参数量/可用卡数/目标步数/语料规模给出**全程总预估**（如 ~6 分钟或 ~2 小时），并填好 9 阶段各自的**预计用时**，写入 `outputs/timing.json` 后打印整张用时表。这是用户看到的第一个进度基准。

### 阶段 2 · 模型与数据集获取
- 模型：本地有就用本地；否则从 modelscope（国内优先，`modelscope` CLI 或 git clone）或 `hf-mirror.com`（`git clone` + `git-lfs`）下载。`huggingface.co` 国内通常不可达。**大权重文件优先 ModelScope 的 `resolve/master/<file>` 直链（`wget -c`）**：新版 `hf` CLI 从 hf-mirror 拉大文件默认走 Xet 后端会 `401 Unauthorized`（`cas-server.xethub.hf.co`），且美区 CDN 慢；必须用 hf-mirror 时设 `HF_HUB_DISABLE_XET=1` 回退普通 LFS（pitfalls #38）。下完比对 `model.safetensors.index.json` 元数据字节数校验。
- 语料：同上。`/mnt/share/...` 等用户给的理想路径不存在时，搜本机或下载。
- 数据集若只有 train 分割、用户要“验证集”：用 `seed` 重建训练划分取 held-out（见 references/eval-metrics.md）。

### 阶段 3 · 语料格式转换与打包
- 读语料（jsonl/json/parquet/csv 均要支持）。
- **格式判定**：
  - 若是 `{"messages":[...]}` chat 格式 → `tokenizer.apply_chat_template` 转连续 token 流（CPT 用）。
  - 若是 `{"text": "..."}` / 纯文本 → 直接 tokenize。
  - 其它 → 取可读文本字段拼接。
- 打包成 `seq_len` 的定长块（`input_ids.pt`，shape `[N, seq_len]`）。不足步数需求则循环重采样补齐（记录 epoch 数）。
- 数据预处理边界（去重/长文档分块/多语料混合/packing vs padding/tokenizer）见 `references/data-prep.md`。
- 用 `scripts/prepare_data.py.tmpl`。**务必打印**：总样本/子集样本/总 token/原始块数/训练块数/epoch 估计。
- **多卡必设 `WORLD_SIZE=N`**：`need = NUM_STEPS × BATCH_SIZE × WORLD_SIZE`，`prepare_data` 据此生成块数。若多卡训练却留 `WORLD_SIZE=1`（默认），need 会算少 → 每卡仅 1/N 块 → 训练内循环重复采样（见 pitfalls）。单卡不用设。

### 阶段 4 · 训练方式自动选型
按 `references/parallel-strategy.md` 的决策树，据**卡间互联速度**、**模型参数量**、**单卡显存**、**语料规模**、**用户要求**选。**先探测卡间互联**（`hccn_tool -i <devid> -ip -g` 是否配 RoCE IP）：

```
卡间互联慢（RoCE 未配 / PCIe ~GB/s）？
├─ 是 → 大模型(单卡装不下) → 【模型并行 device_map="auto"】拆层到多卡，
│       无 all-gather，优化器放 NPU（实测 8× 加速，见 parallel-strategy.md「模型并行」）
└─ 否（互联正常）→ 常规决策树：
    单卡能装下整套(权重+优化器状态+激活)？
    ├─ 能 → 想多卡提速？
    │      ├─ 是且步数够多(>~150) → DDP（每卡持完整参数/梯度/优化器，hccl 后端）
    │      └─ 否/步数极少 → 单卡 Eager
    └─ 不能 → FSDP2（fully_shard，参数/梯度/优化器分片）
```

经验阈值（单张 ~65GB 卡，bf16 权重+fp32 AdamW 状态，近似）：
- 权重占单卡 ≤ ~40% 显存 → 单卡；想提速走 DDP。
- 0.5–3B 常单卡或 DDP；7–14B 单卡勉强→DDP 或 FSDP2/模型并行；≥30B → FSDP2 或模型并行。
- **步数 < ~150 且用图模式不划算**（torchair 首图编译 ~15min 摊销不了）→ 走 Eager+融合优化器。
- **卡间互联慢时大模型别用 FSDP2**：其 all-gather/reduce-scatter 会通信-bound（实测 97% 时间在通信），改模型并行。

### 阶段 5 · 超参自动择优（`references/hyperparam-selection.md`）
- **precision**：fp32 主权重 + bf16 autocast（NPU 原生 bf16，数值稳）。**不要**纯 bf16 前向（混合线性注意力模型会数值崩，见 pitfalls.md）。
- **lr**：CPT 基线 1e-5；全局 batch 越大按 sqrt 缩放上调（DDP 8×→ 可 2e-5）。短训练(100步)用保守值避免发散。
- **warmup**：~10% 步数；cosine 衰减到 0。
- **optimizer**：`NpuFusedAdamW`（融合，替代 torch.optim.AdamW）；betas=(0.9,0.95)，wd=0.01，eps=1e-8，grad_clip=1.0。
- **batch_size**：先按“激活显存预算”估上限，2 步 smoke 验不 OOM；OOM 则降 per-rank bs 或开梯度检查点。优先开**梯度检查点**而非降 bs（保有效 batch）。
- **seq_len**：用户指定优先；否则据语料平均长度选 512/1024（长上下文模型可更大，但显存↑）。
- **attention**：full_attention 层走 SDPA（→NPU fusion attention 自动路由）；不要轻易改 eager（会慢且图模式才需要）。
- **梯度检查点**：线性注意力/hybrid 模型 fallback 激活显存大，默认开（`use_reentrant=False`）。

### 阶段 6 · 生成训练脚本并 smoke
- 按选型用 `scripts/cpt_train.py.tmpl`（**单卡 + DDP 自动检测**，由 RANK env 决定）/ `cpt_fsdp.py.tmpl`（FSDP2）。单卡 `python cpt_train.py`；DDP `bash launch_ddp.sh`（torchrun）。
- 模板通用化：`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=float32)`；多模态走 references/multimodal-remap.md。
- 模板已支持断点续训（`RESUME=1` 开关，默认关闭）与梯度累积（`GRAD_ACCUM=N`），见 references/resume.md。
- **smoke**：2 步、2 卡（DDP）或单卡，确认前向+反向+优化器 step 都通过、loss 合理（初始 ~模型典型值）再上正式。
- **smoke 的 s/step 不可直接外推正式训练用时**：smoke 的 `s/step = 累计耗时/(step+1)` 把首次 import(~90s)+多卡加载模型+FSDP2/DDP 初始化全摊进前几步，前几步 s/step 虚高。预估正式训练取正式 run 稳态步的 s/step（累计平均越往后越准），或 `总耗时/NUM_STEPS`。
- 踩坑先看 references/pitfalls.md（已在脚本模板里规避了多数）。

> **范围声明**：本 skill 覆盖**单机多卡**（1–8 卡）。**多机（multi-node）** CPT 需 `torchrun --nnodes` + RDMA/网络配置（HCCL 跨机），属另一层复杂度，本 skill 不含；如需多机，在单机跑通后另加 `--nnodes`/`--rdzv` 与 HCCL 网络配置。

### 阶段 7 · 正式训练 + loss 曲线 + 公网直链
- 逐 step 记录 `step, loss, lr, elapsed, s/step, tok/s` 到 `logs/step_loss.jsonl` + stdout。
- **心跳输出**：`print(..., flush=True)` 的间隔按**时间**折算（见核心原则 7），保证屏幕 ≤2–3 分钟必有输出（如 50s/step 时每 2–3 步打一次，而非固定每 5 步）。
- **剩余用时外推**：训练期间每 ~30s 或每个心跳行，据已耗 step 数与 `s/step` 外推**剩余训练用时 ETA = (NUM_STEPS − step) × s_per_step**，连同当前 step/loss/已耗时长一并打印；并刷新用时表阶段 7 行为 `⏳doing`（如 `50/100步 ETA1.5min`）。可用 `timing_table.py --doing 7 "50/100步 ETA1.5min"`。
- 训完出 `train_summary.json`。
- **保存 CPT ckpt**（`cpt_model_state.pt`）供阶段 8 评估用（见核心原则 5；FSDP2 用 full_tensor 聚合）。
- **保存 resume ckpt**（`ckpt_latest.pt`：模型+优化器+step）。断点续训开关 `RESUME=1` **默认关闭**；开启后按**时间基准**周期保存（间隔 `= max(15分钟, 预估总时长/5)`，训练期间 **≤5 次**、每次间隔 **≥15 分钟**），训练结束**总是**存一次最终 ckpt（见 `references/resume.md`）。
- 画 loss 曲线 png（`scripts/plot_loss.py.tmpl`，matplotlib，EMA 平滑），**尝试上传公网直链**：catbox.moe → 0x0.st → uguu.se（按可达性依次试；catbox/0x0 常不可用，uguu.se 通常可用）。校验链接 HTTP 200 后告知用户。外网全不通则用表格展示 loss 收敛。
- 所有产物存到 `<模型名>-cpt/` 子目录（见核心原则 8 与「产物目录约定」）。

### 阶段 8 · 训练前后域内评估
- 用 `scripts/eval_cpt.py.tmpl`：取 held-out 样本，对 **base** 与 **CPT** 两套算：
  - PPL = exp(mean NLL)、mean NLL、next-token acc（全序列 token 级，从 logits argmax）。
  - **chat 数据**：末轮 assistant 贪心生 128 token，token 级 Precision/Recall/F1/EM。
  - **纯文本数据**（无 messages，如 base 模型喂网页/代码文本）：只算 PPL/NLL/next-token acc（无"末轮 assistant"可生成比对）。
- 指标选择要"据训练数据与模型合理"：方言/领域语料用域内 PPL/acc/生成 F1；**不要**强行套英文 MMLU/HellaSwag（不对齐且需额外下载 lm-eval-harness）。若用户模型是**通用知识型**（非领域方言）且明确要英文基准，按 `references/eval-metrics.md` 的 MMLU how-to 跑小子集并说明局限。
- **NLL 公式务必取负**：`nll = -log_probs.gather(...).mean()`（漏取负会得负值，见 pitfalls.md）。
- 出 `eval_results.json` + 结论表，给"训练是否有效"结论（Δ 方向 + 是否过拟合）。
- **FSDP2 大模型评估**：ckpt 是 `full_tensor()` 聚合的全量 state_dict，单卡直接 `load_state_dict` 即可（评估不需 FSDP2/DDP）。base 多模态走 remap。完整 9B FSDP2 实战案例（含结果表 + 与 0.8B 对比）见 `references/eval-metrics.md`。
- **多卡训练后别立刻跑单卡评估**：8 卡 FSDP2/DDP 训练退出后 NPU driver **异步回收显存有延迟**（pitfalls #32），立刻 `model.to(npu)` 易 OOM（card 仅剩几百 MB free 但无残留进程）。训练退出后等几秒、`npu-smi` 确认卡空闲（或 `torch.empty(40GB)` 实测可分配）再跑评估；评估脚本载入前先 `torch.npu.empty_cache()`。
- **ckpt 转 HF 可复用目录**（可选）：若用户想用 `from_pretrained` 直接加载 CPT 模型做推理/当新 base，把 `cpt_model_state.pt` 存成 HF 目录（拷 config.json+tokenizer 到 `outputs/cpt_hf_model/` + 存权重为 `model.safetensors`）。否则评估用 `load_state_dict` 即可。

### 阶段 9 · 概要总结报告
输出（Markdown）含：任务/模型/语料、并行策略、融合 API、超参表、loss 收敛(first5→last5, min, delta)、用时表(每阶段实际/预计)、公网直链、关键修正记录、复跑命令。全部归档到 `${WS_DIR}/`（即 `training-ws/<模型名>-cpt/`）。

## 自动选型速查（写进脚本头注释）

| 输入 | 选型 |
|---|---|
| 小模型(<3B) + 单卡装下 | 单卡 Eager + NpuFusedAdamW + SDPA |
| 小/中模型 + 想提速 + 步数>150 | DDP(每卡完整参数) + NpuFusedAdamW + hccl |
| 大模型(单卡装不下优化器状态) + 互联正常 | FSDP2(fully_shard) |
| 大模型 + **卡间互联慢**(RoCE 未配/PCIe ~GB/s) | **模型并行 device_map="auto"**（fp32 权重，无 all-gather，8× 加速） |
| 短训练(<150步) | 不要图模式，Eager+融合优化器 |
| 长训练/推理 + 想极致融合 | torchair 图模式（见 references/fusion-api.md，含 converter 补全清单） |

## 产物目录约定

所有产物统一归档到 SLAI-AscendBridge2 仓库根目录下 **`training-ws/<模型名>-cpt/`** 子目录（如 `training-ws/gemma-4-12B-it-cpt/`），**不得**散落到仓库根目录或与仓库平行的位置。`training-ws/` 无则新建。

```
<REPO_ROOT>/training-ws/<模型名>-cpt/
├── run_env.sh              # NPU 环境入口 (WS_DIR 指向本目录)
├── env_probe.json          # 软硬探测结果
├── prepare_data.py         # 语料转换打包
├── cpt_train.py            # 训练(单卡/DDP 按选型)
├── cpt_mp.py               # 训练(模型并行 device_map，卡间互联慢时)
├── cpt_fsdp.py             # 训练(FSDP2)
├── launch_ddp.sh           # DDP 启动(torchrun)
├── eval_cpt.py             # 前后评估
├── plot_loss.py            # 曲线+上传
├── timing_table.py         # 全程用时表渲染器
├── README.md               # 总结报告
├── logs/{step_loss.jsonl, *_stdout.log}
└── outputs/{input_ids.pt, losses.json, train_summary.json,
            timing.json, loss_curve.png, public_links.json,
            cpt_*_state.pt, eval_results.json, val_samples.json}
```

## 用时表（全程实时刷新，格式化输出）

整个 CPT 过程**必须**在屏幕上维护一张实时用时表（核心原则 9），让用户随时知道：每阶段预计/实际用时、整体总预估、剩余预估。用 `scripts/timing_table.py.tmpl` 渲染（读 `outputs/timing.json`）。

### 要求
- **整体预估**：阶段 0–1 勘察完后，据模型规模/卡数/目标步数/语料量给出**全程总预估**，写入 `timing.json` 的 `overall_estimate_s`，并填好 9 阶段各自 `est_s`，打印第一张完整用时表。
- **每阶段**：进入阶段前确认预计用时；完成时记录**实际用时**（`--set <id> actual <s>` 自动置 done）。
- **剩余预估**：阶段 7 长跑期间，据 `s/step × (NUM_STEPS − step)` 外推**剩余训练用时**，每 ~30s 或每个心跳行刷新阶段 7 为 `⏳doing` 并带 ETA 说明。
- **刷新时机**：每完成一阶段、训练每 N 步、评估每模型切换时，重新打印整张表（Markdown，`print(..., flush=True)`）。

### 表格格式（屏幕实时打印；列：阶段 / 预计 / 实际 / 状态 / 说明）

```
| 阶段 | 预计 | 实际 | 状态 | 说明 |
|---|---|---|---|---|
| 0 意图确认与路径核对 | 1.0min | 0.5min | ✅done | 路径已核对 |
| 1 环境/依赖勘察 | 3.0min | 3.2min | ✅done | 8卡 Ascend910 64GB |
| 2 模型与数据集获取 | 1.0min | 0.3min | ✅done | 本地有 |
| 3 语料格式转换与打包 | 1.0min | 0.8min | ✅done | 800块 |
| 4 训练方式选型 | 1.0min | 0.2min | ✅done | 单卡Eager |
| 5 超参自动择优 | 1.0min | 0.1min | ✅done | lr1e-5 |
| 6 生成脚本并smoke | 3.0min | 2.5min | ✅done | 2步smoke通过 |
| 7 正式训练+曲线 | 6.0min | 2.9min | ⏳doing | 50/100步 ETA1.5min |
| 8 训练前后评估 | 4.0min | — | ⏳pending | base+cpt |
| 9 概要总结报告 | 1.0min | — | ⏳pending | README |
| **合计** | **~22min** | **10.5min** | **进行中** | 剩余 ~11min |
```

长跑阶段（7）每次刷新可只追加一行 "当前 step / s·step / 剩余 ETA"；其余阶段完成后补 actual 列、状态置 ✅done。全程合计行始终显示"已实际 / 整体预估 / 剩余"。

## references（按需读）
- `references/fusion-api.md` — torch_npu 融合 API 清单 + torchair 图模式 + 缺失 converter 补全（softplus/eye/softplus_backward 等）
- `references/pitfalls.md` — 踩坑清单与解法（必读，避免重犯）
- `references/parallel-strategy.md` — 单卡/DDP/FSDP2/**模型并行(device_map)** 选型与代码骨架 + 卡间互联探测
- `references/hyperparam-selection.md` — 超参自动择优 + OOM 回退阶梯 + 梯度累积
- `references/data-prep.md` — 数据预处理边界（去重/分块/混合/packing vs padding/tokenizer）
- `references/resume.md` — 断点续训（存/载 optimizer+step+sampler）
- `references/eval-metrics.md` — PPL/acc/F1/Recall/EM 定义 + held-out 重建 + FSDP2 实战案例 + MMLU how-to
- `references/multimodal-remap.md` — 多模态 checkpoint → 文本头权重重映射

## scripts（标准模板，新模型微调即用）
见 `scripts/*.tmpl`。生成时复制到 `${WS_DIR}/`（即 `training-ws/<模型名>-cpt/`）并按当前模型/语料/选型替换占位。模板已规避多数踩坑（set_to_none=False、gradient_as_bucket_view=False、expandable_segments、autocast、grad-ckpt）。其中 `run_env.sh.tmpl` 的 `WS_DIR` 已自动指向 `training-ws/<模型名>-cpt/` 并 `mkdir -p`；`timing_table.py.tmpl` 用于全程实时用时表（见核心原则 9）。
