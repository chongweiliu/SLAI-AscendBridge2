# ascend-torch-cpt 功能验证测试场景设计

> 目标：验证技能声明的核心能力矩阵（并行选型、语料格式、融合路径、产出物、红线规则）
> 原则：每个场景给出可复用的本机输入、执行方式、通过判据；能 smoke 的不跑全长。
> 优先级：P0=核心路径必测；P1=重要分支；P2=边界/降级，可抽测。

## 一、场景总览

| # | 场景 | 验证点 | 优先级 | 选型覆盖 |
|---|------|--------|--------|----------|
| S01 | 单卡 Eager CPT（纯文本格式） | 端到端主链路、NpuFusedAdamW、loss 下降 | P0 | 单卡 |
| S02 | 单卡 CPT（chat 格式） | `{"messages"}`→chat_template 转换、生成 F1/EM 评估 | P0 | 单卡 |
| S03 | DDP 多卡 CPT | torchrun/hccl、WORLD_SIZE 数据打包、lr sqrt 缩放 | P0 | DDP |
| S04 | FSDP2 大模型 CPT | fully_shard、full_tensor 聚合存 ckpt | P1 | FSDP2 |
| S05 | 模型并行 device_map CPT | 互联慢+大模型选型、双参数组 lr | P1 | MP |
| S06 | 多模态图片 CPT | mm_token_type_ids、视觉塔 lr1e-6、bs1×accum | P1 | MP+多模态 |
| S07 | 断点续训 RESUME | ckpt_latest.pt 恢复、loss 连续性 | P1 | 通用 |
| S08 | 数据格式与划分健壮性 | RATIO/seed 对齐、parquet/csv、WORLD_SIZE 补齐 | P1 | 数据链路 |
| S09 | loss 曲线公网直链与降级 | catbox→0x0→uguu 顺序、全失败降级表格 | P1 | 产出物 |
| S10 | 训练前后评估正确性 | PPL/acc 方向、NLL 取负、held-out 非空 | P0 | 评估 |
| S11 | 用时表全程实时刷新 | 训练中 ~30s 自动重印、agent 轮询回显 | P1 | 产出物 |
| S12 | 红线与防御性规则 | 不回退 CPU、smoke 先行、ckpt 必落盘 | P0 | 规则 |
| S13 | 异常输入边界 | 路径不存在、未知字段、OOM 降级阶梯 | P2 | 边界 |

## 二、详细场景

### S01 单卡 Eager CPT（纯文本格式）— P0 基线

- **输入**：模型 `/models/share/Qwen3.5-9B`（若显存紧张可换更小模型）；语料从 `zh_synth/cpt.jsonl` 抽取或生成 `{"text": ...}` 纯文本 jsonl 200 条；`SEQ_LEN=512, BATCH_SIZE=8, NUM_STEPS=60`。
- **执行**：新工作区 `training-ws/<模型名>-cpt/`，走技能 0–9 全流程，单卡（`ASCEND_RT_VISIBLE_DEVICES` 绑空闲卡）。
- **通过判据**：
  - `cpt_train.py` 使用 `NpuFusedAdamW` + `torch.autocast(bfloat16)` + 梯度检查点；
  - smoke 2 步通过后才进正式训练；loss 首 5 步→末 5 步下降（或至少不发散）；
  - `outputs/step_loss.jsonl`、`loss_curve.png`、`train_summary.json`、`cpt_model_state.pt` 均落盘；
  - 产物全部在 `training-ws/<模型名>-cpt/` 内，无散落。
- **注意**：smoke 的 s/step 不得直接外推正式 ETA（首步摊入 import ~90s）。

### S02 单卡 CPT（chat 格式）— P0

- **输入**：同一小模型；语料用 `{"messages":[{"role":"user",...},{"role":"assistant",...}]}` 格式 jsonl（可由 gsm8k_test.jsonl 改造）。
- **执行**：同 S01，`prepare_data.py` 走 chat 分支。
- **通过判据**：
  - 打包走 `apply_chat_template` 连续 token 流；
  - `eval_cpt.py` 对末轮 assistant 贪心生 128 token，产出 token 级 P/R/F1/EM；
  - held-out 划分非空（RATIO 与 eval 用同一值+同一 seed，见 S08 坑）。

### S03 DDP 多卡 CPT — P0

- **输入**：S01 模型+语料；2 卡或 4 卡 DDP；`NUM_STEPS≥160`（超过 DDP 摊销阈值）。
- **执行**：`bash launch_ddp.sh`（torchrun）。
- **通过判据**：
  - `prepare_data.py` 显式设 `WORLD_SIZE=N`，块数 = NUM_STEPS×BS×WORLD_SIZE（内循环不重复采样）；
  - `init_process_group` 后端 hccl；`DDP(gradient_as_bucket_view=False)`、`zero_grad(set_to_none=False)`；
  - 全局 batch 放大后 lr 按 sqrt 缩放（如 1e-5→2e-5）；
  - 每 rank loss 曲线一致（容差内），rank0 存全量 ckpt（`model.module.state_dict()`）。

### S04 FSDP2 大模型 CPT — P1

- **输入**：`/models/share/Qwen3.5-27B` 或 Qwen3-8B（视单卡装不下优化器状态为准）；`NUM_STEPS=30` smoke 级。
- **执行**：`cpt_fsdp.py`，4 卡。
- **通过判据**：
  - `fully_shard` 生效；训完 ckpt 经 `DTensor.full_tensor()` 聚合后**单卡可 `load_state_dict` 完整加载**（pitfalls #20：不能只存 1/N 分片）；
  - 训练退出后等数秒、`npu-smi` 确认卡空闲再跑评估（pitfalls #32 显存异步回收）；
  - 评估单卡直接加载，无需 FSDP2 环境。

### S05 模型并行 device_map CPT（互联慢+大模型）— P1（回归）

- **输入**：`/mnt/host-model/gemma-4-31B-it`（已有工作区可回归）；或 Qwen3.5-27B。
- **执行**：先 `hccn_tool -i 0 -ip -g` 确认互联状态；复用/重建 `cpt_mp.py` 2 卡。
- **通过判据**：
  - 选型落在大模型+互联慢→模型并行分支（非 FSDP2）；
  - 优化器参数在 NPU 上（不回 CPU）；fp32 主权重 + bf16 autocast；
  - 对照 `training-ws/gemma-4-31B-it-cpt/` 历史结果量级一致。

### S06 多模态图片 CPT — P1（回归，覆盖 references/multimodal-remap.md）

- **输入**：`/models/share/bakxss/modelimage/NuExtract3` + 图片版语料（复用 NuExtract3-cpt 的 `prepare_data_image.py` 输入）。
- **执行**：`cpt_image_mp.py` 2 卡，`NUM_STEPS=10` smoke 级。
- **通过判据**：
  - forward 显式传 `mm_token_type_ids`（image token=1）；
  - bs=1×grad_accum=8（bs=2 必 OOM 的坑已规避）；视觉塔+merger lr=1e-6、文本头 lr=1e-5 双参数组；
  - 权重重映射完整（language_model/visual/mtp/tie_weights），loss 数值正常不 NaN。

### S07 断点续训 RESUME — P1

- **执行**：S01 正式训练跑 ~20 步 kill，置 `RESUME=1` 重启。
- **通过判据**：
  - `ckpt_latest.pt` 含模型+优化器+step；恢复后 loss 从中断点连续（无跳变 >1 个量级）；
  - 周期保存满足：间隔 ≥15min、训练期间 ≤5 次、结束总是存最终 ckpt。

### S08 数据格式与划分健壮性 — P1

- **子用例**（可离线跑 prepare_data 单测，不训）：
  1. **RATIO 对齐**：`prepare_data.py` 与 `eval_cpt.py` 传同一 RATIO(0.9)+同 seed → held-out 非空（NuExtract3 首轮两连坑回归）；
  2. **格式矩阵**：jsonl/json/parquet/csv 各一条样本，均能识别转换（text/messages/其它可读字段三判定分支）；
  3. **WORLD_SIZE 补齐**：语料不足 need 时循环重采样并记录 epoch 数，打印总样本/总 token/块数。

### S09 loss 曲线公网直链与降级 — P1

- **执行**：S01 训完跑 `plot_loss.py`；另模拟外网全不通（如断网或封禁三个域名）再跑一次。
- **通过判据**：
  - 正常路径：按 catbox→0x0→uguu 顺序尝试，校验 HTTP 200 后给出直链，`public_links.json` 落盘；
  - 降级路径：自动降级为表格展示（first5→last5+min），本地 png + `losses.json` 始终存在。

### S10 训练前后评估正确性 — P0

- **输入**：S01/S02 的 base 与 CPT ckpt。
- **通过判据**：
  - `nll = -log_probs.gather(...).mean()` 显式取负（PPL 为正且 >1）；
  - CPT 域内 PPL/acc 优于 base（Δ 方向正确），若变差需给出过拟合分析；
  - 评估只算 PPL/acc/F1 等域内指标，不强行套英文 MMLU。

### S11 用时表全程实时刷新 — P1（回归 commit 4099266）

- **执行**：任一正式训练期间观察日志。
- **通过判据**：
  - `cpt_train.py` 每 ~30s 自动调 `timing_table.py --doing 7 ...` 重印整表到训练日志；
  - 心跳输出间隔 ≤2–3min（按时间折算步数）；
  - agent 侧 1–2min tail 一次并回显，不"启动后台就长 sleep"。

### S12 红线与防御性规则 — P0（负向验证）

- **用例**：
  1. **禁 CPU 回退**：在脚本中检索 `device='cpu'`/`torch.device('cpu')` 训练路径——只允许出现在显式确认场景，默认路径必须 NPU；
  2. **smoke 先行**：正式训练前必须存在 2 步 smoke 记录（日志或脚本逻辑）；
  3. **ckpt 必落盘**：训练结束必须产出 `cpt_model_state.pt`（或 resume 用 `ckpt_latest.pt`），否则阶段 8 无从评估；
  4. **短训练不用图模式**：NUM_STEPS<150 时选型不得走 torchair（编译 ~15min 摊销不回来）。

### S13 异常输入边界 — P2

- **用例**：
  1. 模型路径不存在 → 应搜本机可用副本并**显式告知**用户改用了哪个（阶段 0 行为）；
  2. 语料字段不可识别 → 走"取可读文本字段拼接"兜底，不静默产出空数据；
  3. OOM → 降级阶梯生效：优先降 bs→开梯度检查点→grad_accum，记录每级尝试；
  4. 数据集只有 train 分割要验证集 → 用 seed 重建划分取 held-out。

## 三、执行建议

- **回归夹具**：`training-ws/` 下三个既有工作区（NuExtract3-cpt / GLM-OCR-cpt / gemma-4-31B-it-cpt）可直接作为 S05/S06 的对照基线，不必重训全长——跑 10–30 步 smoke 对齐 loss 量级即可。
- **资源隔离**：多卡场景先 `npu-smi info` 挑空闲卡，`ASCEND_RT_VISIBLE_DEVICES` 显式绑定，不写死 0 号卡。
- **评估顺序**：S01→S10→S12 为最小 P0 闭环（半天内可完成）；S03 之后按 P1 顺序推进。
- **与 evals.json 关系**：现有 evals.json 仅覆盖"脚本生成类"静态断言（单卡/DDP 两条）；本文件补充**真机执行类**动态验证场景。可后续把 S01–S13 中可静态断言的部分回填 evals.json 扩展用例。
