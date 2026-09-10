# LoRA 超参自动选择（模型尺寸/数据量感知）

> route_select.py 内置这些规则；本文解释依据，便于人工覆盖时判断。

## 学习率 LR（按模型参数量）

| 模型大小 | 推荐 LR | 依据 |
|---|---|---|
| < 1B | **2e-4** | 小模型容量小，LoRA 需更快注入；QLoRA/常见 SFT 实践 |
| 1B – 30B | **1e-4** | LoRA SFT 最常用值；27B 实测（Qwen3.5-27B, eff batch 8, 150 步）loss 0.38→0.24 收敛良好 |
| > 30B | **5e-5** | 大模型对扰动更敏感；QLoRA 论文 33B/65B 档 |

- LoRA 的 LR 比 full-ft 高一个量级（只更新低秩增量），1e-4~2e-4 是安全带；若训练 loss 震荡不降 → 减半；若下降过快过拟合（验证指标变差）→ 减半并加 early stop。
- eff batch ≥ 16 时可上浮 1.5×；eff batch 8 以下按下表保守。

## LoRA 秩 r / alpha / dropout

| 任务类型 | r | alpha=2r | 说明 |
|---|---|---|---|
| 风格/格式/对话风格（默认） | 16 | 32 | 0.8B 教学对话与 27B hint-tuning 格式学习均验证有效 |
| 知识/技能注入（新领域推理） | 32–64 | 2r | 容量更大；数据需充足（≥数万样本），否则易过拟合 |
| 轻量风格微调 | 8 | 16 | 最省显存，迭代最快 |

- dropout 0.05（默认）；数据 < 1k 样本建议 0.1。
- 目标模块：模板自动发现全部注意力/线性注意力/MLP 投影（含 in_proj_* 系列），无需手填。

## warmup / 调度

- **warmup = 10% × 总步数**（最少 10 步）。150 步 → 15；60 步 → 10。
- cosine 衰减到 ~0（模板已内置）。

## 有效 batch 与 grad_accum

- 目标**有效 batch ≈ 8**（LoRA SFT 稳定下限）：`grad_accum = ceil(8 / (batch_per_card × world_size))`。
- 多卡数据并行时 eff batch 自然增大（16 卡×1 = 16），无需额外 accum；eff batch 8–32 都在安全带。

## 步数（steps/epochs 换算）

- 用户给 steps 用 steps；给 epochs 则 `steps = ceil(epochs × n_samples / eff_batch)`。
- 经验下限：格式/风格任务 ≥ 100 步；知识注入 ≥ 500 步且看验证指标 early stop。
- 数据 < 1k 样本：多 epoch 会过拟合，建议 epochs ≤ 3 并以 held-out 验证曲线选点。

## 显存估算公式（route_select 内置，bf16 + grad-ckpt）

```
权重     = params × 2B                     (FSDP2: /N; device_map: /N; 单卡: 全量; MoE=全专家总参数)
优化器   = params × 0.8% × 12B             (LoRA AdamW fp32 状态; MoE 融合专家挂不上 LoRA 时实际更小)
激活     ≈ [L×seq×H×2 + seq×(4H+2I)×2 + seq×V×10] × batch × 1.5   (GC 生效: 各层输入+单层重算+logits/CE)
           I: dense=intermediate; MoE=top_k×moe_intermediate+shared (有效激活中间维)
           FSDP2: 每卡全量(数据并行); device_map: /N(流水线分摊); 单卡: 全量
可用预算 = 0.85 × 卡 HBM
```
实测对照（--probe 遥测账本持续校准）：
- 27B dense/seq2048/16卡: 估 17.4GB, 运行正常; 4卡估 27.8GB vs 实测 ~25GB（偏保守，安全）
- 35.95B MoE/seq1024/16卡: 估 14.8GB vs **实测 9.25GB**（偏保守 37%，MoE 激活高估是主因之一）

**步时/ETA 必须用稳态口径**：首步含 NPU 算子编译（大模型可达 38~210s，稳态的 2~10 倍），
ETA = 末步增量 × 总步数。probe 已内置（曾实测平均法把 45min 报成 285min）。

## 精度安全默认（所有路线一致，不可为性能牺牲）

- bf16 autocast + grad-ckpt（`use_reentrant=False`）+ `model.train()`（GC 生效前提）
- eager attention 基线（CANN 9.0.0 可试 sdpa，对混合注意力架构收益近零）
- 训练前标签自检（n_label>0 且无 user token 泄漏）→ 训练后 base vs LoRA 验证门
- 固定 seed 可复现；FSDP2 的 reduce bf16 与单卡数值等价级

## sweep 择优协议（2026-09-10，4 模型批量实测沉淀）

**两阶段 sweep**（每组合 60 步、800 条 probe 子集、eff batch 与正式训练一致）：
1. 阶段1：lr ∈ {5e-5, 1e-4, 2e-4} @ r=16；
2. 阶段2：最优 lr 下 r ∈ {8, 32}（r=16 已有）。
**判据** = 64 条 held-out 的 assistant-token 加权平均 CE（与训练同口径的字符偏移掩码；token 加权而非样本平均，长回复不欠权）。

- 数据修复/换数据分布后不必全量重扫：用固定组合**确认模式**复验（如 `SWEEP_COMBOS="0.0002:32,0.0001:16"` 只跑最优 vs 亚优），本轮修复切分 bug 后复验结论不变。
- sweep 用的 val 集与最终 10 条验证集**分开**（64 条择优 + 10 条终验，都从训练池外取）。

**实测数据点（4 模型 × 200 步，全部一致选出 lr=2e-4, r=32, α=64）**：

| 模型 | 参数量 | 任务 | 最优 val CE（vs 次优） |
|---|---|---|---|
| Qwen3-1.7B | 2.0B | 中文维基续写 | 2.669（vs 1e-4/r16: 2.677） |
| OLMoE-1B-7B-0924 | 6.9B MoE | 英文维基续写 | 2.099（vs 1e-4/r16: 2.158） |
| Qwen2.5-Coder-1.5B | 1.5B | 代码补全 | 0.976（vs 1e-4/r16: 1.008） |
| SmolLM2-1.7B-Instruct | 1.7B | 指令跟随 | 0.981（vs 1e-4/r16: 0.994） |

**对上表规则的修订**：1~2B 档在 **200 步短程 SFT** 下 2e-4 优于默认 1e-4（短程需要更快注入）；知识注入类任务（维基/代码/领域语料）**r=32 优于 16/8**。原 LR 表仍适用于 >2B 或长程（≥500 步）训练。**注意 pitfalls #33**：CE 判据偏好尖锐分布——开放式生成任务在择优后必须补生成指标双证据，必要时跑备选配置完整对照。
