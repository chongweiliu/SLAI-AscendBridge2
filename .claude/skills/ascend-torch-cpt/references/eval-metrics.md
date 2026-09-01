# 评估指标定义与 held-out 重建

## 指标选择原则
按“训练数据与模型”选合理指标，不强套不相关基准。
- **方言/领域语料 CPT**：域内 PPL / next-token acc / 生成 F1·Recall·EM（本文档）。
- **英文知识/通用能力**：MMLU/HellaSwag 是英文知识/常识基准，需 lm-eval-harness + 数据下载；与领域方言 CPT 不对齐，默认**不**跑。若用户明确要，下载小子集 0-shot/5-shot 跑并说明局限。

## 域内指标（手算，无需外部库；本机常无 datasets/lm_eval/rouge）

### 1. PPL（perplexity，核心）
- PPL = exp(mean NLL)，全序列 token 级。
- 衡量 LM 对该语料分布的拟合度，越低越好。
- 同时给 mean NLL（对数版，看绝对降幅）。

### 2. mean NLL
- `nll = -log_softmax(logits).gather(-1, targets).mean()`。**务必取负**（pitfalls #5）。
- 软指标，看概率分布质量，与 acc 互证。

### 3. next-token accuracy
- `pred = logits.argmax(-1); acc = (pred==targets).mean()`。
- 硬指标，看 top-1 命中率。比 PPL 直观但丢概率细节。

### 4. 生成 Precision/Recall/F1/EM（SQuAD 式 token 级）
取每条对话**末轮 assistant** 作 gold，前文作 prompt，贪心生 128 token，比对：
- tokenize 双方为 subword token。
- `Precision = common/len(gen)`：生成不乱编（切题度）。
- `Recall = common/len(gold)`：生成覆盖度。
- `F1 = 2PR/(P+R)`：综合（被短板拖累）。
- `EM = 1 if gen==gold else 0`：精确匹配（自由生成常 0，信息量低）。

## held-out 重建（数据集无独立 validation 时）
数据集常只有 train.jsonl。用训练划分同 seed 重建，取未参与训练部分：
```python
rng = random.Random(SEED)
idx = list(range(total)); rng.shuffle(idx)
n_train = int(total * RATIO)
train_idx = set(idx[:n_train])        # 训练用过的
held_idx = idx[n_train:]              # 未用
val_idx = held_idx[:N_VAL]            # 取前 N 条验证
```
保证“预训练未使用”。

## 对比设计
- **base**：原始权重（多模态走 multimodal-remap.md remap 到文本头）。
- **CPT**：训练产物 ckpt（state_dict，直接 load）。
- 两套同一 10 条 held-out、同一 prompt、同一解码（贪心）。
- 输出：per-sample + 汇总均值 + Δ(cpt-base)，Δ 方向注明（nll/ppl↓改善，acc/f1↑改善）。
- **严格同条件对比（生成式/采样类模型必做）**：base 与 CPT 用**相同种子**驱动一切随机源（解码顺序 randn、采样 RNG、dropout 关闭、数据顺序），保证唯一变量是权重；对比前先做**确定性自检**——同一权重同种子跑两轮，结果应**逐位一致**，不一致说明有未固定的随机源（多 GPU 种子、cudnn 类等价物、worker RNG）。
- **协议正确性锚定**：base（CPT 前权重）的关键指标应与论文/官方报告量级交叉验证（如 recovery/PPL/acc 与发表数字同量级）——锚上了才能证明自己的评估协议没写错；锚不上先查评估脚本再谈 CPT 效果。

## 结论判定
- PPL↓ + acc↑ + NLL↓ 同向 → 训练有效且扎实（非偶然）。
- 10/10 样本全改善 → 未退化。
- Recall/EM 不动 → 生成覆盖/精确匹配是短板（指标诚实指明）。
- epoch 多(小数据集) + held-out 仍改善 → 学到通用模式非纯记忆（但需注明过拟合风险）。
- **"持平"本身可能是正确结论**：若 CPT 语料就是基座**原训练分布的子集**且基座已充分收敛（如官方已训 100+ epochs），短程 CPT 在留出集上**预期就是持平**（±1% 内波动），报告为"符合预期"而非失败——此时要拿到真实增益需换分布语料（新数据快照/新领域）或以基座未见数据重划分（ProteinMPNN CPT 实证：同分布 6 epochs，recovery 45.13% vs base 45.66%，训练收敛良好但无净增益）。
- **发散与持平要区分开报告**：train loss 上升 = lr 配置问题（pitfalls #82），不属"持平"结论；train 收敛 + held-out 持平 = 任务结构性持平。

## 评估代码骨架
见 `scripts/eval_cpt.py.tmpl`。关键：
- fp32 模型 + autocast(bf16) 前向（pitfalls #4）。
- logits[:−1] 对 targets[1:]。
- 生成包 `torch.autocast("npu", bfloat16)`。

## FSDP2 实战案例（Qwen3.5-9B，已验证）
**场景**：9B 大模型 FSDP2 8卡 CPT 后评估。本案例覆盖大模型评估的特殊处理。

### 关键点（小模型评估没有、大模型必须）
1. **ckpt 加载**：FSDP2 训练保存的 `cpt_model_state.pt` 是 `full_tensor()` 聚合后的全量 bf16 state_dict（见 pitfalls #20）。评估时单卡直接 `model.load_state_dict(sd, strict=False)` 即可（不需要再切分）。
2. **base 加载**：多模态模型需 remap 文本头（见 multimodal-remap.md）。9B base = strip `model.language_model.`→`model.`，丢 visual/mtp，`Qwen3_5ForCausalLM(text_config)` + `tie_weights()`。
3. **显存**：9B fp32 模型单卡 65GB 可装下评估（无优化器状态，只需权重+激活）。eval 用 fp32 主权重 + autocast bf16 前向（与训练一致，避免纯 bf16 数值崩 pitfalls #4）。
4. **评估是单卡**（不需要 FSDP2/DDP）：训练用 8 卡 FSDP2，评估只需 1 卡加载全量 ckpt。这降低复杂度。
5. **30 条 held-out**：seed42 重建 60% 训练划分，取未用 40% 的前 30 条。比 10 条更稳健。

### 实测结果（30 条 held-out，9B FSDP2 CPT）
| 指标 | base | CPT | Δ | 结论 |
|---|---|---|---|---|
| PPL | 30.42 | 5.78 | **-81%** | ✅ 大幅下降 |
| next-token acc | 0.441 | 0.634 | +44%相对 | ✅ 提升 |
| mean NLL | 3.415 | 1.754 | -1.66 | ✅ 下降 |
| 生成 F1 | 0.059 | 0.318 | 5.4× | ✅ 显著提升 |
| 生成 Recall | 0.134 | 0.328 | +0.19 | ✅ 覆盖提升（优于 0.8B 的停滞） |
| EM | 0.0 | 0.0 | 0 | ⚪ 预期 |

### 结论模板（供复用）
1. PPL↓ + acc↑ + NLL↓ 同向 → 训练有效扎实。
2. 大模型(9B) base PPL(30)天然低于小模型(0.8B,91)，CPT 仍带来 81% 域内提升 → 大模型 CPT 同样收益明显。
3. 9B 的 Recall 也显著提升（不同于 0.8B 停滞）→ 模型容量充足，生成覆盖改善更充分。
4. held-out 未训练数据改善 → 泛化非记忆。

### 与 0.8B DDP 案例对比
| 维度 | 0.8B DDP | 9B FSDP2 |
|---|---|---|
| 训练并行 | DDP 8卡（每卡持全量参数） | FSDP2 8卡（每卡持 1/8 分片） |
| 优化器 | NpuFusedAdamW（融合，可用） | AdamW（NpuFusedAdamW 与 FSDP2 不兼容） |
| ckpt 保存 | `model.module.state_dict()` rank0 存 | `DTensor.full_tensor()` 聚合全量存（pitfalls #20） |
| base PPL | 91.2 | 30.4 |
| CPT PPL | 12.6 | 5.8 |
| PPL 降幅 | -86% | -81% |
| Recall 改善 | 微升(+0.012) | 显著(+0.194) |

## MMLU/HellaSwag how-to（仅通用知识型模型，用户明确要求时）

**前提**：模型是通用知识型（非领域方言 CPT），用户明确要英文基准。否则用域内 PPL/acc（本 skill 默认）。

**两种方式**：
1. **lm-eval-harness**（标准，需安装）：`pip install lm-eval`；`lm_eval --model hf --model_args pretrained=<path> --tasks mmlu,hellaswag --num_fewshot 5`。国内需 `--load` 离线数据集。本机若已装可一行跑。
2. **手写 few-shot**（本机无 lm-eval 时）：从 HF 下载 MMLU 子集（`cais/mmlu`，国内用 hf-mirror），每个问题拼 5 个 few-shot 例子作 prompt，模型算"正确选项"的 logprob，取 argmax 选项对不对，算 acc。HellaSwag 同理（选结尾句）。跑小样本（如每 subject 20 题 × 5 subject = 100 题）即可看趋势，不必跑全量 14k。

**局限说明**（务必告知用户）：
- MMLU/HellaSwag 是**英文知识/常识**基准；对领域方言 CPT 不对齐（方言 CPT 后这些指标基本不动甚至略降是正常的，因为没训英文知识，反而可能因容量挤占略降）。
- 短训练(100步)对 MMLU 影响极小——MMLU 衡量的是预训练沉淀的知识，100 步 CPT 不足以改变。
- 正确用法：CPT 前后跑 MMLU 主要是**监控是否遗忘**（Δ 不应显著为负），而非看提升。

**手写 few-shot 骨架**：
```python
# 加载 MMLU (hf-mirror)
from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test")  # 或 --revision
# 5-shot prompt = 5 个示例 + 待测题；模型算 ABCD 各 token logprob，argmax
# acc = (argmax==gold) 均值
```
