# 文本 LM / 音频 LLM 的 RL 后训练（RLHF/GRPO，可选）

> 适用：论文含 "RLHF" / "GRPO" / "preference alignment" / "reward model" 的文本 LM 或音频 LLM。
> 与扩散 RL（`generative-diffusion-cpt.md`「RL 后训练」）是**不同范式**，不可混用。
> 仅当用户/论文明确要求 RL 后训练时执行，否则 CPT 在阶段 8-9 结束。

## 核心方法论

文本 LM 的 RL（RLHF/GRPO）与扩散 RL 共享 **RL 骨架**（policy gradient + group-relative advantage + KL-to-ref + frozen reward model），但**策略参数化、轨迹、奖励、KL 计算完全不同**：

### 1. 策略（Policy）
- 文本 LM：**离散 token 分布** p(token | context)，CausalLM 的 `log_softmax(logits)`
- 音频 LLM（audio→text 转写模型）：同文本 LM——policy 是 language_model 的 token 分布，audio 是输入上下文（mel features），不参与策略

### 2. 轨迹（Trajectory）
- 文本 LM：**token 序列**（离散、变长），每个 token 是一个 action
- 音频 LLM：同文本 LM（token 序列，audio 固定输入）

### 3. 奖励（Reward）
- 文本 LM：**文本偏好奖励模型**（reward model 对完整回复打分），或**规则奖励**（数学正确性、代码执行通过、格式合规）
- 音频 LLM：**WER（词错率）**或**转写偏好奖励**（转写文本 vs 参考）

### 4. 组相对优势（GRPO 核心，与扩散 RL 通用）
- 每 prompt 采 G 个不同回复（temperature > 0，不同随机采样）
- `advantage = (r_i - mean(r)) / (std(r) + ε)`（组内相对，替代价值函数）

### 5. 策略梯度
```
log_prob = Σ_t log p_policy(token_t | token_<t, context)   # 每个回复的 token log-prob
loss = -mean(advantage.detach() * log_prob)                # REINFORCE/GRPO surrogate
```
- 通过 **token log-prob** 反传（与扩散 RL 的可微去噪采样反传不同）
- **标准 GRPO 目标**用 PPO-style clipped ratio 代替纯 log-prob（见下「NPU 注意事项」PPO clip），实践优于纯 REINFORCE

### 6. KL-to-参考策略正则
```
log_ratio = log_prob_policy(response) - log_prob_ref(response)   # log-prob 比率
kl = mean(log_ratio)  # 或 PPO-style: mean(exp(log_ratio) - 1 - log_ratio)
loss += β * kl
```
- 参考策略 = SFT/CPT 模型（冻结，`requires_grad=False`）
- **与扩散 RL 的 KL 不同**：扩散用速度 L2 `‖v_policy - v_ref‖²`，文本 LM 用 token log-prob 比率

### 7. 评估
- 文本 LM：held-out prompt 上的平均奖励、奖励分布、生成多样性(entropy)
- 音频 LLM：held-out 音频上的 WER 改善
- **不要套扩散 RL 的 velocity MSE / faithfulness 分类器**（范式不匹配）

## 文本 LM RL vs 扩散 RL 对比（不可混用）

| 维度 | 文本 LM RL (RLHF/GRPO) | 扩散 RL (DDPO/Flow-GRPO) |
|---|---|---|
| 策略 | 离散 token 分布 | 连续速度场 v(z_t,t,cond) |
| 轨迹 | token 序列（变长） | 去噪路径（定长 latent，K 步 Euler） |
| 奖励 | 文本偏好模型/规则 | latent/image 分类器(faithfulness) |
| 策略梯度 | token log-prob × advantage | 可微采样轨迹 × advantage |
| KL | token log-prob 比率 | 速度预测 L2 |
| GRPO 采样 | G 个不同回复(temperature>0) | G 个不同噪声 z0 |
| 代码 | `model(input_ids).log_softmax` | `dit(z, t, cond)` 可微 Euler |

## 音频 LLM RL

音频 LLM（audio-text-to-text，如 Qwen2-Audio/MOSS）的 RL **与文本 LM RL 完全一致**：
- policy = language_model 的 token 分布（audio_tower + projector 冻结）
- 轨迹 = token 序列（audio 是固定输入上下文）
- 奖励 = WER / 转写偏好
- KL = token log-prob 比率
- **唯一区别**：prompt 含 audio input（mel features + audio special tokens），但策略梯度仍在 token 级

**例外**：若音频 LLM 是**语音生成模型**（text→audio, diffusion-based），则走扩散 RL（见 `generative-diffusion-cpt.md`），不走本文档。

## 在 NPU 上的注意事项

- **NpuFusedAdamW**：同 CPT，`zero_grad(set_to_none=False)`（pitfalls #67）
- **bf16 autocast**：policy 模型 forward + backward 在 bf16 autocast 下跑
- **log_prob 精度**：token log-prob 在 bf16 下可能有数值精度问题，建议 log_softmax 在 fp32 计算（`logits.float().log_softmax(-1)`）
- **参考模型显存**：需同时加载 policy + reference 两个模型（2× 显存），小模型(1-7B)单卡可行，大模型需模型并行
- **GRPO group 采样**：G=4-8 个回复并行生成，可用 `torch.cat` 拼成 batch 一次 forward
- **PPO clip（标准 GRPO 目标，优于纯 REINFORCE）**：`ratio = exp(log_ratio); clipped = clamp(ratio, 1-ε, 1+ε); loss = -mean(min(advantage.detach()*ratio, advantage.detach()*clipped))`

## 实施流程（在 CPT 阶段 9 之后，可选）

1. 确认用户/论文明确要求 RL（否则止于 CPT 阶段 9）
2. 冻结 CPT 模型作为 reference
3. 复制 CPT 模型作为可训 policy
4. 准备 reward model（偏好模型 或 规则奖励函数）
5. GRPO 循环：sample G responses → reward → advantage → policy gradient + KL → update
6. 评估：held-out 奖励/WER 改善 + 生成质量

## 踩坑速查（详见 pitfalls.md #73-76）
- #73 文本 LM RL 的 KL 用 token log-prob 比率，不用速度 L2（与扩散 RL 不可混用）
- #74 文本 LM reward 是偏好模型/规则，不用 latent 分类器
- #75 GRPO group = G 个不同回复(temperature>0)，不用 G 个不同噪声
- #76 音频 LLM RL = 文本 LM RL（token 级，audio 是上下文）；仅语音生成(diffusion)走扩散 RL
