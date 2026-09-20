---
name: lora-v1-cat1-batch
description: 台账第1大类4模型LoRA批量微调实战（sweep择优/3个流程bug与修复/PPL与ROUGE背离案例）
metadata:
  type: project
---

# 台账第1大类 4 模型 LoRA 批量微调实战（2026-09-10）

4 模型（Qwen3-1.7B/OLMoE-1B-7B/Qwen2.5-Coder-1.5B/SmolLM2-1.7B-Instruct）× 配套数据集 30%，ascend-torch-lora skill，全部成功。总报告 `lora-ws/.cat1/LoRA-v1-验证报告.md`。

## 结果速记
- 4/4 held-out CE 下降 9.1%~26.7%；3/4 ROUGE-L 改善（SmolLM2+ultrafeedback 最大 +0.152）
- sweep 一致选出 **lr=2e-4, r=32, α=64**（1.5~7B 模型、200 步、知识注入类任务）
- Qwen3+中文维基续写：CE -26.7% 但贪心 ROUGE 持平 + 个别样本重复退化（分布过锐）——开放式续写任务 CE 是可靠证据、ROUGE 噪声大；对照 lr=1e-4/r16 更差，维持主配置

## 三个流程 bug（都已修复，防复发）
1. **前缀截断信息间隙**（最重要）：切分器把 prefix 截到 cap，但 GT continuation 从未截断 prefix 末尾开始 → codeparrot 83.7%/wikitext 53.1%/wiki-zh 37.4% 样本存在模型不可见间隙 → 模型学会「跳段生成」（Coder 首轮 ROUGE -0.21）。修法：**切点受 prefix_cap 约束 + 前缀不做事后截断（所见即所续）+ 切分器内置自检 assert prefix ≤ cap×1.15**。修复后 Coder ROUGE 转正 +0.025。
2. **validate.py SAMPLE_RATIO 默认 0.5**：给 VAL_SRC 传预切分好的 held-out 文件时，模板会再抽一半（10 条只评 5 条，且静默）。**必须 SAMPLE_RATIO=0.0**。
3. **循环 source 不同 env.sh 的 env 泄漏**：OLMoE 的 ASSISTANT_START='Assistant: ' export 后残留到后续 Coder/SmolLM2 迭代（其 env.sh 不设该变量）→ ChatML 渲染找不到标记 → val_loss n=0 静默空结果。**多工作区顺序批处理必须用子 shell `( source ./env.sh; ... )` 隔离**。

另：validate.py 的 OUT_JSON 硬编码 outputs/validation_results.json，**对照实验的验证会覆盖主结果**（本轮 Qwen3 主验证被 alt 覆盖后重跑）——对照实验要么改 OUT_JSON，要么跑完先备份。

## 模型适配要点
- **OLMoE-0924 无 chat template**（eos=<|endoftext|>，连 im_start 都没有）→ 注入 `{role|capitalize}: {content}{eos}` 模板 + ASSISTANT_START='Assistant: '/ASSISTANT_END='<|endoftext|>'；MoE 路由专家是 nn.Linear，LoRA 正常覆盖（可训练 10.1M 参数）
- **Qwen2.5-Coder base 自带 ChatML 模板**但 eos 是 <|endoftext|>，生成需 VAL_EOS_ID=151645（<|im_end|>）才停得住
- **Qwen3 生成验证必须 enable_thinking=False**（否则生成满篇 <think>；<think> 非 special token 不会被 skip_special_tokens 剥掉）+ 指标前正则剥离 <think>...</think>（base/LoRA 同口径）
- Qwen3 模板渲染 assistant 轮自带空 `<think>\n\n</think>\n\n` 前缀，属于训练格式的一部分，不影响掩码

## 复用资产
`lora-ws/.cat1/`：make_split.py（4 种数据格式→chat 化切分器，含间隙自检）、sweep.py（两阶段+SWEEP_COMBOS 确认模式）、val_loss.py（assistant CE 评测）、gen_report.py（报告生成）、全套日志。验证协议：64 条 held-out 做 sweep 判据（CE），10 条 held-out 做最终 base vs LoRA（ROUGE-L/字符重叠/CE 三证据）。
