---
name: vllm-ascend-consistency-validation
description: Validate that vLLM-Ascend service variants preserve API semantics and stable outputs across single-node, multi-node, PD, dtype, or optimization changes. Use after deployment smoke tests or when comparing service revisions.
---

# 服务一致性验证

一致性验证比较“同一模型制品、同一请求契约”下的服务行为，不把字符串完全相同误认为唯一
正确标准，也不把 HTTP 200 当作通过。先确认 tokenizer、采样参数、chat template、task 和
多模态资产一致，再根据任务选择断言。

## 验证层次

1. API：`/v1/models`、错误码、响应 schema、usage 字段和流式事件序列。
2. 语义：生成输出非空且满足确定性断言；Embedding 维度/数值容差；Reranker 分数排序；
   Reward 数值；多模态使用真实资产。
3. 稳定性：重复请求、并发下无异常、无静默 CPU fallback、PD Proxy 的 KV transfer 成功。
4. 性能：单独记录 TTFT、TPOT、吞吐、P95/P99、显存和 NPU 利用率；性能变化不能自动推断
   为功能不一致或寻优成功。

使用 `scripts/check_openai_consistency.py --baseline URL --candidate URL --prompt TEXT` 做
小规模 HTTP 对比。不同任务与流式响应的判定读取
[references/contracts.md](references/contracts.md)。外部服务请求只在用户已授权的目标环境
执行；脚本默认不改变服务。
