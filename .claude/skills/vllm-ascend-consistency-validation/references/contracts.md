# 一致性契约

| 任务 | 必查输出 | 比较方式 |
|---|---|---|
| Chat/Completion | finish reason、文本、token usage | greedy 可严格文本比较；跨实现可做规范化与语义断言 |
| Embedding | 向量维度、有限值、归一化约定 | cosine/L2 容差由可信基线和业务需求定义 |
| Reranker | 每项分数、顺序、有限值 | 排序一致，分值使用明确绝对/相对容差 |
| Reward | `reward_score` schema、有限值 | 数值容差与排序断言 |
| 多模态/ASR | 输入资产、文本/时间戳 schema | 使用同一真实资产和 processor；纯文本不能代替 |
| Streaming | event 顺序、增量内容、终止帧、usage | 拼接结果与非流式语义一致，同时验证 SSE 结构 |

至少包含短输入、接近目标长度的长输入、Unicode/特殊 token、并发重复请求和业务代表样本。
确定性测试固定 temperature、top_p、seed（实现支持时）、max tokens 和 chat template。服务间
模型别名可以不同，但必须解析到同一制品哈希。

把“重复稳定”“跨后端一致”“答案正确”“性能达标”分成四个字段。轻量测试只证明部署链路，
不能替代完整精度 benchmark；不得以若干常识题非空响应宣称模型精度通过。
