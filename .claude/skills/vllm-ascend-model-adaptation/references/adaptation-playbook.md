# vLLM-Ascend 模型适配手册

## 制品分析

读取 `config.json`、嵌套 `text_config`、`architectures`、`auto_map`、tokenizer/processor、
safetensors index 和 generation config。记录 attention 类型、KV heads、RoPE、MoE experts、
量化、多模态 encoder/projector、MTP 与 task。先核对目标版本 vLLM 是否已有该架构；
vLLM core 未支持时，单改 Ascend backend 通常不够。

## 最小适配层

- **Config/registry**：类实现兼容但名称或字段不同。新增规范化和注册映射，保留原模型 ID。
- **Backbone 提取**：复合 checkpoint 中的语言 backbone 已被 vLLM 支持，可生成独立 LLM
  制品。按 safetensors index 过滤 key、重写索引与哈希、复制 tokenizer，并声明丢弃的
  encoder 能力；不能把该制品继续宣称为原多模态模型。
- **vLLM plugin/model executor**：forward、attention、MoE 或 processor 语义不同。实现最小
  注册插件，优先复用 vLLM 接口和 vLLM-Ascend 已有算子，不 fork 无关代码。
- **Backend/operator**：算子在 NPU 缺失或精度不一致。先做 eager correctness，再实现或替换
  kernel，最后验证图模式；CPU fallback 必须显式记录。

## 权重门禁与验证

转换前后列出每个 tensor 的 name、shape、dtype，总参数量与遗漏/新增 key。vocab、embedding、
lm_head、expert 数和 TP 分片规则必须一致。不要用 `strict=False` 隐藏大范围缺失。

1. config/registry 与 dummy load，证明框架识别架构。
2. 单卡真实权重 eager forward，证明权重和基础算子正确。
3. 目标 dtype 与固定提示，和可信 CPU/GPU/原框架基线做任务级比较。
4. 目标 TP/DP/EP、ACLGraph、量化、多模态或 MTP，各特性单独开启并留证。
5. OpenAI API、流式响应、并发和长上下文；多机再做 HCCL，PD 再做 KV transfer。

结果区分 `upstream_supported`、`local_plugin`、`backbone_only` 和 `experimental`。本地适配通过
不能改写官方支持矩阵。
