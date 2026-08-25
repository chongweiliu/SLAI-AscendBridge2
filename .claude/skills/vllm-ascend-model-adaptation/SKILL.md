---
name: vllm-ascend-model-adaptation
description: Adapt a model that is not directly covered by the official vLLM-Ascend model recipes. Use when architecture, task, custom code, quantization, or multimodal processors are unsupported or fail during first forward.
---

# vLLM-Ascend 模型适配

先判定缺口属于配置、模型注册、权重格式、算子/attention、processor 还是 vLLM API 语义，
再选择最小适配层。不得把“能 import”或“服务启动”当作适配完成。

## 路径选择

实施改动前读取 [references/adaptation-playbook.md](references/adaptation-playbook.md)，选择 config、
backbone 或 plugin 路线并保留上游贡献边界。

- 官方模型教程和支持矩阵命中：复用官方 recipe，只做硬件/版本/权重门禁。
- Hugging Face/Transformers 架构可映射到已有 vLLM backbone：提取并注册等价 backbone，
  保留 tokenizer、processor、权重 key 和 task 语义，跑 deterministic smoke test。
- 仅少量参数名或 config 差异：优先 config/model registry/权重转换层，不复制整套模型实现。
- 自定义 `modeling_*.py`、特殊 attention、MoE 路由或多模态输入：先做最小 vLLM plugin，
  明确未覆盖的算子与 CPU fallback；必要时回到完整模型适配，不伪装成官方支持。

## 验收

至少记录模型 config、transformers/vLLM/vLLM-Ascend 版本、权重转换哈希、变更文件、task、
dtype/TP、输入样例和输出断言。依次通过 config 加载、单层/小 batch forward、单卡真实权重、
目标 TP 和 API 语义验收；多模态必须使用真实图片/音频资产。任何 fallback、精度差异或未覆盖
算子都写入报告，再交给一致性验证 Skill。

这条路径是适配与验证，不等于自动性能寻优；性能候选必须有同制品、同拓扑的 benchmark 证据。
