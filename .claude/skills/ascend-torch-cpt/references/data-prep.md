# 数据预处理边界（CPT 数据准备）

prepare_data.py.tmpl 处理常见格式，但通用 CPT 数据还有这些边界要注意。

## 1. 去重（dedup）
- 训练数据有重复样本会让模型记忆而非泛化，还浪费算力。
- 简单做法：按文本内容 hash 去重（`hash(text)` set），跨文件去重。大语料可用 MinHashLSH（需 datasketch 库）。
- CPT 常见：同一文档多次出现、模板化重复（FAQ/日志）。**建议对 >万级语料做 hash 去重**。

## 2. 长文档分块（chunking）
- 单文档超 seq_len 时，需切成 seq_len 块。两种策略：
  - **硬切**（按 token 数直接切）：简单，可能把句子/语义切两半。prepare_data 打包即此。
  - **按段落/分隔符切**：在 `\n\n` 或文档边界处切，块不跨文档——语义更干净。CPT 推荐。
- 多文档打包到同一块（当前 prepare_data 的做法：所有 token 拼一个大池再切）会让"块内跨文档"——对 CPT 可接受（标准做法），但若要严格同文档，按文档逐个切块、短文档 padding 或拼接到 seq_len。
- **权衡**：拼池利用率 100% 但块内跨文档；逐文档干净但利用率低。CPT 一般选拼池（利用率优先）。

## 3. 多语料混合配比（mixing）
- 多个语料源要按比例混合（如 70% 领域 + 30% 通用，防止遗忘）。
- 做法：按目标比例从各源采 token，拼成大池再打包；或按源分别打包后轮流采样。
- CPT 经验：领域语料占比过高会**灾难性遗忘**通用能力；建议混 10–30% 原始/通用数据。本 skill 评估时用 PPL 监控遗忘。

## 4. packing vs padding
- **packing**（当前做法）：把所有 token 拼成定长块，无浪费，利用率 100%。块内可能跨文档（见上）。
- **padding**：每个样本单独 padding 到 seq_len，块内同文档但有 pad token 浪费算力（attention mask 忽略）。
- CPT 默认 **packing**（算力效率优先）；若模型对"跨文档注意力"敏感（少见），改 padding 或加 attention_mask 阻断跨文档。

## 5. tokenizer 边界
- `trust_remote_code=True`：模型带自定义 modeling/tokenizer（如 Qwen3.5）必须设；内置模型可不设。
- pad_token：未设时用 eos_token 代（本 skill 默认）。
- chat_template：`apply_chat_template` 需模型目录有 `chat_template.jinja` 或 tokenizer_config 内嵌；缺失则退回手动 `<|im_start|>...` 拼接（见 prepare_data.tmpl 的 try/except）。
- vocab 检查：打包后确认 token id < vocab_size（异常 tokenizer 偶有越界，会崩 embedding）。

## 6. 数据质量监控（打包时打印）
prepare_data 已打印：总样本/子集/总 token/原始块数/训练块数/epoch 估计。
**额外建议**（大语料时）：打印 token 长度分布（p50/p95/max），过长样本单独处理；打印去重前后样本数差。

## 7. 样本不足补救
- 语料 < 步数需求时，prepare_data 循环重采样补齐（已实现），但 epoch 数会很高（如 24×）→ 过拟合风险。
- **更好**：样本不足先扩充语料（下更多数据/用候选数据集），而非 24× 循环。本仓库 `scripts/dataset_mapping.py --candidates` + `download_datasets.py` 可找候选。
- 评估时若 held-out 改善但 train loss 远低于 held-out，是过拟合信号——需加数据或减 epoch。

## 8. 重预处理数据集：一次性并行缓存替代 per-epoch 重建
非 tokenize-即用型语料（科学数据/图结构/需逐条解析的原始文件）常带重预处理（解析→特征化每条几十~几百 ms）。若基座训练管线是 **per-epoch 重建数据**（每 epoch 重采样/重解析），在 NPU 机器上单轮构建可能远超训练本身（实证 30–60min/epoch vs 训练 9min/epoch，见 pitfalls #83）。
- **做法**：`multiprocessing.Pool(N)` **fork** 一次性预处理全部样本——worker 内完成"原始文件→模型输入张量"全链路，返回 numpy 数组（IPC 友好）；主进程落盘 `.pt` 缓存，训练每 epoch 只做分组/洗牌。
- **实测收益**：21088 样本 20s 建完（1300+/s，64 worker），后续 epoch 数据成本归零；训练总时长从数据绑架（≈构建时间×epoch 数）降为纯计算时间。
- **细节**：①worker 数取 CPU 核数的 1/10~1/8 起步（IO 混合型再调）；②传给 worker 的大字典用 **fork 继承全局变量**（父进程 Pool 前赋值），勿 pickle 进 `initargs`（N worker × 大 dict 全量重复传）；③缓存加载 `torch.load(..., weights_only=False)`（含 numpy，pitfalls #81）；④预处理脚本**不 import torch_npu**（纯 CPU，fork 子进程干净，也不会误触 NPU fork 警告）。
- **代价声明**：固定了每 epoch 的采样（失去重采样多样性）——短程 CPT（≤10 epoch）可接受，README/notes 必须写明；依赖 per-epoch 重采样的长训练不适用。
- **判据**：数据构建时间 > 单 epoch 训练时间 → 转此模式。
