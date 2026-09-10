# 原始语料 → LoRA SFT 任务化（raw corpus to SFT）

> 场景：用户要做 LoRA SFT，但给的数据集是**原始语料**（维基条目/文章行/代码文件/偏好对）而非 chat 对话。
> 处理原则：先任务化转成单轮 chat（user=任务指令+上文，assistant=目标文本），再走标准 SFT 流程。
> 一条命令完成：`scripts/corpus_to_sft.py.tmpl`（内置 4 种范式 + 无间隙切分自检 + 30%/held-out 协议）。

## 四种已验证的任务化范式（2026-09-10，4 模型实测）

| 语料形态 | 实例 | 任务化 | user | assistant |
|---|---|---|---|---|
| 中文章节正文（`{completion:...}` JSONL） | wikipedia-cn-20230720-filtered | 百科续写 | 「请续写下面的中文维基百科条目。只输出续写的正文，不要解释：\n{前缀≤400字}」 | 后文（≤800字） |
| 英文文章行（parquet `text` 列） | wikitext-103-raw-v1 | 文章续写 | "Continue the following article. Output only the continuation:\n{前缀≤700字符}" | 后文（≤1300字符） |
| 代码文件（`{content:...}` json.gz） | codeparrot-clean | 代码补全 | 「请补全下面的 Python 代码，从截断处继续写，只输出代码：\n{前半≤900字符}」 | 后半（≤1700字符） |
| 偏好对（`{instruction, chosen, rejected}` json） | ultrafeedback | 指令跟随 | instruction（≤600字符） | chosen（≤1500字符） |

切分边界：中文按句末标点（。！？）、英文按句号+空格、代码按行——**切点受 prefix_cap 约束，前缀不做事后截断**（pitfalls #26 的信息间隙 bug 及其修复）。

## 切分与 held-out 协议（与 30% 训练池惯例一致）

```
全量索引 seed=42 洗牌
├── 前 30% = 训练池 ──(池内随机取 N_TRAIN 条, seed=42)──> train.jsonl（建议 ≤3000，控制 tokenize 时长）
│                          └── 前 800 条 ──> probe_train.jsonl（sweep 用）
└── 其余 70%
    ├── 随机 64 条（seed=7）──> val_loss.jsonl（超参择优判据）
    └── 随机 10 条（seed=123）──> val10.jsonl（最终 base vs LoRA 验证）
```

- 三份验证/训练索引**集合运算验证严格不相交**，全部落盘 `split_meta.json` 可审计；
- 训练消耗量 = steps × eff_batch（如 200×8=1600 条），从池内随机有放回抽样——池大于消耗量是正常的，30% 定义的是「训练数据来源域」。

## 验证阶段注意（坑都在 pitfalls）

- val10.jsonl 是**预切分文件**：validate.py 对小文件（≤N_VAL×2）自动全量使用；老版模板须显式 `SAMPLE_RATIO=0.0`（#27）；
- Qwen3 系模型生成验证必须关思考块 + 指标剥离 `<think>`（#32）；
- base 模型「模板收尾 token ≠ eos_token」时用 `VAL_EOS_ID` 显式指定停止符（#31）；
- 开放式续写任务 ROUGE 噪声大：报告给 CE + 生成指标双证据，逐条看 10 条 per-sample（#33）。
