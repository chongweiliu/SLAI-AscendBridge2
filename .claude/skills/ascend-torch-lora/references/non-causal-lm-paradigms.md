# 非 CausalLM 五范式 LoRA（encoder / encoder-decoder 模型）

> 场景：目标是 BERT 类 encoder、双塔嵌入、重排器、seq2seq 翻译或 token 分类模型，而不是 CausalLM。
> 2026-09-10 台账第 2 大类 5 模型全链路实测（bge-m3 / nllb-200 / bge-reranker-v2-m3 / distilbert-sst2 / bert-base-NER）。
> 训练器模板：`scripts/lora_train_task.py.tmpl`（范式经 `PARADIGM` 环境变量切换）；完整配套（切分/择优/验证）
> 实战副本在 `lora-ws/.cat2/`（make_split2.py / sweep2.py / val_loss_task.py / validate_task.py）。

## 五范式速查

| PARADIGM | 模型加载 | 数据格式（jsonl 每行） | loss | 真实 batch | 验证主指标 |
|---|---|---|---|---|---|
| embed | `AutoModel`（bge-m3 类） | `{"anchor","positive","negative"}` | in-batch InfoNCE：`CE(anchor@[pos;neg].T×20, arange)`，CLS 池化+L2 | 16（in-batch 负例依赖真 batch） | hit@1 / MRR（候选=10 正+10 负） |
| seq2seq | `AutoModelForSeq2SeqLM`（nllb/M2M100） | `{"en","zh"}` | 标准 CE（labels=目标语 token，pad→-100） | 8 | 生成 ROUGE-L/字符重叠 vs 参考译文 |
| rerank | `AutoModelForSequenceClassification`（num_labels=1） | `{"query","positive","negative"}` | 成对排序 `softplus(-(s_pos−s_neg)).mean()` | 8（正负对成 16 编码） | hit@1 / MRR（1 正+99 采样负） |
| cls | `AutoModelForSequenceClassification` | `{"text","label"}` | 分类 CE | 16 | 准确率 |
| ner | `AutoModelForTokenClassification` | `{"tokens","tags"}` | token CE（**首子词对齐**：word_ids，续子词 -100） | 16 | 实体级 F1（BIO 解码 span 精确匹配） |

## LoRA 目标跨架构自动发现

候选末名并集一次覆盖全部架构：
```
q_proj,k_proj,v_proj,out_proj,fc1,fc2        # M2M100（enc+dec）
query,key,value,dense                         # XLMRoberta / BERT（attention + FFN + 分类头 dense/out_proj）
q_lin,k_lin,v_lin,out_lin,lin1,lin2           # DistilBERT
```
扫 `named_modules()` 取 `nn.Linear` 末名与候选交集即可，无需按架构手填。注意：RoBERTa 系分类头的
`dense`+`out_proj` 会被自动挂上 LoRA（有利）；DistilBERT 分类头（`classifier`/`pre_classifier`）不在候选内。

## 各范式实测要点（含结果量级）

- **embed**：bge-m3 + all-nli triplet 30%，val InfoNCE 0.643→0.383（-40.4%），MRR@20 0.859→0.883。归一化后算相似度、scale=20（sentence-transformers MNRL 默认）。
- **seq2seq**：nllb-200-600M + opus100 30%，val CE 2.93→2.13（-27.4%），ROUGE-L 0.487→0.555（base 长句截断/漏译，LoRA 译文更完整）。**必须** `tok.src_lang`+`tok.tgt_lang` 都设（#35）、生成不按源长切片（#36）。
- **rerank**：bge-reranker + mmarco-zh，**无提升**（见下）。
- **cls**：distilbert-sst2 + imdb 30%，val CE 0.257→0.170（-33.9%），准确率 0.80→0.90（SST-2 短文本→IMDB 长影评域迁移被修正）。
- **ner**：bert-base-NER + conll2003，F1 0.90 持平（见下）。conll txt 的 `B-organisation` 系标签须映射到模型的 `B-ORG` 空间。

## 「无提升」的两类正确结论（避免误判为失败）

1. **数据量不足 → 过拟合**：mmarco-zh 本地档仅 100 query（30%=30 query），sweep 60 步点 val loss ≈ 基线
   （0.0538 vs 0.0539），200 步正式训练反升（MRR -0.042）——30 个 query 的重复采样撑不起 1600 样本消耗。
   判据：sweep 点≈基线 + 训练 loss→0 但 val loss 上升。
2. **完美饱和**：bert-base-NER 原生训练数据就是 conll2003 train（base val loss 0.000148=已记忆），
   F1 0.90 持平。判据：base val loss 接近 0。
   
与 CPT 批次同口径：**持平可能是正确结论**——报告如实记录根因，不要为了「有提升」加步数/加 lr。

## 超参模式（sweep 60 步 × 5 组合，held-out val loss 判据）

- 弱基座/域差距大（embed/seq2seq/cls）→ 选激进组合（lr=2e-4，r=8~32）
- 强基座/数据少（rerank/ner）→ sweep 自动选保守组合（lr=5e-5, r=8）——即便如此仍无提升空间（见上）

## 数据切分注意（接 corpus_to_sft / cat1 协议）

- 同样是 30% 池（seed=42）+ held-out（64 择优 + 10 终验）严格不相交，索引落盘可审计
- **"dev-only"数据集**（如 mmarco 只有 100 query 的 dev）：以 query 为切分单位（30% query 训练），
  训练对由负采样扩展（每 query 采样 N 负例），**不要**把 999 个负例全部塞进训练
- 训练样本写盘前必须洗牌（#34：源文件有序 + 索引排序 → probe 全同标签）
