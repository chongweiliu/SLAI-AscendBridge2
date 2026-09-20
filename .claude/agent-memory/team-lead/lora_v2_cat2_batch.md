---
name: lora-v2-cat2-batch
description: 台账第2大类5模型（非CausalLM范式）LoRA微调实战（5范式适配/3个新bug/reranker与NER饱和判定）
metadata:
  type: project
---

# 台账第 2 大类 5 模型 LoRA 批量微调实战（2026-09-10）

bge-m3 / nllb-200-distilled-600M / bge-reranker-v2-m3 / distilbert-sst2 / bert-base-NER × 各配套数据集 30%，全部为**非 CausalLM**（encoder/encoder-decoder），按范式适配训练脚本（协议与 cat1 一致）。总报告 `lora-ws/.cat2/LoRA-v2-验证报告.md`。

## 结果速记
- **3/5 双证据改善**：bge-m3 检索（val loss -40.4%、MRR@20 +0.024）、nllb 翻译（-27.4%、ROUGE-L +0.068）、distilbert 情感（-33.9%、准确率 0.80→0.90，SST-2→IMDB 域迁移修正）
- **bge-reranker 无提升**：mmarco-zh 本地档仅 100 query（30%=30 query），60 步 sweep 点≈基线、200 步过拟合（MRR -0.042）——数据量不足撑 1600 步样本消耗
- **bert-NER 完美饱和**：模型原生训练数据就是 conll2003 train（base val loss 0.000148=已记忆），F1 0.90 持平——「持平可能是正确结论」
- 超参模式：弱基座/域差距大选激进（lr2e-4），强基座/数据少 sweep 自动选保守（lr5e-5/r8）

## 五范式适配要点（lora-ws/.cat2/ 全套可复用）
- LoRA 目标跨架构自动发现：候选末名并集 {q/k/v_proj, out_proj, fc1/2, query/key/value/dense, q/k/v_lin, out_lin, lin1/lin2} 覆盖 XLMRoberta/M2M100/DistilBert/BERT
- embed=CLS 池化+L2+in-batch InfoNCE（anchor×[pos;neg], scale20, 真batch16）；rerank=softplus(-(s_pos-s_neg)) 成对；seq2seq/cls/ner=标准 CE（ner 首子词对齐 word_ids）
- 验证指标按范式：检索 hit@1/MRR（候选=10正+10负 或 1正+99负）、翻译 ROUGE-L、分类准确率、NER 实体级 F1（BIO 解码 span 精确匹配）
- mmarco 类"dev-only"数据集：以 query 为切分单位（30% query 训练+负采样扩展），不要把 999 负例全塞训练

## 三个新 bug（cat2 专属，均已在脚本修复）
1. **probe 偏斜**：源文件按 label 排序（imdb 前 12500 全 0）+ train_idx 排序 → probe 前 800 条全同标签（sweep 判据失真）。修法：train_recs 写盘前 `random.Random(43).shuffle`。
2. **nllb text_target 缺 tgt_lang**：只设 src_lang 时 labels 带 eng_Latn 头（错语言码）。修法：`tok.tgt_lang="zho_Hans"`（labels 头与生成 forced_bos 一致）。
3. **seq2seq 生成按源长度切片**：`generate()` 对 encoder-decoder 只返回解码器 token，按 CausalLM 习惯 `gen[0][input_len:]` 切片把译文全切掉（ROUGE 0.0095 假象，base 实际 0.4867）。修法：直接 decode 全序列。

## 关联
[[lora-v1-cat1-batch]]（cat1 流程坑已合入 skill MR !62）；cat2 的三个新坑 + 五范式适配尚未合入 skill（候选后续 MR）。
