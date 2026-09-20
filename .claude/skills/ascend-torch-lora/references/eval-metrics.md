# 验证指标：ROUGE-L / 字符重叠 / 生成长度

> base vs LoRA 验证三指标，从不同角度衡量"生成 teacher 回复"与"GT teacher 回复"的吻合度，互补使用，避免单一指标盲区。

## ROUGE-L（char-level F1）

- **算法**：生成文本与 GT 取字符级最长公共子序列 LCS；`precision=LCS/生成长度`、`recall=LCS/GT长度`、`F1=2PR/(P+R)`。
- **视角**：字符**顺序**重合。要求字符"按相同顺序出现"才贡献。
- **意义**：文本生成经典标准指标（翻译/摘要通用）。对教学对话是措辞吻合度的底线度量。
- **局限**：对中文开放式对话绝对值偏低——同义但不同字的表达（"你觉得呢？" vs "你能想到吗？"）得分很低，即使语义都对。所以**绝对值不高不可怕，看相对 base 的提升**。
- **实现**：纯 Python LCS（`lcs_len`），无需外部依赖。见 validate.py.tmpl。

## 字符重叠（Jaccard multiset）

- **算法**：两段文本看成字符多重集，交集字符数 / 并集字符数，**不考虑顺序**。
- **视角**：词袋（用了多少相同字符/词汇）。
- **意义**：与 ROUGE-L 互补——ROUGE-L 受顺序约束，字符重叠只看"是否出现"。两者同向提升说明不是"顺序碰巧对"的虚高，而是真的用到相同字符。
- **实现**：`collections.Counter` 交并，纯 Python。

## 生成长度（字符数均值）

- **算法**：每条生成字符数取平均。
- **视角**：**行为/风格**（不是质量指标）。
- **意义**：揭示模型是否学会"该说多长的话"。GT 教学回复常 20-45 字（短问句引导）；base 常冗长跑题（200+ 字，自我介绍/复述题面/讲大道理）；微调后应回到 GT 量级。
- **关键性**：对教学/对话类任务，长度本身就是风格核心——苏格拉底教学是"短问句引导"非"长篇说教"。最能直观看出是否学到正确"语气"。

## 三者为何要一起看（三角验证）

| 指标 | 视角 | 抗什么假象 |
|---|---|---|
| ROUGE-L | 顺序+措辞 | 单看会被"乱序但同字"误导 |
| 字符重叠 | 词袋(无序) | 单看会被"顺序不同但同义"低估 |
| 生成长度 | 风格/行为 | 质量指标无法捕捉"该不该长篇" |

**结论判据**：三者同向 + 生成样例肉眼可见更贴合 → 微调有效。例：base ROUGE-L 0.130→LoRA 0.370（+184%），字符重叠 0.098→0.305，生成长度 242→38（与 GT 同量级），样例从"自我介绍跑题"变"启发式提问"。

## 进阶：语义相似度（可选）

开放式对话字面重叠天然低，若想更公平评估同义表达，可加：
- embedding 余弦相似度（用 BGE/m3e 等 sentence transformer；需额外依赖与 NPU/CPU 推理）
- 或 LLM-as-judge（让大模型打分）

默认三指标已足够判"微调是否有效"；要量化"语义质量"再上 embedding/judge。

## teacher-forced 多轮评估法

- 逐 assistant 轮：用 **GT 的 student/assistant 历史** + 当前 student 作上下文，`add_generation_prompt=True` 生成 teacher，与 GT teacher 对比。
- 即"喂正确历史，只看当前轮生成质量"，避免误差累积，公平对比 base/LoRA。
- 每条记录贡献多个 assistant 轮，取所有轮的平均。

## 多模态范式的指标适配（vlm / diffusion / video，2026-09-11 实测）

> 详见 references/multimodal-paradigms.md；实现见 scripts/validate_multimodal.py.tmpl。

- **vlm（图像描述/视觉指令）**：沿用 ROUGE-L/字符重叠，两点适配——① **一图多参考取 max**：
  GT 有多条 caption（flickr30k 每图 5 条）时，预测与**全部**参考逐一算分取最大值（多参考生成的标准做法，
  任一参考吻合即得分）；② 逐条肉眼核对描述具体性（"穿霓虹绿帽条纹衫" vs 泛泛"表演特技"）——
  均值相近时样例质量可能已显著变化。
- **diffusion / video（生成式）**：主指标 = **固定噪声 MSE**（越低越好，报告用 neg_mse 保持"越大越好"
  统一方向）。三要素缺一不可：① **per-sample seed** 的 `torch.Generator().manual_seed(BASE_SEED+i)`
  生成噪声与时间步——base 与 LoRA 喂**同一份**，否则 MSE 差是噪声差不是模型差（与 mlm 固定掩码、
  mim 固定内部掩码同一原则，pitfalls #40③）；② latents 来自与训练同源的**预编码缓存**；
  ③ 逐样本列出 base/LoRA MSE 对（本轮 SDXL 10/10、Wan 10/10 全改善——全样本同向比均值更有说服力）。
- **口径诚实**：指标只覆盖训练时的条件设定。Wan 用零文本嵌入训练/验证，测的是「无条件帧去噪」——
  改善真实（MSE 2.22→0.99，base 劣于零预测器 1.0 被修正到噪声方差下界附近），但**不得**表述为
  "文生视频能力提升"；报告必须写明口径与完整任务的差异。

## 视觉/时序/语音范式的指标适配（cat5/cat6 八范式，2026-09-18 实测）

> 详见 references/vision-ts-paradigms.md 与 speech-audio-paradigms.md；实现见
> scripts/validate_vision_ts.py.tmpl / validate_speech.py.tmpl。

- **img_cls / audio_cls（分类）**：top1 命中 + CE 双指标；10 条样本的 top1 粒度粗（每命中 1 条 +0.1），
  CE 是更灵敏的主证据（dinov2 hit 0→4/10 但 CE -40%）。base 侧=**head-only 线性探针**（新头范式），
  报告写明对比问题是"LoRA 是否优于线性探针"。
- **det（目标检测）**：主指标 F1@IoU0.5（score_thr 0.3、贪心按分数匹配、类别必须一致），辅 precision/
  recall/n_pred 分解——**n_pred 剧降+precision 剧升=学会抑制误报**（rtdetr 15.3→1.9 张/图），
  recall 未涨说明预算内还没学会找全目标，两者要分开表述。
- **layout_ner / ner（实体）**：实体级 F1（BIO span 精确匹配）+ token CE；逐文档列出 tp/n_ent_gt
  （span 级 tp 求和比 F1 均值更能反映大文档权重）。
- **depth（深度回归）**：**闭式对齐后**的 AbsRel（主）+ δ1(<1.25) + RMSE，深度空间计算、有效掩码
  gt>1e-3；相对深度模型必须对齐（scale+shift）后才可与米制 GT 比——对齐本身不是作弊，是该任务族
  的标准评测口径。
- **forecast（时序）**：**MASE**（MAE / mean|Δcontext|）为尺度无关主指标——跨序列可平均；辅以原始尺度
  MAE 与归一化 MSE。base MASE≈1.0 的含义是"与朴素预测相当"，改善幅度要对照该基线解读
  （timesfm -1.6% 属近同源小幅改善，如实记录）。
- **asr（语音识别）**：**CER**（字符编辑距离/GT 长度，主）+ char ROUGE-L/字符重叠（辅，与文本范式
  口径连续）；GT 与预测**双侧规范化**（小写、去标点、压空格）后再算。强基座场景（whisper 在
  LibriSpeech base CER 已 0.9%）报告必须写明 base 已近满分，"相对减半"与"绝对提升 0.46pp"并列表述。
- **tts（语音合成）**：teacher-forced **mel MSE**（+L1）；无 vocoder 归档时口径限定 mel 域，
  不表述为音质提升。逐条列帧数（frames）——MSE 与谱长无关才可跨条平均。


