# 视觉与时序五范式 LoRA（img_cls / det / layout_ner / depth / forecast）

> 场景：目标是图像分类骨干（ViT/DINO 系）、目标检测（DETR 系）、文档版面 token 分类（LayoutLM 系）、
> 单目深度估计（Depth-Anything/DPT 系）或时序预测基座（TimesFm 系），而不是 CausalLM/encoder-NLP/多模态生成。
> 实测：2026-09-18 台账第 5 大类 5 模型，**5/5 改善**——dinov2-base+tiny-imagenet（CE -40%）、
> rtdetr_r50vd+cppe-5（F1 0.023→0.147，预算内）、layoutlmv3-base+funsd（实体 F1 0.519→0.847）、
> Depth-Anything-V2-Small+nyu（AbsRel -44%/δ1 0.961）、timesfm-2.0-500m+chronos（MASE -1.6%，近同源小幅）。
> 训练器模板：`scripts/lora_train_vision_ts.py.tmpl`、验证模板：`scripts/validate_vision_ts.py.tmpl`
> （范式经 `PARADIGM` 环境变量切换）；完整配套（切分/sweep/val_loss）实战副本在 `lora-ws/.cat5/`。

## 五范式速查

| PARADIGM | 模型加载 | 数据格式（jsonl 每行） | 任务头 | loss | 真实 batch | 验证主指标 |
|---|---|---|---|---|---|---|
| img_cls | `Dinov2Model`（骨干，无头） | `{"image":绝对路径,"label":int}` | **外置** `nn.Linear(hidden,N)`（HEAD_SEED 初始化，存 head.pt） | 分类 CE（CLS 池化） | 16 | top1 命中 + CE |
| det | `RTDetrForObjectDetection`（config num_labels=K + `ignore_mismatched_sizes=True` 重初始化头） | `{"image","width","height","objects":{"bbox":[[x,y,w,h]绝对],"category":[int]}}` | **模型内** class_embed.{0-5}/enc_score_head/denoising_class_embed（全程直训，不进 LoRA） | 内置 DETR loss（匈牙利匹配；**FP32=1** 训练） | 2 | F1@IoU0.5（score_thr 0.3 贪心匹配）+ DETR val loss |
| layout_ner | `LayoutLMv3ForTokenClassification`（num_labels=7 重初始化） | `{"tokens":[str],"bboxes":[[x0,y0,x1,y1]],"ner_tags":[int]}`（词级） | **模型内** classifier（`.float()` 保 fp32） | token CE（tokenizer 传 `word_labels` 自动首子词对齐） | 8 | 实体级 F1（BIO span 精确匹配，同 ner 范式）+ CE |
| depth | `DepthAnythingForDepthEstimation`（预训练 DPT 头，无需换头） | `{"image","depth":.npy 绝对路径}`（float32 米制） | – | **闭式 scale+shift 对齐视差 MSE**（见下） | 4 | AbsRel + δ1(<1.25) + RMSE（对齐后深度空间） |
| forecast | `TimesFmModelForPrediction`（transformers ≥5.x 原生支持） | `{"id","context":[512 float],"horizon":[128 float]}` | –（预训练输出层） | 内置 future_values MSE（**per-series \|mean\| 归一化**后传入；FP32） | 8 | MASE（MAE/mean\|Δcontext\|，尺度无关主指标）+ MAE + 归一化 MSE |

## head-only 基线协议（新头范式的 base 侧，本批新沉淀）

任务头新初始化的模型（原头类别数不匹配：200类/5类/7类头），"base vs LoRA" 的公平对比是：
- **base 侧 = head-only run**（`LORA_DISABLE=1`：冻结骨干只训头 = 线性探针），与 LoRA run
  **同 HEAD_SEED（同头初始化）、同步数、同 lr（取 sweep 最优）、同数据**；
- **LoRA 侧 = LoRA+头联合训练**；两侧头权重各自存 `head.pt`，validate 用 `BASE_RUN_DIR`/`LORA_RUN_DIR` 分别加载。
- 对比回答的问题是「**LoRA 适配骨干是否优于纯线性探针**」，而不是「是否优于随机头」。
- 预训练头存在的范式（depth/forecast/以及 cat2 的 cls/ner 等）不需要此协议，base=原模型直接对比。

实测参照：layoutlmv3 head-only F1 0.519（线性探针已不弱）→ LoRA 0.847；dinov2 head-only val CE 5.44 → LoRA 3.25。

## 共性模式（跨范式复用，这是本文件的核心）

1. **模型内新头 × peft × 融合优化器三件套**（pitfalls #45/#46，血的教训）：
   ① `get_peft_model` 会冻结**所有**非 LoRA 参数——新头必须在包装后**重新 `requires_grad=True`**；
   ② trainable 列表 = `[模型内 requires_grad 参数] + head_params` **按 id 去重**（head-only 模式曾重复入组）；
   ③ **模型内头统一 `.float()`**——NpuFusedAdamW 拒收纯 bf16 参数组（head-only run trainable 只剩 bf16 头 → 首步 `TypeError: must be float32 or float16`）；autocast 下 fp32 头正常参与 bf16 前向。
   外置头（img_cls 的 `nn.Linear`）天然免疫 ①③，但仍要过 ②。
2. **新头重初始化只动 Linear，绝不动 LayerNorm**（pitfalls #47）：ASTMLPHead 型结构
   （LayerNorm→Dropout→dense）里把 layernorm.weight 归零 → 特征湮灭 → logits 恒 0 → CE 恒 ln(类别数)
   且 dense 梯度恒 0。按**名字后缀**（`dense.weight`/`dense.bias` 或 `classifier.weight`）精确重初始化。
3. **LoRA 候选名会撞任务头**：候选并集里的 `dense` 撞上 `classifier.dense`（AST）→ 头被 LoRA 包裹 →
   head.pt 键名变 `base_layer.*` → validate `strict=False` **静默不加载** → 随机头评估假退化。
   防御三件：任务头范式从候选**排除撞名项**；头保存时键名规范化（`.base_layer.`→`.`、过滤 `lora_` 键）；
   `strict=False` 加载后**校验命中数>0**。
4. **单批过拟合测试是新头范式的入场券**：1 批 × 20-30 步 @ lr=1e-3，loss 必须显著下降才进 sweep。
   loss 钉死在 ln(类别数) → 头没在训练（冻结/梯度为 0/特征湮灭三选一排查）。AST 未做此测试浪费两轮 sweep。
5. **数值稳定性按范式选精度**：det（匈牙利匹配）与 forecast（TimesFm 权重原生 fp32）用 `FP32=1` 关 autocast；
   其余 bf16 autocast 默认。小模型（<100M）fp32 完全放得下，别为省显存冒险。
6. **对齐式回归 loss（depth）**：模型输出相对视差（任意尺度），GT 是米制深度 → 在**视差空间**做
   per-image 闭式最小二乘 `s=cov(p,g)/var(p), t=ḡ-s·p̄`，loss=对齐后 MSE——可微、免调尺度超参；
   验证指标（AbsRel/δ1/RMSE）在**深度空间**算（`d̂=1/(s·p+t)`），有效掩码 `gt>1e-3`。
7. **时序归一化与尺度无关指标（forecast）**：per-series 除以 `mean|context|` 后训练/评估（原始尺度序列
   跨域差 1e3 倍）；主指标用 **MASE**（MAE / mean|Δcontext|）——跨序列可平均、与量纲无关；
   `past_values` 传 list[1-D tensor]（变长友好），`freq=0`（高频档）统一并记录口径。

## 各范式实测要点（含结果量级）

- **img_cls**（dinov2-base+tiny-imagenet 30%池33000/训3200）：200 类、每类仅 16 图的少样本设定，
  top1 0/10→4/10、CE 5.44→3.25（-40%）；head-only 探针 train loss 末10均 4.64 vs LoRA 1.55——
  **LoRA 骨干共适应让头学得更快**是主要增益来源。CLS 池化用 `last_hidden_state[:,0]`。
- **det**（rtdetr_r50vd+cppe-5 池300）：F1 0.023→0.147、precision 0.014→0.267（n_pred 15.3→1.9 张/图：
  学会抑制误报）、val DETR loss 165.6→97.9（sweep 判据）。**诚实结论范式**：改善真实但绝对水平低——
  新检测头+300 图+200 步预算对 DETR 类模型（通常 36+ epoch 收敛）远远不够，报告写「预算内正确方向、
  未收敛」，不写「无效」也不夸大。标注格式=COCO dict（`{"image_id","annotations":[{"bbox"(绝对xywh),
  "category_id","area","iscrowd"}]}`），HF image processor 自动归一化。
- **layout_ner**（layoutlmv3+funsd 池59文档）：F1 0.519→**0.847**（本批最强），仅 59 文档+200 步逼近
  FUNSD 全量微调发表水平（~0.90）。transformers 5.x 的 LayoutLMv3Tokenizer **不需要**
  `is_split_into_words`：`tokenizer(text=[words], boxes=, word_labels=)` 词列表隐式处理且返回对齐好的
  `labels`（首子词），比 cat2 ner 的手动 word_ids 对齐更省。
- **depth**（DAv2-Small+nyu 池364）：δ1 0.913→0.961、AbsRel -44%、RMSE -32%；困难样本（低 δ1）改善最大。
  GT 深度存 **.npy**（PIL mode-F 保存不可靠）；518×518 输入（DAv2 原生分辨率）。
- **forecast**（timesfm+chronos 池336序列）：MASE 1.099→1.081、MAE -3.2%、归一化 MSE -8.3%，6/10 改善。
  **近同源判读**（同 esm2 -1.5% 先例）：TimesFm 是预训练时序基座、chronos 子集与其预训练分布重叠，
  base MASE≈1.1 即"与朴素预测相当"，小幅改善是合理信号，不夸大。窗口固定取序列尾部
  （context=v[-640:-128]，horizon=v[-128:]）保证可复现。

## 超参模式（sweep 60 步 × 5 组合，64 条 held-out val loss 判据）

- **新头+域差大** → 激进组合 lr=2e-4/r=32（dinov2/rtdetr/layoutlmv3 三模型一致）；
- **预训练近同源**（depth 的 DAv2 本就是深度模型、forecast 的 TimesFm 本就是预测基座）→ 保守
  lr=1e-4/r=16 且**全组合近平**（depth 0.0043-0.0049、forecast 0.1137-0.1147）——饱和型任务的正确表征，
  与 cat2 rerank/ner、cat3 esm2/Prithvi 同模式；
- **sweep 判据被实现 bug 污染时必须确认复验**（pitfalls #50）：rtdetr/layoutlmv3 首轮 sweep 在冻头 bug 下
  完成（排序侥幸一致），修复后复验组合不变、val 大幅下移（165.6→97.9），正式 run 才有效。

## 数据切分注意（接 30%/held-out 协议）

- 图像/npy/音频一律**绝对路径**落盘（#41）；只提取被选中的样本（tiny-imagenet 110k 只落 ~3.3k 张）；
- parquet 多分片先合并再切（tiny-imagenet train 100k+valid 10k；funsd 149+50 按**文档**为单位切）；
- det 数据集自带的 test 分片（cppe-5 29 行）**不用**——held-out 一律从自己的 70% 里取，保证与训练池
  同分布且审计链完整；
- 时序按**序列**为单位切分（不是按窗口滑动采样——同序列多窗口会泄漏）；
- 其余同 cat1-4 协议：seed 42/43/7/123、写盘前洗牌（#34）、split_meta.json 索引审计、不相交 assert。
