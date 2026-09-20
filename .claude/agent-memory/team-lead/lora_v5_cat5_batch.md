---
name: lora-v5-cat5-batch
description: 台账第5大类5模型(视觉与时序)LoRA实战（img_cls/det/layout_ner/depth/forecast五新范式/peft冻结模型内新头等4 bug/head-only基线协议）
metadata:
  type: project
---

# 台账第 5 大类 5 模型（计算机视觉与时序）LoRA 批量实战（2026-09-18）

dinov2/tiny-imagenet(img_cls)、rtdetr_r50vd/cppe-5(det)、layoutlmv3/funsd(layout_ner)、DepthAnythingV2-Small/nyu(depth)、timesfm-500m/chronos(forecast)。总报告 `lora-ws/.cat5/LoRA-v5-验证报告.md`。

## 结果速记（5/5 改善）
- **layoutlmv3 最强**：实体 F1 0.519→**0.847**（head-only 探针基线 0.52，逼近 FUNSD 全量微调发表水平 ~0.90，仅 59 文档+200 步）
- **depth 强改善**：AbsRel -44%、RMSE -32%、δ1 0.913→0.961（困难样本改善最大）
- **dinov2**：200 类少样本（每类16图）top1 0/10→4/10、CE -40%（vs head-only train loss 4.64 vs 1.55）
- **det 预算内改善**：F1 0.023→0.147、precision ×19，但绝对水平低——新 DETR 头+300 图+200 步不收敛（DETR 需 36+ epoch），如实记录不判"无效"
- **timesfm 小幅**：MASE -1.6%/MAE -3.2%（base 是预训练时序基座，chronos 近同源——与 esm2 -1.5% 同类判读）
- 超参模式：新头/域差大→2e-4/r32（dinov2/rtdetr/layoutlmv3）；预训练近同源→1e-4/r16 且全组合近平（depth/timesfm）

## 关键协议创新：head-only 基线（新头范式的 base 侧）
任务头新初始化的模型（200类/5类/7类头与原头不匹配），base 侧 = **冻结骨干只训头**（LORA_DISABLE=1），与 LoRA run 同 HEAD_SEED/步数/lr——对比回答"LoRA 是否优于线性探针"。外置头（dinov2 nn.Linear）或模型内头（classifier/class_embed）均支持，头权重存 head.pt 供 validate 加载。

## 四个新 bug（模型内新头 × peft/融合优化器交互，均已修）
1. **peft 冻结模型内新头**：get_peft_model 把所有非 LoRA 参数 requires_grad=False——新头被冻结（LoRA run 随机头）、head-only run trainable 重复。修：peft 后对 head_params **重新 requires_grad=True + 按 id 去重**
2. **NpuFusedAdamW 拒纯 bf16 参数组**：head-only trainable 只剩 bf16 classifier → `must be float32 or float16` 首步崩。修：**模型内头统一 .float()**（autocast 下正常）
3. **validate LOWER_BETTER 字典漏 det 键** KeyError（写完新分支要全键自测）
4. **sweep 判据被污染需确认复验**：rtdetr/layoutlmv3 首轮 sweep 在冻头 bug 下完成（排序仍选出同组合），修复后按 cat4-vlm 先例**确认复验**——组合不变、val 大幅下移（165.6→97.9），正式 run 有效

## 范式实现要点
- det：RTDetrConfig(num_labels=5)+ignore_mismatched_sizes 重初始化头；标注=COCO dict `{"image_id","annotations":[{"bbox"(绝对xywh),"category_id","area","iscrowd"}]}`；FP32 训练（匈牙利匹配稳定性）；LoRA 候选排除 class_embed/'0'-'5'/sampling_offsets
- depth：GT 米制深度→视差 1/d，**闭式 scale+shift 最小二乘对齐**后 MSE（可微）；指标 AbsRel/δ1/RMSE（对齐后深度空间）；depth 存 .npy（PIL mode-F 不可靠）
- forecast：TimesFmModelForPrediction(transformers 5.16 原生支持) forward(past_values=list[1-D tensor], freq=zeros, future_values) 内置 MSE loss；per-series |mean| 归一化；MASE=MAE/mean|Δctx| 做尺度无关主指标
- layout_ner：transformers 5.x LayoutLMv3Tokenizer **不需要 is_split_into_words**（词列表隐式），`tokenizer(text=[words], boxes=, word_labels=)` 直接返回对齐好的 labels
- img_cls：Dinov2Model+外置 Linear(CLS 池化)；tiny-imagenet train+valid 110k 合并切分

## 复用资产
`lora-ws/.cat5/`：make_split5.py(5切分)、scripts/{lora_train5(五范式+LORA_DISABLE+FP32+HEAD_SEED),val_loss5,validate5,sweep5}、run_official5.sh。候选 skill 沉淀：五范式+head-only 协议+bug1/2（见 [[lora-v6-cat6-batch]] 同类）。

## 关联
[[lora-v4-cat4-batch]]（多模态三范式已合入 skill）；环境抢修见 [[venv-interpreter-loss-recovery]]。
