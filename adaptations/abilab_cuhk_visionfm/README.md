# VisionFM — Ascend NPU 适配

> **模型**: `ABILab-CUHK/VisionFM` — ViT-B/16 fundus (NEJM AI 2024)
> **模式**: 青光眼诊断 (PAPILA, 3 类) + 眼底特征提取 (768 维)
> **License**: CC BY-NC 4.0 (非商业)

## 模型说明

VisionFM 是 CUHK ABILab 在 340 万张眼科图像上预训练的视觉基础模型 (NEJM AI 2024).
本 adaptation 同时部署两种能力:

1. **青光眼诊断 (默认 --classify)**: 加载 PAPILA 微调权重 `checkpoint_papila.pth` (1.10GB,
   含 fine-tuned encoder + ClsHead), 输入眼底图 → 3 类概率 (正常 / 疑似青光眼 / 确诊青光眼).
2. **特征提取 (默认)**: 加载预训练 encoder `VFM_Fundus_weights.pth` (1.40GB, iBOT 自监督),
   输出 768 维 CLS token 特征向量.

- 架构: ViT-B/16, embed_dim=768, depth=12, num_heads=12, patch_size=16, input 224

## 推理流程 (研读 GitHub 官方代码后实现)

### 分类模式 (--classify)
1. **encoder**: `vit_base(patch_size=16, num_classes=0)`; 从微调 ckpt 的
   `visionfm_state_dict` (150 keys, 无前缀) 加载, 0 missing.
2. **分类头**: `ClsHead(embed_dim*4=3072, num_classes=3, layers=3)` —
   `channel_bn`(BatchNorm2d 3072) + `classifier`(Linear 3072→1536→768→3, GELU+Dropout);
   从 `classifier_state_dict` (11 keys, strip `module.` 前缀) 加载, 0 missing.
3. **前向**: `model.get_intermediate_layers(x, n=4)` 取最后 4 个 block 的 CLS token
   拼成 [1,3072] → ClsHead → [1,3] logits → softmax → 3 类概率.
4. **PAPILA 类别**: ['正常', '疑似青光眼', '确诊青光眼'].

### 特征模式 (默认)
1. **encoder**: 从预训练 ckpt 的 `teacher` (160 keys, 带 `backbone.` 前缀) 加载, 剥前缀
   strict=False → 0 missing, 10 unexpected (iBOT head 丢弃). `weights_only=False` (旧 ckpt
   含 numpy scalar).
2. **前向**: `model(x)` 返回 CLS token [1,768].
3. **预处理**: PIL RGB → resize 224 (BICUBIC) → /255 → Normalize(Fundus).
   Fundus 归一化: mean=(0.424,0.261,0.128), std=(0.295,0.202,0.137).

## 运行

```bash
uv sync --extra ascend

# 青光眼诊断 (真实分类)
uv run python demo.py --classify

# 特征提取
uv run python demo.py

# Dry run (随机权重, 验证 NPU 架构兼容)
uv run python demo.py --dry-run
```

## 输入

`sample_images/fundus_*.png` — 真实眼底彩照 (512x512 RGB), 来自公开
`YoussefAboelwafa/Retina_Blood_Vessel_Segmentation` 数据集.

## 输出

- 分类模式: 3 类概率 + 预测类别, 写入 `classification_probs.npy`
- 特征模式: 768 维特征向量, 写入 `embedding_fundus_01.npy`

## 自定义模型库说明

vendor 了官方 `models/vision_transformer.py` (原样) + 极简 `utils.py` shim
(仅 `trunc_normal_` + Fundus 归一化常量). `ClsHead` 内联于 `demo.py` (严格匹配官方
`models/head.py` 的 layers=3 结构). 源码位于 `visionfm_src/`.

## 精度评测 (PAPILA test set, 98 张)

`eval_papila.py` 对 PAPILA/test 全集前向, 复现官方 `inference_visionfm_for_multiclass_classification.py`
口径 `roc_auc_score(average='macro', multi_class='ovr')` + `average_precision_score(average='macro')`.

| 口径 | AUROC | AUPR |
|---|---|---|
| raw-logit (官方脚本字面, 旧 sklearn) | **0.8561** | 0.7612 |
| softmax 概率 (sklearn 1.2+ 标准) | 0.8804 | 0.7765 |

每类 OvR AUROC: anormal 0.878, bsuspectglaucoma 0.895, cglaucoma 0.795 (raw-logit).

### 数值对齐校验 (parity)

`parity_check.py` 用官方 `vision_transformer.py` + 官方 `head.py` 原始 ClsHead 跑同一图/权重,
与本 adaptation 内联 ClsHead 比对:

- ClsHead 权重最大差异 = **0.00e+00**
- logits 最大差异 = **0.00e+00** (`[2.036156, -3.356451, 0.710639]` 双方一致)
- softmax probs 最大差异 = **0.00e+00**

**结论**: 本 NPU 适配与官方代码路径 bit-level 等价, 零数值偏差. PAPILA AUROC 0.8561
等同于官方脚本在此权重+数据上的产出. (论文摘要未单独列出 PAPILA 数; 摘要 headline
0.945/0.974 为 DR/AMD 不同任务, 不可直接对比.)

## 设备

NPU: Huawei Ascend910, torch_npu 2.7.1. 边界: 所有产物仅限本 adaptation 目录.
