---
name: cpt-v2-24model-batch-run
description: 24 新 SOTA 模型批量 CPT 实战（2026-09-09）：v2 数据管线、各范式坑（ASR 特征100倍数/SD3.5仅T5/DepthPro禁融合优化器/Wan VAE bf16/flickr列表陷阱/gliner键映射/FSDP2 mesh+HCCL超时/Adafactor-DTensor不兼容/保存死锁）
metadata:
  type: project
---

# 24 新模型批量 CPT 实战记录（2026-09-09，v2 轮）

6 大类 24 个新 SOTA 模型（Qwen3.5/3.6/Coder-Next、SD3.5、Wan2.2、dinov3、chronos-2、Qwen3-ASR、CosyVoice3 等）在 8×Ascend910 上的批量 CPT：30% 语料训练 + 10 条 held-out 验证。**结果：19 改善 / 4 持平（9B已对齐、Prithvi/DepthPro饱和、SD3.5微变化）/ 1 特殊（ChemBERTa回归头）**。

**Why:** v2 轮换模型后大量"同名不同构"陷阱；本文沉淀可复用的判定与修法，避免 v3 再踩。

**How to apply:**

## 数据切分管线（可复用）
- `training-ws/.v2tools/make_split.py`：全量语料 → train30.jsonl(30%) + val_sample.jsonl(10条来自剩余70%，seed 42 保证不重叠)。字段抽取按数据集 lambda 配置。
- **坑**：flickr30k 的 caption 是**列表**（每图5条）——抽到 {"text": list} 会让 tokenize 全空 → prepare_data 的补块 while 循环**死循环**（表现为 prep 99% CPU 十分钟无输出）。py-spy dump 定位在 line 84 死循环。
- mmarco-zh 无 train 分片只有 corpus/queries/dev——v1 用 corpus 段落做 CPT 文本。
- chemos chronos m4_daily 目录可能为空——用 electricity_15min。

## 各模型范式与坑（关键修法）
- **Qwen3-ASR**：① 原始 repo（ModelScope）与 transformers 不兼容（projector 键错位）→ 必须用官方 **`-hf` 后缀 repo**（Qwen/Qwen3-ASR-1.7B-hf）；② 特征长须为 100 倍数——**波形 np.pad 到整秒**（16000 采样 = 100 帧）最稳；③ forward 需 `input_features_mask`（处理器原生输出）；④ label 掩码用 prompt-only 模板长度（am.sum() 掩全序列是经典错误→loss=nan）。
- **SD3.5-medium**：ModelScope 非递归下载只有单文件格式！组件需递归补齐 transformer/vae/text_encoder×2/tokenizer×2；**context 只吃 T5 维 4096**（joint_attention_dim=4096，不是 6144 拼接）；pooled=cat(clip1,clip2)=2048。
- **DepthPro**：① 必须 processor 1536 路径（PIL 直传）；② **NpuFusedAdamW step 在 DepthPro 上崩**（aclnnInplaceAdd 广播错 647141441≠646551617）→ 换 plain AdamW；模型在 nyu 上已饱和（abs_rel 0.02）。
- **Wan2.2**：VAE bf16 → 输入须 .to(bfloat16)；14B 单卡 bf16+Adafactor 可训；eval 前**必须 del optim + p.grad=None** 再加载 base（否则 OOM）。
- **gliner2**：ckpt 键 `encoder.*` → DeBERTa `deberta.*` 前缀映射；词表 128011（以 embedding shape 为准）；MLM 头随机初始化 → 部分 val 样本 NaN（评估均值需排除 NaN）。
- **dinov3**：AutoModelForImageClassification 不认 DINOv3ViTConfig → DINOv3ViTModel + 手动线性头；数据是 tiny-imagenet **200 类**（不是 imagenette 10 类——NPU 的 CE 不做越界检查会输出假 loss 0.0！）。教训：**loss 恰为 0.0/持平时先查标签范围**。
- **ChemBERTa-77M-MTR**：回归头模型无 MLM 头——base 对照无意义（随机头 loss 14.5），如实记录"口径特殊"。
- **CosyVoice3**：原生格式（llm.pt/flow.pt/hift.pt）→ llm.pt 是 **Qwen2-0.5B** 骨架（键 `llm.model.model.*`→`model.*`）remap + ljspeech 文本 CE。
- **GOT-OCR**：config model_type "GOT"（旧名）→ 改 "got_ocr2" + architectures GotOcr2ForConditionalGeneration；verovio 依赖。
- **chronos-2**：需 `chronos-forecasting` 包；forward(context, context_mask, group_ids, num_output_patches=H/16, future_target) 返回 loss+quantile_preds[·,21,·]。
- **twitter-roberta/gliner 分类头模型**：AutoModelForMaskedLM 自动兼容（roberta.* 前缀剥离）。
- **Qwen3.5-9B**：已对齐数据（ultrafeedback）上 200 步 CPT 持平是正确结论（Adafactor lr 3e-5 也一样）；对照 4B-wikipedia 大幅改善。
- **FSDP2 大模型（36B/Coder-Next-80B）**：
  - torch 2.10 `fully_shard` 无 device_id 参数 → 用 `init_device_mesh("npu",(w,))` + mesh= 传参；
  - **160GB 级加载偏斜 > HCCL 120s 连接超时** → `HCCL_CONNECT_TIMEOUT=1800` 必设 + 加载后 `dist.barrier()`；
  - **Adafactor 与 FSDP2 DTensor 不兼容**（内部 in-place pow_）→ AdamW；80B 全训 AdamW 装不下 → **冻结下 36 层**（20.4B 可训练，~26GB/卡 25.7s/步）；
  - FP8 ckpt：块状 128×128 scale，键名是 `X.weight`→`X.weight_scale_inv`（**替换非拼接**）；反量化后 config 需删 `quantization_config`；
  - **保存阶段死锁再现**（36B：逐 tensor full_tensor() 30min+ 挂死，#127 变体）→ 训练指标从 step_loss.jsonl 抢救；后续大模型保存应在训练后立即写 summary、再尝试聚合，或改用 DCP。
- kill -9 大模型进程后 NPU 显存不释放 → 等 1-2 分钟再启新任务，否则 TsdOpen failed。

## 环境事实
- 主 python3.12.13：torch2.10+torch_npu2.10+transformers 5.5.4（qwen3_5/moe/next/vl/got/dinov3/depth_pro ✓，qwen3_asr ✗）
- lora-ws venv：transformers 5.16.1（qwen3_asr ✓）——大模型与 ASR 用它
- 2TB 内存机器可同时 8 rank × 160GB 加载
