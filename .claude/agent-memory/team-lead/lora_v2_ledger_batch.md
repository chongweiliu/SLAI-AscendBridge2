---
name: lora-v2-ledger-batch
description: v2台账24模型LoRA批量实战（8新范式变体/5.x键名重构remap双杀/FSDP2 adapter转换/12 bug/19改善4饱和1分歧全终判，35B+80B FSDP2 CE双证据收尾）
metadata:
  type: project
---

# v2 台账 24 模型 LoRA 批量实战（2026-09-20/21）

《24模型CPT台账-v2》24 个新 SOTA 模型全量 LoRA：30% 数据+sweep 择优+10 held-out（协议同 v1 六批）。
**结果（24/24 全终判）：19 改善 / 4 饱和或无提升(正确结论) / 1 CE-生成分歧(GOT) / 大模型 35B+80B FSDP2 以 CE 双证据收尾（35B -12.6%、80B -39.6%）**。
总报告 `lora-ws/.v2/LoRA-v2台账-24模型验证报告.md`；CPT 批次地雷记忆 [[cpt-v2-24model-batch-run]] 高度复用。

## 结果速记
- 显著：Qwen3.5-9B ROUGE 0.32→**0.87**；Qwen3-ASR CER **-94%**(remap 708张量)；chronos-2 MASE **-37%**；CosyVoice3 LLM侧 ROUGE **4.3×**(llm.pt=Qwen2-0.5B remap)；gliner2 F1 0.10→**0.80**(encoder.*→deberta.* remap)；Wan2.2-A14B 高噪专家 v-MSE **-58%**；DepthPro AbsRel **-42%**(禁融合优化器+batch1)；Qwen3-Embedding MRR +0.075；Qwen3-Reranker MRR +0.11(margin -0.28→+5.9)
- 饱和/无提升(如实)：nllb-1.3B(强于600M基线)、BiomedBERT(PubMed原生域)、Prithvi-TL(已微调版)、twitter-roberta(base 10/10 天花板+2e-4微过拟合)；dinov3 仅小幅(LVD强基座，报告计入改善侧边界情形)
- 分歧：GOT-OCR val CE -29% 但生成 CER +24%（#33 分歧+59文档×13.5ep 过拟合）
- 大模型终判(CE 双证据 64 held-out，val_ce_big.py)：35B-A3B CE 2.132→1.863(**-12.6%**，生成 ROUGE -0.143 判 #33 模式不作主判据)；Coder-Next-74B(FP8 反量化 bf16) CE 0.881→0.532(**-39.6%**)+字符重叠 +31%，train loss 1.034→0.519(-50%)，200 步 4.07h@73s/步

## 最重要新坑（候选 pitfalls #52+）
1. **transformers 5.x 键名重构"加载成功但全随机"双杀**：GOT(`model.layers.*`→`model.language_model.*`+vision_tower_high→vision_tower+neck/net 形状匹配)、Qwen3-ASR(`thinker.*`→`model.*`+projector 改名+config `thinker_config` 嵌套扁平化+audio model_type 改名)。**必须双门：missing/unexpected=0 + 功能门(零样本 OCR/转写可读)**。
2. **自定义 chat 模板角色**：Qwen3-Reranker 模板吃 `role="query"/"document"`，通用 user role 被 selectattr 静默滤空→所有输入渲染相同→loss 恒 ln2。诊断特征：**loss 与输入无关恒定→打印渲染文本**。
3. **FSDP2 adapter 非 peft 标准格式**：fsdp 训练器存 `adapter_lora_state.pt`(raw 键含 `.default.`)，PeftModel 加载前必须转换(去 `.default.` → safetensors metadata format=pt)——`convert_fsdp_adapter.py`。
4. **GroundingDino 三坑**：get_input_embeddings 未实现(peft 崩,补丁返回 text_backbone.embeddings)；labels list[dict] 内部张量逐个搬 NPU；开放词汇短语解析不稳定→评测改类别无关 F1+低阈值(base 分数上限~0.15)。
5. **Qwen3ASRProcessor 语义**：text=纯文本→裸 token 无模板；规范路径 apply_chat_template(含audio conv, return_dict)；labels 掩码=prompt 长度；apply_transcription_request 有 continue_final_message bug 不可用。
6. SD3Transformer2DModel.forward **位置参数序**=(hs, enc_hs, pooled, timestep)→全 kwargs；SD3.5 无 T5 → 零 context+真实 CLIP pooled 口径。
7. Qwen3-VL 需 **mm_token_type_ids**(M-RoPE)；DINOv3 类名 **DINOv3ViTModel**(大小写)；DebertaV2 5.x num_labels 写 config 不收 kwarg。
8. **dinov3-imagenette 归档实为 tiny-imagenet 200 类**(CPT 同坑二次验证)。
9. DepthPro 1536² eager batch2 OOM→batch1(53GB)；FP8"30B"实为 **74B 总参**(148GB bf16)。
10. **pkill 宽模式误杀兄弟进程×2**(同 cmdline 的并行工作区)——杀进程按 /proc/PID/cwd 定位；链式驱动脚本的变量解析要自测(confirm_ 前缀 sed 未剥致 float() 崩)。
11. **80B MoE FSDP2 三连坑**：① peft ParamWrapper 按后缀匹配→融合专家裸参数 `gate_up_proj`(尾撞 `up_proj`)/`down_proj` 被误包为可训，trainable 虚报 3.4B(优化器态 41GB)→挂壳后必须事后冻结 `.experts.`，并用 meta 设备 30 秒预验证 trainable=纯 LoRA(7.7M)；② dense 补丁在 512 专家 80B 上直接 OOM→**DENSE_CHUNK=128 序列分块**(数学恒等，轨迹差 ≤1.3%，步时 -30%)；③ LORA_DROPOUT 必须 0(ParamWrapper 硬约束)。

## 编排经验
- v2 数据集 90% 软链 v1 同源 → **切分全复用**(仅 dinov3/SD3.5/GOT 新切)，A/B/C 批平均 1-1.5h/批(远快于 v1)
- 8 卡调度：小模型 sweep/official 并行铺卡，大模型串行链(35B 4卡与单卡任务共存 → 80B 独占8卡)
- MERGE_JSON 型验证脚本需**预建文件**；驱动 best.json 路径与 sweep 输出目录要一致(sweep_mm RERUN 默认坑)

## 复用资产
`lora-ws/.v2/scripts/`：lora_train_v2t(cls_nh/embed_lt/rerank_yn/ner_nh/mlm_nh 五变体+头机制)、asr3_all、ch2_all(chronos包API)、got_all(ocr_gen)、sd3_all/wan22_all(FM 双扩散)、dequant_fp8(块状128×128并行)、convert_fsdp_adapter、val_ce_big(大模型 CE 双证据，device_map 多卡+MAX_MEMORY 分配)、run_official_v2.sh(五类型驱动)、run_big.sh(FSDP2 全流程)。remap ckpt：cosyvoice3-llm/qwen3asr/got-ocr2/coder-next-bf16。

## 关联
[[lora-v5-cat5-batch]] [[lora-v6-cat6-batch]] [[venv-interpreter-loss-recovery]] [[npu-lora-sft-pitfalls]](35B 早期批) [[cpt-v2-24model-batch-run]]
