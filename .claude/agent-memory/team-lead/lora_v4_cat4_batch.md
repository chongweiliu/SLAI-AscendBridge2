---
name: lora-v4-cat4-batch
description: 台账第4大类3模型(多模态)LoRA实战（vlm/diffusion/video三新范式/3-3全改善/4个新坑含就地改文件截断事故）
metadata:
  type: project
---

# 台账第 4 大类 3 模型（多模态）LoRA 批量实战（2026-09-11）

Qwen2.5-VL-3B/flickr30k（vlm）、SDXL-base/nouns（diffusion）、Wan2.1-T2V-1.3B/ucf101（video）。台账第 4 大类实际只有 3 个模型（编号 14-16）。总报告 `lora-ws/.cat4/LoRA-v4-验证报告.md`。

## 结果速记（3/3 全改善）
- **vlm**：ROUGE-L 0.531→0.593（+0.062），val CE -5.4%；逐条可见描述更具体（"穿霓虹绿帽条纹衫" vs base 泛泛）
- **diffusion (SDXL UNet LoRA)**：固定噪声 MSE 0.098→0.051（-47.6%）
- **video (Wan DiT LoRA)**：MSE 2.224→0.993（-55.4%）——**base 在零文本条件下预测劣于零预测器（2.22>1.0）**，LoRA 使模型适应该条件设定；改善真实但口径是「无条件帧去噪」（T5 11.4GB 跳过用零嵌入，沿 CPT 先例），与完整 T2V 有差异

## 三范式要点（lora-ws/.cat4/ 全套可复用）
- **vlm**：ChatML 渲染+processor 展开 image_pad；labels 用 **token-id 序列定位 assistant 头**（`<|im_start|>assistant\n` 的 id 序列在 input_ids 里 find）掩 prompt；批处理 **concat pixel_values（非 stack）+ cat image_grid_thw**；视觉塔（qkv 融合命名）天然不被语言侧候选命中→目标自动隔离
- **diffusion**：SDXL 组件直载（variant=fp16）+ latents/双编码器嵌入**预编码缓存**（固定路径按 DATA_FILE 哈希，sweep 复用）；LoRA 目标用 peft **后缀匹配** `["to_q","to_k","to_v","to_out.0"]` 一行覆盖全部 attention
- **video**：Wan DiT/VAE 均 **shape-based key remap**（diffusers 0.40 vs ckpt 0.30 命名）；VAE 输入 `[B,C=3,T=1,H,W]`
- 验证：diffusion/video 用**固定噪声/时间步**（per-sample seed Generator），vlm 生成 greedy ROUGE-L

## 四个新坑（全修复）
1. **切分器写相对路径**：训练器 cd 到工作区后图全打不开（vlm FileNotFoundError、diffusion/video 编码 0 样本静默）——切分产物一律 abspath
2. **就地改文件的事故写法**：修复脚本 `lines=read(); open(f,'w')` 截断后才处理，遇 NameError 崩溃 → **flickr train.jsonl 被清空**。教训：就地改必须「写临时文件+os.replace 原子替换」；且修复脚本崩了要立刻核对首个文件是否已损坏
3. **flickr CSV raw 列是字符串化列表**（`'["caption", ...]'`）——json.loads 取首元素，strip('"') 不够
4. **Qwen2.5-VL pixel_values[0] 切片错**：单图 processor 返回 [256,1176]，`[0]` 取的是首行 patch（[1176]）→ 视觉塔 `shape '[0,4,-1]'` 报错（只剩 2 个视觉 token）。诊断特征：视觉 token 数异常小

## 依赖
- venv 装 diffusers 用 `uv pip install diffusers==0.40.0 --no-deps`（与系统 python 同版本）+ 立即 torch_npu 回归（[[lora-v3-cat3-batch]] #39 纪律）；peft 对 diffusers 模型（UNet/DiT）get_peft_model/save_pretrained 正常

## 关联
[[lora-v3-cat3-batch]]（mlm/mim 已合入 MR !65）；**cat4 三范式已合入 skill（2026-09-18）**：新增 `references/multimodal-paradigms.md` + `scripts/lora_train_multimodal.py.tmpl` + `scripts/validate_multimodal.py.tmpl`，pitfalls 追加 #41-#44（绝对路径+静默0样本/原子替换/pixel_values三坑/字符串化列表+VAE维度序），SKILL.md description+通用性矩阵+核心原则13+文件清单同步，eval-metrics.md 追加多模态指标节（固定噪声MSE/多参考max ROUGE/口径诚实）。回归验证按 [[skill-change-verify-protocol]] 全过（其余 8 tmpl+6 ref diff=0、12 关键词保留、pitfalls 前 40 标题一致、10 tmpl ast.parse、引用完整）。四批累计 16 模型（cat1 4 + cat2 5 + cat3 4 + cat4 3）。改动在工作区未提交，待走 MR 流程（对照 !62/!64/!65）。**同日全量重跑完成**（用户确认范围含第4大类）：新模板+全5组合sweep+200步+val10，3/3 复现改善——SDXL/Wan 与 9-11 逐位一致（确定性复现），VL 选出 r32 略优（ROUGE 0.5977 vs 0.5934）；产物在各 ws `outputs_rerun_0918/`，报告 `lora-ws/.cat4r/LoRA-v4r-验证报告.md`。重跑期间遇 venv 解释器丢失（[[venv-interpreter-loss-recovery]]）。
