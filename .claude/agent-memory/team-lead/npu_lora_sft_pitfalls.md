---
name: npu-lora-sft-pitfalls
description: 昇腾 NPU 上做 LoRA SFT 的关键踩坑（CANN 9.0.0 选择/transformers 5.16 支持 qwen3_5/chat template assistant_masks 全 0/NpuFusedAdamW zero_grad）+ Qwen3.5 混合注意力可训练
metadata:
  type: project
---

在 Ascend910 + CANN 9.0.0 + torch_npu 2.10.0 上对 Qwen3.5-0.8B（混合 linear+full attention 多模态架构）做 LoRA SFT 全流程跑通（2026-08-28）。

**关键踩坑（都会重复出现，务必记住）：**

1. **CANN 默认 latest 指向 8.3.RC1（无 FA 内核），必须显式 source 9.0.0**：`source /usr/local/Ascend/cann-9.0.0/set_env.sh`。8.3.RC1 的 OPP 里 FlashAttention 只有头文件无 .o 内核，`F.scaled_dot_product_attention` 会报 "Cannot find binary for op FlashAttentionScore"。本机 `/usr/local/Ascend/cann-9.0.0/` 已装但不是默认。
   - **How to apply:** 任何 attention 模型训练/推理前，先 source cann-9.0.0；用 `attn_implementation='eager'` 也可绕过 FA 依赖（纯 torch 算子，NPU 可用，稍慢）。

2. **torch 2.10.0+cpu + torch_npu 2.10.0 配 CANN 9.0.0**；torch_npu 走 ascend-repo（`https://repo.huaweicloud.com/repository/pypi/simple`），torch 走 download.pytorch.org/whl/cpu（绝不要用 ascend-repo 的 torch，是残缺 stub）。

3. **uv sync 在跨文件系统（/root/.cache/uv → /mnt/model/...）时 hardlink 失败退化为 full copy，torch（~900MB unpacked）复制要 5 分钟**。设 `UV_LINK_MODE=copy` 抑制警告；首次卡在 "Resolved" 不动可能是 torch copy 中。venv 移动（mv 同 fs）后 python prefix 会自动更新，venv 仍可用。

4. **Qwen3.5 (model_type=qwen3_5) 需要 transformers ≥ 5.16.1**（旧版无此 model 类）。AutoModelForCausalLM 加载的实为 text 语言模型（top children = model/lm_head，无 vision tower），24 层：18 层 linear_attn（gated delta，projections = in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/out_proj）+ 6 层 full attn（q/k/v/o_proj）+ 所有层 mlp（gate/up/down_proj）。eager attention 在 NPU 跑通。

5. **`apply_chat_template(tokenize=True)` 返回 BatchEncoding（dict），`len()`=键数 2（input_ids/attention_mask）而非 token 数！** 取 token 数要用 `seg['input_ids']`。

6. **Qwen3.5 的 chat template `return_assistant_tokens_mask=True` 返回的 assistant_masks 全为 0（不可用）**。SFT label 掩码改用**字符偏移映射法**：`rendered = apply_chat_template(tokenize=False)` → `tok(rendered, return_offsets_mapping=True, add_special_tokens=False)` → 在 rendered 中 find `<|im_start|>assistant\n`...`<|im_end|>\n` 字符区间 → token 的 offset 与区间重叠即为 assistant token（标 label）。此法稳健，不依赖模板实现。增量前缀法（encode(msgs[:i])）在 Qwen3.5 上边界不对齐（会捕获 user token）。

7. **`NpuFusedAdamW.zero_grad(set_to_none=True)` 报错 "set_to_none is not supported in fused optimizers"**，必须 `set_to_none=False`。

**验证方法（teacher-forced 多轮）**：逐 assistant 轮，用 GT student/assistant 历史 + 当前 student 作上下文 add_generation_prompt 生成 teacher，与 GT 算 char-level ROUGE-L。LoRA 150 步后 base 0.130 → LoRA 0.370（+184% 相对），且 base 生成冗长跑题(242字)、LoRA 生成精炼贴合教学风格(38字)。

工作目录: `/mnt/model/jiyg/SLAI-AscendBridge2/lora-ws/`，含 scripts/(prepare_data/lora_train/plot_loss/validate/gen_report) + outputs/(train.jsonl/loss/adapter/validation_*)。

**追加（2026-08-29，FSDP2 攻坚最关键教训）：**
- **expandable_segments 与 FSDP2 不兼容**：`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（cpt/lora skill 惯例必设）会破坏 FSDP2 逐层 all-gather buffer 跨层复用 → 64 层 buffer 累积 ≈ 全模型 → 假性"fully_shard 未分片"OOM（曾据此误判 torch_npu 不实现真分片，被用户质疑后逐阶段显存测量推翻：分片本身正常，27B 4-die shard 11.34GB/rank）。FSDP2 必须用 `max_split_size_mb:256`。已在 cpt_fsdp.py.tmpl（守卫）+ lora_train_fsdp.py.tmpl（守卫）+ 两 skill 的 pitfalls 固化。
- **漏调 model.train() 致 GC 失效**：transformers 的 gradient_checkpointing 只在 training=True 生效；from_pretrained 默认 eval。无 GC 时全层激活（27B seq2048 ≈43GB）在 FSDP2（每卡过全模型）下压爆单卡；device_map 流水线版同 bug 被"激活分摊各卡"掩盖。训练脚本必须 model.train()。
- **方法论**：性能/显存问题表象（"多卡不降显存"）可能由多个独立 bug 叠加（分配器配置 + train 模式），不逐阶段测量（加载后/shard后/前向后分别打 memory_allocated）就会误判为后端缺陷。诊断脚本要排除自己按惯例设的环境变量。
- FSDP2 可用配置：bf16 CPU low_cpu_mem 加载（不 .to(dev)）→ 只逐层 fully_shard（不切顶层）→ 顶层参数手动上 NPU → AdamW + enable_input_require_grads + model.train()。27B 8-die 实测 1.45s/样本（device_map 的 4.7×）。

**追加（2026-08-29，Qwen3.6-35B-A3B MoE 80%数据/100步实测）**：
- **MoE 100 步 @ LR 5e-5 的 LoRA 太轻，改不了生成行为**：loss 2.35→0.72（teacher-forced 似然大降）但 ROUGE-L 持平（0.0238→0.0241），两路生成仍走 base 固有英文思维链风格。根因：①融合路由专家挂不上 LoRA（仅 21.2M/35.95B=0.06% 参数可训）；②LR 5e-5 保守；③100 步仅 1400 样本。**已验证加强版（2026-08-31）：500 步 @ LR 1e-4 → Δ/W 0.66%→1.71%，ROUGE-L 0.024→0.212（8.9×），生成从英文思维链转为中文切题回复（273 vs 717 字，59/59 轮全变）。35B MoE 的 LoRA 行为改变门槛≈Δ/W 1.7%，即 500 步 @ 1e-4 量级**。判据：训后查 ‖ΔW‖/‖W‖；base 与 LoRA 逐字对比（有差异=adapter 生效，风格未变=幅度不足——两回事，勿混淆）。
- **FSDP2 训练脚本存的 adapter_lora_state.pt 需转 peft 标准格式**：keys 去 `.default.` 后缀存 `adapter_model.safetensors`（metadata format=pt），PeftModel.from_pretrained 才能直接加载；Qwen3.6 的 base checkpoint 键名是 `model.language_model.layers...` 而内存模块名是 `model.layers...`（transformers 自动重映射），对比权重要注意换算。
- **多卡训练挂死若伴随全卡 HBM 膨胀到 ~64GB 而 torch 峰值正常**，是 HCCL 通信缓冲+挂死重试累积的表象，先查个别芯片中毒（见 [[npu-poisoned-chip-hccl-hang]]）。

**追加（2026-08-31，35B MoE 训练提速实测，归因→优化闭环）**：
- **MoE LoRA 训练慢的根因是 transformers eager 专家分发 = Python 逐专家 for 循环**：qwen3_5_moe 的 `Qwen3_5MoeExperts.forward` 里 `for expert_idx in expert_hit:`，短序列(290 token×top-8 命中 ~180 专家)下 40 层×180 迭代×~10 小算子 ≈ 7万+ 微小内核/前向，反向翻倍。步时分解：fwd 40% / bwd 58% / opt+comm 2%，吞吐仅 ~15 tok/s/rank——纯调度开销，与算力无关。
- **稠密 MoE 补丁（数学等价，-34% 步时）**：monkey-patch `Qwen3_5MoeExperts.forward` 为 2 次批量 bmm 算全部 256 专家再 gather top-k（多算的丢弃）。等价性验证：同种子 4 步 loss 差 0.3~0.5%（bf16 内核舍入级）。步时 20.5→13.5s，显存 10.3→12.0GB。产物：`lora-ws/Qwen3.6-35B-A3B-lora/scripts/lora_train_fsdp_fast.py`（含补丁+右填充批处理）。
- **[反直觉] 35B MoE 上关 GC 反而更慢**（20.5→31.2s）：全激活保留致 NPU 分配器碎片化与压力（峰值 10.3→20.9GB），逐专家小张量分配变慢。GC 开既省显存又更快。
- **reshard_after_forward=False 在 35B+稠密激活下必 OOM**（40 层全聚合 35GB+激活>61GB）；transformers 的 `batched_mm` 专家实现会物化 S×份专家权重（200token×top8 → ~60GB）不可用；`grouped_mm` 依赖 CUDA 的 torch._grouped_mm，NPU 无此内核。
- **bs=4（右填充，pad label=-100）= 3.25× 吞吐**（25.3s/步处理 4 样本 = 6.3s/样本，显存 18GB）；代价是有效 batch 14→56，训练动力学有变。
- 计时探针模板：`scripts/step_probe.py`（fwd/bwd/opt/comm 分相计时 + PROBE_GC_OFF/PROBE_RESHARD_OFF/PROBE_DENSE_MOE/BATCH_SIZE 开关）。教训：优化前先分相实测，别按直觉改（GC 直觉上该关，实测该开）。
- **端到端验证（2026-08-31，fast 200 步正式运行，同超参）**：14.46s/步（-27.4%，探针短序列子集预测 -34%，全量数据略缓）；loss 轨迹与 eager 前 200 步窗口均值偏差 -0.4%~+1.6%（残余=LoRA A 随机初始化差异+bf16 噪声）；200 步 Δ/W=1.19% 落在 eager 100 步(0.66%)→500 步(1.71%) 插值区间。**稠密补丁正式训练成立**。注意：训练脚本无 torch.manual_seed，lora_B 初始为 0 所以 step1 loss 跨运行必然一致，step2+ 必然有小幅随机偏差，属正常重跑差异，勿误判为不等价。

**追加（2026-09-01，grouped GEMM 正解打通 + skill 沉淀）：**
- **#22 旧结论"NPU 无 grouped GEMM 内核"已被推翻**：按三步算子发现法（本机 CANN 接口盘点→gitcode.com/cann 搜索→文档案例）找到 torch_npu 2.10 内置 `npu_grouped_matmul`（无反向）+ cann/torchtitan-npu 的 aten::_grouped_mm PrivateUse1 桥接（反向免费，dx/dw 逐位一致实测）。详见 [[npu-grouped-gemm-moe-ops]]。
- **GMM 训练性能反转**：单层全 T 碾压 dense（T=290: 10 vs 27ms），但真实训练（短序列+14进程）分发链小算子开销放大 → 17.4 vs 13.5s/步；bs=4 打平且显存省 30%（12.8 vs 18.1GB，∝topK×T 而非 E×T）。**选型：短序列 dense / 长序列(T≥1024)+大batch+显存受限 gmm**。
- **已沉淀进 ascend-torch-lora skill（2026-09-01）**：① `scripts/lora_train_fsdp.py.tmpl` 新增 `MOE_IMPL=eager|dense|gmm` 开关（含 gmm 自定义 autograd + DTensor to_local + 右填充 collate，冒烟实测 dense/gmm diff=8e-6）；② 新 reference `references/moe-optimization.md`（双路线代码/选型表/等价性验证协议/已试错清单）；③ 新 reference `references/npu-op-discovery.md`（三步算子发现法，用户强调的方法论：报"昇腾不支持"前必须本机接口盘点→gitcode.com/cann→文档案例，结论要写明排查范围）；④ pitfalls 修正 #22 + 新增 #23（GMM 正解）/#24（三步法）/#25（端口占用/HCCL 瞬断/芯片中毒 device↔chipId 映射/LoRA 初始化等价性判读）；⑤ SKILL.md 核心原则增至 10 条 + 通用性矩阵 + 触发描述加 MoE 提速关键词。回归验证全过（ast.parse/关键词保留/引用完整/冒烟）。
