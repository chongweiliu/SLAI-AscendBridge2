# Team-Lead Agent Memory

> 索引。详细内容一律在主题文件中；执行细则以 `.claude/agents/team-lead.md` 系统提示词为准。

## 环境与硬件

- [transformers 5.x 加载老 remote code 的 3 个 patch 点](transformers5_remote_code_patches.md) — tokenizer 特殊 token 赋值顺序 / ROPE_INIT_FUNCTIONS 无 'default' / all_tied_weights_keys 缺失；附 NPU CPU-bool-mask 索引坑（"N elements cannot be converted to Scalar"）

- [昇腾 LoRA SFT 全流程踩坑](npu_lora_sft_pitfalls.md) — CANN 必须 9.0.0/latest 是 8.3.RC1 无 FA 内核；transformers≥5.16 支持 qwen3_5；assistant_masks 全 0 用字符偏移法标 label；NpuFusedAdamW zero_grad 须 set_to_none=False；FSDP2 禁 expandable_segments；必须 model.train()；35B MoE 100 步@5e-5 太轻（Δ/W 0.66%），500 步@1e-4 达门槛（1.71%，ROUGE-L 8.9×）；MoE 慢根因=eager 逐专家 Python 循环，稠密 bmm 补丁 -27~34% 数学等价；adapter 需转 peft 格式
- [多卡 HCCL 挂死=芯片中毒](npu_poisoned_chip_hccl_hang.md) — RunAicpuKfcResInitV2 失败/全卡 HBM 假膨胀 64GB 而 torch 峰值正常 → 按错误里 chipId/dieId 排除该芯片两 die 重跑（device N↔chip N/2, die N%2）
- [昇腾 grouped GEMM/MoE 算子调研](npu_grouped_gemm_moe_ops.md) — torch_npu 2.10 已内置 npu_grouped_matmul（实测调用格式：x/w 包 List、group_list 前缀和、group_type=0 须 split_item=2；前向精确 bf16）；permute/unpermute grad 已内置；GMM 反向无，可前向组合自包 autograd；源码 cann/ops-transformer + cann/catlass（昇腾版 CUTLASS）
- [CPT 环境踩坑](ascend_cpt_env_pitfalls.md) — 950PR/950DT 新芯片需升级 torch_npu；容器内存限制致 torch.load 大 ckpt OOM；torch2.12 后 torchvision ABI 错配（已沉淀 ascend-torch-cpt skill pitfalls #42-45）
- [950PR 容器 32GB cgroup](cpt_950pr_32gb_cgroup.md) — 多模态 remap 必须搬 NPU，不留 CPU 堆副本（SIGKILL 137）
- [HF 下载技巧](hf_mirror_download_technique.md) — DNS 污染→/etc/hosts 固定 hf-mirror IP→curl -L -C - 断点续传+12 路并行；弃用易卡死的 hf download Xet；hf-mirror 单节点宕机时备选：ModelScope(Name字段搜索)/Zenodo(12路Range)/codeload.github.com
- [并行Range下载损坏修复](parallel_range_download_corruption.md) — 双实例并发追加=尺寸对内容坏的块；必须gzip -t终检；探针抽样会漏，全量流式比对才收敛；杀进程树按 主脚本→xargs→bash→curl 逐层

## 工作流与协作

- [团队编排踩坑合集](team_orchestration_pitfalls.md) — 逐个 spawn+30s；心跳 active≠存活，分配前验 inbox 文件存在；TeamDelete 卡死直接改 config.json；compaction 后 "Shut down" 无限循环是 orchestrator bug 不自理；board_ops 传 JSON notes 用 Python 直调不走 shell
- [skill 改动回归验证协议](skill_change_verify_protocol.md) — 未改文件 diff=0/纯追加/关键词保留/ast.parse/引用完整
- [gitcode MR 流程](gitcode_mr_workflow.md) — 令牌安全红线、v5 API 创建+合并 MR
- [共享存储多会话互扰](shared_storage_multi_session_interference.md) — 下载产物放带前缀独立子目录，中断后先全盘 find 定位
- [spawn 最佳实践](spawn_best_practices.md) / [故障容错](fault_tolerance.md) / [优化精度对比](optimization_accuracy.md) / [Git/CI 运维](git_ci_ops.md) / [批次记录](batch_records.md)

- [24 新模型批量 CPT 实战](cpt_v2_24model_batch.md) — v2 轮 19改善/4持平/1特殊；ASR用-hf repo+整秒补零、SD3.5仅T5维度4096、DepthPro禁融合优化器、Wan VAE bf16、flickr caption列表死循环、dinov3假loss0查标签、FSDP2 mesh/HCCL超时1800/Adafactor-DTensor不兼容→冻结36层+AdamW、FP8 scale键替换非拼接、36B保存死锁抢救
- [台账第1大类 4 模型 LoRA 批量微调](lora_v1_cat1_batch.md) — 4/4 CE 降 9~27%、SmolLM2+ultrafeedback ROUGE +0.152；三流程坑：前缀截断信息间隙(83.7%样本→跳段生成,切点受cap约束+自检)/SAMPLE_RATIO默认0.5静默抽半/env.sh循环source泄漏须子shell隔离；OLMoE无模板注入User:/Assistant:；Qwen3验证须enable_thinking=False
- [台账第2大类 5 模型 LoRA（非CausalLM范式）](lora_v2_cat2_batch.md) — 3/5 双证据改善(bge-m3检索/nllb翻译/distilbert情感)、reranker 30query过拟合无提升、bert-NER原生即conll饱和；五范式训练器(embed InfoNCE/rerank softplus/seq2seq/cls/ner)可复用；三新坑：sorted源probe偏斜须shuffle/nllb须设tgt_lang/seq2seq生成不能按源长切片
- [台账第3大类 4 模型 LoRA（AI4Science）](lora_v3_cat3_batch.md) — 3 mlm全改善(ChemBERTa -57.7%/ClinicalBERT -12.8%/esm2 -1.5%同源近饱和)、Prithvi mim饱和持平；mlm/mim范式沉淀(固定掩码验证/peft对非HF模块)；四新坑：pubmedqa context是dict非str/hls台账3233实为804影像(AppleDouble+mask)/timm安装拖torch 2.14断链须pin回滚/mim无tokenizer须None守卫
- [台账第4大类 3 模型 LoRA（多模态）](lora_v4_cat4_batch.md) — 3/3全改善(VL描述ROUGE+0.062/SDXL MSE-47.6%/Wan MSE-55.4% base零文本预测劣于零预测器)；vlm/diffusion/video三范式沉淀(pixel concat/预编码缓存/peft后缀匹配to_q/shape-remap)；四新坑：切分须写绝对路径/就地改文件必须临时文件+原子替换(截断事故)/flickr caption字符串化列表/pixel_values[0]切片错只剩2视觉token；**2026-09-18 三范式已合入 skill（multimodal-paradigms.md+两模板+#41-44）并用模板全量重跑复现（SDXL/Wan 逐位一致，VL全5组合r32略优）**
- [台账第5大类 5 模型 LoRA（视觉与时序）](lora_v5_cat5_batch.md) — 5/5改善：layoutlmv3 F1 0.52→0.85最强/depth AbsRel-44%/dinov2 CE-40%/det预算内改善(新头+300图+200步不收敛如实记)/timesfm近同源小幅；**head-only基线协议**(新头范式base侧=冻骨干只训头,同HEAD_SEED)；四新bug：peft冻结模型内新头须重新requires_grad+去重/NpuFusedAdamW拒纯bf16参数组头须.float()/validate字典漏键/sweep污染须确认复验
- [台账第6大类 3 模型 LoRA（语音音频）](lora_v6_cat6_batch.md) — 3/3改善：whisper CER减半(0.0091→0.0045主场仍降)/AST top1 7→10 CE-91.5%/speecht5 mel MSE-37%；transformers 5.x语音栈五坑：WhisperFE max_length变采样点语义须手动pad3000帧/SpeechT5偶帧+guided_attn崩+pos_weight留CPU/ASTMLPHead layernorm被误清零致CE恒ln(50)+dense候选撞classifier.dense；**纪律：新头范式先单批过拟合测试再sweep**

## 模型适配案例

- [PowerFlowNet NPU CPT 坑](powerflownet_npu_cpt_pitfalls.md) — main与历史ckpt代码断代(save_logs时间戳→git历史commit对齐); surfdrive五坑(限流封禁≤24路/PROPFIND zip错位逐文件查/urllib hang→curl后端/尾段截断换文件/token认证); 同形bool索引1-D展平; regular优先namespace补__init__; 断点续传r+b+seek精确写
- [EquiformerV2/fairchem NPU CPT 坑](equiformer_v2_npu_cpt_pitfalls.md) — fairchem 1.10 pin torch2.4→no-deps+精确注册; OCPTrainer全栈NPU死锁(pin_memory吃满HBM静默挂死)→MINREPRO证模型仅0.5GB→自管轻量循环; 能量per-element参考口径(extxyz raw vs 模型去参考差500eV,最小二乘反解ref表,误差∝原子数是指纹); ase读xz O(n²)先解压100×; ase_read_multi双bug; EMA污染final ckpt差5×; 等变batch超线性; 力指标聚合口径敏感5×
- [MatterSim 昇腾 CPT 全流程坑](mattersim_npu_cpt_pitfalls.md) — 官方代码3处NPU patch(batch_to_dict静默降级cpu/threebody缺文件shim/mp_api懒加载)；MLIP评估禁no_grad(力=-dE/dx需图)；re_normalize只重拟合shift保力基线(scale重设破坏F=-scale·draw/dx)；high_level_water.xyz在ModelScope镜像data/而非GitHub；mattersim依赖双pin防torch升级
- [ProteinBERT Keras→PyTorch CPT](proteinbert_keras2torch_cpt.md) — pkl变量序非代码序(dense-global-input在前)；同形权重交换消融验证；替换噪声恢复率~11%即健康(勿按掩码MLM预期误判)；NpuFusedAdamW跨阶段重建优化器必崩；signalP_binary全70aa在protein_bert仓库内
- [ProteinMPNN 昇腾 CPT 全流程坑](proteinmpnn_npu_cpt_pitfalls.md) — checkpoint反传507035(双侧才崩,去掉即好)、变长batch必开expandable_segments(否则0.37→5s/step)、NoamOpt factor=2 CPT发散(降0.25/500)、per-epoch数据重建不可用(fork池预处理缓存1300/s)
- [CONFLUX 3D 胸部 CT 扩散模型迁移](conflux_npu_adaptation.md) — VAE3D+DiT3D 修正流迁移 910C；CANN 版本串非 UTF8 patch；FA 内核缺手动 bf16 matmul attention；全 bf16 1.24× 加速

## 下载与容错
- [hf-mirror 宕机 ModelScope 兜底](hf_mirror_outage_modelscope_fallback.md) — MS 模型/数据集 API+下载端点全覆盖；worker 停滞删重下+必须精确字节；pkill 自匹配 exit 144 连环坑
- [venv 解释器丢失原路径重装](venv_interpreter_loss_recovery.md) — uv 托管 python(/root) 跨会话被清而 /mnt venv site-packages 完好 → pip 装 uv + UV_PYTHON_INSTALL_MIRROR=gh-proxy 装回原路径即零依赖恢复；勿重建 venv

## 硬约束（不可遗忘）

- `prompts/task_*.txt` 写明的人数/并发数是最高优先级契约：写几个就是几个，不得因"机器空闲/想提速"增减；`wait_cuda` 是模型状态不是 worker 占用，prompt 写 1 个 business-benchmark 就一直复用它
- 第四阶段验收不能只看工件齐全：VLM 漂成纯文本集、latency 微秒级但 wall-clock 秒级、有 model_files/ 但 npu_perf 无 patch 继承证据 → 一律打回重跑
- 不得自行退出主循环/TeamDelete，除非用户明确要求；nopua：同一 action 失败 2+ 次即调用，不被动等待
