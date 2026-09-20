---
name: proteinmpnn-npu-cpt-pitfalls
description: ProteinMPNN 在昇腾 910 上 CPT 的三个关键坑：torch.utils.checkpoint 反传 vector core 异常(507035)、变长形状必须 expandable_segments、官方 NoamOpt factor=2 对 CPT 发散
metadata:
  type: project
---

# ProteinMPNN 昇腾 CPT 三个关键坑（2026-09-01 实测，910 + torch 2.10.0+torch_npu 2.10.0 + CANN 8.3.RC1）

**1. `torch.utils.checkpoint` 反传 → vector core exception (EE9999/507035)**
- 现象：ProteinMPNN 官方 model_utils.py 的 enc/dec 层 checkpoint 包裹，在特定 batch（如 B=4 L_max=2305）**确定性**崩 "The vector core execution is abnormal"；forward 单独 OK、无 checkpoint 的 fwd+bwd OK、**单侧 ckpt（只 enc 或只 dec）OK、双侧 ckpt 才崩**——与 reentrant/preserve_rng_state 配置无关。
- 修复：去掉 checkpoint 直调 layer（protein_mpnn 规模 1.66M 参数，64GB HBM 无 ckpt 峰值 ~16GB 完全装得下，且提速 15%：0.372→0.316 s/step）。
- 定位方法论：逐步批次形状插桩（每步先落盘 B/L_max/first-name 再执行）→ 崩溃即知凶手批 → 离线单批复现 → 按 phase（features/encoder/forward/loss/backward）和 ckpt 配置矩阵二分。

**2. 变长形状（L_max 200~9852 逐批变化）必须 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`**
- 关闭后：分配器碎片化 → s/step 从 0.37 **渐进劣化到 5+**，plog 里 207001 OOM 重试风暴（单次申请 700MB-1.2GB 失败）。与 [[npu_lora_sft_pitfalls]] 的 "FSDP2 禁 expandable_segments" 相反方向——**非 FSDP 的变长 batch 场景必须开**。

**3. 官方 NoamOpt(factor=2, warmup=4000) 是从零训练配置，CPT 必发散**
- v_48_020 已收敛（官方 200 epochs），峰值 lr 2.8e-3 直接把模型推飞：train/valid ppl **双升**（5.15→5.96 / 4.72→5.17）。
- CPT 合适配置：**factor=0.25, warmup=500**（峰值 ~1e-3 → 收尾 ~2e-4），收敛正常（train 5.19→4.96）。
- 教训：CPT "官方 optimizer 配置" 不能照抄，lr 量级要按起点收敛状态重选；发散特征=训练集 ppl 也升（区别于过拟合的 train 降 valid 升）。

**4. 数据管线：官方 per-epoch 重建在 NPU/网络 FS 场景不可用**
- 官方 ProcessPoolExecutor+DataLoader 每 epoch 重建（spawn 慢启动+get_pdbs 串行后处理+70 万小文件）要 40+min/epoch。
- 改一次性 fork Pool(64) 预处理缓存（worker 内组装→修剪→featurize，numpy 数组 IPC）：1300+ 组装体/s，缓存后训练**秒级加载**。8GB train_cache 供 6 epochs 复用（组内重组装采样换成了固定采样+批序洗牌，CPT 场景可接受）。
- torch 2.10 `torch.load` 默认 weights_only=True，加载含 numpy 的缓存要 `weights_only=False`。

**5. 结论口径：同分布+已收敛基线的 CPT 预期是持平**
- ProteinMPNN v_48_020 在自己训练分布的 80% 上短程 CPT（6ep/9918 步）：100 条留出链 recovery@T0.1 45.13% vs base 45.66%（-0.53pp）、NLL +0.009——**无增益**。base recovery 45.66% 与论文量级一致（协议正确）。
- 要真实增益需换分布数据（新 PDB 快照）或 base 未见过簇的重划分。蛋白逆折叠模型评估协议：单链设计 + `model.sample()` 自回归采样 + T=0.1 recovery + teacher-forced NLL/acc。

**How to apply**：任何 GNN/MPNN 类模型（protein_mpnn/EneRFold 等）上 NPU 训练，先查这三件套：去 checkpoint、开 expandable_segments、CPT lr 降一档。批次插桩+离线单批复现是定位确定性崩溃的标准手段。

**已通用化贡献到 ascend-torch-cpt skill（2026-09-01）**：本文件 1-5 条的通用部分 = skill pitfalls #78（多区域 ckpt 反传 507035）/ #79（崩溃定位四步法）/ #80（变长 batch expandable_segments）/ #81（weights_only）/ #82（从零调度 CPT 发散）/ #83（per-epoch 重建→一次性缓存）；评估三原则（同种子确定性自检/论文锚定/同分布持平预期）进 SKILL.md 阶段 8 + eval-metrics.md；数据缓存模式进 data-prep.md §8；lr 原则进 hyperparam-selection.md。ProteinMPNN 专属细节（批次 869、v_48_020 数字等）仅留本文件。

关联：[[ascend-cpt-env-pitfalls]] [[hf-mirror-download-technique]] [[parallel-range-download-corruption]]
