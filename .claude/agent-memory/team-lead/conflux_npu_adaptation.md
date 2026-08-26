---
name: conflux-npu-adaptation
description: CONFLUX 3D胸部CT潜在扩散模型(VAE3D+DiT3D修正流)迁移到Ascend 910C NPU的适配方案与踩坑
metadata:
  type: project
---

# CONFLUX 昇腾 NPU 适配（2026-08-26，Ascend910C/A3，CANN8.3RC1）

**任务**：斯坦福 gevaertlab/conflux（3D胸部CT潜在扩散：VAE3D压缩 + 条件修正流DiT3D + RL后训练GRPO）迁移到昇腾910C，精度对齐+性能优化+10样本验证。

## 环境关键事实
- 设备：npu-smi 显示 Ascend910，8 die × 64GB HBM，CANN 8.3.0.1 RC1
- **可用 Python 环境：`/usr/local/python3.12.13/bin/python3`**（系统 python3.10 无 torch）。已装 torch 2.10.0+cpu + torch_npu 2.10.0 + numpy/safetensors/huggingface_hub；补装 nibabel+matplotlib
- uv venv 方案失败：ascend-repo(huaweicloud) 的 torch 2.5.1 wheel 残缺（缺 _vendor/_strobelight/_C 仅132KB stub）；PyPI torch 2.5.1 aarch64 也只下到残缺 wheel。最终直接用系统 python3.12.13

## 踩坑1：torch_npu 2.10.0 加载报 `Failed to load backend extension: torch_npu`
- 根因：`torch_npu._C._get_cann_version(module)` 返回含非UTF-8字节（0xfe/0xe8，CANN版本串带中文）的 str，C 侧内部 `PyUnicode_DecodeUTF8` 抛 UnicodeDecodeError
- locale（LC_ALL=C.UTF-8）无效（C 函数返回非UTF8字节，与locale无关）
- **解法**：patch `/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/npu/utils.py` 的 `get_cann_version`，用 try/except UnicodeDecodeError 包住 `torch_npu._C._get_cann_version(module)` 调用，失败返回 "8.3.0.1"。注意 isinstance(bytes) 检查没用——异常在 C 调用内部抛出，必须 try/except 调用本身。清 .pyc 缓存
- 此 patch 是本机 torch_npu 可用的前提

## 踩坑2：F.scaled_dot_product_attention → Ascend FlashAttention 无内核
- 症状：`aclnnFlashAttentionScore failed, error 161001, Cannot find any bin for input 18, integral key 0/1/|27,2,27,2,27,2,0,`
- 根因：Ascend FA 内核对 DiT(heads=12,head_dim=64,seq_len~2002) 和 VAE AttnBlock3D(num_heads=1,head_dim=512) 的 head 配置无覆盖；**且 FA 不支持 fp32**（RMSNorm/QKNorm 内部 x.float() 把 q,k 转回 fp32，bf16 autocast 无效）
- **解法**：dit.py SingleStreamBlock + vae.py AttnBlock3D 的 SDPA 改为手动 bf16 matmul attention：`q,k,v → bf16; attn=softmax(q@k.T*scale)@v; .to(x.dtype)`。数学等价 SDPA，NPU bf16 matmul 全支持，seq_len 2002/14850 内存无压力
- RoPE 的 float64 在 NPU 上能跑（_rope 用 float64 arange），无需改

## 精度对齐
- 确定性：同 seed 两次 max|diff|=0（完全可复现）
- 全bf16 vs fp32+autocast：std 相同(0.4152)，mean|diff|=0.0036（精度无损）
- 10样本验证（用真实样本标签作条件生成，对比真实体数据统计）：gen std 均值0.435 vs 真实0.427，mean|std diff|=0.020，mean|mean diff|=0.046；全部 finite、std>0.05

## 性能
- bf16 autocast：50步 sample 58s + decode 12.5s = 70.5s/体
- **全bf16权重（dit.to(bf16)+vae.to(bf16)+flow_sample dtype=bf16）：1.24× 加速**（26s vs 32s/20步），锁定为生产配置
- TASK_QUEUE_ENABLE=1（异步算子下发）已开
- HBM 峰值占满（free 6.1G/65.8G），64GB 刚好容纳全分辨率(216×176×200) VAE decode

## 产物位置
- 适配工程：`adaptations/conflux_chest_ct/`（demo.py NPU入口、bench.py、validate.py、cflx/ 已patch）
- 模型权重：`/mnt/model/jiyg/conflux/chest-ct/`
- 数据集：`/mnt/model/jiyg/training-data/conflux-chest-ct/`（521样本+metadata.csv）
- 验证报告：`adaptations/conflux_chest_ct/validation_report.json`
- torch_npu patch 备份：`/usr/local/python3.12.13/.../torch_npu/npu/utils.py.bak`

## 后续可优化（未做）
- TorchAir 图模式（torch.compile+torchair backend）融合 DiT 12层减少 dispatch 开销，目前 1.3s/step 离 910C bf16 峰值仍有差距
- 探索 npu_fusion_attention 对 head_dim 重排的可用配置（当前手动 attention softmax 是主要耗时点之一）
- VAE decoder 全分辨率 conv3d 可考虑分块/分步 decode 降 HBM 峰值

## 2026-08-26 续：CANN 9.0.0 升级 + FA 可用 + 真瓶颈修正（重要，纠正前述 attention 结论）

### CANN 9.0.0 已装，无需安装，只需切环境
- `/usr/local/Ascend/cann-9.0.0` 已存在（version 9.0.0, inner V100R001C10SPC001B250），`set_env.sh` 可直接 source
- 当前默认 `ascend-toolkit/latest`=8.3.RC1 的 OPP **无 FlashAttention 编译内核**（headers 有但 .o 无）
- **CANN 9.0.0 的 OPP 有 ascend910_93 的 FA 内核二进制**：`incre_flash_attention`、`prompt_flash_attention`、`rain_fusion_attention`（在 opp/built-in/.../ascend910_93/ 与 vendors/batch_invariant/）
- 芯片 soc = **ascend910_93**（910C/A3），驱动 25.5.2 / 固件 7.8.0.7.220（兼容 9.0.0）
- **解法**：`source /usr/local/Ascend/cann-9.0.0/set_env.sh`（把 ASCEND_OPP_PATH/ASCEND_HOME_PATH 指到 9.0.0），FA 立即可用。封装在 `adaptations/conflux_chest_ct/run.sh`
- FA 验证：`F.scaled_dot_product_attention` bf16 (1,12,2002,64) → 0.37ms, cos=0.996 ✓；`npu_fusion_attention BNSD` 输出错误(maxerr 3.07)弃用

### ⚠️ 纠正：attention 不是瓶颈（前述"98.7%"是测量假象）
- 之前 profile 在 attention 里放了 `torch.npu.synchronize()`，sync 会等待前面所有排队的异步算子，把整 block 开销全记到 attention 头上 → 误判 attention 占 98.7%
- 真实：attention 单次 0.37ms（<0.1%/步）；恢复原生 SDPA 后采样时间几乎不变（57→56s）
- 真瓶颈：完整 DiT block 93ms、整步 1126ms，有效算力仅 ~0.5 TFLOPS（910C bf16 峰值 ~300 TFLOPS 的 0.17%）；隔离 matmul 能跑到 240 TFLOPS（近峰值），但链式 block 执行慢 500× —— 差距来自 ~180 个小 op/步 的 dispatch 串行化 + RoPE/RMSNorm 的 `.float()` fp32 回退路径 + 缺乏算子融合
- 真优化方向：**TorchAir 图模式(torch.compile) 算子融合**，不是 cannbot（cannbot 是补缺失算子；CANN 9.0.0 下所有算子都已存在）

### 当前最优配置与性能
- 配置：CANN 9.0.0 + 原生 SDPA（撤销手动 matmul attention）+ 全 bf16 权重
- dit.py/vae.py 已恢复 `F.scaled_dot_product_attention`（保留 bf16 cast，因 NPU FA 不支持 fp32）
- demo.py 用全 bf16（`vae/dit.to(device=DEVICE, dtype=bf16)`）
- 性能：采样 56.4s(50步,1.13s/步) + 解码 2.8s = **59.2s/体**（比初始 70.5s 降 16%，主要来自 bf16 解码 9.37→2.81s；采样阶段基本未变因 attention 非瓶颈）
- 精度不变：std=0.426，10样本对齐仍有效

### 用户决策树结论（CANN升级→gitcode→cannbot）
- ① CANN 9.0.0 升级：✅ 成功，FA 算子可用，**无需进入②③**
- ② gitcode.com/cann 找现成算子：未到（9.0.0 已自带）
- ③ cannbot-adapter 生成融合 attention：**不需要**（算子缺口已解决；且 attention 非瓶颈，即使写融合 FA 对端到端提速有限）
- 后续若要再提速：TorchAir 图模式融合（torchair-graph-mode skill），预期 2-5×

## 2026-08-26 续2：torchair 失败 + fp32 RoPE 才是真优化（18× 加速，精度无损）

### torchair 图编译失败（已回退）
- 最小 probe(x+y) 通过，但 DiT 整模编译失败
- 错误1：`npu.npu_fusion_attention_v3 缺 AscendIR Converter`（torch_npu 2.10.0 未注册 SDPA→GE 转换器）→ 被迫改手动 matmul attention（标准 ATen 算子有 Converter）
- 错误2：手动 attention 后仍 `EZ9999 Inner Error / Compile graph failed (1343225857)`（GE 无法编译此自定义 DiT，meshgrid/RoPE/动态 padding 无法干净入图）
- 结论：torchair 图模式对此自定义 DiT 不成立（skill 警告的"高风险，不承诺整图成功"）。已回退 dit.py 到原生 SDPA

### ⚠️ 真优化：RoPE fp64→fp32（18× 加速，精度完全相同）
- 为图编译把 `_rope` 的 `torch.arange(dtype=torch.float64)` 改 fp32，图编译虽失败但重测发现**速度从 56s→1.04s/50步**
- A/B 铁证（同 seed 同权重，仅 RoPE dtype 不同）：
  - fp32 RoPE：sample 1.04s + decode 2.79s = 3.82s/体，std=0.426
  - fp64 RoPE：sample 56.32s + decode 2.81s = 59.12s/体，std=0.426
  - 输出 std/mean/min/max **完全一致**（RoPE fp32 vs fp64 差异在 bf16 量化之下）
- 根因：**Ascend NPU 无原生 fp64**，fp64 算子（arange/cos/sin）走极慢模拟回退，是真瓶颈。改 fp32（原生）后 50 步采样 1.04s
- **教训**：之前所有 profile（attention 98.7%、block 93ms）都被 fp64 RoPE 污染——fp64 慢路径隐藏在 apply_rope 里，被各种 sync 假象掩盖。最终靠 A/B 对照才定位
- 教训2：NPU 上凡 float64 一律先转 float32（除非真需 fp64 精度，医学影像 bf16 推理不需要）

### 最终最优配置与性能
- CANN 9.0.0 + 原生 SDPA + 全 bf16 权重 + **fp32 RoPE**
- demo.py 3.82s/体；lean 验证 1.7s/体（首样本 3.8s 含初始化）
- **端到端 70.5s → 3.82s，18× 加速**，精度无损
- 10 真实 + 5 模拟样本全部通过：all finite、all std>0.05、gen std 0.435≈real 0.427

### 最终产物
- dit.py：原生 SDPA + fp32 RoPE（_rope arange dtype 改 float32）
- 最终验证：adaptations/conflux_chest_ct/final_validation.json
