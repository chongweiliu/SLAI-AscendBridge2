# 文生视频/文生图 diffusers 扩散模型继续预训练(CPT)

> 适用：diffusers pipeline 类生成模型——文生视频(T2V)、文生图(T2I)、图生视频(I2V)。
> 典型组件：`transformer/`(DiT) 或 `unet/` + `vae/` + `text_encoder/` + `scheduler/` + `tokenizer/`，有 `model_index.json`。
> 本文档从 Wan2.1-T2V-1.3B / FastMetal-1.3B-QAD + MixKit 实战提炼，对同类扩散模型普遍适用。

## 核心方法论（先内化，再动手）

扩散 CPT 与文本 LM CPT 是**不同范式**，但骨架一致——"拆出可训练 backbone + 冻结编解码器，用原生损失训练"：

1. **判定格式**：是标准 PyTorch diffusers，还是 MLX(Apple) / 量化格式？MLX 须换 PyTorch 基座（见下"坑1"）。
2. **拆组件**：可训练 = 生成 backbone（DiT/U-Net）；冻结 = VAE（编解码 latent）+ text_encoder（编码条件）+ scheduler（噪声调度）。只训练 backbone，其余 `requires_grad=False`。
3. **数据流水线**：decode(视频 decord / 图片 PIL) → resize 到原生分辨率 → **VAE 编码上 NPU** → latent；text_encoder 编码 caption → emb；**预计算 + 缓存 latent+emb 到磁盘**，训练阶段只跑 backbone 前向反向（快、省显存）。
4. **损失**：流匹配(`sigma~U[0,1], noisy=(1-σ)·lat+σ·noise, target=noise-lat=velocity`) 或 DDPM epsilon。backbone 预测，MSE loss。
5. **评估**：固定 σ 算 velocity MSE（base vs CPT，越低越好）+ 采样生成（noise→N 步去噪→VAE 解码→帧）。**不要套 PPL/next-token**。

## 全流程（9 阶段，与 SKILL.md 对齐）

### 阶段 0-1 · 判定 + 环境
- **范式判定**：`ls MODEL_DIR`——有 `model_index.json` + `transformer/`+`vae/`+`text_encoder/`+`scheduler/` → 扩散生成式范式。`architectures` 含 `WanTransformer3DModel`/`UNet`/`DiT`/`*Transformer` 等。
- **格式判定**：出现 `mlx_*.json`/`mlx_*.safetensors` 或 `library_name=mlx` → MLX 格式，NPU 不可直接加载，换同源 PyTorch 基座（坑1）。
- 依赖：`pip install diffusers decord imageio-ffmpeg`（diffusers 需含目标模型支持，如 Wan 需 ≥0.32，建议装最新如 0.40）。`decord` 自带 ffmpeg，无需系统 ffmpeg。
- cgroup 检查（核心原则 9 的 cgroup 条）：VAE 编码/UMT5 加载全上 NPU，不在 CPU 堆。

### 阶段 2 · 获取（含坑1-3）
- **MLX 格式（坑1）**：如 `FastVideo/FastMetal-1.3B-QAD` 是 `mlx_dit.*`（MLX int8 量化，Apple 专用）。昇腾 NPU 用 transformers/diffusers 无法加载训练。**解法**：改用同源 PyTorch 基座（如 `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` 的 `transformer/`）。FastMetal 特定 finetune 权重(MLX)无法继承，须在 README 注明替换。
- **下载 flaky 大文件（坑2）**：`hf_hub` snapshot_download 在 flaky 连接上对大文件(>2GB)会**校验失败后从头重下**（.incomplete 从 4.98GB 回到 52MB，死亡螺旋，永远下不完）。**解法**：停 hf_hub，改 `wget -c --tries=0 --timeout=30 --retry-connrefused --waitretry=3`（断点续传，连接断从断点续，不重头）。大文件分片可并行多 wget。
- **优先级下载**：hf_hub 默认无优先级，可能先下巨大的 text_encoder(fp32,11-23GB) 再下 backbone。用 `allow_patterns` 指定 backbone(`transformer/*`,`vae/*`) 优先，text_encoder 后置（或换同族 bf16 版省一半）。
- **text_encoder 可能巨大（坑3）**：UMT5-XXL 等 text_encoder 可达 11-23GB。下载受阻时见下"文本编码兜底"。

### 阶段 3 · 数据流水线（用 `prepare_generative_data.py.tmpl`）
- **视频 decode**：`decord.VideoReader(path, width=W, height=H)`，按步长取 N 帧（N 满足 VAE 时间压缩因子 k：`(N-1)%k==0`，如 Wan k=4 → N∈{1,5,9,13,17,21,...}）。归一化 `[-1,1]`：`frames/127.5-1`。
- **VAE 编码（坑4/#52）**：输入 **`[B,C,T,H,W]`**（视频）/ `[B,C,H,W]`（图片），**不是** `[B,T,C,H,W]`。`decode_video` 返回 `[T,3,H,W]` → `permute(1,0,2,3).unsqueeze(0)` 成 `[1,3,T,H,W]`。`vae.encode(v)`：返回 `.latent_dist`(.sample()) 或 `.latents` 或直接 tensor，三种兼容。latent 通道数 = config `in_channels`（Wan=16）。
- **latent 归一化（坑52，必读）**：`AutoencoderKLWan.encode().latent_dist.sample()` 返回 **raw latent（无自动归一化）**。diffusers Wan pipeline 的 decode 是 `raw = stored * config.latents_std + config.latents_mean`，故 DiT 原生 latent 空间 = **`stored = (raw - latents_mean) / latents_std`（除，单位方差）**。prepare 阶段编码后应做此归一化（`prepare_generative_data.py.tmpl` 默认 `LATENT_NORM=1` 开启），让训练 latent 与预训练 DiT 对齐 + CPT ckpt 可直接 drop-in 原 pipeline 生成。**符号陷阱**：是**除以** `latents_std`，**不是乘**——乘反会把 loss 虚高 ~10×（实测 21 vs 正确 ~2）。`LATENT_NORM=0` 存 raw（原 950PR 方式，能跑但与预训练 scale 不对齐、ckpt 不能直接进原 pipeline）。eval 生成时 `vae.decode(lat)` 前须逆归一化 `lat = lat*latents_std + latents_mean`（`eval_diffusion.py.tmpl` 据 cached `latent_norm` 标志自动处理）。
- **VAE 编码上 NPU**：VAE encode 的 3D conv 中间激活大，CPU 侧撞 cgroup OOM；视频 decode 后小 tensor `.to(npu)` 再 encode，CPU 峰值≈0。
- **分辨率**：用模型原生分辨率（Wan 832×480）。显存紧张可降分辨率/减帧，但 H/W 须整除 VAE 空间压缩因子(Wan=8)。
- **视频解码兜底**：`decord` 装不上（无 aarch64 wheel / 缺 ffmpeg）时，`decode_video` 回退 `imageio.v3.imread(path, index=None)`（读全帧→抽帧+resize），imageio-ffmpeg 已在依赖里。**按文件扩展名**判视频/图片，不按 decord 是否可用。
- **captionless 视频数据集**（MixKit/stock footage 无标注）：`caption_for` 从文件名派生（`mixkit-airplane-arriving-at-air-terminal-4095_clip_1.mp4` → "airplane arriving at air terminal"，去 mixkit-/数字 id/_clip_N），或用类别名；无文本来源则退回零嵌入无条件（坑50）。
- **文本编码（坑3/坑53 兜底）**：
  - 正常：`text_encoder(input_ids).last_hidden_state` → `[1,L,D]`（D=DiT 期望的 cross-attn 维，如 UMT5-XXL D=4096）。
  - **UMT5 必须用 `UMT5EncoderModel`（坑53）**，**不要用 `T5EncoderModel`**——UMT5 每层都有 `relative_attention_bias`，T5EncoderModel 只在 block 0 有，用 T5 加载会丢 block1-23 的偏置、embedding 静默退化。CLIP 用 CLIPTextModel。
  - **预计算+缓存+释放**：编码完全部 caption 后 `del text_encoder; torch.npu.empty_cache()`，训练时只用缓存 emb，省 11GB 显存。
  - **兜底（零嵌入近似无条件）**：text_encoder 下载受阻/未就绪时，用 `torch.zeros(1,L,D)` 近似无条件训练。DiT 流匹配 velocity 预测仍正常学习视频分布，loss 正常下降；代价是无文本条件对齐。**须在 README 注明**。TE 仍可后台下载，就绪后重跑带真实 caption 版本。
- **缓存**：`torch.save({"latents":[...],"embs":[...],"captions":[...]}, "video_latents.pt")`。latent `[1,C,Tl,Hl,Wl]`、emb `[1,L,D]` 各占几 MB，60 个约几十 MB。
- **held-out**：留后 N% 样本不参与训练，用于阶段 8 velocity MSE 评估。

### 阶段 4-5 · 选型 + 超参（**自动**, 对齐 parallel-strategy.md / hyperparam-selection.md）
**通用加载**：`cpt_diffusion.py.tmpl`/`prepare_generative_data.py.tmpl`/`eval_diffusion.py.tmpl` **读 `model_index.json` 动态按类名 import** backbone(`transformer`/`unet`→WanTransformer3DModel/UNet2DConditionModel/FluxTransformer2DModel/SD3Transformer2DModel/CogVideoXTransformer3DModel/…)、VAE(`AutoencoderKLWan`/`AutoencoderKL`/`AutoencoderKLSD`/`VQModel`/…)、text_encoder(`UMT5EncoderModel`/`CLIPTextModel`/`T5EncoderModel`)——**不写死 Wan 类，换 diffusers 视频/图模型不改模板**。`backbone_dir` 据 model_index key 自动定(`transformer` vs `unet`)。
**自动并行**（探测 `device_count`+`free_mem`，估 `8×P` bf16/`16×P` fp32 GB）：`8×P≤free×0.8`→单卡；单卡装不下+`n_card≥2`→2卡模型并行(`max_memory` 按 `free×0.5` 自动算)；仍装不下→LoRA 回退。`MODE=auto|single|mp2|lora` 可强制。
**自动超参**：lr=`1e-5×sqrt(global_batch/32)` clamp `[5e-6,5e-5]`；warmup~10%；bs OOM 回退阶梯(bs//2→1)；grad_accum 凑有效 batch。fp32/bf16 master + autocast + grad-ckpt(`use_reentrant=False`)。`from_pretrained` 撞 `keep_in_fp32_modules`(#54) → `from_config+load_state_dict` 回退。
- 视频 DiT 单步显存大，常 batch=1；大 DiT(14B) 自动走模型并行，小 DiT(1.3B) 单卡。

### 阶段 6 · 脚本 + smoke（坑5）
- 用 `cpt_diffusion.py.tmpl`：读缓存 latent+emb，流匹配，NpuFusedAdamW，存 `dit_cpt_state.pt`。
- **smoke 顺序（坑5，先单组件后合练）**：
  1. **backbone 前向反向 smoke**：dummy latent `[1,C,Tl,Hl,Wl]` + dummy text emb `[1,L,D]`，`transformer(hidden_states=, timestep=, encoder_hidden_states=, return_dict=True)`，前向看 pred shape 对、反向不报错。**早抓 NPU 算子问题**（attention/3D conv 是否支持）。
  2. **VAE encode smoke**：dummy `[1,3,T,H,W]`，`vae.encode` 看 latent shape 对（坑4 layout）。
  3. 两者通过再合练 2 步。
- **DiT forward 返回兼容（坑5）**：`out` 可能是 tensor 或 dict(`.sample`) 或 dataclass——`isinstance(out,dict): out=out['sample']; elif hasattr(out,'sample'): out=out.sample; else: out`（直接是 tensor）。**不要**无脑 `out[0]`（tensor 索引会丢 batch 维）。
- **checkpoint 权重未完全用警告**：`Some weights were not used: ...to_gate_compress...` 是 diffusers 版本与 ckpt 的 minor 差异（多/少几个 gate 键），forward 正常即忽略。

### 阶段 7 · 正式训练
- 流匹配每步：`sigma~U[0,1]; noisy=(1-σ)·lat+σ·noise; target=noise-lat; pred=Dit(noisy, t=σ·1000, emb); loss=mse(pred, target)`。timestep 常乘 1000 对齐 scheduler 尺度。
- 心跳/用时表同文本 CPT（核心原则 7/9）。存 `dit_cpt_state.pt`(backbone state_dict) + `train_summary.json` + `losses.json`。
- 画 loss 曲线 + 公网直链（plot_loss.py 通用，读 `step_loss.jsonl`）。

### 阶段 8 · 评估（用 `eval_diffusion.py.tmpl`）
- **量化 velocity MSE**：固定 σ（如 0.5），对 base vs CPT 在 held-out latent 上算 MSE(pred, target)。`delta<0` 训练有效。注意 base 也用同 σ 同 noise seed 公平对比。
- **定性采样生成**：`lat=randn; for s in N: t=(N-1-s)/N; v=Dit(lat, t·1000, emb); lat=lat+dt·v`（Euler，从 t=1 降到 0）；`vae.decode(lat).sample` → `[1,3,T,H,W]` → 帧 → `imageio-ffmpeg` 存 mp4（imageio 常缺失，用 `imageio_ffmpeg.get_ffmpeg_exe()`+rawvideo pipe 存 mp4，或存 `.npy`+PNG）。
- VAE decode 后 `clamp(-1,1)` 再 `(+1)/2*255` → uint8 帧。

## 踩坑速查（详见 pitfalls.md #47-54）
- #47 MLX 格式 → 换 PyTorch 基座
- #48 hf_hub flaky 大文件死亡螺旋 → wget -c
- #49 VAE 输入 layout [B,C,T,H,W]
- #50 text_encoder 巨大 → 预计算缓存/零嵌入兜底
- #51 DiT forward 返回 tensor|dict 兼容 + 组件分离 smoke 顺序
- #52 VAE latent 归一化符号：`(raw-mean)/latents_std` 除(非乘)，乘反 loss 虚高 ~10×
- #53 UMT5 用 `UMT5EncoderModel`(非 T5EncoderModel)，否则丢 block1-23 的 relative_attention_bias
- #54 transformers5.x + `_keep_in_fp32_modules` 的 DiT，`from_pretrained` 崩 → `from_config`+`load_state_dict` 回退
- #44/#46 cgroup 32GB → VAE/UMT5 加载编码全上 NPU（通用，非扩散专属）

## 实战数据点（跨芯片交叉验证）
- **Ascend950PR 单卡 128GB**（torch 2.12 + torch_npu 2.12）：DiT 1.419B，fp32+bf16 autocast+NpuFusedAdamW+grad-ckpt；latent `[1,16,6,60,104]`（21 帧 832×480）；50 步 45s(~0.9s/step)，loss 1.63→0.41，velocity MSE base 1.99→CPT 0.37（降 81%）。全程 NPU 算子通过。
- **Ascend910 单卡 64GB**（torch 2.8.0+cpu + torch_npu 2.8.0.post4 + transformers 5.15.1 + diffusers 0.40，cross-chip 回归验证）：同一 FastWan2.1-T2V-1.3B-Diffusers，MixKit 80 视频(16帧 256×448)；latent `[1,16,4,32,56]`；50 步 63s(~1.3s/step)，loss 21.14→8.77，velocity MSE 17.63→6.27（降 64%）。**910 暴露了 950PR 未遇的 3 个跨栈 gap**：#52(那次 latent 归一化乘反致 loss ~10×虚高)、#53(UMT5 用 T5EncoderModel 丢偏置)、#54(`from_pretrained`+keep_in_fp32 崩，950PR 的 transformers 版未触发)。说明：**技能每支持一种新芯片/版本栈都应在该栈实跑回归**（见 skill_change_verify_protocol）。
