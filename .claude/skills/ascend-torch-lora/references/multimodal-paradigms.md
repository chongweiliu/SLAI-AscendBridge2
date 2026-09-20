# 多模态与生成式三范式 LoRA（vlm / diffusion / video）

> 场景：目标是视觉语言模型（图像描述/视觉指令 SFT）、文生图扩散模型（UNet LoRA）或视频生成模型（DiT LoRA），
> 而不是纯文本 CausalLM 或 encoder 类模型。
> 实测：2026-09-11 台账第 4 大类 3 模型，**3/3 全部改善**——
> Qwen2.5-VL-3B-Instruct + flickr30k（vlm，ROUGE-L 0.531→0.593）；
> stable-diffusion-xl-base-1.0 + nouns（diffusion，固定噪声 MSE -47.6%）；
> Wan2.1-T2V-1.3B + ucf101 帧级（video，MSE -55.4%）。
> 训练器模板：`scripts/lora_train_multimodal.py.tmpl`、验证模板：`scripts/validate_multimodal.py.tmpl`
> （范式经 `PARADIGM` 环境变量切换）；完整配套（切分/sweep/报告）实战副本在 `lora-ws/.cat4/`。

## 三范式速查

| PARADIGM | 模型加载 | 数据格式（jsonl 每行） | LoRA 目标 | loss | 真实 batch | 验证主指标 |
|---|---|---|---|---|---|---|
| vlm | `<Model>ForConditionalGeneration`（Qwen2.5-VL 等）+ `AutoProcessor` | `{"image","caption"}`（验证档 `{"image","captions":[...]}`） | 语言侧 `q/k/v/o/gate/up/down_proj` 自动发现（视觉塔融合命名天然不命中→自动隔离） | assistant CE（**token-id 序列定位 assistant 头**掩 prompt，caption+`<\|im_end\|>` 计 loss） | 2（图像大） | 贪心生成描述 vs 全部 GT captions 取 **max ROUGE-L**/字符重叠 |
| diffusion | 组件直载：`UNet2DConditionModel` + `AutoencoderKL` + 双 CLIP 文本编码器（`variant="fp16"`） | `{"image","prompt"}` | peft **后缀匹配** `["to_q","to_k","to_v","to_out.0"]` 一行覆盖 UNet 全部 attention | DDPM epsilon MSE（`add_noise` 后预测噪声） | 4 | **固定噪声/时间步**（per-sample seed）下的逐样本 epsilon MSE |
| video | `WanTransformer3DModel` + `AutoencoderKLWan`（均 **shape-based key remap** 加载） | `{"image","label"}`（帧级） | 同 diffusion（DiT attention 后缀匹配） | DDPM noise MSE（**零文本嵌入**，见口径说明） | 4 | 同 diffusion（固定噪声 MSE） |

三范式共用文本范式同款训练骨架：NpuFusedAdamW + bf16 autocast + cosine/warmup + grad clip + loss.jsonl 逐步记录，
仅 `forward_loss` 按范式不同。200 步/1000 样本量级、单卡即可（3B vlm 2.3min、SDXL 2.4min、Wan 1.0min）。

## 共性模式（跨范式复用，这是本文件的核心）

1. **预编码缓存（sweep 成本杀手）**：VAE latents + 文本嵌入**一次编码、全量缓存**，cache key =
   `md5(DATA_FILE|PARADIGM)` 前 10 位，路径固定在 `<WS>/outputs/latent_cache_<key>.pt`（**不随 OUT_DIR 变**）——
   sweep 各超参组合与正式训练全部复用，不重复编码。SDXL 1024px VAE + 双编码器编码是耗时大头，缓存后 sweep
   5 组合 ≈ 纯 UNet 前反向时间。缓存 sample 存 CPU（`.cpu()`），训练时逐 batch `.to(DEV)`。
2. **LoRA 目标两种打法**：
   - vlm（HF transformers 模型）：与语言模型同款自动发现——扫 `named_modules()` 取 `nn.Linear` 末名 ∩ 候选
     `{q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}`。视觉塔的 qkv 融合命名（如 `qkv`/`fc1` 风格）
     不在语言侧候选内，**天然不被命中**→ 语言侧 LoRA 自动隔离，无需手排除。
   - diffusion/video（diffusers UNet/DiT）：peft 的 `target_modules` 是**后缀匹配**，
     `["to_q","to_k","to_v","to_out.0"]` 一行覆盖全部 attention 块（含 `to_out.0` 这种带序号末名）。
     peft 对 UNet/DiT 的 `get_peft_model`/`save_pretrained` 均正常（与 timm MAE 同结论）。
3. **固定随机性验证（base/LoRA 严格可比）**：diffusion/video 验证必须用 per-sample seed 的
   `torch.Generator().manual_seed(BASE_SEED+i)` 生成**同一份**噪声与时间步喂给 base 和 LoRA——
   与 mlm 固定掩码、mim 固定 MAE 内部随机掩码同一原则（pitfalls #40③）。不固定则双方噪声不同，MSE 差是噪声差。
4. **按【图】切分防泄漏**：一图多 caption 的数据集（flickr30k 每图 5 条）必须以**图**为切分单位
   （30% 图训练，held-out 图的全部 caption 留验），按 caption 行切会让同一图像同时出现在训练与验证。
5. **切分产物一律绝对路径**：训练器/验证器可能 cd 到别的目录，相对路径图片全打不开——vlm 会报
   FileNotFoundError（显性），diffusion/video 预编码 try/except 跳过坏样本后**静默 0 样本**（隐性，更危险）。
   见 pitfalls #41。

## vlm 范式细节（Qwen2.5-VL-3B 实测）

- **渲染**：`processor.apply_chat_template(msgs, tokenize=False)` 渲染 ChatML 字符串（user 轮含
  `{"type":"image"}` 占位），再 `processor(text=[rendered], images=[img], return_tensors="pt")` 展开 image token。
- **labels 用 token-id 序列定位 assistant 头**（不用字符偏移法）：processor 展开图像 token 后字符偏移与
  token 序列不再对齐，改为把 `<|im_start|>assistant\n` 单独 encode 成 id 序列，在 `input_ids` 里滑窗
  `torch.equal` 找**最后一次**出现位置，头之前全部 -100，caption+`<|im_end|>` 计 loss。自检 `n_label>0`。
- **批处理三要点**（pitfalls #43）：① `pixel_values` 用 **concat（cat dim=0）不是 stack**——每图 patch 数
  可以不同，stack 要求同形；② `image_grid_thw` 同样 cat；③ per-example 取 `pixel_values[0]` 是**首行 patch**
  （`[1176]`）不是整图（`[256,1176]`），错切后视觉塔报 `shape '[0,4,-1]'`，诊断特征=视觉 token 数异常小。
- **文本 padding**：input_ids/labels 右 pad（pad token 位 label=-100），attention_mask 对应置 0。
- **验证**：`add_generation_prompt=True` 渲染到 assistant 头，greedy `generate(max_new_tokens=64)`，
  切掉 prompt 前缀 decode；与 GT 的**全部** captions 逐一算 ROUGE-L 取 max（一图多参考的标准做法）。
- **实测**：flickr30k 30%（9304 图池，训 1000）200 步 lr=1e-4/r=16 → val CE 1.939→1.835（-5.4%），
  ROUGE-L 0.531→0.593（+0.062）。逐条肉眼可见描述更具体（"穿霓虹绿帽条纹衫滑轨" vs base 泛泛"表演特技"）。

## diffusion 范式细节（SDXL 实测）

- **组件直载不用 pipeline**：`UNet2DConditionModel.from_pretrained(f"{MODEL_DIR}/unet", variant="fp16")`
  （torch_dtype=float32 + variant=fp16 = 加载 fp16 权重重.cast，省显存）；VAE/双文本编码器同款，
  编码完即 `torch.npu.empty_cache()` 释放，训练只驻留 UNet。
- **SDXL 双编码器嵌入拼接**：`emb = cat([CLIPTextModel.last_hidden_state,
  CLIPTextModelWithProjection(output_hidden_states=True).hidden_states[-1]], dim=-1)`（77×2048），
  pooled = `text_embeds`；前向传 `added_cond_kwargs={"text_embeds": pooled, "time_ids": [1024,1024,0,0,1024,1024]}`。
- **训练步**：`noise=randn_like(latents)`；`ts=randint(0,1000)`；`noisy=DDPMScheduler.add_noise(latents,noise,ts)`；
  `pred=unet(noisy,ts,encoder_hidden_states=emb,added_cond_kwargs=...).sample`；`loss=MSE(pred.float(),noise.float())`。
  latents 编码时已乘 `vae.config.scaling_factor`。
- **实测**：nouns（NFT 图+prompt）30% 训 1000，200 步 lr=2e-4/r=32 → 固定噪声 MSE 0.100→0.052（**-47.6%**），
  10/10 样本全部改善（0.296→0.098 最大）。

## video 范式细节（Wan2.1-T2V-1.3B 实测）

- **shape-based key remap 加载**：ckpt（diffusers 0.30 命名）与本地 diffusers 0.40 的 `WanTransformer3DModel`/
  `AutoencoderKLWan` 键名不一致——按「shape 分桶 + 桶内 sorted 对位」remap 后 `load_state_dict(strict=False)`
  （与 CPT 批次同款实现，模板内置 `remap_load()`）。DiT 构造参数照抄模型卡（patch_size/num_layers/ffn_dim/
  qk_norm 等）。
- **VAE 输入维度序 `[B,C=3,T=1,H,W]`**：`permute(2,0,1).unsqueeze(0).unsqueeze(2)`——unsqueeze 顺序错会报
  `expected 3 channels got 1`（pitfalls #44）。单帧当 T=1 的视频编码。
- **零文本嵌入（口径限制，必须如实记录）**：T5 文本编码器 11.4GB，沿 CPT 先例跳过，
  `encoder_hidden_states=zeros(B,512,4096)`。评测口径因此是「**无条件帧去噪**」而非完整 T2V——
  base 在零文本条件下 MSE=2.22 **劣于零预测器（1.0）**，LoRA 使模型适应该条件设定（→0.99，接近噪声方差
  下界）。改善真实且可解释，但报告必须写明与完整 T2V 的差异，不得宣称"文生视频能力提升"。
- **加噪公式**（Wan 线性 beta 手写）：`alpha_t = 1 - 0.02*(t+1)/1000`；
  `noisy = sqrt(alpha_t)*latents + sqrt(1-alpha_t)*noise`。
- **实测**：ucf101 帧级 30% 训 1000，200 步 lr=1e-4/r=8 → MSE 2.224→0.993（**-55.4%**），10/10 全改善。

## 超参模式（sweep 60 步 × 5 组合，held-out val loss 判据）

- **diffusion**：域分布差异大（nouns NFT 画风 vs SDXL 预训练分布）→ 选激进组合 lr=2e-4/r=32（与文本知识注入同模式）。
- **vlm**：sweep 两组合 val loss 打平（1.8373 vs 1.8373）→ 按协议取**保守**组合 lr=1e-4/r=16。
  数据修复后（caption 字符串化列表解析错，#44）必须**确认复验**再正式训练。
- **video（任务饱和型）**：全组合 val loss 0.9965~1.0044 几乎打平，sweep 区分度低——「学无条件去噪」任务
  本身简单，取中位保守组合即可，不要为拉开差距加 lr。

## 数据切分（make_split4 模式，接 30%/held-out 协议）

- **parquet 内嵌图像字节**（nouns/ucf101）：`pq.read_table().to_pylist()` 取 `image.bytes` 落盘 jpg
  （文件名用 `分片名:行号` 防重），按**行**切分；**zip+CSV**（flickr30k）：按**图**切分（见共性模式 4）。
- 产物四件套：`train.jsonl`（≤1000）/ `probe_train.jsonl`（前 300，sweep 用）/ `val_loss.jsonl`（64）/
  `val10.jsonl`（10，终验；vlm 存全部 captions）+ `split_meta.json`（全部索引落盘可审计 + 不相交 assert）。
- 图像路径写**绝对路径**（#41）；训练样本写盘前洗牌（#34）。
- CSV 的 caption 列可能是**字符串化列表**（`'["caption", ...]'`）：`json.loads` 取首元素，
  只 `strip('"')` 洗不干净（#44）——残留 `["` 会污染训练文本与 sweep 判据。
