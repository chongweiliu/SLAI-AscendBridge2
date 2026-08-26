---
name: ascend-adaptation
description: 将 PyTorch 模型适配到 Ascend NPU。提供设备选择、环境配置、验证流程的模板与最佳实践。
---

# Ascend 适配 Skill

本 skill 指导你将 PyTorch 模型适配到华为 Ascend NPU。完整规则见 `.claude/agents/adapter.md`。

**边界**：若目标模型是 `diffusers` pipeline（如 `FLUX`、`Stable Diffusion`、`Wan`、`text-to-image`、`text-to-video`、`image-to-video`），优先切换到 `ascend-diffusers-adaptation`。本 skill 保持聚焦于通用 PyTorch / transformers / 自定义仓库适配。

## 核心流程

1. **环境配置**：确保 `pyproject.toml` 正确配置（参考 `uv-env-setup` skill）。**必须**执行 `uv sync --extra ascend` 或 `uv sync --extra cuda` 之一。
2. **代码适配**：使用 `demo.py.j2` 模板生成 `demo.py`。
3. **验证**：Dry Run 跑通，`check_adaptation.py` 通过。

## 1. 模板与设备检测

使用 Jinja2 模板 `.claude/skills/ascend-adaptation/templates/demo.py.j2` 生成 `demo.py`，模板变量为 `{{ model_id }}`。

```python
# 设备检测（模板内）：NPU > CUDA > CPU
def get_device():
    """Returns (device_str, device_count)."""
```

**验收**：NPU 或 CUDA 均可，不允许 CPU 回退。

## 2. Dry Run 策略（快速验证）

**目的**：不下载权重，仅验证架构与代码路径在 NPU/CUDA 上的兼容性。

**实现**：

- `from_config(config)` 创建随机权重模型；
- `shrink_config_for_dry_run(config)` 保守缩小（仅层数减为 2）；
- `infer_auto_device_map(model)` + `dispatch_model(model, device_map)` 分布到多卡（需 accelerate）。

```bash
uv run python demo.py --dry-run
```

**验证失败情况**（任一即不通过）：

- 输出出现 `using CPU` 或 `No accelerator detected`
- 未加载模型、未执行推理、运行报错或退出码非 0

## 3. Full Run（加载真实权重）

- `from_pretrained(..., device_map="auto", cache_dir=CACHE_DIR)`；
- 模型与 tokenizer 缓存到 `adaptations/<name>/models/`。

## 4. 保存输出与验证

```bash
uv run python demo.py > output.txt 2>&1
```

适配完成后运行 `adaptation/scripts/check_adaptation.py --adapt "{adapt_name}"` 做完整验证（adapt_name 由 model_id 经 model_id_to_safe_name 得到；可用 `--skip-status` 跳过 .status.json）。

## 5. 依赖与配置

- **必需**：`accelerate`、`torch`、`transformers`、`safetensors`
- **generate**：`do_sample=False` 避免 NPU 上 ArgSort 的 AiCpu 回退警告

## 6. 断言与校验

模板内置断言：设备为 NPU 或 CUDA；模型首层在 NPU/CUDA 上；`generate()` 输出非空。

## 7. 非 LLM 模型

若适配非因果语言模型，需在生成后的 `demo.py` 中修改：

- **序列分类**：`AutoModelForCausalLM` → `AutoModelForSequenceClassification`
- **视觉**：使用 `AutoImageProcessor` + `AutoModelForImageClassification`
- **Pipeline**：若文档推荐 `pipeline`，优先 `pipeline(..., device=device)`

## 8. 其他优化与避坑（NPU 性能/精度共性经验）

### 8.1 FlashAttention：CANN 版本决定可用性

- `F.scaled_dot_product_attention` 在 NPU 上映射到 `aclnnFlashAttentionScore`，但**该算子的编译内核(.o)只在 CANN 8.5.0+ 才有**；8.3.RC1 及更早只有头文件、无内核，所有 head 配置都报 `Cannot find binary for op FlashAttentionScore`。
- **先确认 CANN 版本**（`cat /usr/local/Ascend/cann-9.0.0/version.cfg` 或 `ls /usr/local/Ascend/`），不是 9.0.0/8.5.0 就先切环境（见 uv-env-setup skill 1.5 节）。
- FA **不支持 fp32**：若模型里 RMSNorm/QKNorm 等用 `.float()` 把 q/k 转回 fp32，bf16 autocast 也救不了 SDPA。需在 SDPA 前显式 `q=q.to(torch.bfloat16)` 等。
- 回退方案：手动 `matmul` attention（`softmax(QKᵀ·scale)·V`）数学等价 SDPA，但会物化 L×L 矩阵、慢且耗内存——只在没有 FA 内核时临时用。

### 8.2 ⚠️ float64 是 NPU 上的隐藏杀手

- **Ascend NPU 无原生 fp64**，`torch.arange(dtype=torch.float64)` / `cos/sin` 等 fp64 算子走**极慢模拟回退**，可让单步耗时暴增几十倍。
- 高发场景：**RoPE**（很多实现用 `dtype=torch.float64` 求高精度频率）、某些 norm/位置编码。
- CONFLUX 实测：RoPE fp64→fp32，采样 **56s→1.04s/50步（54× 加速），输出 std 完全相同**（fp32 与 fp64 差异在 bf16 量化之下）。
- **规则**：NPU 上凡 `float64` 一律先转 `float32`（医学影像/生成模型 bf16 推理不需要 fp64 精度）。grep 模型代码 `dtype=torch.float64\|\.double()\|float64` 全部改 float32。

### 8.3 性能 profiling 的 sync 假象——用 A/B 对照定位真瓶颈

- 在某算子里插 `torch.npu.synchronize()` 计时，会**等待前面所有排队的异步算子**，把整段开销都记到该算子头上 → 误判瓶颈。
- 正确做法：(a) 隔离计时每个算子要 `warmup + N 轮 + 末尾单次 sync`；(b) **定位瓶颈用 A/B 对照**：改一个变量（如 RoPE 的 dtype），同 seed 同权重跑两次比耗时——差异即该变量的真实影响。
- 隔离 matmul 计时可达近峰值（240 TFLOPS），但链式 block 执行可能慢几百倍——差距来自 dispatch/fp32 回退，不是单算子慢。

### 8.4 TorchAir 图模式（torch.compile）的限制

- 自定义模型整模 `torch.compile(backend=torchair.get_npu_backend(...))` 常失败：
  - **SDPA 缺 Converter**：图里 `npu.npu_fusion_attention_v3` 在 torch_npu 2.10.0 未注册 AscendIR 转换器 → 编译报错。手动 matmul attention 只用标准 ATen 算子（有 Converter），能入图。
  - meshgrid / 动态 padding / 自定义算子常导致 GE `EZ9999 Inner Error / Compile graph failed`。
- Skill 经验：对自定义非标准模型，**不承诺整图成功**；先最小 compile probe，失败则保留 Eager。图模式适合标准 transformers/LLM，自定义 DiT/VAE 等优先靠 8.2 的 fp64→fp32 等低风险优化。

### 8.5 通用验证与调优方法论（模型无关）

适配/优化后建议按以下范式验证，适用于所有 NPU 模型：

1. **精度门 = 确定性复现**：同 seed 跑两次，`max|diff|` 应为 0（或 < 1e-3）。NPU 推理应可复现；不可复现说明有非确定性算子（如某些 atomic 算子）需排查。这是精度可信的前提。
2. **优化前后 A/B 对照**（不是单次计时）：改一变量（dtype/算子/配置），同 seed 同权重跑两次比耗时与输出差异。`max|diff| < 容差` 才算"精度无损"。CONFLUX 的 fp64→fp32 就靠此法证明 54× 加速且输出 std 完全一致。
3. **真/模拟双样本验证**：从数据集抽真实样本、用其标签作条件生成、对比生成体与真实体统计（std/mean/air/值域）+ 随机构造模拟样本验证泛化。比"只跑一个固定 prompt"更全面。
4. **warmup 后再计时**：首个样本含编译/初始化/缓存预热开销（CONFLUX 首样本 3.8s vs 稳态 1.5s）。基准测试丢弃前 2-3 次再取均值。
5. **bf16 解码降显存**：VAE/diffusion decoder 的全分辨率 conv 是 HBM 峰值来源，bf16 解码省一半显存且精度无损（CONFLUX VAE 解码 fp32 12.5s→bf16 2.8s，且 std 不变）。
6. **bulk 下载前估总量**：下 1 个样本测大小 × 总数估总规模，避免 TB 级数据盲目全下（CONFLUX 数据集 200K 样本 ≈ 1.68TB，按需取子集）。

## 9. 自定义模型库（Custom Repo Models）

**触发条件**：`get_model_info.py` 返回 `transformers_info == {}` 且 `model_type == "Custom"`（可选：额外获取 README 检查 `git clone`）。

**流程**（详细规则见 adapter.md 2.7）：

1. **获取仓库 URL**：从 HF README/model card 提取 `git clone https://github.com/xxx/yyy.git`
2. **克隆**：`git clone --depth 1` 到 `adaptations/{name}/<repo-name>/`
3. **安装**：使用 `uv add --editable ./<repo-name>/packages/<pkg>` 或路径依赖，**禁止**在 demo.py 中 `pip install -e`
4. **替换 CUDA**：在克隆的源码中将 `torch.cuda.*` 替换为设备无关版本（见 adapter.md 2.7.4）
5. **demo.py**：从本地安装的包导入（如 `from ltx_core.*`），不使用 `transformers.AutoModel`
6. **README**：记录「自定义模型库」及 `repo_name`、`packages` 列表，便于 benchmark-runner、npu-optimizer 识别
