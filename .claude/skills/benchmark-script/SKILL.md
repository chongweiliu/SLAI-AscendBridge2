---
name: benchmark-script
description: 模型评测脚本生成与执行。生成 accuracy_run.py，产出 outputs_*.pt、benchmark_metrics_*.json、trace_*.json。
---

# Benchmark Script Skill

本 skill 指导你生成并执行模型评测脚本。完整规则见 `.claude/agents/benchmark-runner.md`。

## 核心流程（两步执行）

评测脚本采用 **两步流程**：

| 步骤 | 输入 | 输出 | 目的 |
|------|------|------|------|
| **Step 1** | 单样本 | `trace_*.json` + `benchmark_metrics_*.json` | 性能分析 + Profiler trace |
| **Step 2** | 全样本 | `outputs_*.pt` | 精度测试（可对比 CUDA/NPU） |

**为什么分两步？**

- Step 1 开启 Profiler，会有性能开销，仅用于单样本分析
- Step 2 关闭 Profiler，获得真实的全样本推理精度

## 1. 模板使用

使用 Jinja2 模板 `.claude/skills/benchmark-script/templates/accuracy_run.py.j2` 生成 `accuracy_run.py`。

### 1.1 模板变量

| 变量 | 必须 | 说明 |
|------|------|------|
| `{{ model_id }}` | ✅ | 模型 ID，如 `Qwen/Qwen2.5-1.5B-Instruct` |
| `{{ safe_name }}` | ❌ | 安全名称，用于注释 |
| `{{ model_type }}` | ❌ | 模型类型（见下表），默认 `causal_lm` |
| `{{ dataset_key }}` | ❌ | 推荐数据集 key，默认 `wikitext` |

### 1.2 模板直接支持的标准模型类型

| model_type | AutoModel 类 | 输出类型 | 推荐数据集 |
|------------|-------------|---------|-----------|
| `causal_lm` | `AutoModelForCausalLM` | generated_text | wikitext |
| `bert` | `AutoModel` | cls_embeddings | wikitext |
| `vision_classification` | `AutoModelForImageClassification` | class_labels | cifar100 |
| `asr` | `AutoModelForSpeechSeq2Seq` | transcriptions | librispeech |
| `seq2seq` | `AutoModelForSeq2SeqLM` | generated_text | cnn_dailymail |

**边界说明**：

- 本模板面向标准 `transformers` benchmark 路线
- 若 `scripts/dataset_mapping.py` 返回 `model_type=diffusion` 或 `model_type=video`，**不要**继续套本模板；改用 `.claude/skills/ascend-diffusers-benchmark/SKILL.md`

### 1.3 生成流程

1. **确定模型类型**：调用 `scripts/dataset_mapping.py` 获取

   ```bash
   uv run python scripts/dataset_mapping.py --model_id "Qwen/Qwen2-0.5B" --json
   # 输出: {"model_id": "Qwen/Qwen2-0.5B", "model_type": "causal_lm", "dataset_key": "wikitext"}
   ```

2. **读取模板**：`.claude/skills/benchmark-script/templates/accuracy_run.py.j2`

3. **替换变量**：使用 Jinja2 渲染

4. **写入目标**：`adaptations/{safe_name}/accuracy_run.py`

### 1.4 生成命令示例

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader(".claude/skills/benchmark-script/templates"))
template = env.get_template("accuracy_run.py.j2")

script_content = template.render(
    model_id="Qwen/Qwen2-0.5B",
    safe_name="qwen_qwen2-0_5b",
    model_type="causal_lm",
    dataset_key="wikitext"
)

# 写入 adaptations/qwen_qwen2-0_5b/accuracy_run.py
```

## 2. 产出文件

### 2.1 文件命名规范

所有产出文件使用统一命名格式：

```
{type}_{device}_{dtype}_{mode}_{dataset}.{ext}
```

| 字段 | 说明 | 示例 |
|------|------|------|
| `type` | 文件类型 | `trace`, `benchmark_metrics`, `outputs` |
| `device` | 设备类型 | `npu`, `cuda` |
| `dtype` | 数据精度 | `fp32`, `fp16`, `bf16` |
| `mode` | 加载模式 | `config` (Tier1), `pretrained` (Tier2) |
| `dataset` | 数据集名称 | `wikitext`, `cifar100`, `librispeech`, `random` |

**必须动态计算**（禁止硬编码）：

- `mode`：必须反映**实际执行模式**。当脚本不存在 fallback 时，可等价写为 `mode_str = "pretrained" if args.use_pretrained else "config"`
- `dtype`：根据**模型实际加载的 dtype** 决定，`dtype_str = get_dtype_str(next(model.parameters()).dtype)`，映射 `torch.float32`→`fp32`、`torch.float16`→`fp16`、`torch.bfloat16`→`bf16`；有 `--dtype` 时可用 `args.dtype`

**示例**：

- `outputs_npu_bf16_config_wikitext.pt`
- `benchmark_metrics_cuda_fp16_pretrained_cifar100.json`
- `trace_npu_fp32_config_librispeech.json`

### 2.2 必需文件

| 文件 | 必须 | 说明 |
|------|------|------|
| `accuracy_run.py` | ✅ | 评测脚本 |
| `outputs_*.pt` | ✅ | 模型输出（字典格式，见 2.3 节） |
| `benchmark_metrics_*.json` | ✅ | 性能指标（含 output_type 字段） |
| `trace_*.json` | ✅ | PyTorch Profiler trace 文件 |

### 2.3 outputs_*.pt 格式（生成式模型）

对于生成式模型（CausalLM、Seq2Seq），outputs_*.pt 包含字典：

```python
{
  "generated_text": ["文本1", "文本2", ...],           # 解码后的文本列表
  "logits": [tensor([vocab_size]), ...],             # 每个 prompt 最后一个 token 的 logits
  "perplexity": [15.23, 12.45, ...]                  # 每个样本的困惑度值
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `generated_text` | `List[str]` | 解码后的文本列表 |
| `logits` | `List[Tensor]` | 每个 prompt 最后一个 token 的 logits 张量（用于算子精度验证） |
| `perplexity` | `List[float]` | 每个样本的困惑度值（PPL = exp(cross_entropy_loss)） |

**ASR 模型**：outputs_*.pt 包含 `generated_text`（转录文本）和 `logits`（最后一个 token 的 logits，用于 CUDA/NPU 算子精度对比），无 perplexity

**非生成式模型**（BERT、Vision）：
- BERT: `List[Tensor]` - [CLS] token 向量列表
- Vision: `List[str]` - 类别标签列表

### 2.4 benchmark_metrics 格式

```json
{
  "start_time": "2026-02-25T20:45:30.123456",
  "end_time": "2026-02-25T20:50:31.234567",
  "latency_s": 0.63,
  "peak_memory_mb": 970.34,
  "num_samples": 250,
  "device": "npu:0",
  "device_model": "Ascend910B1",
  "mode": "config",
  "output_type": "generated_text",
  "ttft_ms": 15.234,
  "tpot_ms": 8.567,
  "packages": {
    "torch": "2.1.0",
    "transformers": "4.36.0",
    "torch_npu": "2.1.0"
  }
}
```

- `start_time`: 评测开始时间（ISO 8601 格式）
- `end_time`: 脚本结束时间（ISO 8601 格式，包含 Step1 + Step2 完整时间）
- `latency_s`: 当前 benchmark 结果对应工作负载的平均推理延迟（秒）
- `num_samples`: 本次 benchmark 实际参与统计的样本数；completed 记录要求 `num_samples >= 50`，默认应与 `--max-samples` 保持一致（通常为 250）
- `peak_memory_mb`: 峰值内存使用（MB），CPU 模式下为 0
- `ttft_ms`: 平均首字延迟（毫秒），仅生成式模型有值（CausalLM/Seq2Seq/ASR），基于 Step 2 全样本计算
- `tpot_ms`: 平均每 token 生成时间（毫秒），仅生成式模型有值，基于 Step 2 全样本计算
- `output_type` 可能的值：
  - `generated_text`: CausalLM/Seq2Seq 解码后文本
  - `cls_embeddings`: BERT [CLS] token 向量
  - `class_labels`: Vision 类别标签
  - `transcriptions`: ASR 转录文本（mixed 格式下用 `generated_text` 存储，与 CausalLM 一致）

**注意**：`ttft_ms` 和 `tpot_ms` 对于非生成式模型（BERT、Vision）为 `null`

## 3. 依赖管理（R2）

生成 `accuracy_run.py` 前，**必须检查并更新** `pyproject.toml`：

**必需依赖**：

```toml
dependencies = [
    # ... 原有依赖 ...
    "datasets>=2.14",
    "numpy>=1.24",
]
```

**Vision 模型额外依赖**：

```toml
"Pillow>=9.0",  # 图像处理
```

**ASR 模型额外依赖**：

```toml
"librosa>=0.10",  # 音频加载（torchcodec 不支持 aarch64）
```

**更新流程**：

```bash
# 1. 检查
grep -E "datasets|numpy|librosa" adaptations/{sanitized_name}/pyproject.toml

# 2. 若缺失，使用 Edit 工具添加

# 3. 同步
cd adaptations/{sanitized_name} && uv sync --extra ascend
```

## 4. 样本数限制（R1）

**所有模型类型评测样本数统一限制为 250 个**，避免产出文件过大。

```python
parser.add_argument("--max-samples", type=int, default=250)
texts = texts[:args.max_samples]
```

**产出文件大小预估**（250 样本）：

| 模型类型 | 典型产出大小 |
|---------|-------------|
| CausalLM | ~100-500KB |
| BERT | ~2-5MB |
| Vision | ~10-50KB |
| ASR | ~50-200KB |

## 5. 内存优化策略

### 5.1 立即移动到 CPU

每个样本推理完成后，**立即**将输出移动到 CPU：

```python
all_outputs.append(output_tensor.cpu())
del out, output_tensor, inputs
```

### 5.2 定期清理缓存

每 32 个样本清理 NPU/CUDA 缓存：

```python
if (i + 1) % 32 == 0:
    if device_short == "npu":
        torch.npu.empty_cache()
    elif device_short == "cuda":
        torch.cuda.empty_cache()
```

## 6. 动态 Profiler 选择

根据硬件动态选择原生 Profiler：

### 6.1 NPU Profiler

```python
def get_profiler_context(device_short):
    if device_short == "npu":
        from torch_npu.profiler import ProfilerActivity as NPUActivity
        from torch_npu.profiler import profile as npu_profile

        activities = [NPUActivity.CPU, NPUActivity.NPU]
        return npu_profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True)
```

### 6.2 CUDA Profiler

```python
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device_short == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        return torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True)
```

## 7. 执行方式

**Tier1（随机权重，快速验证）**：

```bash
cd adaptations/{sanitized_model_name}
uv run python accuracy_run.py
```

**Tier2（真实权重）**：

```bash
cd adaptations/{sanitized_model_name}
uv run python accuracy_run.py --use-pretrained
```

**CPU 模式（强制 CPU 推理）**：

```bash
cd adaptations/{sanitized_model_name}
uv run python accuracy_run.py --cpu
```

> **注意**：使用 `--cpu` 时，`device` 将为 `cpu`，`peak_memory_mb` 为 0。此模式用于无 NPU/CUDA 环境下的基本验证。

**命令行参数**：

| 参数 | 说明 |
|------|------|
| `--use-pretrained` | 加载预训练权重（Tier2） |
| `--max-samples` | 最大样本数（默认 250） |
| `--cpu` | 强制 CPU 推理（禁用 NPU/CUDA） |

## 8. 验收标准

- `device` 字段以 `npu`、`cuda` 或 `cpu` 开头
- `benchmark_metrics_*.json` 包含 `latency_s`、`peak_memory_mb`、`device`、`output_type`、`ttft_ms`、`tpot_ms`、`end_time`
- `outputs_*.pt` 文件存在且可加载
- `trace_*.json` 文件存在
- `end_time` 应为脚本结束时间（晚于 `start_time`）

### 8.0 outputs_*.pt 格式验证（生成式模型）

对于生成式模型（CausalLM、Seq2Seq），验证 outputs_*.pt 包含正确的字典格式：

```python
import torch

# 加载 outputs
outputs = torch.load("outputs_*.pt")

# 验证是字典格式
assert isinstance(outputs, dict), "outputs 应为字典格式"

# 验证必需字段
assert "generated_text" in outputs, "缺少 generated_text 字段"
assert "logits" in outputs, "缺少 logits 字段"
assert "perplexity" in outputs, "缺少 perplexity 字段"

# 验证字段类型
assert isinstance(outputs["generated_text"], list), "generated_text 应为列表"
assert isinstance(outputs["logits"], list), "logits 应为列表"
assert isinstance(outputs["perplexity"], list), "perplexity 应为列表"

print("✅ outputs 格式验证通过")
```

### 8.1 输出一致性验证（R3）

**目标**: 确保相同输入下多次运行输出完全一致（确定性推理）。

**验证方法**:

```bash
# 1. 运行第一次
uv run python accuracy_run.py --use-pretrained --max-samples 10
# 复制结果
cp outputs_npu_*.pt outputs_cuda_*.pt

# 2. 运行第二次
rm outputs_npu_*.pt benchmark_metrics_npu_*.json trace_npu_*.json
uv run python accuracy_run.py --use-pretrained --max-samples 10

# 3. 对比 MD5
md5sum outputs_cuda_*.pt outputs_npu_*.pt
# 必须完全一致！

# 4. 使用 benchmark_tool.py 验证
uv run python benchmark/scripts/benchmark_tool.py compare outputs_cuda_*.pt outputs_npu_*.pt
# match_rate 必须为 100%
```

**验收标准**:

| 指标 | 要求 |
|------|------|
| MD5 校验 | 两次运行 MD5 完全相同 |
| match_rate | 100%（generated_text / class_labels） |
| cosine_similarity | 1.0（cls_embeddings / logits） |
| max_abs_error | 0.0（cls_embeddings / logits） |

**常见不一致原因**:

1. 缺少 `torch.use_deterministic_algorithms(True, warn_only=True)`
2. 数据集加载顺序未使用 `sorted()`
3. 生成模型未设置 `do_sample=False`
4. 随机图像/音频生成未设置种子

## 9. 非模板路线模型类型

对于模板未覆盖的模型类型，需手动改写 `accuracy_run.py`。

### 9.1 支持的模型类型

| model_type | AutoModel 类 | 输出类型 | 推荐数据集 |
|------------|-------------|---------|-----------|
| `causal_lm` | `AutoModelForCausalLM` | generated_text | wikitext |
| `bert` | `AutoModel` | cls_embeddings | wikitext |
| `vision_classification` | `AutoModelForImageClassification` | class_labels | cifar100 |
| `asr` | `AutoModelForSpeechSeq2Seq` | transcriptions | librispeech |
| `seq2seq` | `AutoModelForSeq2SeqLM` | generated_text | cnn_dailymail |

### 9.2 不走本模板的模型类型

以下模型类型不应直接套用本模板：

| model_type | 说明 | 推荐数据集 |
|------------|------|-----------|
| `vlm` | 视觉语言模型 (LLaVA, Qwen-VL, InternVL) | scienceqa |
| `diffusion` | 改用 `ascend-diffusers-benchmark`，按 diffusers 手写 benchmark 路线处理 | builtin / latency-only |
| `tts` | 语音合成 (XTTS, SpeechT5) | None |
| `video` | 改用 `ascend-diffusers-benchmark`，按 video benchmark 路线处理 | builtin / latency-only |
| `token_classification` | NER/Token 分类 | conll2003 |
| `embedding` | 语义嵌入 (BGE, E5, GTE) | wikitext |
| `reranker` | 检索排序 | ms_marco |
| `vision_detection` | 目标检测 (YOLO, DETR) | coco |
| `audio_embedding` | 音频嵌入 (CLAP, WavLM) | librispeech |
| `biomedical_nlp` | 生物医学 NLP | pubmed_qa |
| `specialized` | 默认回退 | wikitext |

### 9.3 改写时必须保留

对于 `diffusion` / `video`，优先参考 `.claude/skills/ascend-diffusers-benchmark/SKILL.md`，不要再从本模板逆向删改。

- 随机数固定（`torch.manual_seed(42)`）
- **确定性算法**（`torch.use_deterministic_algorithms(True, warn_only=True)`）
- **数据集顺序固定**（`sorted()`）
- 设备检测（NPU > CUDA > CPU）
- 性能监控（内联 `_PerfMonitor`）
- Trace 导出（`prof.export_chrome_trace()`）
- 两步流程（Step1: trace/metrics, Step2: outputs）
- 内存优化（`.cpu()` + `empty_cache()`）
- 统一输出文件名（`outputs_*.pt`）
- metrics 中记录 `output_type`
- **生成参数显式设置**（`do_sample=False, temperature=1.0, top_p=1.0, top_k=50`）

### 9.4 手动编写规范（必须遵守）

| 项目 | 必须 | 禁止 |
|------|------|------|
| CACHE_DIR | `(Path(__file__).resolve().parent / "models").as_posix()` | `"./models"` 或任何相对路径 |
| use_pretrained | `setup_model(use_pretrained, ...)` 根据 args 分支 | 定义参数但不参与模型加载 |
| 数据集加载 | 优先 `load_from_disk(DATASET_DIR / "xxx")` | `load_dataset(..., trust_remote_code=True)` |
| 输出命名 | `{type}_{device}_{dtype}_{mode}_{dataset}.{ext}` | 缺少 dataset 后缀或与实际不符 |
| mode/dtype | 动态 `mode_str`、`dtype_str`（dtype 来自 `next(model.parameters()).dtype`，mode 必须反映真实执行模式） | 硬编码 `config`、`fp32`、`fp16`；dtype 按设备推断；把请求 pretrained 误记成实际 pretrained |
| max_samples | `default=250` | 默认值 10 或其他 |
| 函数命名 | `load_benchmark_texts`、`load_benchmark_images` | `load_dataset`（与 datasets 库冲突） |
| shrink 函数 | **严禁** | `shrink_config_for_dry_run` 等任何 shrink 函数 |
| 模型设备 | `model = model.to(device)`，模型必须在 device 上推理 | config 分支中 `model = model.cpu()` |
| pretrained 失败处理 | 立即报错退出，并上报失败 | `except` 后 silent fallback 到 config 继续产出结果 |

### 9.5 常见错误与避免

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| `cache_dir = "./models"` | 依赖 cwd，benchmark-runner 从 adaptation 目录运行时路径错误 | `CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()` |
| `--use-pretrained` 未参与加载 | Tier1/Tier2 无法区分，始终加载权重 | `if use_pretrained: from_pretrained(...) else: from_config(...)` |
| `from_pretrained(...)` 失败后 fallback 到 config | 产出“名义 pretrained、实际 config”的伪 benchmark 结果 | 直接抛错并标记 benchmark 失败，修复后重跑 |
| `load_dataset(..., trust_remote_code=True)` | HF 2.16+ 自定义脚本数据集报错 "Dataset scripts are no longer supported" | 优先 `load_from_disk` 加载项目级 `datasets/` |
| 输出文件 `trace_npu_0_fp32_config.json`（无 dataset） | 不符合命名规范，聚合工具无法正确识别 | `trace_npu_0_fp32_config_wikitext.json` |
| dataset 后缀与实际不符（如用 wikitext 却写 `_sst2`） | 误导聚合与对比 | 使用实际加载的 dataset_name |
| `len(texts)==0` 未检查 | IndexError 当数据集为空 | `text = texts[0] if texts else "fallback"` |
| 自定义函数 `def load_dataset(...)` | 与 `datasets.load_dataset` 同名，易混淆 | `def load_benchmark_texts()` 等 |
| 硬编码 `_config_` 或 `_fp32_` | 输出文件名不符合 mode/dtype 动态规范 | 使用 `mode_str`、`dtype_str` 变量 |
| `dtype_str` 按设备推断（`device.startswith("npu")` 等） | 与实际模型 dtype 可能不符 | `dtype_str = get_dtype_str(next(model.parameters()).dtype)` |
| 使用 `shrink_config_for_dry_run` 等 shrink 函数 | accuracy_run.py 严禁使用 shrink | 直接 `from_config(config)`，不修改 config |
| config 分支中 `model = model.cpu()` | 模型必须在 device（NPU/CUDA）上推理 | `model = model.to(device)` |
| `init_empty_weights` 创建后丢弃再 `from_config` | 前者无效、后者全量加载易 OOM | 直接 `from_config` + `model.to(device)` |

### 9.6 强制检查脚本（必须执行）

**规则强制**：生成或修改 `accuracy_run.py` 后，**必须**运行 `benchmark/scripts/check_accuracy_run.py` 校验，**禁止跳过**。违规时 `exit 1`，必须修复后重新检查直至通过。

```bash
uv run python benchmark/scripts/check_accuracy_run.py              # 检查全部，违规 exit 1
uv run python benchmark/scripts/check_accuracy_run.py --adapt xxx  # 仅检查指定 adaptation
```

校验项：get_dtype_str、dtype 来自模型、无 shrink、mode_str 正确、max_samples=250、CACHE_DIR、load_benchmark_* 命名、禁止 model.cpu()（模型必须在 device 上）等。

## 10. 模板特殊处理

### 10.1 随机数据生成降级逻辑

模板为每种模型类型实现了数据集加载的降级策略：

| 模型类型 | 首选数据集 | 降级策略 |
|---------|-----------|---------|
| CausalLM/BERT/Seq2Seq | wikitext | cnn_dailymail → imdb → 内置文本 |
| Vision | cifar100 | 生成随机图像 (224×224 RGB) |
| ASR | librispeech | 生成随机音频 (1s @ 16kHz) |

**降级原因**：

- 数据集未下载或路径不匹配时自动降级

- 确保评测脚本可在无外部数据时运行

### 10.2 ASR attention_mask 特殊处理

ASR 模型（如 Whisper）需要显式设置 `attention_mask` 以避免 pad_token 警告：

```python
# 显式设置 attention_mask 避免 pad_token 警告
if "attention_mask" not in inputs:
    inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0], dtype=torch.long)
```

### 10.3 ASR librosa 延迟导入设计

由于 `librosa` 依赖较重且 `torchcodec` 不支持 aarch64 架构，模板使用**延迟导入**：

```python
def load_benchmark_audio():
    # 仅在需要时导入 librosa
    import librosa

    # 使用 librosa 从文件加载音频
    audio, sr = librosa.load(f, sr=16000, mono=True)
```

**设计原因**：

- 避免非 ASR 模型加载不必要的依赖

- 兼容 aarch64 架构（torchcodec 不支持）
- 使用模糊匹配查找音频文件（`*.flac`, `*.wav`）

### 10.4 ASR max_length 与 max_new_tokens 冲突处理

ASR 模型使用 `generate()` 时，需要显式禁用 `max_length` 避免与 `max_new_tokens` 冲突：

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    max_length=None,  # 显式禁用 max_length 避免警告
)
```

### 10.5 ASR logits 提取（用于 CUDA/NPU 算子精度对比）

ASR 模型（encoder-decoder 架构）通过二次 forward 提取 logits：

1. `generate()` 获取 `generated_ids`
2. `decoder_input_ids = generated_ids[:, :-1]`（去掉最后一个 token）
3. 调用 `model(**inputs, decoder_input_ids=decoder_input_ids)`，取 `outputs.logits[0, -1, :]` 作为最后一个 token 的 logits
4. 输出格式与 CausalLM 一致：`{"generated_text": [...], "logits": [...]}`，便于 `benchmark_tool compare` 做 CUDA/NPU 输出对比
