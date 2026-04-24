# Benchmark-Runner: accuracy_run.py 模式与反模式

本文档记录 accuracy_run.py 编写时的反模式（禁止）与正确示例，供 benchmark-runner 在生成或手动改写脚本时参考。

## MoE / 大模型 benchmark 策略

**Qwen3-Coder-Next-Base (512 experts MoE)**:
- flash-linear-attention 未安装时退回纯 PyTorch 实现，极慢
- 单样本 forward ~112s (NPU x4, 41GB peak), generate ~18s/sample (max_new_tokens=8, truncation=128)
- 50 samples (含前10个PPL计算) 总耗时 ~10min (Step1 ~4min + Step2 ~7min)
- 优化: 使用 `generate(return_dict_in_generate=True, output_scores=True)` 合并 forward+generate
- 仅前10个样本做额外 forward 计算 PPL
- 使用 `local_files_only=True` 避免 HuggingFace 在线访问

## 反模式清单（严禁）

| 反模式 | 问题 | 正确做法 |
|--------|------|----------|
| `cache_dir = "./models"` | 相对路径依赖 cwd | `CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()` |
| 定义 `--use-pretrained` 但不分支 | Tier1/Tier2 无法区分 | `if use_pretrained: from_pretrained(...) else: from_config(...)` |
| `load_dataset(..., trust_remote_code=True)` | HF 2.16+ 自定义脚本数据集已弃用 | `load_from_disk(DATASET_DIR / "xxx")` |
| 输出文件无 dataset 后缀 | 不符合命名规范 | `trace_npu_0_fp32_config_wikitext.json` |
| dataset 后缀与实际不符 | 误导聚合 | 使用实际加载的 dataset_name |
| 硬编码 `_config_` 或 `_fp32_` | mode/dtype 不符实际 | 使用 `mode_str`、`dtype_str` 动态 |
| `dtype_str` 按设备推断（`device.startswith("npu")` 等） | 与实际模型 dtype 可能不符 | `dtype_str = get_dtype_str(next(model.parameters()).dtype)` |
| `max_samples` 默认 10 | 与 R1 规则冲突 | `default=250` |
| `def load_dataset(...)` | 与 datasets 库冲突 | `def load_benchmark_texts()` |
| 未检查 `len(texts)==0` | 空数据集 IndexError | `text = texts[0] if texts else "fallback"` |
| **shrink 函数**（如 `shrink_config_for_dry_run`） | **严禁** | 直接 `from_config(config)`，不修改 config |
| **config 分支中 `model = model.cpu()`** | 模型必须在 device（NPU/CUDA）上推理 | `model = model.to(device)` |
| **init_empty_weights 创建后丢弃再 from_config** | 前者无效、后者全量加载 | 直接 `from_config` + `model.to(device)` |

## 正确示例

### CACHE_DIR

```python
from pathlib import Path

CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()
```

### setup_model 与 use_pretrained 分支

```python
def setup_model(use_pretrained: bool, device, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=cache_dir)

    if use_pretrained:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto", cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer
```

### 数据集加载（优先 load_from_disk）

```python
from datasets import load_from_disk

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

def load_benchmark_texts() -> tuple[list[str], str]:
    wikitext_path = DATASET_DIR / "wikitext___wikitext-2-raw-v1"
    if wikitext_path.exists():
        ds = load_from_disk(str(wikitext_path))
        texts = sorted([s["text"] for s in ds if s.get("text", "").strip()])
        return texts, "wikitext"
    # fallback
    return ["Hello, benchmark."], "builtin"
```

### 输出文件命名（mode/dtype 必须动态）

```python
def get_dtype_str(dtype: torch.dtype) -> str:
    """Convert torch dtype to short string (fp32, fp16, bf16)."""
    dtype_map = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    return dtype_map.get(dtype, "fp32")

adapt_dir = Path(__file__).resolve().parent  # 与 CACHE_DIR 同目录
device_short = device.replace(":", "_")  # npu:0 -> npu_0
dataset_name = "wikitext"  # 与实际加载一致
mode_str = "pretrained" if args.use_pretrained else "config"
# dtype_str 必须根据模型实际加载的 dtype，禁止按设备推断
dtype_str = get_dtype_str(next(model.parameters()).dtype)  # 有 --dtype 时可用 args.dtype
trace_path = adapt_dir / f"trace_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.json"
outputs_path = adapt_dir / f"outputs_{device_short}_{dtype_str}_{mode_str}_{dataset_name}.pt"
```

### setup_model 与 shrink 严禁

- **严禁** `shrink_config_for_dry_run` 等任何 shrink 函数
- DRY RUN 模式直接 `from_config(config)`，不修改 config

## 强制检查

运行 `benchmark/scripts/check_accuracy_run.py` 可批量校验所有 accuracy_run.py 是否符合规范：

```bash
uv run python benchmark/scripts/check_accuracy_run.py              # 检查全部，违规时 exit 1
uv run python benchmark/scripts/check_accuracy_run.py --warn-only  # 仅输出，不失败
uv run python benchmark/scripts/check_accuracy_run.py --adapt xxx  # 仅检查指定 adaptation
```

**Benchmark-Runner 禁止**：**禁止**调用 `update_benchmark_status`；仅通过 SendMessage 报告结果，由 team-lead 统一更新看板并执行 git commit（避免重复 commit）。

**保证执行**（规则强制）：
- **Agent 规则**：benchmark-runner.md 2.11、benchmark-script/SKILL.md 9.6 规定：生成或修改 accuracy_run.py 后**必须**执行 `check_accuracy_run.py`，禁止跳过；违规 exit 1 时必须修复后重跑直至通过
- **CI**：仓库已移除 `check-accuracy-run.yml` 自动检查；须在本地或 team-lead 流程中执行 `check_accuracy_run.py` 直至通过
- **benchmark 电子狗**：每轮启动前必跑，输出写入 `logs/team_lead_*.log`


## seq2seq 模型特殊处理（T5 等）

**重要**: T5 等_encoder-decoder 模型（`AutoModelForSeq2SeqLM`）与模板默认的 `model(**inputs)` 调用方式不适用，会报错：
```
ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds
```

**解决方案**:
1. **Step 1**: 使用 `generate()` 方法而非直接 `model(**inputs)`
2. **Step 2**: 使用 `generate()` 方法获取 generated_ids，再通过 forward 提取 logits

**示例**:
```python
# Step 1: 使用 generate()
generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

# Step 2: 通过 forward 提取 logits
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=first_device)
forward_inputs = {k: v for k, v in inputs.items()}
forward_inputs["decoder_input_ids"] = decoder_input_ids
logits_output = model(**forward_inputs)
last_token_logits = logits_output.logits[0, -1, :].cpu()
```

**NPU OOM 处理**: 如果 NPU 上 OOM，改用 `--cpu` 参数在 CPU 上运行。

```python
uv run python accuracy_run.py --max-samples 10 --cpu
```

## 参考

- benchmark-runner.md 2.10 稡板强制检查
- benchmark-script/SKILL.md 9.4 手动编写规范、 9.5 娡板常见错误- benchmark-runner.md 2.10 禁用手册
- benchmark-script/SKILL.md 9.4 手动编写规范、9.5 常见错误
- dataset-mapping/SKILL.md 4.1 数据集加载方式


## ESPnet ASR 模型特殊处理

**重要**: ESPnet ASR 模型（如 reazonspeech-espnet-v2）不使用 transformers 标准模型，而是使用 ESPnet 库的 `Speech2Text` 类。

**特点**:
1. 模型加载需要从 HuggingFace 下载 config 和 checkpoint 文件
2. 使用 `Speech2Text` 类而非 `AutoModelForSpeechSeq2Seq`
3. 需要安装 `espnet` 和 `librosa` 依赖（通过 `pip install librosa==0.10.0`）
4. 与 Python 3.12 不兼容（llvmlite 需要 Python 3.9-3.11）

**解决方案**:
1. 添加 `--cpu` 参数支持强制 CPU 推理
2. 在 `load_benchmark_audio` 函数中使用 librosa 生成合成音频
3. 不使用 transformers 标准模型，而是使用 ESPnet 的 `Speech2Text` 类

**示例**:
```python
# 模型加载
from espnet2.bin.asr_inference import Speech2Text

speech2text = Speech2Text(
    asr_train_config="https://huggingface.co/.../config.yaml",
    asr_model_file="https://huggingface.co/.../valid.acc.ave_10best.pth",
    device=str(torch_device),
)

# 推理
result = speech2text(audio)  # 返回 List[Tuple]
transcription = result[0][0] if result else ""
```
**兼容性问题**:
- llvmlite==1.36.0 需要 Python 3.9-3.11，在 Python 3.12 上安装失败
- librosa 安装可能需要编译依赖

**建议**: 对 ESPnet 模型使用 `--cpu` 参数在 CPU 上运行评测

## TTS 模型特殊处理（Qwen3-TTS 等）

**重要**: TTS 模型使用 `qwen-tts` 库加载，不走标准 transformers 模板。

**特点**:
1. 使用 `Qwen3TTSModel.from_pretrained()` 加载预训练模型
2. 使用 `Qwen3TTSForConditionalGeneration(config)` 做架构验证（config mode）
3. 推理用 `model.generate_custom_voice(text=..., speaker=..., language=...)`
4. 无需外部数据集，用内置 TTS prompts 即可（dataset_name = "synthetic"）
5. `qwen-tts` 内部依赖 `torchaudio.compliance.kaldi`（需要 ONNX runtime）

**NPU 兼容性问题（已知）**:
- `qwen-tts` 的 `generate_custom_voice` 在 NPU 上因 `MultinomialWithReplacement` AICPU kernel 崩溃（errcode 0x2a）
- 设置 `do_sample=False` 不能解决，因为内部 `code_predictor.generate()` 有独立的采样逻辑
- CPU 模式可用但极慢（单样本 >820s），ONNX tokenizer 开销巨大
- **建议**: TTS 模型在 NPU 上只能跑 config mode；pretrained 模式需要 CPU 或 CUDA

**torchaudio 在 NPU 环境的兼容性**:
- NPU 环境无 CUDA runtime，`torchaudio` 的 C 扩展加载失败（`libcudart.so.13` 缺失）
- 可通过 patch `torchaudio/_extension/__init__.py`，将 `_load_lib("_torchaudio")` 包在 try/except OSError 中解决
- patch 后 `torchaudio._IS_TORCHAUDIO_EXT_AVAILABLE = False`，`torchaudio` 可导入但 C 扩展不可用

**accuracy_run.py 编写要点**:
- 变量名用小写 `dataset_name`（不要用 `DATASET_NAME`），因为 check_accuracy_run.py 的 regex 匹配 `{dataset_name}`
- output_type 用 `tts_audio_stats`（pretrained）或 `tts_architecture_test`（config）
- outputs_*.pt 格式：`{"tts_audio_stats": [dict, ...]}` 或 `{"tts_architecture_outputs": [dict, ...]}`
- config 模式下做 talker forward pass 验证架构

## ⚠️ nopua skill — 遇到困境必须调用

**nopua 不会自动触发**，需要主动 `Skill("nopua")`。

**触发条件**：同一 action 失败 2+ 次 / 陷入等待循环 / 被动等待而不改变策略。

**正确用法**：1. 停止当前循环 2. 查询 board.db 获取真实状态 3. 根据状态决定下一步 4. 写教训到 MEMORY。

**反面教训**：benchmark-runner 若 check_accuracy_run.py 反复失败，应立即读取脚本源码查根因，而非重试 5+ 次。
