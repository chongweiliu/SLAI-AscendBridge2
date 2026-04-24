---
name: model-files-override
description: 在 adaptations 下创建 model_files 目录，用于性能优化测试。不改动 models 缓存，demo.py 和 accuracy_run.py 均保持不变。支持 trust_remote_code 模型（有 modeling_*.py）与内置 transformers 模型（无自定义代码）。触发词：model_files、性能提升测试、优化版 accuracy、不修改 models、内置模型、monkey patch。
---

# Model Files Override Skill

在 adaptation 目录（即具体模型目录，如 `adaptations/dutir_bionlp_taiyi_llm`）下创建 `model_files/`，作为模型加载的本地覆盖目录。**demo.py 和 accuracy_run.py 均保持不变**，新建的 accuracy 变体（如 `accuracy_run_perf.py`）从 `model_files` 加载，用于性能提升测试。

**权责**：`model_files/` 与 `accuracy_run_perf.py` 必须且仅能由 **npu-optimizer** 创建。adapter、benchmark-runner 不得使用本 skill 创建 model_files。

## 1. 适用场景

- 需修改 `modeling_*.py` 做 NPU 算子优化验证
- 不想改动 `models/` 缓存（保持 HF 缓存干净）
- 新建独立 accuracy 脚本做性能对比测试

## 2. 目录结构

adaptation 目录 = 具体模型目录 = `adaptations/{sanitized_name}/`：

```
adaptations/{sanitized_name}/   # 即 adaptation 目录 / 具体模型目录
├── demo.py              # 不变
├── accuracy_run.py      # 不变
├── accuracy_run_perf.py # 新建，从 model_files 加载
├── model_files/         # 新建，本地覆盖目录（扁平结构，见下）
│   ├── config.json
│   ├── modeling_*.py    # 可修改
│   ├── configuration_*.py
│   ├── tokenization_*.py
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── generation_config.json
│   ├── *.tiktoken / *.model  # tokenizer 词表
│   ├── model-*.safetensors   # 符号链接 → models/.../snapshots/
│   └── model.safetensors.index.json
└── models/              # HF 缓存，不修改
```

**禁止**：`model_files/` 必须是扁平结构，**不得**在 model_files 下出现 HF 缓存目录（`models--xxx/blobs/`、`refs/`、`snapshots/` 等）。若将 `model_files` 作为 `cache_dir` 传入 `from_pretrained()`，HF 会在 model_files 下创建完整缓存，导致数 GB 大文件被误提交。

**加载规范**：
- 从本地 model_files 加载（优化版）：`from_pretrained(MODEL_PATH)`，其中 `MODEL_PATH = adaptation_dir / "model_files"`，**勿传 `cache_dir`**。
- 从 Hub 下载（需 cache 时）：`from_pretrained(MODEL_ID, cache_dir=str(adaptation_dir / "models"))`，**`cache_dir` 必须指向 `models/`，禁止指向 `model_files/`**。

## 3. 创建 model_files 流程

### 3.1 定位 snapshot 路径

HF 缓存结构：`models/models--{org}--{model}/snapshots/{hash}/`

```bash
ADAPT_DIR="adaptations/dutir_bionlp_taiyi_llm"
MODEL_ID="DUTIR-BioNLP/Taiyi-LLM"

# 方式 A：从 refs/main 获取 hash
HASH=$(cat "$ADAPT_DIR/models/models--${MODEL_ID//\//--}/refs/main")
SNAPSHOT="$ADAPT_DIR/models/models--${MODEL_ID//\//--}/snapshots/$HASH"

# 方式 B：直接取唯一 snapshot
SNAPSHOT=$(find "$ADAPT_DIR/models" -path "*/snapshots/*" -type d | head -1)
```

### 3.2 创建目录并填充文件

```bash
TARGET="$ADAPT_DIR/model_files"
mkdir -p "$TARGET"

# 复制小文件（config、tokenizer、自定义 Python 模块）
for f in config.json configuration_*.py modeling_*.py tokenization_*.py \
         *_generation_utils.py tokenizer_config.json special_tokens_map.json \
         generation_config.json *.tiktoken *.model model.safetensors.index.json \
         pytorch_model.bin.index.json; do
  for p in "$SNAPSHOT"/$f; do
    [ -e "$p" ] && cp "$p" "$TARGET/"
  done
done

# 权重用符号链接，避免重复占用空间
for p in "$SNAPSHOT"/model-*.safetensors "$SNAPSHOT"/pytorch_model-*.bin; do
  [ -e "$p" ] && ln -sf "$(realpath "$p")" "$TARGET/$(basename "$p")"
done
```

**注意**：`for f in ...` 中部分 glob 可能无匹配，`[ -e "$p" ]` 确保只处理存在的文件。

### 3.3 通用脚本（适配不同模型）

不同模型文件各异。**权重必须用符号链接**，否则会重复占用大量空间。

```bash
# 1. 权重与词表用符号链接（优先处理，避免被复制）
for pat in model-*.safetensors pytorch_model-*.bin *.tiktoken *.model; do
  for f in "$SNAPSHOT"/$pat; do
    [ -e "$f" ] && ln -sf "$(realpath "$f")" "$TARGET/$(basename "$f")"
  done
done

# 2. 其余文件复制（config、*.py、*.json 等）
for f in "$SNAPSHOT"/*; do
  [ -d "$f" ] && continue
  base=$(basename "$f")
  [ -e "$TARGET/$base" ] && continue  # 已链接的权重跳过
  cp "$f" "$TARGET/"
done
```

## 4. accuracy 变体命名规范（强制）

**变体脚本名称固定为 `accuracy_run_perf.py`，禁止使用其他名称。**

| 允许 | 禁止 |
|------|------|
| `accuracy_run_perf.py` | `accuracy_run_opt.py`、`accuracy_perf.py`、`perf_run.py` 等任意其他名称 |

产出文件后缀统一为 `_perf`，如 `benchmark_metrics_*_perf.json`、`outputs_*_perf.pt`。

## 5. 模板使用

使用 Jinja2 模板 `.claude/skills/model-files-override/templates/accuracy_run_perf.py.j2` 生成 `accuracy_run_perf.py`。

### 5.1 模板变量

| 变量 | 必须 | 说明 |
|------|------|------|
| `{{ model_id }}` | ✅ | 模型 ID，如 `DUTIR-BioNLP/Taiyi-LLM` |
| `{{ model_type }}` | ❌ | 模型类型，默认 `causal_lm` |
| `{{ dataset_key }}` | ❌ | 推荐数据集 key，默认 `wikitext` |
| `{{ safe_name }}` | ❌ | 安全名称，用于注释 |

### 5.2 支持的模型类型

| model_type | AutoModel 类 | 输出类型 | 推荐数据集 |
|------------|-------------|---------|-----------|
| `causal_lm` | `AutoModelForCausalLM` | generated_text | wikitext |
| `bert` | `AutoModel` | cls_embeddings | wikitext |
| `vision_classification` | `AutoModelForImageClassification` | class_labels | cifar100 |
| `asr` | `AutoModelForSpeechSeq2Seq` | transcriptions | librispeech |
| `seq2seq` | `AutoModelForSeq2SeqLM` | generated_text | cnn_dailymail |
| `token_classification` | `AutoModelForTokenClassification` | token_labels | conll2003 |

与 benchmark 模板一致，共 6 种类型。`dataset_mapping.py` 可能返回其他类型（如 `vlm`、`embedding`、`diffusion`），此时需手动指定 `model_type` 为上述之一或参考 benchmark 模板扩展。

### 5.3 生成流程

1. **确定模型类型**：调用 `scripts/dataset_mapping.py` 获取

   ```bash
   uv run python scripts/dataset_mapping.py --model_id "prajjwal1/bert-small" --json
   # 输出: {"model_id": "prajjwal1/bert-small", "model_type": "bert", "dataset_key": "wikitext"}
   ```

2. **读取模板**：`.claude/skills/model-files-override/templates/accuracy_run_perf.py.j2`

3. **替换变量**：使用 Jinja2 渲染

4. **写入目标**：`adaptations/{safe_name}/accuracy_run_perf.py`

### 5.4 生成命令示例

**使用辅助脚本**（推荐）：

```bash
# 从项目根目录执行，model_type/dataset_key 自动从 dataset_mapping 获取
uv run python .claude/skills/model-files-override/scripts/generate_accuracy_run_perf.py \
    --model_id "prajjwal1/bert-small" \
    --safe_name prajjwal1_bert_small

# 或显式指定
uv run python .claude/skills/model-files-override/scripts/generate_accuracy_run_perf.py \
    --model_id "prajjwal1/bert-small" \
    --safe_name prajjwal1_bert_small \
    --model_type bert \
    --dataset_key wikitext
```

**使用 Jinja2 直接渲染**：

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader(".claude/skills/model-files-override/templates"))
template = env.get_template("accuracy_run_perf.py.j2")

script_content = template.render(
    model_id="prajjwal1/bert-small",
    safe_name="prajjwal1_bert_small",
    model_type="bert",
    dataset_key="wikitext"
)

# 写入 adaptations/prajjwal1_bert_small/accuracy_run_perf.py
```

### 5.5 手动新建（基于 accuracy_run.py 复制）

若不使用模板，可基于现有 `accuracy_run.py` 复制，**仅修改加载路径**：

```python
# 原 accuracy_run.py
MODEL_ID = "DUTIR-BioNLP/Taiyi-LLM"
CACHE_DIR = (Path(__file__).resolve().parent / "models").as_posix()

# accuracy_run_perf.py 修改为
MODEL_PATH = (Path(__file__).resolve().parent / "model_files").as_posix()

# setup_model 中
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
# from_config 分支同样用 MODEL_PATH 加载 config
```

**产出命名**：输出文件名必须加 `_perf` 后缀，如 `benchmark_metrics_npu_bf16_pretrained_wikitext_perf.json`、`outputs_npu_bf16_pretrained_wikitext_perf.pt`，避免与 accuracy_run.py 产出冲突。

**优化前后对比**：若同目录存在 baseline 文件（`benchmark_metrics_{device}_{dtype}_{mode}_{dataset}.json`，即 accuracy_run.py 产出），accuracy_run_perf.py 会在写入 perf 时自动合并对比字段：`baseline_latency_s`、`speedup_ratio`、`latency_reduction_pct`、`memory_reduction_pct`、`ttft_speedup_ratio`、`tpot_speedup_ratio` 等，便于 `optimization_tool aggregate` 聚合统计。

**产出对比**：若存在 baseline `outputs_*.pt`（accuracy_run.py 产出），run 结束时会自动对比 baseline 与 perf 的 outputs，并将结果写入 `benchmark_metrics_*_perf.json` 的 `output_compare` 字段（含 `cosine_similarity`、`match_rate`、`max_abs_error` 等），便于统计优化前后精度一致性。**精度对比必须使用 pretrained 权重**，baseline 与 perf 均需 `--use-pretrained`。

## 6. 通用性检查清单

| 项目 | 有自定义代码（auto_map） | 无自定义代码（内置） |
|------|--------------------------|------------------------|
| config.json | 必有，含 auto_map | 必有，可无 auto_map |
| modeling_*.py | 必有，可修改 | 无，需 Monkey patch 或复制+auto_map |
| configuration_*.py | auto_map 引用时必有 | 通常无 |
| 权重格式 | safetensors 或 pytorch_model-*.bin | 同上 |
| 词表文件 | *.tiktoken、*.model 等因模型而异 | 同上 |

不同模型（Qwen、LLaMA、DeepSeek 等）文件名不同，创建脚本时以 snapshot 实际列表为准。

## 7. 模型无自定义代码（内置 transformers）时的修改方式

部分模型（如 BERT、Mistral、部分 LLaMA）的 config.json **无 auto_map**，代码仅在 transformers 库内部。**默认使用「复制 + auto_map」**，与有自定义代码的模型统一为「改 model_files 中的 modeling 文件」。

### 7.1 复制 + auto_map（默认）

1. 从 transformers 源码复制 modeling 文件到 model_files：
   ```bash
   # 以 LLaMA 为例，model_type 与目录名对应
   cp $(python -c "import transformers; print(transformers.__path__[0])")/models/llama/modeling_llama.py \
      model_files/
   ```

2. 在 config.json 中增加 auto_map（若原本没有）：
   ```json
   "auto_map": {
     "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"
   }
   ```

3. 修改 `model_files/modeling_llama.py` 后，用 `from_pretrained(MODEL_PATH, trust_remote_code=True)` 加载。

### 7.2 Monkey Patching（可选，小范围修改）

仅替换单个类（如 Attention）时可用，在 `accuracy_run_perf.py` 中、from_pretrained 之前注册：

```python
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.monkey_patching import register_patch_mapping

class CustomLlamaAttention(LlamaAttention):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

register_patch_mapping(mapping={"LlamaAttention": CustomLlamaAttention})
```

### 7.3 如何判断模型类型

| config.json 特征 | 类型 | 修改方式 |
|------------------|------|----------|
| 有 `auto_map` 且含 `AutoModelForCausalLM` 等 | 自定义代码 | 直接改 model_files 中的 modeling_*.py |
| 无 `auto_map`，仅有 `model_type`、`architectures` | 内置模型 | 复制+auto_map（默认），或 Monkey patch |

## 8. 与 benchmark 规范的关系

- `accuracy_run_perf.py` 为**变体脚本**，不纳入 `check_accuracy_run.py` 检查，但**必须**通过 `optimization/scripts/check_accuracy_run_perf.py` 检查
- **board_ops 拦截**：`update_optimization_status --optimization_status completed` 前会执行 check_accuracy_run_perf.py，未通过则拒绝更新、exit 1
- 使用 MODEL_PATH 时，等价于「本地路径作为模型根目录」，符合 `from_pretrained(path)` 用法
- 产出格式与 `accuracy_run.py` 一致，便于 `benchmark_tool compare` 对比

## 9. 对比（compare）——参考 benchmark_tool.py

`accuracy_run_perf.py` 产出的 `outputs_*_perf.pt` 可直接用 `benchmark/scripts/benchmark_tool.py compare` 对比，逻辑与标准 accuracy 一致。

### 9.1 产出格式要求

产出格式须与 `benchmark_tool.compare_outputs` 的 `detect_output_type` 兼容，见 `benchmark_tool.py` 第 129–197 行：

| output_type | 数据结构 | 对比函数 |
|-------------|----------|----------|
| `mixed` | `{"generated_text": [...], "logits": [...], "perplexity": [...]}` | `compare_mixed_outputs` |
| `cls_embeddings` | `list[Tensor]` 或 `{"cls_embeddings": [...]}` | `compare_cls_embeddings` |
| `class_labels` | `list[str]` | `compare_class_labels` |
| `generated_text` | `list[str]` | `compare_text_outputs` |

### 9.2 对比命令

**方式一：在 adaptation 目录内用 accuracy_run_perf.py（推荐）**

```bash
cd adaptations/xxx
uv run python accuracy_run_perf.py compare                    # 自动查找 *_perf.pt 对
uv run python accuracy_run_perf.py compare cuda.pt npu.pt      # 指定文件对
```

**方式二：用 benchmark_tool.py**

```bash
# 单文件对（显式指定 CUDA/NPU 的 _perf.pt）
uv run python benchmark/scripts/benchmark_tool.py compare \
  adaptations/xxx/outputs_cuda_bf16_pretrained_wikitext_perf.pt \
  adaptations/xxx/outputs_npu_bf16_pretrained_wikitext_perf.pt

# 指定 adaptation（自动匹配 outputs_*_perf.pt 对）
uv run python benchmark/scripts/benchmark_tool.py compare --adaptation xxx

# 对比所有（含 perf 产出）
uv run python benchmark/scripts/benchmark_tool.py compare --all --format table
```

### 9.3 文件名与配对

`find_compare_pairs` 通过 `BenchmarkFilename.parse` 解析 `outputs_{device}_{precision}_{mode}_{dataset}.pt`。`_perf` 作为 dataset 后缀（如 `wikitext_perf`）可被正确解析，CUDA 与 NPU 的 `*_perf.pt` 会按 `(precision, mode, dataset)` 自动配对。

### 9.4 验收标准（与 benchmark-analysis skill 一致）

| 输出类型 | 指标 | 要求 |
|---------|------|------|
| generated_text / class_labels | match_rate | 100% |
| cls_embeddings / logits | cosine_similarity | ≥ 0.999 |
| cls_embeddings / logits | max_abs_error | < 0.001 |
| mixed (perplexity) | ppl_avg_rel_diff | < 5% |

详见 `benchmark/scripts/benchmark_tool.py`：`compare_outputs`、`detect_output_type`、`find_compare_pairs`。

## 10. 快速参考

**使用辅助脚本**（推荐）：

```bash
# 从项目根目录执行
bash .claude/skills/model-files-override/scripts/create_model_files.sh adaptations/dutir_bionlp_taiyi_llm
```

**手动执行**：

```bash
ADAPT="adaptations/dutir_bionlp_taiyi_llm"
SNAPSHOT=$(find "$ADAPT/models" -path "*/snapshots/*" -type d | head -1)
mkdir -p "$ADAPT/model_files"
# 小文件复制、权重链接（见 3.3）

# 修改 model_files/modeling_qwen.py 后，运行性能测试
cd "$ADAPT" && uv run python accuracy_run_perf.py run --use-pretrained --max-samples 50
```
