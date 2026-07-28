> 来源：vllm-ascend docs（main 分支，抓取于 2026-07-28），整合自多个源文件

# vLLM-Ascend 核心特性知识库

本文件整合 vllm-ascend 仓库 `docs/source/user_guide/feature_guide/` 下 7 个特性文档，覆盖 Flash Attention、Graph Mode、Quantization、Speculative Decoding、LoRA、Sleep Mode、Structured Output 七大核心特性。各章节按"是什么 / 何时使用 / 如何启用"组织；所有命令、CLI flags、参数名、环境变量、版本/硬件要求、配置片段均保持英文原文。

---

## Flash Attention

### 是什么

Flash Attention 3 (FA3) on Ascend 是 vLLM-Ascend 提供的、与训练侧 Flash Attention 保持语义一致的推理 attention backend。默认的 Fused Infer Attention (FIA) 实现与训练侧 Flash Attention 存在差异，可能导致训练-推理不一致；FA3 用于在 RL 训练框架（如 veRL）等场景下消除这种差异。

FA3 当前处于 beta 阶段，依赖 `flash_attn_npu` 包（已在 GitHub 开源：[flash-attention-npu repository](https://github.com/MinghuasLab/flash-attention-npu)）。

### 特性矩阵（`flash_attn_with_kvcache`）

| Feature | GPU FA3 | NPU FA3 |
|---------|---------|---------|
| FP16 (float16) | ✅ | ✅ |
| BF16 (bfloat16) | ✅ | ✅ |
| Causal Attention | ✅ | ✅ |
| Sliding Window Attention | ✅ | - |
| MQA/GQA | ✅ | ✅ |
| Paged KV Cache | ✅ | ✅ |
| Rotary Position Embedding (RoPE) | ✅ | - |
| ALiBi | - | - |
| Softcapping | ✅ | - |
| FP8 Quantization | ✅ | - |
| Variable-length Sequences | ✅ | ✅ |

### 何时使用

- **Training-inference consistency**：RL 工作流（如 veRL）中推理结果用于计算训练信号，必须保证 attention 一致。
- **Framework debugging**：消除训练/推理 attention 差异，便于调试。
- **Reinforcement Learning (RL)**：RL 训练需要确定且一致的 rollout 以保证可复现与稳定训练。

### 硬件 / 软件要求

- Hardware: FA3 currently requires Ascend Atlas A2 and A3 inference products NPUs.
- Software: FA3 requires the `flash_attn_npu` package, which provides the `flash_attn_npu_v3` module with the `flash_attn_with_kvcache` operator.
- Installation: refer to <https://github.com/MinghuasLab/flash-attention-npu/blob/main/README.md#installation>.

### 如何启用

两步：(1) Set the environment variable `export VLLM_BATCH_INVARIANT=1` to enable batch invariant mode; (2) Specify the attention backend as `FLASH_ATTN` via the LLM parameter `attention_backend="FLASH_ATTN"`。

Online Inference (Server Mode):

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-8B \
  --attention-backend FLASH_ATTN \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

Offline Inference:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    attention_backend="FLASH_ATTN",
    compilation_config={"cudagraph_mode": "PIECEWISE"},
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

### 限制

- **Package not yet open-sourced**: The `flash_attn_npu` package required for FA3 has not yet been released. External users cannot use FA3 until the package is available.
- **Sliding window not supported**: 需要滑动窗口的模型应使用默认 FIA backend。
- **ACL graph capture not supported**: tiling 在 host 侧处理，不支持 ACL graph capture，必须使用 `compilation_config={"cudagraph_mode": "PIECEWISE"}`。
- **RoPE not supported**: FA3 attention kernel 内部不支持 RoPE，vLLM-Ascend 通过 PyTorch native RoPE fallback 补丁处理。
- **ALiBi not supported** / **Softcapping not supported** / **FP8 quantization not supported**。
- **MLA and SFA not supported**：不支持 Multi-head Latent Attention (MLA) 或 Sparse Flash Attention (SFA)。
- Enabling FA3 may cause performance degradation compared to the default FIA backend. This trade-off is intentional to guarantee training-inference consistency.

### 已测试模型

- Qwen3 (Dense): `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`
- Qwen3 (MoE): `Qwen/Qwen3-30B-A3B`

---

## Graph Mode

### 是什么

Graph mode 是 vLLM 在 Ascend 上的图执行机制，结合编译期 FX 图优化与运行期 capture/replay 以降低 kernel launch 开销。vLLM 上游已提供通用 graph-mode 架构（见 [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/) 与 [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/)），本节聚焦 Ascend 专属视图。

Current Status on Ascend:

- Graph mode is currently available only on the **V1 Engine**.
- **ACLGraph** (capture/replay via `torch.npu.NPUGraph`) is the runtime graph execution mechanism used by the default graph path on Ascend.
- **Npugraph_ex** is a compile-time FX graph optimization layer, enabled by default in FULL/FULL_DECODE_ONLY modes.
- **XliteGraph** is an optional graph path for selected model families and environments.
- In context parallel scenarios, `cudagraph_mode="FULL"` is not sufficiently supported yet.

### Graph Paths on Ascend

| Graph Path | Default | Description | Since |
|---|---|---|---|
| ACLGraph (+ Npugraph_ex) | Yes | Compile-time FX optimization (Npugraph_ex) + runtime capture/replay (ACLGraph) | v0.9.0rc1 (Npugraph_ex since v0.15.0rc1) |
| XliteGraph | No | Preconfigured graph path for selected model families. Requires separate installation | v0.11.0 |

### cudagraph_mode 语义

- **FULL_AND_PIECEWISE**: Default mode, same as the upstream vLLM strategy. Compile-time PIECEWISE compilation; runtime may still use full-graph behavior for uniform decode batches.
- **FULL / FULL_DECODE_ONLY**: Npugraph_ex FX graph optimization (`force_eager=True`, compile-time only, no capture). The optimized callable is then captured and replayed by ACLGraph at runtime.
- **PIECEWISE**: Npugraph_ex is disabled. Only basic FX fusion passes are applied at compile-time. ACLGraph captures and replays the resulting callable at runtime.
- **NONE**: No compilation or graph capture. The model runs in eager mode.

| `cudagraph_mode` | Compile-time | Runtime | Npugraph_ex |
|---|---|---|---|
| FULL_AND_PIECEWISE | Piecewise compilation path | Mixed: PIECEWISE for mixed batches, FULL-capable for uniform decode batches | Disabled |
| FULL / FULL_DECODE_ONLY | Npugraph_ex FX optimization | ACLGraph capture/replay | Enabled |
| PIECEWISE | Fusion pass only | ACLGraph capture/replay | Disabled |
| NONE | None | Eager execution | Disabled |

### 何时使用

- 默认 ACLGraph 路径适合大多数场景；uniform decode 为主的工作负载优先 `FULL` / `FULL_DECODE_ONLY`。
- Npugraph_ex 主要通过算子融合（如 add + rms_norm → npu_add_rms_norm）减少 kernel launch 开销。
- Static kernel compilation 适合静态/近静态 shape 的网络，能减少运行时开销，但会显著增加启动时间。
- XliteGraph 适合 Llama、Qwen dense、Qwen MoE、Qwen3-VL 等模型族，作为 ACLGraph 的可选替代路径。

### 如何启用

ACLGraph 在 `cudagraph_mode` 不为 `NONE` 时自动启用，无需显式配置。

CLI example:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

Python example:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    compilation_config={"cudagraph_mode": "PIECEWISE"},
)
```

#### Npugraph_ex

Atlas inference products and Atlas 200I Pro do not support `enable_npugraph_ex`. Set `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'`.

Npugraph_ex is **enabled by default** when `cudagraph_mode` is `FULL` or `FULL_DECODE_ONLY`. 自动在 `PIECEWISE` 或 `NONE` 下禁用。

Explicit enable (online):

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true}}'
```

Explicit disable (online):

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

Offline explicit:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

#### Static kernel compilation（可选，默认关闭）

Enabling static kernel triggers a compilation pass during the graph capture phase at service startup. This may add **several minutes to tens of minutes** to the startup time depending on the number of operators to compile and model complexity.

Online:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

Offline:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": True,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

验证 static kernel 是否生效：通过 Ascend Profiling 收集 trace，打开生成的 `op_statistic.csv`，若 `op_type` 或 `name` 列包含关键字 `static_kernel` 即生效。编译期会看到 Python warning：

```text
Starting static kernel compilation, the build directory is <path>
```

#### XliteGraph

Install Xlite first:

```bash
pip install xlite
```

Offline example (Xlite supports decode-only by default; full mode via `"full_mode": True`):

```python
from vllm import LLM

llm = LLM(
    model="path/to/Qwen3-32B",
    tensor_parallel_size=8,
    additional_config={
        "xlite_graph_config": {
            "enabled": True,
            "full_mode": True,
        }
    },
)
outputs = llm.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve path/to/Qwen3-32B \
  --tensor-parallel-size 8 \
  --additional-config '{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

### Attention backend 兼容性

vLLM 会在兼容性检查阶段根据 backend 支持级别自动调整 `cudagraph_mode`，可能将 full-graph 降为 mixed/piecewise，甚至禁用 graph mode。

| Attention backend | Declared support | Practical meaning |
|---|---|---|
| `attention_v1` | `ALWAYS` | Supports graph execution for mixed prefill/decode batches |
| `context_parallel/attention_cp` | `ALWAYS` | Supports graph execution for mixed prefill/decode batches |
| `mla_v1` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `context_parallel/mla_cp` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `sfa_v1` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `context_parallel/sfa_cp` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |

### Troubleshooting：capture 资源耗尽

若 ACLGraph capture 因配置的 graph sizes 超过当前 stack 可用运行时资源而失败，vLLM Ascend 会抛出带缓解指引的专用错误。主要措施：

- upgrade to a newer HDK/CANN stack if one is available;
- reduce `cudagraph_capture_sizes` or `max_cudagraph_capture_size`;
- prefer `FULL` or `FULL_DECODE_ONLY` when the workload is mostly uniform decode;
- temporarily disable graph mode to confirm the issue is capture-related.

最常出现在 `PIECEWISE` 或 `FULL_AND_PIECEWISE` 配置下。若错误文本中含 `207008` together with `Stream resources are insufficient` or `Insufficient_Stream_Resources`，即为此类失败。

### Fallback to Eager Mode

遇问题时可设置 `enforce_eager=True` 临时退回 eager。

Offline:

```python
from vllm import LLM

llm = LLM(model="path/to/your/model", enforce_eager=True)
outputs = llm.generate("Hello, how are you?")
```

Online:

```bash
vllm serve path/to/your/model --enforce-eager
```

### Common Limitations and Caveats

- XliteGraph 应视为可选替代路径，并非在所有场景下都能替代 ACLGraph。
- Encoder-decoder models currently do not keep `FULL_AND_PIECEWISE`; on Ascend they fall back to `PIECEWISE` or `NONE` depending on compilation support.

---

## Quantization

### 是什么

模型量化通过降低权重和激活的数值精度来减小模型体积与计算开销，从而节省显存、提升推理速度。vLLM Ascend 支持两类量化工具：**ModelSlim**（推荐）与 **LLM-Compressor**。

### 何时使用

- 想要更小模型体积、更低显存、更快推理时使用。
- 可自行转换，也可使用官方上传的量化模型，例如 <https://www.modelscope.cn/models/vllm-ascend/Kimi-K2-Instruct-W8A8>。
- Before you quantize a model, ensure sufficient RAM is available.

### 如何启用

#### 1. ModelSlim (Recommended)

[ModelSlim](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) 是面向 Ascend 硬件的压缩工具，覆盖 dense、MoE、多模态理解/生成等模型。

Installation:

```bash
# Install 26.0.0 version, this is currently the latest stable branch
git clone https://gitcode.com/Ascend/msmodelslim.git -b 26.0.0

cd msmodelslim

bash install.sh
```

Model Quantization（W8A8 for Qwen3-MoE 示例）:

```bash
cd example/Qwen3-MOE

# Support multi-card quantization
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# Set model and save paths
export MODEL_PATH="/path/to/your/model"
export SAVE_PATH="/path/to/your/quantized_model"

# Run quantization script
python3 quant_qwen_moe_w8a8.py --model_path $MODEL_PATH \
--save_path $SAVE_PATH \
--anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
--calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
--trust_remote_code True
```

更多示例见 <https://gitcode.com/Ascend/msmodelslim/tree/master/example>。

#### 2. LLM-Compressor

[LLM-Compressor](https://github.com/vllm-project/llm-compressor) 是面向 vLLM 推理的统一压缩模型库。

Installation:

```bash
pip install llmcompressor
```

Dense Quantization (W8A8 dynamic):

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic.py
```

MoE Quantization (W8A8 dynamic):

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic_moe.py
```

LLM-Compressor 当前支持的量化类型可在 `vllm_ascend/quantization/compressed_tensors_config.py` 查看。

### 运行量化模型

ModelSlim 量化模型需指定 `--quantization ascend`；LLM-Compressor 量化模型无需该参数。

Offline Inference:

```python
import torch

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Set sampling parameters
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/path/to/your/quantized_model",
          max_model_len=4096,
          trust_remote_code=True,
          # Set appropriate TP and DP values
          tensor_parallel_size=2,
          data_parallel_size=1,
          # Set an unused port
          port=8000,
          # Set serving model name
          served_model_name="quantized_model",
          # Specify `quantization="ascend"` to enable quantization for models quantized by ModelSlim
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Online Inference:

```bash
# Corresponding to offline inference
python -m vllm.entrypoints.api_server \
    --model /path/to/your/quantized_model \
    --max-model-len 4096 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --served-model-name quantized_model \
    --trust-remote-code 
```

---

## Speculative Decoding

### 是什么

Speculative decoding 是一种通过 proposer-verifier 架构降低 memory-bound LLM 推理 inter-token 延迟的技术：

1. **Proposer** (`vllm_ascend/spec_decode/`): Generates draft (speculative) tokens using various methods — from simple n-gram matching to neural-network-based draft models.
2. **Rejection Sampler** (`vllm_ascend/sample/`): Verifies draft tokens against the target model's output, accepting matches and rejecting mismatches, with optional optimizations including Block Verify and Entropy Verify.

支持的方法：

| Method | Description |
| ------ | ----------- |
| `ngram` | Match n-grams from the prompt |
| `suffix` | Suffix-based pattern matching (requires Arctic Inference) |
| `medusa` | Medusa heads embedded in the target model |
| `eagle` | EAGLE-based draft model |
| `eagle3` | EAGLE-3 based draft model |
| `mtp` | Multi-Token Prediction with shared embedding head |
| `dflash` | Block diffusion-based parallel draft model |
| `draft_model` | Generic external draft LLM |
| `extract_hidden_states` | Extract hidden states for EAGLE training |

### 何时使用

- 需要降低 inter-token 延迟、提升 memory-bound 推理吞吐的场景。
- `ngram` / `suffix`：高重复率任务（代码编辑、agentic loops、RL rollouts）。
- `eagle` / `eagle3`：有可用 draft 模型时。
- `mtp`：DeepSeek 系列等多 token 预测模型，无输出质量损失。
- `extract_hidden_states`：不真正解码，专门用于采集 EAGLE 训练数据。

### 如何启用

All speculative decoding methods are configured through the `speculative_config` parameter when initializing the model or starting the server.

通用参数：

- **`method`** (str, required): The speculative decoding method. Must be one of the supported method names listed in the table above.
- **`num_speculative_tokens`** (int, required): Number of speculative tokens to generate per forward pass. Auto-filled from the draft model's `n_predict` config (e.g., MTP) or `suffix_decoding_max_tree_depth` (suffix method) when available.
  - PD Separation 约束：(1) Hybrid Mamba models (e.g., Qwen-Next and Qwen3.5 series): `num_speculative_tokens` should be equal on P nodes and D nodes. (2) Other models: `num_speculative_tokens` on P nodes should be 1, and `num_speculative_tokens` on D nodes should be greater or equal to 1.
- **`model`** (str, optional): Path or HF repo ID for the draft model. Required for `eagle`, `eagle3`, `dflash`, `medusa`, and `draft_model`. Automatically resolved for `mtp` (reuses target model), `ngram`, `suffix`, and `extract_hidden_states`.
- **`draft_tensor_parallel_size`** (int, optional): Tensor parallelism size for the draft model. Can only be `1` or the same as the target model's tensor parallel size.
- **`disable_padded_drafter_batch`** (bool, default: `False`): Disable input padding for speculative decoding. Only effective with `eagle`, `eagle3`, `mtp`, `dflash`, `draft_model`, and `extract_hidden_states` methods.

> On Ascend NPUs, the `npu_fused_infer_attention_score` operator supports a maximum of 16 tokens per decode round. Therefore, `(num_speculative_tokens + 1)` must be ≤ 16.

Offline:

```python
from vllm import LLM

llm = LLM(
    model="path/to/target/model",
    speculative_config={
        "method": "eagle3",
        "model": "path/to/draft/model",
        "num_speculative_tokens": 3,
    },
)
```

Online:

```shell
vllm serve path/to/target/model \
  --speculative-config '{"method": "eagle3", "model": "path/to/draft/model", "num_speculative_tokens": 3}'
```

#### ngram 示例

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

#### EAGLE 示例与注意事项

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=4,
    distributed_executor_backend="mp",
    enforce_eager=True,
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

注意事项：

1. The EAGLE draft models available in the [HF repository for EAGLE models](https://huggingface.co/yuhuili) should be loaded and used directly by vLLM. This functionality was added in PR [#4893](https://github.com/vllm-project/vllm-ascend/pull/4893). 若 vLLM 版本早于此 PR 合入，需升级。
2. The EAGLE based draft models need to be run without tensor parallelism (i.e. `draft_tensor_parallel_size` is set to 1)，主模型仍可使用 tensor parallelism。
3. When using EAGLE-3 based draft model, option "method" must be set to "eagle3".
4. 启用 EAGLE 后，main model 需在一次解码中校验 `(1 + K)` 个 tokens；fullgraph 模式会固定验证阶段 token 数，因此 `cudagraph_capture_sizes` 必须是 capture sizes 列表，每个 size 为 `n * (K + 1)`。例如 batch sizes 1~4、`num_speculative_tokens = 4` 时，`cudagraph_capture_sizes = [5, 10, 15, 20]`。

#### MTP 示例（DeepSeek-V3.2-Exp-W8A8）

```shell
vllm serve /deepseek-ai/DeepSeek-V3.2-Exp-W8A8 \
    --port 20004 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name dsv3 \
    --max-model-len 36768 \
    --max-num-batched-tokens 5000 \
    --max-num-seqs 10 \
    --quantization ascend \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 2, "method":"mtp", "disable_padded_drafter_batch": false}'
```

> Due to the fact that only a single layer of weights is exposed in DeepSeek's MTP, accuracy and performance are not effectively guaranteed in scenarios where `num_speculative_tokens > 1` (especially ≥ 3).
>
> In the fullgraph mode with `num_speculative_tokens > 1`, the capture size of each ACLGraph must be an integer multiple of `(num_speculative_tokens + 1)`.

#### Suffix Decoding 示例

> Suffix Decoding requires Arctic Inference. You can install it with `pip install arctic-inference`.

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    enforce_eager=True,
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 15,
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

#### extract_hidden_states 示例

This method produces only 1 output token per request. The primary output is the hidden states saved to disk, not the generated text.

```python
import tempfile

from safetensors import safe_open
from vllm import LLM, SamplingParams

def main():
    with tempfile.TemporaryDirectory() as tmpdirname:
        llm = LLM(
            model="Qwen/Qwen3-8B",
            tensor_parallel_size=1,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        # Layer indices to extract hidden states from
                        "eagle_aux_hidden_state_layer_ids": [2, 18, 34],
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": tmpdirname,
                },
            },
        )

        prompts = ["Hello, how are you?", "What is machine learning?"]
        sampling_params = SamplingParams(max_tokens=1)
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            print("Prompt:", output.prompt)
            print("Prompt token ids:", output.prompt_token_ids)

            hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
            print("Hidden states saved to:", hidden_states_path)

            with safe_open(hidden_states_path, "pt") as f:
                token_ids = f.get_tensor("token_ids")
                hidden_states = f.get_tensor("hidden_states")
                print("Shape:", hidden_states.shape)
                # Shape: (num_tokens, num_layers, hidden_size)

if __name__ == "__main__":
    main()
```

Key configuration parameters:

1. **`num_speculative_tokens`**: Must be set to `1`.
2. **`eagle_aux_hidden_state_layer_ids`**: List of layer indices, e.g., `[2, 18, 34]`.
3. **`kv_connector`**: Must be set to `"ExampleHiddenStatesConnector"`.
4. **`kv_role`**: Must be set to `"kv_producer"` for the extraction mode.
5. **`shared_storage_path`**: Directory where hidden states will be saved as `.safetensors` files (one per request).

### Block Verify and Entropy Verify

两个可选的 rejection sampler 优化，以少量精度换取吞吐提升。Both modify the token acceptance criteria and may cause minor precision degradation.

- **Block Verify**: 将所有 draft tokens 作为 block 用累积概率乘积评估，而非逐 token 独立校验，在 `num_speculative_tokens >= 3` 时尤其有效。
- **Entropy Verify**: 根据目标分布熵值调整接受阈值。High entropy → lower effective threshold → more tokens accepted; Low entropy → higher effective threshold → stricter rejection。参数：
  - **`posterior_threshold`** (default: `0.95`, range: `(0, 1]`): 修改后阈值的上限。
  - **`posterior_alpha`** (default: `0.4`, range: `>= 0`): 控制熵对阈值的影响强度；alpha 为 `0` 时阈值等于 `posterior_threshold`。

Usage (online):

```shell
vllm serve <model> --additional-config \
    '{"rejection_sampler_config": {"enable_block_verify": true, \
    "enable_entropy_verify": true, "posterior_threshold": 0.95, \
    "posterior_alpha": 0.4}}'
```

Usage (offline):

```python
llm = LLM(
    model,
    additional_config={
        "rejection_sampler_config": {
            "enable_block_verify": True,
            "enable_entropy_verify": True,
            "posterior_threshold": 0.95,
            "posterior_alpha": 0.4,
        }
    },
)
```

Both features can be enabled independently or together. When used together, the cumulative acceptance from Block Verify is combined with the entropy-adjusted threshold from Entropy Verify.

---

## LoRA

### 是什么

vllm-ascend 与 vLLM 一样支持 LoRA，且现已可与 ACLGraph mode 一起运行（参见 [Graph Mode Guide](./graph_mode.md) 获得更好性能）。实现层面已内置 LoRA 相关 AscendC 算子（`bgmv_shrink`、`bgmv_expand`、`sgmv_shrink`、`sgmv_expand`），位于 [vllm-ascend repo](https://github.com/vllm-project/vllm-ascend/tree/main/csrc/kernels) 的 `csrc/kernels` 目录。支持 dense 与 mixture-of-experts (MoE) 模型（[PR #10977](https://github.com/vllm-project/vllm-ascend/pull/10977)），但 MoE+LoRA 暂不支持 expert-parallel (EP) 或 quantization。

### 何时使用

- 需要在同一 base model 上同时服务多个 LoRA adapter（如 sql-lora 测试适配器）。
- ACLGraph mode 默认启用，可获得更好性能。

### 如何启用

下载地址：

- base model: <https://www.modelscope.cn/models/vllm-ascend/Llama-2-7b-hf/files>
- loRA model: <https://www.modelscope.cn/models/vllm-ascend/llama-2-7b-sql-lora-test/files>

Example（默认启用 ACLGraph mode）:

```shell
vllm serve meta-llama/Llama-2-7b \
    --enable-lora \
    --lora-modules '{"name": "sql-lora", "path": "/path/to/lora", "base_model_name": "meta-llama/Llama-2-7b"}'
```

更多用法见 [vLLM official document](https://docs.vllm.ai/en/latest/features/lora/)；支持 LoRA 的模型列表见 [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)。

---

## Sleep Mode

### 是什么

Sleep Mode 是一套用于将 model weights 卸载到 CPU 并丢弃 KV cache 的 API，主要面向 RL post-training 工作负载（如 PPO/GRPO/DPO 等 online 算法）。在训练中，policy model 通常用 vLLM 做 autoregressive generation，再做 forward/backward；由于生成与训练阶段可能采用不同模型并行策略，需要在训练期间释放 vLLM 内的 KV cache 甚至 model parameters，以高效利用显存、避免资源争抢。

由于该特性使用底层 API [AscendCL](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/appdevgapi/appdevgapi_07_0000.html)，需按 [installation guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) 从源码构建；若使用 < v0.12.0rc1，需设置 `export COMPILE_CUSTOM_KERNELS=1`。

### 何时使用

- **Level 1 Sleep**：Offloads model weights and discards the KV cache. Model weights 移至 CPU 内存，KV cache 被丢弃。适合稍后复用同一模型（需保证 CPU 内存足够容纳模型权重）。
- **Level 2 Sleep**：Discards both model weights and KV cache. 两者内容均被遗忘。适合切换到不同模型或更新当前模型。

### 如何启用

With `enable_sleep_mode=True`，vLLM 内存管理（malloc/free）在特定内存池下进行；模型加载与 KV cache 初始化阶段将内存标记为 `{"weight": data, "kv_cache": data}`。

#### 可选的 extra cleanup

默认 sleep mode 仅释放由 sleep-mode allocator 管理的内存。RL 工作负载若需将更多 NPU 内存归还 trainer，可启用 `enable_sleep_mode_extra_cleanup`：

Offline:

```python
llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    enable_sleep_mode=True,
    additional_config={"enable_sleep_mode_extra_cleanup": True},
)
```

Online:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --enable-sleep-mode \
    --additional-config '{"enable_sleep_mode_extra_cleanup": true}'
```

When `enable_sleep_mode_extra_cleanup` is enabled, `sleep()` additionally:

- clears ACL graph attention workspaces and invalidates captured ACL graph caches when ACL graph is enabled;
- resets the model runner graph manager so ACL graphs can be captured again after wakeup;
- waits for pending pipeline-parallel send work, synchronizes the NPU, and destroys HCCL process groups.

During `wake_up()`, vLLM Ascend restores the HCCL process groups, refreshes MoE dispatcher HCCL metadata, restores sleep-mode allocator memory, and recaptures ACL graphs when needed.

Extra cleanup trades lower sleep-time NPU memory usage for longer wakeup latency. 若 ACL graph 已启用，`wake_up()` 必须在模型状态恢复后再次调用 `capture_model()`。当更短的 wakeup 延迟比释放 HCCL/ACL graph workspace 内存更重要时，保持 `enable_sleep_mode_extra_cleanup` 关闭。

Level 2 wakeup 可分两阶段：

```python
llm.wake_up(tags=["weights"])
# Reload or update model weights here.
llm.wake_up(tags=["kv_cache"])
```

With extra cleanup enabled, ACL graphs are recaptured only when `tags` is `None` or contains `"kv_cache"`。

#### Expert weight layout restoration

For dense models, `wake_up()` simply restores the model weights to NPU memory; the tensor layout is unchanged.

For **unquantized MoE models** (`quant_config is None`)，fused expert weights 以转置布局存储以适配 `torch_npu.npu_grouped_matmul`。该布局由 `process_weights_after_loading()` 在模型加载时一次性生成：将 `w13_weight` 与 `w2_weight` 的第二、三维 `transpose(1, 2)`。

Sleep-mode allocator 恢复原始（未转置）内存后，当恢复 `"weights"` tag 时 `wake_up()` 对受影响 expert weights 重新应用相同 transpose：

- `w13_weight` (gate/up projection): transposed back to the runtime layout when its second dimension matches `hidden_size`;
- `w2_weight` (down projection): transposed back to the runtime layout when its third dimension matches `hidden_size`.

Dense models（无 expert weights）与 quantized models（权重由量化方法处理）跳过此步。

#### 用法示例

Offline inference:

```python
import os

import torch
from vllm import LLM, SamplingParams
from vllm.utils.mem_constants import GiB_bytes

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"

if __name__ == "__main__":
    prompt = "How are you?"

    free, total = torch.npu.mem_get_info()
    print(f"Free memory before sleep: {free / 1024 ** 3:.2f} GiB")
    # record npu memory use baseline in case other process is running
    used_bytes_baseline = total - free
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", enable_sleep_mode=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    output = llm.generate(prompt, sampling_params)

    llm.sleep(level=1)

    free_npu_bytes_after_sleep, total = torch.npu.mem_get_info()
    print(f"Free memory after sleep: {free_npu_bytes_after_sleep / 1024 ** 3:.2f} GiB")
    used_bytes = total - free_npu_bytes_after_sleep - used_bytes_baseline
    # now the memory usage should be less than the model weights
    # (0.5B model, 1GiB weights)
    assert used_bytes < 1 * GiB_bytes

    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)
    # cmp output
    assert output[0].outputs[0].text == output2[0].outputs[0].text
```

Online serving（须显式指定 dev 环境 `VLLM_SERVER_DEV_MODE` 以暴露 sleep/wake up endpoints）:

```bash
export VLLM_SERVER_DEV_MODE="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_USE_MODELSCOPE="True"
export VLLM_ASCEND_ENABLE_NZ="0"

vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-sleep-mode

# after serving is up, post to these endpoints

# sleep level 1
curl -X POST http://127.0.0.1:8000/sleep \
    -H "Content-Type: application/json" \
    -d '{"level": "1"}'

curl -X GET http://127.0.0.1:8000/is_sleeping

# sleep level 2
curl -X POST http://127.0.0.1:8000/sleep \
    -H "Content-Type: application/json" \
    -d '{"level": "2"}'

# wake up
curl -X POST http://127.0.0.1:8000/wake_up

# wake up with tag, tags must be in ["weights", "kv_cache"]
curl -X POST "http://127.0.0.1:8000/wake_up?tags=weights"

curl -X GET http://127.0.0.1:8000/is_sleeping

# after sleep and wake up, the serving is still available
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "The future of AI is",
        "max_tokens": 7,
        "temperature": 0
    }'
```

---

## Structured Output

### 是什么

Structured Output（又称 Guided Decoding）让 LLM 在保留系统非确定性本质的同时，按用户提供的 schema 生成符合特定结构（如 JSON）的输出。即给 LLM 一份"模板"来"影响"其输出，确保符合期望结构，避免生成合法文本却违反 JSON 规范等问题。

### 何时使用

- 需要模型输出严格符合 JSON / 特定 schema / 特定格式的场景。

### 如何启用

Currently, the usage of structured output feature in vllm-ascend is totally the same as that in vllm. 更多示例与说明见 [vLLM official document](https://docs.vllm.ai/en/stable/features/structured_outputs/)。

---

## 源文件清单

| Source Path | Status |
|---|---|
| docs/source/user_guide/feature_guide/flash_attention.md | OK |
| docs/source/user_guide/feature_guide/graph_mode.md | OK |
| docs/source/user_guide/feature_guide/quantization.md | OK |
| docs/source/user_guide/feature_guide/speculative_decoding.md | OK |
| docs/source/user_guide/feature_guide/lora.md | OK |
| docs/source/user_guide/feature_guide/sleep_mode.md | OK |
| docs/source/user_guide/feature_guide/structured_output.md | OK |

所有 7 个源文件均成功抓取（HTTP 200），无失败项。
