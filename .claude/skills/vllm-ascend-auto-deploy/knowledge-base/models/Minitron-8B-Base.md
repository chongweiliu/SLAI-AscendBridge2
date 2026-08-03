> 来源：vllm-ascend docs/source/tutorials/models/Minitron-8B-Base.md（main 分支，抓取于 2026-07-28）

# Minitron-8B-Base 部署知识库

## 关键信息速览（中文提炼）

- 架构类型：Dense
- 推荐 TP：[1]
- max-model-len：[4096]
- gpu-memory-utilization：['0.9']
- enforce-eager：出现（建议 Eager 模式，见原文上下文）
- 是否有量化：否（以原精度 BF16/FP16 部署为主）
- 相关环境变量：IMAGE

## vllm serve 启动命令（原文保留）

（本教程未直接给出 `vllm serve` 命令块，请参考下方原文的部署章节。）

## 原文教程（完整保留，部署指导权威来源）

# Minitron-8B-Base

## Introduction

The released `Minitron-8B-Base` is a lightweight, efficient large language model developed by NVIDIA. It is designed for general-purpose text generation and reasoning tasks, and can be deployed with vLLM for online serving and evaluation on Ascend NPU hardware through `vllm-ascend`.

This document describes the main verification steps of the model, including supported features, environment preparation, single-node deployment, functional verification, and accuracy evaluation on the GSM8K benchmark.

## Environment Preparation

### Model Weight

`Minitron-8B-Base`(BF16 version): requires 1 Ascend 910B (with 1 x 64G NPUs). [Download model weight](https://www.modelscope.cn/models/nv-community/Minitron-8B-Base)

It is recommended to place the model weight in a shared cache directory, such as `/root/.cache/` or a local model path like `/data/vllm-workspace/models/Minitron-8B-Base`.

### Installation

`Minitron-8B-Base` can be deployed with `vllm-ascend` in a compatible runtime environment.

You can use the official docker image for deployment:

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/vllm-workspace/models:/data/vllm-workspace/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

If you do not want to use the docker image, you can also build from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Deployment

Start the online serving service with the following command:

``` bash
vllm serve "nv-community/Minitron-8B-Base" \
  --served-model-name minitron-8b-base \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --port 8000
```

## Functional Verification

Once your server is started, you can query the model with a simple prompt:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minitron-8b-base",
    "prompt": "Question: If a train travels 60 miles in 2 hours, what is its average speed in miles per hour?\nAnswer:",
    "max_tokens": 64,
    "temperature": 1.0
  }'
```

A valid response indicates that the model is deployed correctly and can generate text outputs.

## Accuracy Evaluation

The GSM8K dataset was used to evaluate the reasoning capability of `Minitron-8B-Base`.

The current evaluation setting is:

- Dataset: `gsm8k`
- Split: `test`
- Number of samples: `1000`
- Few-shot setting: `5-shot`
- `apply_chat_template`: `False`
- `fewshot_as_multiturn`: `False`

The current evaluation results are:

| Category | Dataset | Metric | Result |
|----------|---------|--------|--------|
| Accuracy | gsm8k / test | Total Samples | 1000 |
| Accuracy | gsm8k / test | exact_match,strict-match | 0.5436 |
| Accuracy | gsm8k / test | exact_match,flexible-extract | 0.5451 |

### Remarks on Metrics

- **exact_match,strict-match**: Only predictions that strictly match the expected final-answer extraction format are counted as correct.
- **exact_match,flexible-extract**: Predictions are evaluated with a more flexible answer extraction rule, which tolerates minor formatting differences as long as the final numeric answer is correct.

## Performance

### Baseline Result

`Minitron-8B-Base` can be deployed through `vllm-ascend` for online inference and benchmark evaluation.  
Actual throughput and latency depend on hardware resources, prompt length, output length, concurrency, and runtime configuration.

### Remarks

This document focuses on functional verification and benchmark accuracy on GSM8K.  
Further benchmarking is recommended for:

- request latency
- throughput under concurrency
- long-context inference
- memory utilization
- stability under continuous serving workloads
