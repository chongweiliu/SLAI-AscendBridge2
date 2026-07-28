> 来源：vllm-ascend docs/source/quick_start.md（main 分支，抓取于 2026-07-28）
> 版本快照：vllm_ascend_version=v0.22.1rc1

# 快速开始

以 Qwen3-0.6B 单卡离线推理为例，覆盖容器环境搭建与两种启动方式。

## 1. 支持设备

- Atlas A2 训练系列（Atlas 800T A2、Atlas 900 A2 PoD、Atlas 200T A2 Box16、Atlas 300T A2）
- Atlas 800I A2 推理系列
- Atlas A3 训练系列（Atlas 800T A3、Atlas 900 A3 SuperPoD、Atlas 9000 A3 SuperPoD）
- Atlas 800I A3 推理系列
- Atlas 推理产品（310p）

## 2. 环境要求

- OS：Linux
- Python：`>= 3.10, < 3.13`
- CANN `== 9.0.1`（A2/A3）或 `== 9.1.0-beta.1`（310p）
- torch-npu `== 2.10.0.post2`、torch `== 2.10.0`、NNAL `== 9.0.1`（A2/A3）/ `== 9.1.0-beta.1`（310p）
- 310p 不支持 triton / triton-ascend

> Atlas 推理产品用 `float16`，镜像用 `-310p`（Ubuntu）或 `-310p-openeuler`。
> Atlas 推理产品与 Atlas 200I Pro 不支持 `enable_npugraph_ex`，须设 `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'`。
> Atlas 200I Pro 需额外设备节点与驱动挂载，参见 Installation 的 Docker 章节。

## 3. 用容器搭环境

先装 Docker。按硬件/OS 选镜像：

| 场景                              | IMAGE                                                  |
|-----------------------------------|--------------------------------------------------------|
| Ubuntu A2                         | `quay.io/ascend/vllm-ascend:v0.22.1rc1`                |
| Ubuntu A3                         | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`             |
| Ubuntu Atlas 推理产品             | `quay.io/ascend/vllm-ascend:v0.22.1rc1-310p`           |
| openEuler A2                      | `quay.io/ascend/vllm-ascend:v0.22.1rc1-openeuler`      |
| openEuler A3                      | `quay.io/ascend/vllm-ascend:v0.22.1rc1-a3-openeuler`   |
| openEuler Atlas 推理产品          | `quay.io/ascend/vllm-ascend:v0.22.1rc1-310p-openeuler`|

### 3.1 Ubuntu A2 最小启动

```bash
export DEVICE=/dev/davinci0
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1
docker run --rm \
    --name vllm-ascend --shm-size=1g \
    --device $DEVICE \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash
apt-get update -y && apt-get install -y curl
```

### 3.2 Ubuntu A3（至少 2 卡协同）

```bash
export DEVICE0=/dev/davinci0
export DEVICE1=/dev/davinci1
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
docker run --rm \
    --name vllm-ascend --shm-size=1g \
    --device $DEVICE0 --device $DEVICE1 \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash
apt-get update -y && apt-get install -y curl
```

### 3.3 Ubuntu Atlas 推理产品（310p）

```bash
export DEVICE=/dev/davinci0
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-310p
docker run --rm \
    --name vllm-ascend --shm-size=1g \
    --device $DEVICE \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash
apt-get update -y && apt-get install -y curl
```

> openEuler 版把 `apt-get` 换成 `yum update -y && yum install -y curl`，镜像后缀加 `-openeuler`。
> 默认 workdir `/workspace`，vLLM 与 vLLM Ascend 在 `/vllm-workspace` 以开发模式安装。

## 4. 用法

加速下载用 ModelScope：

```bash
export VLLM_USE_MODELSCOPE=True
```

### 4.1 离线批量推理

`example.py`：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# 首次约 3-5 分钟下载（10 MB/s）
llm = LLM(model="Qwen/Qwen3-0.6B")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

```bash
python example.py
```

HuggingFace 连不上：

```bash
export VLLM_USE_MODELSCOPE=True
pip install modelscope
python example.py
```

### 4.2 OpenAI Completions API（vllm serve）

```bash
# 部署 vLLM 服务（首次约 3-5 分钟下载）
vllm serve Qwen/Qwen3-0.6B &
```

启动成功标志：

```
INFO:     Started server process [3594]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

查询模型列表（健康检查可用）：

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

发推理请求：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Beijing is a",
        "max_completion_tokens": 5,
        "temperature": 0
    }' | python3 -m json.tool
```

优雅停止后台进程（等价于前台 Ctrl-C）：

```bash
VLLM_PID=$(pgrep -f "vllm serve")
kill -2 "$VLLM_PID"
```

停止日志：

```
INFO:     Shutting down FastAPI HTTP server.
INFO:     Application shutdown complete.
```

退出容器：`Ctrl-D`。

## 5. 平台检测成功日志（启动健康标志）

```
INFO ... Available plugins for group vllm.platform_plugins:
INFO ... - ascend -> vllm_ascend:register
INFO ... Platform plugin ascend is activated
```
