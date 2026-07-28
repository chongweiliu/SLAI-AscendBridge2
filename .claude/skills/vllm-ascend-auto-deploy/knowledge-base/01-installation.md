> 来源：vllm-ascend docs/source/installation.md + docs/source/getting_started.md（main 分支，抓取于 2026-07-28）
> 版本快照：vLLM Ascend `v0.22.1rc1` / vLLM `v0.22.1`；main 分支对应 vLLM commit `fe784ff22e630a31fd798f392b01e0a75c18f047`（`.github/vllm-main-verified.commit`）
> mkdocs.yml 宏：`vllm_version=v0.22.1`、`vllm_ascend_version=v0.22.1rc1`、`pip_vllm_version=0.22.1`、`pip_vllm_ascend_version=0.22.1rc1`、`cann_image_tag=9.0.1-910b-ubuntu22.04-py3.12`

# 安装与上手

## 1. vLLM Ascend 是什么

vLLM Ascend plugin（vllm-ascend）是社区维护的硬件插件，让 vLLM 跑在 Ascend NPU 上。它通过硬件可插拔接口解耦 Ascend NPU 与 vLLM 的集成，使主流开源模型能在 Ascend 硬件上无缝运行。

- 入口：`Quick Start`、`Installation`、`Model Tutorials`、`Feature Tutorials`、`FAQs`
- 插件机制：vLLM 启动时打印 `Available plugins for group vllm.platform_plugins`，`ascend -> vllm_ascend:register`，随后 `Platform plugin ascend is activated`。可用 `VLLM_PLUGINS` 控制加载哪些插件。

## 2. 环境要求

- OS：Linux
- Python：`>= 3.10, < 3.13`
- 硬件：Atlas 800 A2 系列及 Atlas 推理产品等带 Ascend NPU 的机器
- 软件：见下表，把 vLLM Ascend / vLLM / PyTorch / torch-npu / CANN / Triton Ascend 当作**一套兼容集**整体对待

### 2.1 软件栈版本矩阵

**Atlas A2 推理产品 / Atlas A3 推理产品**

| Software   | Supported version | Note                                         |
|------------|--------------------|----------------------------------------------|
| Ascend HDK | 参见 CANN 9.0.1 文档 | Required for CANN                            |
| CANN       | `== 9.0.1`         | vllm-ascend 与 torch-npu 必需                 |
| torch-npu  | `== 2.10.0.post2`  | 无需手动装，后续步骤自动安装                 |
| torch      | `== 2.10.0`        | torch-npu 与 vLLM 必需，无需手动装           |
| NNAL       | `== 9.0.1`         | 提供 libatb.so，启用高级张量运算              |

**Atlas 推理产品（310p）**

| Software          | Supported version | Note                                |
|-------------------|--------------------|-------------------------------------|
| Ascend HDK         | 参见 CANN 9.1.0-beta.1 文档 | Required for CANN          |
| CANN               | `== 9.1.0-beta.1` | vllm-ascend 与 torch-npu 必需        |
| torch-npu          | `== 2.10.0.post2` | 自动安装                            |
| torch              | `== 2.10.0`       | 自动安装                            |
| NNAL               | `== 9.1.0-beta.1` | libatb.so                           |
| triton / triton-ascend | Not supported  | `Dockerfile.310p` 中不安装          |

> 重要：把 vLLM Ascend / vLLM / PyTorch / torch-npu / CANN / Triton Ascend 视为一个兼容集。release 安装从 release 兼容矩阵选一整行；main 分支开发用 `.github/vllm-main-verified.commit` 记录的精确 vLLM commit，任意 vLLM tag 或 PyPI release 可能有不同传递依赖。

## 3. 配置 CANN 环境

安装前确保固件/驱动与 CANN 装好（参见 CANN 官网下载页）。校验固件/驱动：

```bash
npu-smi info
```

### 3.1 用 CANN 镜像（推荐，pip 前最省事）

CANN 预置镜像已含 NNAL（libatb.so），无需额外安装。

```bash
# 按 /dev/davinci[0-7] 修改 DEVICE
export DEVICE=/dev/davinci7
export IMAGE=quay.io/ascend/cann:9.0.1-910b-ubuntu22.04-py3.12
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --device $DEVICE \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

### 3.2 手动装 CANN（备选）

若运行时报 `libatb.so not found`，确保 NNAL 已按下面步骤安装。

```bash
# 建虚拟环境
python -m venv vllm-ascend-env
source vllm-ascend-env/bin/activate

# 基础包
python -m pip install --upgrade pip
pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

# CANN toolkit
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-toolkit_9.0.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-toolkit_9.0.1_linux-"$(uname -i)".run
./Ascend-cann-toolkit_9.0.1_linux-"$(uname -i)".run --full
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

# 910b ops
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-910b-ops_9.0.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-910b-ops_9.0.1_linux-"$(uname -i)".run
./Ascend-cann-910b-ops_9.0.1_linux-"$(uname -i)".run --install

# NNAL
wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-nnal_9.0.1_linux-"$(uname -i)".run
chmod +x ./Ascend-cann-nnal_9.0.1_linux-"$(uname -i)".run
./Ascend-cann-nnal_9.0.1_linux-"$(uname -i)".run --install

source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 4. 用 pip 安装 vllm 与 vllm-ascend

先装系统依赖并配 pip 镜像：

```bash
# apt（带镜像）
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update -y && apt-get install -y gcc g++ cmake ninja-build libnuma-dev wget git curl jq
# 或 yum
# yum update -y && yum install -y gcc g++ cmake ninja-build numactl-devel wget git curl jq
# 配 pip 镜像（仅 0.11.0 及更早支持，更新版本不要执行）
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

可选：x86 机器或 torch-npu dev 版需配 extra-index：

```bash
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
```

### 4.1 预构建 wheel（原始）

```bash
# 最新支持版本为 v0.22.1
pip install vllm==0.22.1

pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  vllm-ascend==0.22.1rc1
```

### 4.2 预构建 wheel（uv-wheelnext，增量下载更小）

先装 uv-wheelnext 支持增量 wheel：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sed 's/verify_checksum "$_file"/true/' | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
source $HOME/.local/bin/env
```

再装：

```bash
pip install vllm==0.22.1

uv pip install --system \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  vllm-ascend==0.22.1rc1
```

> 若 `uv pip install` 出错（缓存损坏/旧包数据），先清缓存再重跑：`uv cache clean`

### 4.3 从源码构建

triton-ascend 安装：

```bash
pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
# 用 uv 时务必把 triton-ascend 放在最后装，避免依赖解析冲突
```

```bash
# 装 vLLM
git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -e .
cd ..

# 装 vLLM Ascend
git clone --depth 1 --branch v0.22.1rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -e .
cd ..
```

> 为 Atlas A3 构建自定义算子时，必须手动 `git submodule update --init --recursive`，或确保环境能联网。
> Atlas 推理产品（310p）不支持 triton/triton-ascend，源码安装可能拉入这些包，运行前需移除：`pip uninstall -y triton-ascend triton`

### 4.4 CPU-only 构建校验

仅校验无 NPU 可见时能否构建 Python 包，**不**验证 NPU 运行时加载、推理示例、自定义算子或 NPU 专项测试。仍需 CANN toolkit（读头文件和库）。

装构建后端与原生构建工具：

```bash
python -m pip install --upgrade \
    pip "setuptools>=64" "setuptools-scm>=8" wheel \
    attrs googleapis-common-protos \
    "cmake>=3.26" ninja
```

x86 上先装 CPU 版 PyTorch：

```bash
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu/ \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    torch-npu==2.10.0.post2 triton-ascend==3.2.1
python -m pip install \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    -r requirements.txt
```

显式设构建目标并禁用设备后端自动加载后构建：

```bash
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export COMPILE_CUSTOM_KERNELS=0
export SOC_VERSION=ascend910b1  # Atlas A2；其他产品见下
python -m pip install \
    --no-build-isolation --no-deps \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    -e .
```

构建完成后跑 `python -m pip check` 解决所有冲突，再视为可运行。无设备时跳过推理示例与 NPU 专项测试。

### 4.5 SOC_VERSION 取值参考（CPU-only / 无 npu-smi 时必设）

从 `Dockerfile*` 默认值参考：

- Atlas A2：`export SOC_VERSION=ascend910b1`
- Atlas A3：`export SOC_VERSION=ascend910_9391`
- Atlas 推理产品：`export SOC_VERSION=ascend310p1`
- Ascend 950 Products：`export SOC_VERSION=<value starting with "ascend950">`

> 自定义算子构建需 gcc/g++ > 8 且 C++17+。`pip install -e .` 遇 torch-npu 版本冲突用 `pip install --no-build-isolation -e .`。编译器异常时用 `CXX_COMPILER` / `C_COMPILER` 指定 g++/gcc 路径。
> 启用 batch invariance：构建前 `export VLLM_BATCH_INVARIANT=1`。

## 5. 用 Docker 安装

预置镜像在 [ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags)。

| image name                            | Hardware                | OS        |
|---------------------------------------|-------------------------|-----------|
| vllm-ascend:v0.22.1rc1                | Atlas A2                | Ubuntu    |
| vllm-ascend:v0.22.1rc1-openeuler      | Atlas A2                | openEuler |
| vllm-ascend:v0.22.1rc1-a3             | Atlas A3                | Ubuntu    |
| vllm-ascend:v0.22.1rc1-a3-openeuler   | Atlas A3                | openEuler |
| vllm-ascend:v0.22.1rc1-310p           | Atlas 推理产品          | Ubuntu    |
| vllm-ascend:v0.22.1rc1-310p-openeuler | Atlas 推理产品          | openEuler |

从 Dockerfile 构建：

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
docker build -t vllm-ascend-dev-image:latest -f ./Dockerfile .
```

### 5.1 A2/A3 容器启动

A2：`/dev/davinci[0-7]`；A3：`/dev/davinci[0-15]`。先下载权重到 `/root/.cache`。

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g --net=host \
    --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

默认 workdir `/workspace`，vLLM 与 vLLM Ascend 代码在 `/vllm-workspace` 以开发模式（`pip install -e`）安装，便于即时改代码。

### 5.2 Atlas 推理产品容器启动

按需调整 `/dev/davinci0`。

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
```

### 5.3 Atlas 200I Pro 容器启动

需额外设备节点、驱动库与配置文件，使 `npu-smi` 等驱动命令在容器内可用。

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-310p
docker run --rm --privileged \
    --name vllm-ascend --shm-size=10g \
    --device=/dev/davinci0:/dev/davinci0 \
    --device=/dev/davinci_manager --device=/dev/ascend_manager --device=/dev/user_config \
    -v /etc/sys_version.conf:/etc/sys_version.conf \
    -v /etc/ld.so.conf.d/mind_so.conf:/etc/ld.so.conf.d/mind_so.conf \
    -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg \
    -v /var/dmp_daemon:/var/dmp_daemon \
    -v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
    -v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
    -v /usr/lib/aarch64-linux-gnu/libyaml-0.so.2:/usr/lib64/libyaml-0.so.2 \
    -v /etc/slog.conf:/etc/slog.conf \
    -v /var/slogd:/var/slogd \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so \
    -v /root/.cache:/root/.cache \
    -p 8000:8000 \
    -it $IMAGE bash
```

openEuler：保持命令结构，做以下替换：
- `IMAGE` 改为 `quay.io/ascend/vllm-ascend:v0.22.1rc1-310p-openeuler`
- 加 `-v /usr/lib64/libsemanage.so.2:/usr/lib64/libsemanage.so.2`
- libyaml 挂载改为 `-v /usr/lib64/libyaml-0.so.2.0.9:/usr/lib64/libyaml-0.so.2`

## 6. 校验安装

`example.py`：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen3-0.6B")
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

```bash
python example.py
```

Hugging Face 连不上时用 ModelScope：

```bash
export VLLM_USE_MODELSCOPE=True
pip install modelscope
python example.py
```

平台检测成功的标志日志：

```
INFO ... Available plugins for group vllm.platform_plugins:
INFO ... - ascend -> vllm_ascend:register
INFO ... All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO ... Platform plugin ascend is activated
```

离线推理退出时的 `EngineCore died unexpectedly` 是正常退出后进程结束的副作用，不影响推理。

## 7. 多节点部署

### 7.1 物理层要求
- 各机在同一 LAN、网络互通；所有 NPU 用光模块连接且状态正常。

### 7.2 每节点校验（结果须全 `success`、状态 `UP`）

A2 系列（8 卡）：

```bash
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done   # 远端交换端口
for i in {0..7}; do hccn_tool -i $i -link -g ; done                 # 链路 UP/DOWN
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done           # 网络健康
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done            # 探测 IP 配置
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done              # 网关
cat /etc/hccn.conf                                                  # NPU 网络配置
```

A3 系列把 `{0..7}` 改为 `{0..15}`（共 16 卡）。

### 7.3 互联校验

获取 NPU IP：

```bash
# A2
for i in {0..7}; do hccn_tool -i $i -ip -g | grep ipaddr; done
# A3
for i in {0..15}; do hccn_tool -i $i -ip -g | grep ipaddr; done
```

跨节点 PING：

```bash
hccn_tool -i 0 -ping -g address x.x.x.x
```

### 7.4 每节点起容器

用官方容器跑多节点最高效。A2 用 `v0.22.1rc1`，A3 用 `v0.22.1rc1-a3`，openEuler 加 `-openeuler` 后缀；`--net=host`，并提前暴露网桥端口用于多节点通信。容器命令同 5.1（A2）或对应 A3（16 卡 `/dev/davinci0..15`）。
