---
name: uv-env-setup
description: 用单一 pyproject.toml 配置 uv，支持 NVIDIA CUDA 与华为 Ascend NPU 的 PyTorch。用于双硬件（CUDA/Ascend）依赖、pyproject.toml 或环境配置与排错。触发词："uv适配"、"torch适配"、"昇腾NPU配置"、"CUDA配置"、"记录适配问题"、"更新适配规则"
---

# uv + PyTorch（CUDA 与 Ascend）

配置 `uv`，使一个 `pyproject.toml` 同时支持 NVIDIA GPU 与华为 Ascend NPU 的 PyTorch。

## 1. 初始安装与工具

### 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 自动加载环境（~/.bashrc 或 ~/.zshrc）

在 shell 配置中加入以下内容，以自动加载华为 Ascend 环境。**默认用 CANN 9.0.0**（见下节"CANN 版本选择"）：

```bash
# 优先用 CANN 9.0.0（含 FlashAttention 内核）；若不存在再 fallback 到 ascend-toolkit/latest
if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh > /dev/null 2>&1
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh > /dev/null 2>&1
fi
```

## 1.5 CANN 版本选择（重要）

**默认用 CANN 9.0.0**（最新主流发布版本）。配套组件版本同步：

| CANN | torch-npu | PyTorch | 备注 |
|------|-----------|---------|------|
| **9.0.0（推荐）** | **2.10.0** | **2.10.0** | FA 内核齐全；910C/910B/950 通用 |
| 8.5.0 | 2.10.0rc2 / 2.9.0 | 2.10.0 / 2.9.0 | 表中次新 |
| 8.3.RC1 | 2.8.0 / 2.7.1 / 2.6.0 | 同 | ⚠️ **无 FlashAttention 编译内核** |
| 9.0.0-beta.2 | 2.12.0 | 2.12.0 | Ascend950 系列新芯片需此 |

来源：torch_npu wheel METADATA 的官方映射表（命名约定 `{PyTorch版本}-{Ascend版本}`）。**注意**：torch_npu 2.10.0 的 METADATA 表最高只列到 CANN 8.5.0↔2.10.0rc2，并无 9.0.0 行；上表 9.0.0 行是**实测前向兼容**结论（torch_npu 2.10.0 + torch 2.10.0+cpu + CANN 9.0.0 在 910C 上 FA 全可用、950 系列需 2.12.0）。后续若出更新的 torch_npu，以其自带 METADATA 表为准。

### 国内下载 fallback（torch）

模板默认 torch 从 `download.pytorch.org/whl/cpu` 取，国内约 193KB/s（139MB 需 ~12 分钟）。若太慢或不可达，把 `pyproject.toml` 里 torch 的 index 从 `pytorch-cpu` 换成 `pypi-tuna`（`https://pypi.tuna.tsinghua.edu.cn/simple`，~450KB/s，镜像了完整 torch wheel）。**绝不要**用 ascend-repo 当 torch 源（残缺 stub，见排错）。

### 为什么不能默认用 8.3.RC1

CANN 8.3.RC1 的 OPP 里 **FlashAttention 只有头文件、没有编译好的 .o 内核二进制**。`F.scaled_dot_product_attention` 会报：
```
aclnnFlashAttentionScore failed, error 161001
Cannot find binary for op FlashAttentionScore
```
对所有 head 配置（heads=12/head_dim=64、heads=1/head_dim=512 等）都缺。**9.0.0 的 OPP 含 ascend910_93 下的 incre/prompt/rain FA 内核 .o**，FA 立即可用。所以凡是模型用到 attention 的，必须 9.0.0（或 8.5.0+）。

### 不要相信 `ascend-toolkit/latest` 软链

部署机常把 `/usr/local/Ascend/ascend-toolkit/latest` 指向旧版（如 8.3.RC1），而 cann-9.0.0 已装却不是默认。**适配前先枚举**：
```bash
ls /usr/local/Ascend/ | grep -iE "cann|toolkit"
# 有 cann-9.0.0 就 source 它的 set_env.sh，而不是 ascend-toolkit/latest
source /usr/local/Ascend/cann-9.0.0/set_env.sh
echo $ASCEND_OPP_PATH  # 应指向 .../cann-9.0.0/opp
```
LD_LIBRARY_PATH 里若同时有 8.3 和 9.0.0 的 lib64，且 ASCEND_OPP_PATH 指向 8.3，会出现"libopapi 来自 9.0.0 但 OPP 找不到内核"的版本错配——FA 全失败。让两者都指向同一个 CANN 版本。

### 若 CANN 9.0.0 未安装（官方安装方法）

官方文档（CANN 商用版 9.0.0 安装指南）：
- 安装指南：https://www.hiascend.com/document/detail/zh/canncommercial/latest/softwareinst/instg/instg_0008.html
- 社区版 CANN：https://www.hiascend.com/en/software/cann/community/

CANN 9.0.0 支持 Python 3.7.x–3.13.x、Ubuntu/CentOS/openEuler/Kylin 等。用官方 `.run` 安装包（OBS 源）：
```bash
# aarch64（910C/910B 等ARM机型）
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-aarch64.run
bash ./Ascend-cann-toolkit_9.0.0_linux-aarch64.run --install
# 装完 source 它的 set_env.sh（见 §1）
```
注意：CANN 9.0.0 自身支持 Python 3.13，但**配套 torch_npu 2.10.0 的 PyTorch 只支持 3.10/3.11/3.12**（见 torch_npu METADATA 的 PyTorch↔Python 表），故 pyproject `requires-python` 仍限 `>=3.10, <3.13`，以 torch_npu 为准。

## 2. 项目配置（pyproject.toml）

使用 `tool.uv.index` 和 `tool.uv.sources` 隔离硬件相关依赖。

**重要规则：双 Extra（CUDA 与 Ascend）**

必须定义两个可选依赖（extra）：`cuda` 和 `ascend`。

* **不要** 把 `torch` 放在主 `dependencies` 数组里。
* **不要** 为不同硬件创建单独文件。
* 使用 `conflicts` 确保同一时间只安装一种硬件后端。

`optional-dependencies` 中一般避免精确版本约束（`==`），优先 `>=`。**唯一例外是 torch / torch-npu**：torch-npu 的 `Requires-Dist` 精确 pin torch（如 `torch ==2.10.0`），二者必须严格等版本匹配，故模板对 torch/torch-npu 用 `==`，其余依赖用 `>=`。

### 实现指南

1. **定义索引**：为 CUDA 和 Ascend 分别配置索引。
2. **映射来源**：根据 extra 使用条件化 sources。
3. **处理冲突**：防止同时安装 CUDA 和 Ascend extra。

完整示例见 [templates/pyproject_template.toml](templates/pyproject_template.toml)。

## 3. 硬件同步

根据可用硬件同步项目环境。

### NVIDIA CUDA

```bash
uv sync --extra cuda
```

### Ascend NPU

同步前需先加载驱动环境（见第 1 步/§1.5）。**优先 9.0.0**：

```bash
# 优先 CANN 9.0.0（8.3.RC1 无 FA 内核，见 §1.5）
source /usr/local/Ascend/cann-9.0.0/set_env.sh 2>/dev/null \
  || source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 使用 ascend extra 同步
uv sync --extra ascend
```

## 4. Python 实现

在 Ascend 环境中务必导入 `torch_npu` 以注册 NPU 成员。

```python
import torch
try:
    import torch_npu
except ImportError:
    pass

def get_device():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device("npu:0")
        return "npu:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"
```

## 5. 静态分析与 IDE

排除 history 并指定 venv，以优化 IDE 表现。

```toml
[tool.ruff]
exclude = [".history"]

[tool.pyright]
exclude = [".history"]
```

## 排错

* **`torch-npu` 找不到**：确认 `ascend-repo` 索引已设置 `explicit = true`。
* **NPU 成员错误**：确保在访问 `torch.npu` 前已执行 `import torch_npu`。
* **显存未释放**：调用 `torch.npu.empty_cache()` 或 `torch.cuda.empty_cache()`。
* **`ModuleNotFoundError: No module named 'torch._vendor'` / `torch._strobelight` / `torch._C`**：说明装到了 ascend-repo 的残缺 torch wheel（`torch._C` 仅 ~132KB stub）。**torch 必须从 PyTorch 官方 CPU index 装**：`[tool.uv.sources]` 里 `torch` 指向 `https://download.pytorch.org/whl/cpu`，**不要**指向 ascend-repo。模板已按此配置。
* **`RuntimeError: Failed to load the backend extension: torch_npu` + `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfe`**：torch_npu 的 `_get_cann_version()` 返回的 CANN 版本串含非 UTF-8 字节（CANN 版本串带中文/特殊字符），C 侧内部解码抛异常，发生在 `torch.__init__` 的 `_import_device_backends` 阶段（import 时即崩，无法用 `isinstance(bytes)` 或运行时 monkeypatch 绕过）。**解法（最后手段）**：patch `torch_npu/npu/utils.py` 的 `get_cann_version`，用 try/except 包住 `torch_npu._C._get_cann_version(module)` 调用本身，失败返回硬编码版本串如 `"9.0.0"`，清 .pyc 缓存。**风险提示**：① 这是改系统包文件，torch_npu 重装/升级即丢失，需重新打；② 影响该 python 环境的所有用户；③ 优先检查是否有更新版 torch_npu 已修此 bug（该 bug 是 CANN 版本串含非 UTF-8 字节，新版可能已改用 `errors='replace'` 解码）。locale（LC_ALL=C.UTF-8）无效。
* **`Cannot find binary for op FlashAttentionScore`（FA 算子无内核）**：CANN 版本太老（8.3.RC1 及更早）。切到 CANN 9.0.0（`source /usr/local/Ascend/cann-9.0.0/set_env.sh`），9.0.0 的 OPP 有 ascend910_93 的 FA .o 内核。详见 1.5 节。
* **uv sync 下载 torch 极慢/卡死**：`export UV_LINK_MODE=copy`（cache 与 target 跨文件系统时 hardlink 失败的 fallback），或直接用系统已有的 python（见下）。
* **uv venv torch 装不上/残缺**：aarch64 下若 ascend-repo 与 PyTorch 官方 wheel 都有问题，直接复用系统预装 python（通常 `/usr/local/python3.12.13/bin/python3` 已装 torch+torch_npu，`-m pip install` 补 nibabel/matplotlib 等即可），避免重装 torch。

## 自更新策略

在以下情况更新本 skill：

* 发现新的硬件相关镜像 URL。
* 依赖冲突解决模式有改进。
* Ascend toolkit 需要新的环境变量。

**说明**：将改进直接整合到上述相关章节。下方变更日志仅用于事件记录。

### 变更日志
- 2026-08-26（CONFLUX 910C 适配实测）：默认 CANN 改为 9.0.0（配套 torch-npu 2.10.0 + torch 2.10.0，来源 torch_npu METADATA 官方映射表）。新增 1.5 节说明 8.3.RC1 无 FA 内核、`latest` 软链不可信、版本错配。排错新增：ascend-repo torch wheel 残缺（torch 走 PyTorch 官方 CPU index）、`_get_cann_version` UnicodeDecodeError patch、FA 算子无内核、系统 python 复用。pyproject 模板 torch 改走 pytorch-cpu index。
