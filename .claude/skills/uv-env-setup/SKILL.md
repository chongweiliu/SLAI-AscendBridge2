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

在 shell 配置中加入以下内容，以自动加载华为 Ascend 环境：

```bash
export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
[ -f "$ASCEND_HOME/set_env.sh" ] && source "$ASCEND_HOME/set_env.sh" > /dev/null 2>&1
```

## 2. 项目配置（pyproject.toml）

使用 `tool.uv.index` 和 `tool.uv.sources` 隔离硬件相关依赖。

**重要规则：双 Extra（CUDA 与 Ascend）**

必须定义两个可选依赖（extra）：`cuda` 和 `ascend`。

* **不要** 把 `torch` 放在主 `dependencies` 数组里。
* **不要** 为不同硬件创建单独文件。
* 使用 `conflicts` 确保同一时间只安装一种硬件后端。

`optional-dependencies` 中避免使用精确版本约束（`==`），优先使用 `>=` 以便依赖解析。

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

同步前需先加载驱动环境（见第 1 步）。

```bash
# 若未在 shell 配置中加载，则手动加载
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

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

## 自更新策略

在以下情况更新本 skill：

* 发现新的硬件相关镜像 URL。
* 依赖冲突解决模式有改进。
* Ascend toolkit 需要新的环境变量。

**说明**：将改进直接整合到上述相关章节。下方变更日志仅用于事件记录。

### 变更日志
