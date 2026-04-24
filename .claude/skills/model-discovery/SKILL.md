---
name: model-discovery
description: 使用 get_model_info.py 脚本发现并分析 PyTorch 模型。用于搜索模型、提取模型元数据、分析模型依赖。简短触发词："发现模型"、"搜索模型"、"model discovery"、"获取模型信息"
---

# 模型发现 Skill

本 skill 提供使用项目脚本发现并分析 PyTorch 模型的标准流程。

## 简短触发与执行

**一句话触发示例**：

- 「发现模型 microsoft/biogpt」
- 「获取模型元数据」
- 「model discovery」

**精简执行清单**：

1. 确定模型 ID 和来源（HuggingFace/GitHub）
2. 运行 `scripts/get_model_info.py`
3. 解析返回的 JSON 元数据
4. 提取依赖和配置信息

## 关键节点与日志

| 节点 | 建议输出 |
|------|----------|
| 开始 | `[discovery] 开始模型发现: {model_id}` |
| 执行脚本 | `[discovery] 运行 get_model_info.py...` |
| 解析结果 | `[discovery] 解析模型元数据` |
| 完成 | `[discovery] 模型发现完成` |

## 1. 使用脚本获取信息

**不要自己编写 Python 代码调用 API**。请直接使用项目提供的标准化脚本。

### 命令格式

```bash
.venv/bin/python scripts/get_model_info.py <model_id> --source <source>
```

- **model_id**：模型 ID（例如 `microsoft/biogpt` 或 `owner/repo`）
- **source**：`huggingface`（默认）或 `github`

### 示例

```bash
# HuggingFace（默认）
.venv/bin/python scripts/get_model_info.py microsoft/biogpt

# GitHub
.venv/bin/python scripts/get_model_info.py google-research/bert --source github
```

## 2. 解析元数据（JSON 输出）

脚本将 JSON 结果输出到 stdout。请重点关注以下字段：

```json
{
    "id": "microsoft/biogpt",           // 模型唯一标识
    "source": "huggingface",             // 来源
    "model_type": "CausalLM",            // 决定 demo.py 使用 AutoModelForCausalLM
    "architecture": "BioGPT",           // 架构名称
    "dependencies": [                   // 需添加到 pyproject.toml 的依赖
        "transformers>=4.28.0",
        "torch>=2.0.0"
    ],
    "config": { ... },                  // 原始配置（可选）
    "size": "1.5GB",                    // 预估模型大小
    "tags": ["medical", "nlp"],         // 标签
    "transformers_info": {              // Transformers 自动类推断
        "auto_model": "AutoModelForCausalLM",
        "processor": "AutoTokenizer"
    }
}
```

### 字段用途

- **dependencies**：直接用于 `pyproject.toml` 的 `dependencies` 列表。
- **model_type / transformers_info**：决定 `demo.py` 中导入哪个 AutoModel 类（如 `AutoModelForCausalLM` 与 `AutoModelForSequenceClassification`）。
- **size**：评估是否需要大内存节点。

## 排错

- **脚本报错**：查看 stderr 输出。常见原因：模型 ID 错误、网络问题或 API 限制。
- **依赖缺失**：若 `dependencies` 为空，可尝试从 `config` 或 `tags` 推断，或默认添加 `transformers` 和 `torch`。
- **未知模型类型**：若 `model_type` 为 "Custom" 或 "Unknown"，在 `demo.py` 中使用通用 `AutoModel` 并打印警告。

## 自更新策略

在以下情况更新本 skill：

- `scripts/get_model_info.py` 的参数或输出格式发生变化。
- 需要支持新的元数据字段。
