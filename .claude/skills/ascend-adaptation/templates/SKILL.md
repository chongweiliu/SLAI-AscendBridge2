---
name: ascend-adaptation
description: 将 PyTorch 模型适配到 Ascend NPU。提供设备选择、环境配置、验证流程的模板与最佳实践。
---

# Ascend 适配 Skill

本 skill 指导你将 PyTorch 模型适配到华为 Ascend NPU。完整规则见 `.claude/agents/adapter.md`。

## 核心流程

1. **环境配置**：确保 `pyproject.toml` 正确配置（参考 `uv-env-setup` skill）。
2. **代码适配**：使用 `demo.py.j2` 模板生成 `demo.py`。
3. **验证**：Dry Run 跑通，`check_adaptation.py` 通过。

## 1. 模板与设备检测

使用 Jinja2 模板 `.claude/skills/ascend-adaptation/templates/demo.py.j2` 生成 `demo.py`，模板变量为 `{{ model_id }}`。

```python
# 设备检测（模板内）：NPU > CUDA > CPU
def get_device():
    """Returns (device_str, device_count)."""
```

**验收**：NPU 或 CUDA 均可，不允许 CPU 回退。

## 2. Dry Run 策略（快速验证）

**目的**：不下载权重，仅验证架构与代码路径在 NPU/CUDA 上的兼容性。

**实现**：

- `from_config(config)` 创建随机权重模型；
- `shrink_config_for_dry_run(config)` 保守缩小（仅层数减为 2）；
- `infer_auto_device_map(model)` + `dispatch_model(model, device_map)` 分布到多卡（需 accelerate）。

```bash
uv run python demo.py --dry-run
```

**验证失败情况**（任一即不通过）：

- 输出出现 `using CPU` 或 `No accelerator detected`
- 未加载模型、未执行推理、运行报错或退出码非 0

## 3. Full Run（加载真实权重）

- `from_pretrained(..., device_map="auto", cache_dir=CACHE_DIR)`；
- 模型与 tokenizer 缓存到 `adaptations/<name>/models/`。

## 4. 保存输出与验证

```bash
uv run python demo.py > output.txt 2>&1
```

适配完成后运行 `adaptation/scripts/check_adaptation.py --adapt "{adapt_name}"` 做完整验证（adapt_name 由 model_id 经 model_id_to_safe_name 得到；可用 `--skip-status` 跳过 .status.json）。

## 5. 依赖与配置

- **必需**：`accelerate`、`torch`、`transformers`、`safetensors`
- **generate**：`do_sample=False` 避免 NPU 上 ArgSort 的 AiCpu 回退警告

## 6. 断言与校验

模板内置断言：设备为 NPU 或 CUDA；模型首层在 NPU/CUDA 上；`generate()` 输出非空。

## 7. 非 LLM 模型

若适配非因果语言模型，需在生成后的 `demo.py` 中修改：

- **序列分类**：`AutoModelForCausalLM` → `AutoModelForSequenceClassification`
- **视觉**：使用 `AutoImageProcessor` + `AutoModelForImageClassification`
- **Pipeline**：若文档推荐 `pipeline`，优先 `pipeline(..., device=device)`

## 8. 其他优化

- **Flash Attention**：若模型使用，需确认 NPU 兼容版本或回退到标准 attention
