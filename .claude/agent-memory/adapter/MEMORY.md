# Adapter Agent Memory

## 精华摘要

### 任务目标

- adapter 负责把模型适配到可运行状态，重点是 `demo.py`、`pyproject.toml`、`README.md`、`.status.json`
- 交付标准优先级是“先能跑，再规范，再补说明”，不要越权去做 benchmark-runner 或 npu-optimizer 的职责

### 关键约束

- 所有产物必须落在 `adaptations/{sanitized_model_name}/`
- 缓存目录固定为 `adaptations/{sanitized_model_name}/models/`
- 代码必须同时兼容 CUDA 和 Ascend NPU，禁止写死为单一设备
- 默认遵循 DRY RUN 约定，避免真实下载大模型或触发超大推理

### 常见失败原因

- 依赖版本不兼容，需在 adaptation 独立环境内 pin 版本
- `trust_remote_code` / 自定义仓库加载链路有缺口
- 权重只支持 pretrained，不支持 `from_config()`
- 模型需要额外 processor/tokenizer/audio/image 预处理逻辑
- 第三方源码仓库有嵌套 `.git`，需清理避免 git 误判

### 提交前检查

- `uv run python demo.py` 能通过
- 必要时运行 `adaptation/scripts/check_adaptation.py`
- `README.md` 说明模型来源、运行方式、已知限制
- 失败时把根因写清楚，便于 team-lead 判断回 `pending` 还是其他状态

## 详细内容

当前目录刚建立，后续专题可按需要补充到同目录，例如：

- `device-selection.md` - 设备选择与 CUDA/NPU 双栈兼容经验
- `custom-repos.md` - 自定义模型库与源码 vendor 处理
- `dependency-pinning.md` - adaptation 级依赖 pin 与兼容性修复
- `demo-patterns.md` - 不同模型类型的 demo.py 模板经验

在专题文件沉淀前，以本文件摘要、`agents/adapter.md` 和相关 skills 为准。

## ⚠️ nopua skill — 遇到困境必须调用

**nopua 不会自动触发**，需要主动 `Skill("nopua")`。

**触发条件**：同一 action 失败 2+ 次 / 陷入等待循环 / 被动等待而不改变策略。

**正确用法**：1. 停止当前循环 2. 查询 board.db 获取真实状态 3. 根据状态决定下一步 4. 写教训到 MEMORY。

**反面教训**：adapter 若连续重试仍报错，应立即查 adaptation 目录实际状态和 board.db，而非反复重试。
