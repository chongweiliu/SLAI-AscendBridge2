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

专题文件：

- [dependency-pinning.md](dependency-pinning.md) — 本机已验证的 torch==2.8.0+torch_npu==2.8.0.post4 (cp312/aarch64) 组合、镜像索引选择、虚拟项目写法
- [demo-patterns.md](demo-patterns.md) — transformers 5.x demo.py 要点：from_config 只在 Auto 类、CLIP get_*_features 取 pooler_output、复合 config 收缩、合成输入验证

后续可按需补充：

- `device-selection.md` - 设备选择与 CUDA/NPU 双栈兼容经验
- `custom-repos.md` - 自定义模型库与源码 vendor 处理

其余以本文件摘要、`agents/adapter.md` 和相关 skills 为准。

## ⚠️ nopua skill — 遇到困境必须调用

**nopua 不会自动触发**，需要主动 `Skill("nopua")`。

**触发条件**：同一 action 失败 2+ 次 / 陷入等待循环 / 被动等待而不改变策略。

**正确用法**：1. 停止当前循环 2. 查询 board.db 获取真实状态 3. 根据状态决定下一步 4. 写教训到 MEMORY。

**反面教训**：adapter 若连续重试仍报错，应立即查 adaptation 目录实际状态和 board.db，而非反复重试。

## 专题文件

- [uv-torch-abi-trap.md](uv-torch-abi-trap.md) — uv conflicts 分桶误装新版 torch 致 torch_npu undefined symbol；本机验证组合 torch==2.8.0 + torch-npu==2.8.0.post4
- [qwen3_ascend_stack_verified.md](qwen3_ascend_stack_verified.md) — 2026-08-22 Qwen3-8B 一次通过的可复现依赖区间、选卡模式与 check 用法（裸名）
- [torch_npu_exit_hang.md](torch_npu_exit_hang.md) — [Success] 后进程挂死（futex/线程不退出）的诊断与 flush+os._exit(0) 修法（bge-m3 实案）
- [vlm-demo-pattern.md](vlm-demo-pattern.md) — VLM demo.py 输入路径（apply_chat_template+process_vision_info）；qwen_vl_utils 需 torchvision；transformers>=4.57 from_config→_from_config
- [asr-whisper-pattern.md](asr-whisper-pattern.md) — Whisper ASR：免 torchaudio 合成波形；fp16 mel dtype 对齐；transformers4.57 generation_config 补丁（lang_to_id key 须为 <|en|>），forced_decoder_ids 已废弃
- [diffusers-sdxl-pattern.md](diffusers-sdxl-pattern.md) — SDXL/diffusers：dry-run 从 config 随机权重组装 pipeline；diffusers0.39 to() 静默忽略 torch_dtype=（用位置参数 to(device, dtype)）；单卡整管线
- [cv-timm-pattern.md](cv-timm-pattern.md) — timm 图像分类模型适配模板：import timm 依赖 torchvision，torch==2.8.0↔torchvision==0.23.0 配套，随机图像前向取 Top-1
- [audio-numpy-pattern.md](audio-numpy-pattern.md) — CLAP 类音频-文本对比模型：免 torchaudio（numpy 合成正弦波 + 纯 numpy mel），复合 config 音频分支按 depths 缩层
- [model-families.md](model-families.md) — qwen3_5 族（Qwen3.5/3.8）：transformers>=5.15.1 硬要求、AutoModelForImageTextToText、免 torchaudio、layer_types 截断缩层、27B 跨卡
