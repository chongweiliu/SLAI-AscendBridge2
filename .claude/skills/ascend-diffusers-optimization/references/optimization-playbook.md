# Diffusers Optimization Playbook

## 1. 先做 runtime-only

优先做不改模型语义的优化：

- `TASK_QUEUE_ENABLE=1`
- 合理的 allocator 配置
- warmup 后再计时
- 避免不必要的同步

如果 runtime-only 已有真实提速，这条路径在本仓库里可以是正式结论，但必须按 optimization 规则记录为 `runtime_only`。

## 2. 再做 backend 选择

diffusers attention 优先先试：

```python
os.environ["DIFFUSERS_ATTN_BACKEND"] = "_native_npu"
```

要点：

- 尽量在 import diffusers 前设置
- 这是 diffusers 专属路径，不等价于 LLM 上手写 `npu_fusion_attention`

## 3. 再评估 monkey-patch

当热点明确、并且 patch 语义清晰时，再做 monkey-patch：

- `nn.RMSNorm.forward -> npu_rms_norm`
- `GELU.gelu -> npu_gelu`
- `SwiGLU` 语义完全匹配时再上 `npu_swiglu`

原则：

- 优先 patch 类方法，不改 site-packages 文件
- patch 代码放在 `adaptation_path/model_files/` 本地模块中，默认使用 `model_files/npu_patches.py`
- `accuracy_run_perf.py` 负责从 `model_files/` 导入并调用 patch 入口
- 只有明确走自定义仓库源码改写模式时，才直接改 `adaptation_path/<repo>/...`
- patch 后要能显式打印“命中了哪些模块”

## 4. 不要把 diffusers 当成 decoder-only LLM

常见区别：

- attention 往往是 non-causal
- pipeline 中还有 text encoder、VAE、scheduler、offload
- 端到端时间不只由 transformer 内核决定

## 5. 测量口径

正式 optimization 结论仍遵守本仓库规则：

- baseline / perf 均用 pretrained
- baseline / perf 复用同一组卡
- 不能拿 config-only 结果冒充正式速度结论
- 若融合路径失败，先完成 runtime-only 尝试，再决定状态
