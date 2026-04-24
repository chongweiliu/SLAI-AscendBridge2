# Common Diffusers Fixes

## 1. VAE decode 跨设备

当主干组件被 dispatch 到多卡时，最终 latent 可能不在 VAE 所在设备上。常见修复方式：

- 让 `vae` 固定在主卡
- 在 `vae.decode()` 前把输入 latent 显式搬到目标设备

不要假设 pipeline 会自动处理这个跨设备移动。

## 2. `device_map="balanced"` 不稳定

如果 pipeline 级 `device_map="balanced"` 在 Ascend 环境报初始化或精度模式错误：

- 不要继续在这个入口上反复试
- 改为“先 CPU 加载，再对子组件手动 dispatch”

## 3. complex64 / `torch.polar` / `view_as_complex`

如果模型用复数形式实现 RoPE 或频率编码，Ascend 上常见问题是：

- complex64 索引不兼容
- `view_as_complex` / `torch.polar` 路径报错

常用修法：

- 在 CPU 上完成必要的复数索引
- 尽快转成实数 `cos/sin` 或 `[..., 2]` 形式
- 在 NPU 上改用等价的实数算术

## 4. generator 设备不匹配

在 offload 或多设备场景中，`torch.Generator(device="npu")` 不一定稳定。

若遇到跨设备或 dispatch 相关问题：

- 先尝试把 generator 放到 CPU
- 明确记录这个选择，避免后续 benchmark 和 optimization 口径不一致

## 5. 清理显存

在多阶段流程里，编码器、主干、VAE 常常是分时使用。

每次组件切换后建议：

```python
del obj
gc.collect()
torch.npu.empty_cache()
```

如果要测时间，再补 `torch.npu.synchronize()`。
