# torch_npu 融合 API 与图模式

## Eager 融合路径（首选，简单可靠）

| 能力 | API | 说明 |
|---|---|---|
| 融合优化器 | `from torch_npu.optim import NpuFusedAdamW` | 替代 torch.optim.AdamW，融合 step。注意 `zero_grad(set_to_none=False)`，与 DDP 需 `gradient_as_bucket_view=False`。**与 FSDP2 不兼容**（meta tensor 无 fake impl）→ FSDP2 用 AdamW |
| 融合 attention | `F.scaled_dot_product_attention` | NPU 自动路由到 fusion attention 内核；transformers full_attention 默认走此路径（attn_implementation="sdpa"）。**不要**手改 eager 除非图模式需要 |
| 异步算子下发 | `export TASK_QUEUE_ENABLE=1` | +5~15% 加速 |
| bf16 混合精度 | `torch.autocast(device_type="npu", dtype=torch.bfloat16)` | fp32 主权重 + bf16 前向。模型保持 fp32 |
| 显存碎片 | `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 减碎片，融合优化器额外缓冲需此 |
| 显式融合 attention | `torch_npu.contrib.npu_fused_attention` / `npu_fused_attention_with_layernorm` | attention+layernorm 融合；侵入式 patch，非必要不碰 |
| DDP | `torch.nn.parallel.DistributedDataParallel` + `backend="hccl"` | 每卡持完整参数/梯度/优化器；`torch_npu.distributed.is_hccl_available()` 验证 |
| NPU 显存探测 | `torch.npu.mem_get_info(i)` | 选空闲卡/估 batch |

## 图模式（torchair，仅长训练/推理，短训练<150步不划算）

### 启用顺序
```python
import torch, torch_npu      # torch_npu 必须先
import torchair              # = torch_npu.dynamo.torchair（顶层别名）
config = torchair.CompilerConfig()          # 默认 max-autotune
npu_backend = torchair.get_npu_backend(compiler_config=config)
opt_model = torch.compile(model, backend=npu_backend, dynamic=False)  # 形状固定用 False
```
- 模式：默认 `max-autotune`（图优化全，编译慢）；`reduce-overhead`（capture&replay，host 开销低）。
- `dynamic=False`：形状固定时首选（更小图、更快编译；且规避 constant_pad_nd 的 tensor-pad 未实现分支）。
- 每进程仅 1 张卡。

### torchair 依赖（GE 初始化需要）
torchair 本随 torch_npu 自带（`torch_npu.dynamo.torchair`），但 GE 初始化缺 Python 侧依赖需补：
```
protobuf scipy attrs decorator cloudpickle ml_dtypes tornado
setuptools<82   # ≥82 移除 pkg_resources 破坏 GE
```
装后跑最小 probe 验证：`torch.compile(lambda x,y: x+y, backend=npu_backend)(a.npu(), b.npu())`。

### 缺失 GE Converter 补全（关键）
torchair 对部分 aten op 是空 stub（直接 raise）。Qwen3.5 线性注意力实测缺：

| op | qwen3_5 用途 | 实现（用 NPU 原生算子） |
|---|---|---|
| `aten.softplus.default/.out` | `F.softplus(a+dt_bias)` 门控 | `ge.Softplus(self)`（beta=1；beta≠1 缩放） |
| `aten.softplus_backward.default/.grad_input` | grad-ckpt 反向重算 | `ge.Mul(grad_output, ge.Sigmoid(self))`（softplus'=sigmoid；\|x\|<<threshold 时精确） |
| `aten.eye.default/.m` | `torch.eye(chunk_size)` | `ge.Eye(num_rows=, dtype=)` |

注册方式（覆盖 stub）：`Converter.__call__` 写入 `aten_op._ge_converter`，故重复注册即覆盖：
```python
from torchair._ge_concrete_graph.ge_converter.converter_utils import register_fx_node_ge_converter, ge, torch_type_to_ge_type, DataType
@register_fx_node_ge_converter(torch.ops.aten.softplus.default)
def _softplus(self, beta=1, threshold=20, meta_outputs=None):
    return ge.Softplus(self)
```
其它 qwen3_5 用到的 op（bmm/mm/cat/exp/sigmoid/silu/rsqrt/cumsum/tril/triu/expand/masked_fill.Scalar/_softmax/...）torchair 均自带可用。

### 图模式 full_attention 处理
full_attention 默认 SDPA→`npu_fusion_attention_v3`，torchair 无此 AscendIR 映射。
**图模式时**设 `text_config._attn_implementation = "eager"`（手动 bmm+softmax+masked_fill，全部 op 有 converter）。
Eager 训练**不需要**改（SDPA 自动走 NPU fusion）。

### 编译成本现实
- 首图编译 ~15–18 min（TBE 算子逐个 autotune，CPU 单核）。
- 100 步训练：编译单项已超整轮 Eager，**不划算**。≥~150–250 步或推理才摊销。
- 不要用 `suppress_errors=True` 回退当真实加速比（技能测量纪律）。
