# Diffusers Migration Workflow

## 1. 先确认 pipeline 入口

优先确认以下事实：

- pipeline 类名，如 `FluxPipeline`、`Flux2Pipeline`、`StableDiffusion3Pipeline`、`WanPipeline`
- 主要组件有哪些：`text_encoder`、`transformer` / `unet`、`vae`、`scheduler`、`tokenizer`
- 哪个组件最大，哪个是真正的瓶颈
- 权重是否在本地，是否需要 `trust_remote_code`

## 2. adaptation-local 加载规则

- 所有缓存都写到 `adaptation_path/models/`
- 不要把模型缓存写到项目根 `models/`
- 不要在项目根运行会下载权重的命令

典型加载骨架：

```python
from diffusers import SomePipeline

cache_dir = Path(__file__).resolve().parent / "models"
pipe = SomePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    cache_dir=str(cache_dir),
)
```

## 3. dry run 设计

diffusers 的 dry run 重点是“覆盖真实主路径”，不是“完整生成最终媒体”。

推荐顺序：

1. 实例化主干组件或 pipeline
2. 把主干组件迁移到 NPU / CUDA
3. 构造最小输入做一次前向
4. 记录设备、dtype、关键张量 shape

如果整条 pipeline 太重，可只验证主干：

- 图像模型通常优先验证 `transformer` 或 `unet`
- 视频模型通常优先验证 transformer 主干与 latent 路径

## 4. full run 设计

真实权重路径应尽量接近模型卡推荐方式，但要满足本仓库要求：

- 入口统一放在 `demo.py`
- 同时支持 Ascend 与 CUDA
- 日志里明确输出所选设备与缓存路径

## 5. 何时判定不是标准 diffusers 路线

出现以下情况时，不要继续按标准 pipeline 写法强推：

- 关键目录缺失，无法 `from_pretrained`
- 模型发布格式是单文件量化，不是标准组件目录
- 模型卡明确要求特定 GPU-only 内核

此时应改为：

- 调查是否存在自定义 repo 路线
- 或明确写出该格式为何不适合当前仓库的标准 diffusers 适配流程

注意：

- “不适合标准 diffusers 适配流程” 不等于自动 `not_applicable`
- 先回到自定义流程 / 额外仓库调查；只有命中 adapter 的格式预检或取得强平台不适用证据后，才允许标 `not_applicable`
