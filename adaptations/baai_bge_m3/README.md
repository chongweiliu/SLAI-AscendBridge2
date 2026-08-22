# BAAI/bge-m3 Ascend NPU Adaptation

## 模型信息

- **Model ID**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **架构**: `xlm-roberta`（`XLMRobertaModel`，encoder-only，sentence-transformers 打包）
- **任务**: feature-extraction / sentence-similarity（multilingual retrieval embedding）
- **语言**: 多语言（100+ 语言，含中文）
- **规模**: 约 568M 参数，24 层，hidden 1024，16 heads，vocab 250002，max length 8192，fp32（约 2.3GB 权重）

## 加载方式说明

`get_model_info.py` 对该模型返回 `transformers_info == {}` / `model_type == "Custom"`，
这是 sentence-transformers 打包导致的误判：仓库内 `config.json` 为标准
`xlm-roberta`（architectures=`XLMRobertaModel`），可直接用标准
`transformers.AutoModel` 加载，**无需克隆外部代码仓库**。
本 demo 使用 mean pooling + L2 归一化生成句向量（官方 FlagEmbedding 用
[CLS] pooling，二者均可用于相似度验证）。

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

关键版本约束：

- `transformers>=4.51,<5.0`、`torch>=2.6.0,<2.9` + 匹配 `torch-npu`
- 额外依赖 `sentencepiece`（XLM-R 分词器）
- `requires-python >=3.10,<3.13`

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

不下载权重；层数保守缩小到 2（24 -> 2）；随机权重下相似度无意义，仅验证
架构与代码路径（编码 + 相似度矩阵 + shape 校验）。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（约 2.3GB，`pytorch_model.bin`），缓存到本目录 `models/`；
NPU 下通过 `max_memory` 限制只使用所选单卡。真实权重下额外做质量断言：
语义相近句对（含跨语言）的余弦相似度必须高于无关句对。

## 本机（Ascend）注意事项

- **严禁设置 `ASCEND_RT_VISIBLE_DEVICES`**（本机会触发 `aclInit error 107001` /
  `torch.npu.is_available()=False`）；demo.py 用 `torch.npu.set_device()` 选卡。
- 选卡逻辑：优先环境变量 `NPU_DEVICE_ID`，否则自动选空闲 HBM 最多的卡（不写死 0 号卡）。
- HuggingFace 走镜像：默认 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本（句向量编码 + 相似度），支持 `--dry-run` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra |
| `models/` | 模型缓存目录（自动创建） |
| `output.txt` | Dry run 运行输出（重定向生成） |
| `full_run_output.txt` | Full run 输出（可选证据） |
| `.status.json` | 适配状态记录 |

## 适配状态

- **Dry Run**: 已验证（见 `.status.json` / `output.txt`）
- **Full Run**: 可选（权重约 2.3GB）
- **设备**: Ascend NPU / CUDA 双栈；NPU 多卡时固定单卡运行

## 备注（供后续阶段参考）

- 该模型为 embedding / retrieval 类型（非生成式），benchmark/optimization 阶段
  应选用句向量/检索类评测口径（如相似度、STS、recall@k），而非文本生成指标。
