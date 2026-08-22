# laion/clap-htsat-fused Ascend NPU Adaptation

## 模型信息

- **Model ID**: [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused)
- **架构**: `clap`（CLAP 音频-文本对比模型：HTSAT 音频编码器 + RoBERTa 文本编码器，projection_dim=512）
- **任务**: audio-text contrastive embedding（零样本音频分类/检索）
- **语言**: 英文文本；音频任意
- **规模**: 约 630M 参数（音频分支 4 阶段深度 [2,2,6,2]，文本分支 12 层），fp32

## 环境要求

**必须**安装 NPU 或 CUDA 依赖（二选一）：

```bash
uv sync --extra ascend   # Ascend NPU
# 或
uv sync --extra cuda     # NVIDIA CUDA
```

关键版本约束：

- `transformers>=4.45,<5.0`（CLAP 自 4.27 起原生支持）
- `torch==2.8.0`（两路统一 pin）+ ascend extra 下 `torch-npu==2.8.0.post4`（本机已验证的 ABI 匹配组合）
- **不使用** `[tool.uv] conflicts` 分桶 + `torch>=`：会让 `--extra ascend` 误装最新 torch，与
  torch-npu ABI 不匹配（`undefined symbol: is_contiguous_custom`），此前已实测踩坑
- `requires-python >=3.10,<3.13`

## 验证方式

**不依赖 torchaudio**（本机 torchaudio 损坏，刻意不引入）：demo.py 用 numpy 合成
440Hz 正弦波（1s @48kHz），经 `ClapProcessor` 提取 log-mel 特征（transformers.audio_utils
纯 numpy 实现，无需 librosa/scipy/soundfile），与 3 条候选文本一起编码，输出
`audio_embeds` / `text_embeds` 形状与有限值校验 + 余弦相似度与 Top-1 匹配。

## 使用方式

### Dry Run（随机权重，快速验证）

```bash
uv run python demo.py --dry-run > output.txt 2>&1
```

不下载权重；保守缩小：文本分支 12 -> 2 层，音频分支每阶段深度 [2,2,6,2] -> [1,1,1,1]
（保持阶段数与空间结构不变，projection 维度不受影响）。随机权重下相似度无意义，
仅验证架构与代码路径。

### Full Run（真实权重）

```bash
uv run python demo.py
```

加载预训练权重（`pytorch_model.bin`），缓存到本目录 `models/`。

## 本机（Ascend）注意事项

- **严禁设置 `ASCEND_RT_VISIBLE_DEVICES`**（本机会触发 `aclInit error 107001` /
  `torch.npu.is_available()=False`）；demo.py 用 `torch.npu.set_device()` 选卡。
- 选卡逻辑：优先 `--npu-index` / 环境变量 `NPU_DEVICE_ID`，否则自动选空闲 HBM 最多的卡
  （`torch.npu.mem_get_info`，不写死 0 号卡）。
- HuggingFace 走镜像：默认 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1`。
- 成功结尾使用 `flush + os._exit(0)` 规避 torch_npu 解释器退出挂死（bge-m3 实案沉淀）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo.py` | 主脚本（合成正弦波音频 + 文本对比编码），支持 `--dry-run` |
| `pyproject.toml` | 依赖配置，含 cuda/ascend 可选 extra（torch==2.8.0 pin） |
| `models/` | 模型缓存目录（自动创建） |
| `output.txt` | Dry run 运行输出（重定向生成） |
| `.status.json` | 适配状态记录 |

## 适配状态

- **Dry Run**: 见 `.status.json` / `output.txt`
- **Full Run**: 可选
- **设备**: Ascend NPU / CUDA 双栈；NPU 多卡时固定单卡运行
