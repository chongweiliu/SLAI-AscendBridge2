---
name: lora-v3-cat3-batch
description: 台账第3大类4模型(AI4Science)LoRA实战（mlm/mim两新范式/timm依赖坑致torch断链回滚/AppleDouble数据甄别）
metadata:
  type: project
---

# 台账第 3 大类 4 模型（AI for Science）LoRA 批量实战（2026-09-11）

ClinicalBERT/PubMedQA、ChemBERTa-zinc/zinc250k、esm2_t30_150M/uniref50（均 mlm）+ Prithvi-EO-2.0-300M/hls-burn-scars（mim）。总报告 `lora-ws/.cat3/LoRA-v3-验证报告.md`。

## 结果速记
- **3 mlm 全部双证据改善**：ChemBERTa val loss **-57.7%**（原生预训练即 zinc，本地档分布差异下 LoRA 仍大幅注入）、ClinicalBERT -12.8%、esm2 -1.5%（uniref50 是其原生预训练数据→近饱和）
- **Prithvi mim 完美饱和**：双方 MSE 均 1e-5 量级（与 CPT 批次同结论）；但 MIM 范式全链路打通（timm Block 的 qkv/proj/fc1/fc2 LoRA、peft 对非 HF 模块正常）
- MLM 验证必须**固定掩码**（seed 派生 per-record，base/LoRA 同掩码才可比）；MIM 验证同理（forward 前 `torch.manual_seed` 固定 MAE 内部随机掩码）

## 四个新坑（均修复）
1. **PubMedQA context 是 parquet struct 还原的 dict**（非字符串）——直接 `ctx["contexts"]` 取用，别 eval；只有 str 才需要 eval(np.array)
2. **hls-burn-scars 台账 3233 文件实为 804 张 6 波段影像**：其余是 ~1608 个 macOS AppleDouble 元数据条目（`._*`）+ 804 张单波段 mask 标注（`*.mask.tif`）。切分按 `*_merged.tif` 过滤；tar 里的 `LIBARCHIVE.xattr` warning 无害
3. **timm 安装拖带 torch 升级断链**（最危险）：`uv pip install timm` 把 torch 2.10.0 拖到 2.14.0 + torchvision 0.29，torch_npu 2.10 立即 `undefined symbol` 断链（venv 是 cat1/2/3 共用的！）。修法：`uv pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cpu` 显式 pin 回滚（timm 1.0.29 硬依赖 torchvision，必须装配套 0.25）。**往共享 venv 装新包必须先看会不会动 torch**
4. **mim 范式无 tokenizer**：val/validate 脚本若无条件 `AutoTokenizer.from_pretrained` 会把 Prithvi 目录误判成 timm_wrapper 模型而崩——与训练器一样加 `None if PARADIGM=="mim"` 守卫；另 mlm 验证 device 坑：NPU logits 掩码位索引结果须 `.cpu()` 再与 CPU labels 比较

## 范式沉淀
- 7 范式训练器齐了：cat2 五范式（embed/seq2seq/rerank/cls/ner）+ cat3 两新范式（mlm/mim），`lora-ws/.cat3/scripts/` 为最新版（mlm 掩码 80/10/10、mim 复用 CPT 的 PrithviMAE 加载/归一化/裁剪）
- Prithvi LoRA 目标：timm Block 命名 qkv/proj/fc1/fc2（encoder+decoder 都挂上）；非 HF 模块 get_peft_model/save_pretrained 正常

## 关联
[[lora-v2-cat2-batch]]（五范式+3坑已合入 MR !64）；cat3 的 mlm/mim + 4 新坑可后续 MR。
