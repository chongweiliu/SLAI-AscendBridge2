---
name: hf-mirror-outage-modelscope-fallback
description: hf-mirror 宕机时的 ModelScope 全面兜底清单（模型+数据集 API/下载端点）、断点续传 worker 的坑（停滞重下+精确尺寸）、pkill 自匹配连环坑
metadata:
  type: feedback
---

# hf-mirror 宕机时用 ModelScope 全面兜底 + 下载 worker 实战坑（2026-09-03 实测）

**Why:** hf-mirror.com 是单节点（160.16.86.14），会整段宕机（本次实测 17:00-18:35 宕 1.5h，恢复后 18:43 又抖）。宕机期间 ModelScope 可以覆盖绝大多数 HF 模型与相当多数据集，不必干等。

**How to apply:**

## ModelScope 可用端点（全部实测）
- 模型搜索：`PUT https://modelscope.cn/api/v1/dolphin/models` body `{"PageSize":10,"PageNumber":1,"Name":"关键词"}`（必须 Name 字段）
- 模型文件树：`GET https://modelscope.cn/api/v1/models/{owner}/{name}/repo/files?Revision=master&Root=`（含精确 Size）
- 模型下载：`https://modelscope.cn/models/{owner}/{name}/resolve/master/{path}`（支持 Range 断点续传）
- 数据集搜索：`GET https://modelscope.cn/api/v1/datasets?PageSize=6&PageNumber=1&Query=关键词`（不是 dolphin 端点）
- 数据集文件树：`GET https://modelscope.cn/api/v1/datasets/{ns}/{name}/repo/tree?Revision=master&Root={subdir}`（注意是 /repo/tree 不是 /repo/files；子目录用 Root= 参数逐层展开）
- 数据集下载：`https://modelscope.cn/datasets/{ns}/{name}/resolve/master/{path}`

## ModelScope 镜像覆盖情况（实测 2026-09）
- 模型：Qwen 全系/facebook(esm2,dinov2)/microsoft(speecht5,layoutlmv3)/google(timesfm)/openai-mirror(whisper HF 格式)/depth-anything 都有；**没有**：Prithvi-EO-2.0 基座、medicalai/ClinicalBERT、dslim/bert-base-NER、PekingU/rtdetr_r50vd、OLMoE、SmolLM2、bge 系
- 数据集：Salesforce/wikitext、HuggingFaceH4/ultrafeedback_binarized、sentence-transformers/all-nli、openslr/librispeech_asr、autogluon/chronos_datasets、stanfordnlp/imdb 均为同 repo 镜像；AI-ModelScope/{wikipedia-cn-20230720-filtered}、nv-community/esm2_uniref_pretraining_data、OmniData/ESC-50、VoyagerX/hls-burn-scars-tutorial、chandar-lab/ZINC_250k、lmms-lab/flickr30k(parquet)、vikhyatk/nyu_depth_v2(parquet)、C-MTEB/Mmarco-reranking、MTEB/tiny-imagenet、yingxi/conll2003(原生txt)

## 下载 worker 两个致命坑
1. **传输在固定字节处停滞**：curl `-C -` 续传无法越过停滞点（hf-mirror CDN 签名过期 / MS 边缘节点坏缓存），表现为每次重试 size 不变。**正确做法**：停滞 2 次即 `rm -f` 删掉重新下（新连接通常绕开坏点），最多 fresh 重启 8 次。
2. **绝不能用估算 size**：API 显示 "475.10MB" 是四舍五入，真实 475098550。用估算 size 会导致：完整文件被 check_ok 判失败 → 停滞删除 → 无限重下 8 轮 → FAIL-STUCK 把**已经完整的文件删了**。**必须从 API 拿精确字节数**（modelscope tree/files API 返回精确 Size；hf-mirror tree API 用 lfs.size）。

## pkill 自匹配连环坑（本会话中了 5 次）
harness 把整条命令包进 `bash -c 'eval ...'`，命令文本里出现的任何模式串（如 supervisor.sh 路径、"dl_worker"、"curl"）都在进程 cmdline 里，`pkill -f <pattern>` 会杀掉自己的 shell（exit 144）。**解法**：
- 模式动态拼接：`P="dl_""worker"; pkill -f "$P"`（cmdline 里只有 `dl_""worker` 不匹配正则 dl_worker）
- 或把 pkill 写进脚本文件再 `bash /tmp/kill_all.sh`（但注意脚本内 heredoc 若含模式串，外层命令文本同样会匹配！）
- 最稳：`ps aux | grep xxx | grep -v grep` 先确认，能不 pkill 就不 pkill

## 其它
- pgrep -fc 同样会自匹配（pipeline subshell 的 cmdline 含模式），判断进程数用 `ps aux | grep xxx | grep -v grep | wc -l`
- hf-mirror 的 `/resolve/` 对 `refs/convert/parquet` 分支**完全无法下载**（连接超时），只能走数据集 main 分支或 ModelScope
- hf-mirror API（/api/）与下载（/resolve/）可用性不同步：API 大面积失败时 resolve 可能仍正常；反之亦然
- MS 下载 whisper 用 `openai-mirror/whisper-large-v3-turbo`（HF 格式 safetensors），`iic/Whisper-large-v3-turbo` 是原生 .pt

## 补充（2026-09-03 深夜收尾实测）
- **BigCode 系数据集（the-stack-smol/the-stack-dedup）即使 tree API 能列出文件、resolve 也是 403 gated**（"not in the authorized list"，需在 HF 网页人工接受许可）。tree 可见 ≠ 可下载。替代：**codeparrot/codeparrot-clean**（main 分片 json.gz 开放可下，Python 代码语料）
- **hf-mirror 会发生 tree 元数据与实际 serve 的文件版本偏差**（SDXL te/vae/unet、nouns、layoutlmv3 vocab 均差几十 KB）：tree API 的 lfs.size 可能是旧版。**大文件下载前先 `curl -sIL` HEAD 拿 content-length 作为期望尺寸**，比 tree 元数据可靠
- **worker 的 size=0（不校验）校验是陷阱**：镜像提前断连时 curl 正常退出（exit 0）但文件截断，size=0 会误判 DONE。至少要 HEAD 拿真实尺寸
- **镜像晚高峰（19:00-23:00）速度从 5MB/s 掉到 1.7MB/s**，且大文件（>4GB）在 5GB 附近反复停滞触发删重下循环；此类文件直接切 ModelScope（单文件 6.9GB 走 MS 十几分钟搞定）
- 双 supervisor / 双 worker 并发写同一文件是最危险的坑（te2 被 append 到超尺寸）：**启动队列前先 `ps aux | grep -c` 确认无同路径 worker**；孤儿 curl（父 worker 被杀）会继续写文件，杀 worker 时要把子 curl 一起杀

## 2026-09-08 全站长时间宕机实录
- 05:00 起已爬行（每流 6-11 KB/s，旧 TCP 连接未断但无数据），07:15 实测新连接完全失败（`/dev/tcp/160.16.86.14/443` 不通，speed=0）
- **AliDNS DoH（223.5.5.5/resolve）确认 hf-mirror.com 只有一条 A 记录 160.16.86.14**——单节点无备用 IP，Cloudflare DoH(1.1.1.1) 被墙超时。此时换 IP 无解，只能整站切走
- 切 ModelScope 后每流 1.6-2.1 MB/s（三流并行合计 ~20GB/h），Qwen 全系 + Wan-AI 都有官方 MS repo
- 判定"慢 vs 死"的快速方法：`timeout 15 curl -sL -o /dev/null -w "%{speed_download} %{remote_ip}" --max-time 12 <小文件>` speed=0 且 remote_ip 空 = 新连接全挂；再 `/dev/tcp/IP/443` 握手测试确认 IP 层不可达
- 下载 worker 混源排队时（同一模型既在 hf 队列又在 ms 队列）会在 hf 死后白白排队；排大队列前先确认两个源都活着

## CPT 前置权重完整性检查（2026-09-04 实测：Qwen3-1.7B + OLMoE 双双中招）
- **尺寸校验通过的 safetensors 仍可能是比特级损坏**（Qwen3 lm_head 80 个 NaN + 2.5e38 异常值；OLMoE 两个分片共 1065 NaN，均为下载期静默损坏）
- **症状**：训练第一步 loss=NaN（纯 fp32 前向也 NaN，与 autocast/NPU 无关）；CPU 前向同样 NaN 即可确认是权重问题
- **铁律**：CPT 启动前必须 `safetensors.torch.load_file` 全量扫 `torch.isnan(v).sum()`（2B 模型 ~30s，6.9B ~3min）；smoke 出 NaN 先查权重再查环境
- 修复：删损坏分片后 ModelScope 重下（Qwen 系有镜像）；OLMoE 只在 hf-mirror——镜像恢复期重下仍可靠
- 顺带：smoke 的第一步耗时含 NpuFusedAdamW 状态初始化（Qwen3 首步 12s vs 稳态 1.04s/步），勿用首步外推总时长
