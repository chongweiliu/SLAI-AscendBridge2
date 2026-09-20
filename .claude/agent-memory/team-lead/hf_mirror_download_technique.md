---
name: hf-mirror-download-technique
description: 国内网络下从 HuggingFace 下载 gated 模型/数据集的可靠方法（DNS污染+hf-mirror+curl断点续传+并行）
metadata:
  type: feedback
---

国内机器下载 HuggingFace 资源的可靠方法（2026-08-26 实测下载 gevaertlab/conflux 模型+数据集）。

**Why:** huggingface.co DNS 被污染（解析到 Facebook 段 31.13.87.19），直连不可达；hf-mirror.com DNS 正常但连接不稳定（HTTP:000 频发）；`hf download` 走 Xet 协议经镜像会卡死在半途。

**How to apply:**

1. **DNS 污染诊断**：`python3 -c "import socket; print(socket.gethostbyname('huggingface.co'))"` 若返回 31.13.x（Facebook 段）即被污染。
2. **hf-mirror 固定 IP 绕 DNS**：`echo "160.16.86.14 hf-mirror.com" >> /etc/hosts`（需 root；该 IP 是 hf-mirror 的 Sakura Japan 节点）。
3. **gated 仓库**：`gated:"auto"` 类型只需有效 HF 令牌（Bearer），无需手动填表，认证请求自动批准。令牌持久化写入 `~/.cache/huggingface/token`（权限 600）+ `~/.bashrc` 的 `HF_TOKEN`/`HF_ENDPOINT=https://hf-mirror.com`。
4. **下载大文件用 curl 断点续传 + 重试，不用 `hf download`**：
   ```
   curl -sL --resolve "hf-mirror.com:443:160.16.86.14" \
     --connect-timeout 5 --max-time 90 --speed-time 15 --speed-limit 3000 \
     -C - -o "$DEST" -H "Authorization: Bearer $TOKEN" \
     "https://hf-mirror.com/<org>/<repo>/resolve/main/<path>"
   ```
   - `-L` 跟随 302 重定向到 `us.aws.cdn.hf.co`（CloudFront CDN，未被污染，可正常解析）。
   - `-C -` 断点续传；每次重试自动重新解析签名 CDN URL（签名有 Expires，过期重试即刷新）。
   - `--speed-time 15 --speed-limit 3000`：15s 内 <3KB/s 即判定卡死并重试，穿透镜像掉线窗口。
5. **多文件并行**：`cat filelist.txt | xargs -P 12 -I {} bash dl_worker.sh {}`。worker 内含 40-80 次重试 + 已完成跳过（`stat -c%s >= 期望大小`）。8-12 路并发聚合 ~5MB/s（单连接仅 ~600KB/s-2MB/s）。
6. **关键坑**：`hf download` CLI 经 hf-mirror 下载大 safetensors 会用 Xet 分块协议，在镜像不稳定时**卡死**（.incomplete 文件停在固定字节不再增长），改用 curl 才稳。

文件树结构可用 `curl -s .../api/models/<id>/tree/main?recursive=true` 获取（注意响应可能含控制字符，用 `json.loads(raw, strict=False)`）。

**hf-mirror 宕机时的备选链（2026-09-01 实测 GENERanno 下载）**：
- hf-mirror.com 是**单节点**（所有公共 DNS 只返回 160.16.86.14），该节点整体宕机时 TCP 443 直接超时，无替代 IP。
- 模型备选：**ModelScope**（`https://modelscope.cn` 国内可达）。搜索用 `PUT /api/v1/dolphin/models` body `{"PageSize":10,"PageNumber":1,"Name":"关键词"}`（必须用 `Name` 字段，`SearchKeyword`/`Keyword` 无效会返回全库列表）。文件下载 `GET /api/v1/models/{owner}/{name}/repo?Revision=master&FilePath={file}`（小文件直接返回内容；remote code 的 modeling/configuration/tokenizer.py 同样可取）。
- 数据集备选：**Zenodo**（`https://zenodo.org/api/records/{id}` 可达且快）；大文件用 12 路 Range 并行（`curl -r start-end` 分段 + cat 合并），单线程 ~65KB/s，12 路聚合 ~300KB/s。
- GitHub 备选：`github.com` 不可达但 **`codeload.github.com`（tarball）和 `raw.githubusercontent.com` 可达**——仓库整体下载走 `https://codeload.github.com/{owner}/{repo}/tar.gz/refs/heads/{branch}`。
- `huggingface.co` DNS 污染 IP 已变为 168.143.171.93（旧记录 31.13.x）。
