---
name: gitcode-mr-workflow
description: gitcode 主仓库 SLAI/SLAI-AscendBridge2 的 PR/MR 提交合并流程（git config、push 凭据、v5 API 创建+合并 MR）
metadata:
  type: feedback
---

# gitcode MR 提交合并流程（SLAI/SLAI-AscendBridge2 主仓库）

记录"以后能用"的标准 PR 流程信息。**令牌绝不写入本文件/git config/任何 git 跟踪文件**（agent-memory 是版本控制目录，写令牌=泄露到仓库）。

## 仓库与 git 身份
- **仓库**：`https://gitcode.com/SLAI/SLAI-AscendBridge2.git`（origin）
- **git config（已 --global 设置，以后可直接用）**：
  - `user.name = gcw_WJCkAUuk`
  - `user.email = 86940135@qq.com`

## 令牌安全（红线）
- 令牌在 gitcode 个人设置→访问令牌 生成。
- **绝不** `git config credential` 存令牌、**绝不**写入 agent-memory/任何文件。
- 用法：push 时临时用环境变量 `export GITCODE_TOKEN='<令牌>'` + `git push "https://oauth2:${GITCODE_TOKEN}@gitcode.com/SLAI/SLAI-AscendBridge2.git" HEAD:<分支>`，命令后 `unset GITCODE_TOKEN`。
- **令牌一旦贴进对话/日志即视为泄露，用完立即到 gitcode 撤销/轮换。**

## 标准 PR 流程（已跑通）
1. `git checkout -b feat/<描述>`（不在 main 直接提交）
2. `git add <文件>` + `git commit -m "..." -m "Co-Authored-By: Claude <noreply@anthropic.com>"`
3. push：`export GITCODE_TOKEN='<令牌>'; git push "https://oauth2:${GITCODE_TOKEN}@gitcode.com/SLAI/SLAI-AscendBridge2.git" HEAD:feat/<分支>`（remote config 不留令牌）
4. 创建 MR（gitcode v5 API，**认证用 `PRIVATE-TOKEN` header**，不是 `Authorization: token`/`Bearer`——后者返回 401 token not found）：
   ```bash
   curl -X POST "https://gitcode.com/api/v5/repos/SLAI/SLAI-AscendBridge2/pulls" \
     -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" -H "Content-Type: application/json" \
     -d '{"head":"feat/<分支>","base":"main","title":"...","body":"..."}'
   ```
   返回含 `iid`（MR 编号）、`state:"opened"`。注意：v4/v1 路径 404，**只有 v5 可用**；body 字段是 `head`/`base`（Gitea 风格，非 source_branch/target_branch）。
5. 合并 MR（如需直接合并）：
   ```bash
   curl -X PUT "https://gitcode.com/api/v5/repos/SLAI/SLAI-AscendBridge2/pulls/<iid>/merge" \
     -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" -H "Content-Type: application/json" -d '{"Do":"merge"}'
   ```
   返回 `{"merged":true,"message":"Pull Request 已成功合并"}`。
6. 同步本地：`git checkout main && git pull origin main`（pull public 不需令牌）。

## 实战记录（2026-08-24）
- 分支 `feat/ascend-torch-cpt-950-skill`，commit `1477b58`，MR `!18`，merge commit `3e08dd8`。
- 提交内容：ascend-torch-cpt skill 优化（pitfalls #42-45 + 模板 stub/map_location + SKILL.md + agent-memory），见 [[ascend-cpt-env-pitfalls]]。
- 合并后 main 已含，本地已同步。

### !20 (2026-08-24, stub 策略 MR)
- 分支 `feat/cpt-stub-policy-official`，commit `05b305c`，MR `!20`，merge commit `905d3eb`。
- 提交内容：torchaudio/torchvision 默认走正式匹配版、stub 仅≥3次失败后经 `STUB_MM_FALLBACK=1` 兜底（SKILL.md + pitfalls #39/#43 + cpt_train.py.tmpl/eval_cpt.py.tmpl）。4 文件 +24/-20。
- 合并后远端 main `818435f → 905d3eb`。

### !21 (2026-08-24 晚, 950PR CPT remap cgroup 坑 MR)
- 分支 `feat/cpt-cgroup-remap-pitfall`，commit `f7e3084`，MR `!21`，merge commit `10e5ea6`。
- 提交内容：950PR 单卡 CPT Qwen3.5-4B 实战踩坑沉淀——多模态 remap 在 CPU 三份叠加（model+ckpt+sd fp32 副本≈43GB）撞容器 cgroup 32GB 限制致 OOM 137。新增 pitfalls #46（#44 的 remap 变体，整个 remap 搬 NPU），SKILL.md 阶段1 强制查 cgroup 别信 free，agent-memory [[cpt-950pr-32gb-cgroup]]。

### !22 (2026-08-24 晚, MR 工作流记忆更新)
- MR `!22`，merge commit `ec4e8a9`。提交内容：更新本文件（补 !21 实战 + 令牌凭据存储发现）。

### 仓库卫生修复（2026-08-24，直接 push 非独立 MR）
- 本地 main 比远端多 40 个未推送 optimization/benchmark 提交，其中 `dd38404` 等含 636MB/317MB/149MB/59MB `trace_*.json` 产物（`.gitignore` 只忽略 `trace_*.json.tmp`，漏了无后缀 `trace_*.json`），超 gitcode 100MB 限制挡住整链 push。
- 修复：`git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch adaptations/*/trace_*.json' -- origin/main..main` 从本地 40 提交历史清除大文件（保留与远端共享基线 SHA 不变、不动磁盘产物）；补 `.gitignore` 加 `trace_*.json` 防复发；`git merge origin/main` 合入 !20/!21/!22；fast-forward push 42 提交。备份分支 `backup-main-pre-clean`。
- 教训：`.gitignore` 大产物规则要覆盖无后缀 `trace_*.json`（不只 `.tmp`）；`adaptations/*/` 的 trace/benchmark 产物默认不入库。

### 令牌凭据存储（红线补充）
- 本机 `git config --global credential.helper=store` + `~/.git-credentials`（mode 600，`https://oauth2:<TOKEN>@gitcode.com`）已存令牌。push 直接走凭据存储自动鉴权；v5 API 用 `printf 'protocol=https\nhost=gitcode.com\n\n' | git credential fill`（或 `grep -oP 'oauth2:\K[^@]+' ~/.git-credentials`）取令牌到 shell 变量（不回显），用完 `unset`。**令牌只在凭据存储文件里，不在 agent-memory/对话/日志**。

## 复用命令骨架（下次直接改分支名/文件/标题）
```bash
cd /workspace/SLAI-AscendBridge2
BR=feat/<your-branch>; git checkout -b $BR
git add <files>; git commit -m "<title>" -m "<body>" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
export GITCODE_TOKEN='<令牌>'
git push "https://oauth2:${GITCODE_TOKEN}@gitcode.com/SLAI/SLAI-AscendBridge2.git" HEAD:$BR
curl -X POST "https://gitcode.com/api/v5/repos/SLAI/SLAI-AscendBridge2/pulls" \
  -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" -H "Content-Type: application/json" \
  -d "{\"head\":\"$BR\",\"base\":\"main\",\"title\":\"<title>\",\"body\":\"<body>\"}"
unset GITCODE_TOKEN
```
