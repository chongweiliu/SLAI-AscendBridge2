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
