---
name: skill-change-verify-protocol
description: 对 ascend-torch-cpt skill 的任何改动必须做改动前后回归验证，确保不破坏原有功能
metadata:
  type: feedback
---

任何对 ascend-torch-cpt skill（及仓库内其它 skill）的改动，提交前**必须**做改动前后回归验证，确保新改动不影响之前原有功能。

**Why**：用户 2026-08-25 明确要求"任何对这个 skill 的改动，要对改动前后进行验证评估判断，确保新的改动更新不会影响之前原有功能"。skill 是多模型类型通用的，文本LM路径是已验证可用的基线，新加范式(如扩散生成式)不能破坏它。

**How to apply**（每次改 skill 后、提交 MR 前执行这套检查）：
1. **未改动文件确认**：`git diff <父commit> <本commit> -- <每个原有模板/references文件>` 应为空（diff 行数=0）。本次验证：9 个原有 .tmpl + 7 个原有 references 全部 diff=0 ✅。
2. **追加而非覆盖**：若新增 pitfalls/内容，确认是纯追加——`git diff | grep '^-[^-]'` 的真实删除行应为 0（注意 `--- a/...` 文件头行不算）。标题逐条对照 `diff <(git show 父:文件|grep '^## ') <(git show 本:文件|grep '^## '|head -N)` 应一致。
3. **原有路径关键内容保留**：grep 原有范式的关键词（文本LM：PPL/DDP/FSDP2/input_ids/apply_chat_template/NpuFusedAdamW/ckpt_latest/first5/AutoModelForCausalLM 等）出现次数应≥原值，不缺失。
4. **模板语法**：`python3 -c "import ast; ast.parse(open(f).read())"` 对所有 .py.tmpl 通过。
5. **引用完整性**：SKILL.md/references 里引用的脚本文件必须都存在。
6. **新增分支是加法**：SKILL.md 的范式分支应是"A=原文本LM内容(原措辞保留)+B=新范式"，不是替换。用 `git diff` 确认 A 分支内容仍在。

验证全过才提交 MR。相关：[[gitcode-mr-workflow]]、[[cpt-950pr-32gb-cgroup]]、[[ascend-cpt-env-pitfalls]]。
