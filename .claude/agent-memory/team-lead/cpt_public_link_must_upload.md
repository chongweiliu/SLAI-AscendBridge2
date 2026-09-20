---
name: cpt-public-link-must-upload
description: CPT 训练产出 loss 曲线后必须立即上传公网直链(catbox→0x0→uguu)，不能只存本地就跳到报告
metadata:
  type: feedback
---

# CPT loss 曲线必须上传公网直链（2026-08-26 CONFLUX CPT 漏做教训）

**规则**：CPT 训练产出 `loss_curve.png` 后，**必须立即**尝试上传公网直链（catbox.moe → 0x0.st → uguu.se 顺序），校验 HTTP 200 后告知用户直链，**然后**才进报告阶段。不能画完 png 就跳到 README/评估。

**Why**：CONFLUX CPT 任务中，我画完 loss_curve.png 就直接写报告了，漏了 skill（ascend-torch-cpt 核心原则 + 阶段7）明确要求的公网直链上传。用户追问才发现补上。skill 默认产出就是"loss 曲线 + 公网链接"。

**How to apply**：
- 训练脚本/画图脚本之后，紧跟一个上传步骤（curl -F file=@loss_curve.png catbox.moe/user/api.php / 0x0.st / uguu.se/upload）
- catbox 常无响应、0x0.st 因 spam 停用上传，uguu.se 通常可用——三者依次试
- uguu.se 是临时直链（几小时过期），长期复看以本地 png 为准；链接存 `outputs/public_links.json`
- 把"上传公网直链"作为 CPT 流程的必经步骤写进 checklist，不能依赖记忆

**关联**：[[ascend-torch-cpt]] skill 阶段7、[[conflux-npu-adaptation]]。
