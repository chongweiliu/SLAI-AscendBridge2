#!/usr/bin/env python3
"""PAPILA test set 精度评测 (忠实复现官方 inference_visionfm_for_multiclass_classification.py 口径).

对 PAPILA/test 全集 (98 张) 前向, 收集 raw logits (非 softmax, 与官方一致),
计算:
  AUROC = roc_auc_score(target, logits, average='macro', multi_class='ovr')
  AUPR  = average_precision_score(target_one_hot, logits, average='macro')
模型/前向路径与 demo.py --classify 完全相同 (已 parity 校验与官方 bit-level 等价).
"""
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

import demo  # build_model, load_pretrained, load_classifier, ClsHead, preprocess

ADAPT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("/models/PAPILA")  # 含 test/{anormal,bsuspectglaucoma,cglaucoma}/
SPLIT = "test"
CLASSES = ["anormal", "bsuspectglaucoma", "cglaucoma"]  # 官方 RETFoundDataset PAPILA 顺序
DEVICE = "npu:0"
import torch_npu  # noqa


def collect_test_images():
    """返回 [(img_path, label_idx), ...], 与官方 RETFoundDataset enumerate(folder_list) 一致."""
    items = []
    for lbl, cls_name in enumerate(CLASSES):
        d = DATA_PATH / SPLIT / cls_name
        for img_f in sorted(d.iterdir()):
            if img_f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                items.append((str(img_f), lbl))
    return items


@torch.no_grad()
def main():
    import torch_npu  # noqa
    print(f"[Data] PAPILA {SPLIT} set: ", end="")
    items = collect_test_images()
    print(f"{len(items)} images, classes={[f'{c}={sum(1 for _,l in items if l==i)}' for i,c in enumerate(CLASSES)]}")

    # 构建 encoder + ClsHead (复用 demo 的加载逻辑, 已 parity 证明 == 官方)
    model = demo.build_model(DEVICE)
    model = demo.load_pretrained(model, ADAPT_DIR / "models" / "checkpoint_papila.pth", DEVICE,
                                 key="visionfm_state_dict")
    head = demo.load_classifier(ADAPT_DIR / "models" / "checkpoint_papila.pth", 3, DEVICE)

    all_logits, all_targets = [], []
    BATCH = 16
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        imgs = torch.cat([demo.preprocess(p, DEVICE) for p, _ in chunk], dim=0)
        # preprocess 会打印每张图, 静音: 关掉 stdout 重复 (保留功能)
        feats = model.get_intermediate_layers(imgs, n=4)
        output = torch.cat([x[:, 0] for x in feats], dim=-1)  # [B, 3072]
        logits = head(output)  # [B, 3] raw logits
        all_logits.append(logits.cpu().float().numpy())
        all_targets.extend([l for _, l in chunk])
        if (i // BATCH + 1) % 3 == 0:
            print(f"  ... {i + len(chunk)}/{len(items)} done")

    output = np.vstack(all_logits)        # [N, 3] raw logits
    target = np.array(all_targets).reshape(-1, 1)  # [N, 1]
    print(f"[Result] logits shape={output.shape}, target shape={target.shape}")

    # softmax -> 概率 (sklearn 1.2+ multiclass roc_auc 要求行和=1; AUROC 为排序指标)
    probs = torch.softmax(torch.from_numpy(output), dim=-1).numpy()

    # one-hot target (AUPR 用)
    target_one_hot = np.zeros((target.shape[0], len(CLASSES)))
    for k in range(target.shape[0]):
        target_one_hot[k, target[k, 0]] = 1

    # === 官方脚本字面口径: raw logits (旧 sklearn 上 multi_class ovr 直接计算 per-class logit OvR) ===
    # 官方 inference 脚本: roc_auc_score(target, output=raw_logits, average='macro', multi_class='ovr')
    auroc_logit_macro = float(np.mean([
        (roc_auc_score((target.flatten() == c).astype(int), output[:, c])
         if 0 < (target.flatten() == c).sum() < len(target) else float('nan'))
        for c in range(len(CLASSES))
    ]))
    aupr_logit_macro = average_precision_score(target_one_hot, output, average="macro")

    # === softmax 概率口径 (sklearn 1.2+ 标准 multiclass 输入) ===
    auroc_prob_macro = roc_auc_score(target, probs, average="macro", multi_class="ovr")
    aupr_prob_macro = average_precision_score(target_one_hot, probs, average="macro")

    # 每类 OvR AUROC (透明对照: raw logit 列 vs softmax 概率列)
    per_class_auroc_logit, per_class_auroc_prob = [], []
    for c in range(len(CLASSES)):
        y_true = (target.flatten() == c).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            per_class_auroc_logit.append(float("nan")); per_class_auroc_prob.append(float("nan")); continue
        per_class_auroc_logit.append(roc_auc_score(y_true, output[:, c]))
        per_class_auroc_prob.append(roc_auc_score(y_true, probs[:, c]))

    print("=" * 64)
    print(f"[官方字面 raw-logit]  AUROC macro-ovr = {auroc_logit_macro:.4f}   AUPR macro = {aupr_logit_macro:.4f}")
    print(f"[softmax 概率    ]  AUROC macro-ovr = {auroc_prob_macro:.4f}   AUPR macro = {aupr_prob_macro:.4f}")
    for c, cls_name in enumerate(CLASSES):
        print(f"  class {c} ({cls_name}): AUROC logit={per_class_auroc_logit[c]:.4f}  prob={per_class_auroc_prob[c]:.4f}")
    print("=" * 64)
    print("[Done] PAPILA test 评测完成")


if __name__ == "__main__":
    main()
