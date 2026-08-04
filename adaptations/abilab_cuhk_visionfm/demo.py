#!/usr/bin/env python3
"""VisionFM (ABILab-CUHK) — Ascend NPU 适配 demo.

模型: VisionFM ViT-B/16 fundus encoder (NEJM AI 2024).
两种模式:
  (默认) 特征提取: 加载 VFM_Fundus_weights.pth (iBOT 预训练 encoder), 输出 768 维 CLS 特征.
  --classify 青光眼诊断: 加载 PAPILA 微调权重 (encoder + ClsHead), 输出 3 类概率
                         (正常 / 疑似青光眼 / 确诊青光眼).

设备检测: NPU (torch_npu) > CUDA > CPU. 不允许 CPU 回退.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ---- boundary: 缓存目录固定在 adaptation_path/models, 禁止项目根 models/ ----
ADAPT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ADAPT_DIR / "models" / "VFM_Fundus_weights.pth"
CLS_WEIGHTS_PATH = ADAPT_DIR / "models" / "checkpoint_papila.pth"
SAMPLE_DIR = ADAPT_DIR / "sample_images"

# vendored 源码目录: visionfm_src/models/vision_transformer.py + visionfm_src/utils.py
sys.path.insert(0, str(ADAPT_DIR / "visionfm_src"))

# Fundus 归一化常量 (官方 utils.get_stats("Fundus"))
FUNDUS_MEAN = (0.423737496137619, 0.2609460651874542, 0.128403902053833)
FUNDUS_STD = (0.29482534527778625, 0.20167365670204163, 0.13668020069599152)

# PAPILA 青光眼 3 分类 (官方 inference 脚本 RETFoundDataset 的 dr_folder_list)
PAPILA_CLASSES = ["正常 (Normal)", "疑似青光眼 (Suspect Glaucoma)", "确诊青光眼 (Confirmed Glaucoma)"]


def get_device():
    """Returns (device_str, device_count, device_name). NPU > CUDA > CPU."""
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            cnt = torch.npu.device_count()
            print(f"[Device] Huawei Ascend NPU detected, available count={cnt}")
            return "npu:0", cnt, "Huawei Ascend NPU"
        print("[Device] torch_npu imported but no NPU available")
    except ImportError:
        pass
    if torch.cuda.is_available():
        cnt = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"[Device] NVIDIA CUDA detected: {name}, count={cnt}")
        return "cuda:0", cnt, name
    print("[Device] WARNING: No accelerator detected, falling back to CPU")
    return "cpu", 0, "CPU"


def build_model(device):
    """构建 ViT-B/16 encoder (num_classes=0, head=Identity)."""
    from models.vision_transformer import vit_base
    model = vit_base(patch_size=16, num_classes=0, use_mean_pooling=False)
    model.embed_dim = 768
    return model


class ClsHead(nn.Module):
    """官方 models/head.py ClsHead (layers=3), 原样内联以精确匹配权重 key.

    channel_bn: BatchNorm2d(embed_dim)
    classifier: Linear(3072->1536) -> GELU -> Dropout -> Linear(1536->768) -> GELU -> Dropout -> Linear(768->3)
    """
    def __init__(self, embed_dim, num_classes, layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = layers
        if layers == 3:
            channels = [embed_dim, embed_dim // 2, embed_dim // 4, num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]), nn.GELU(), nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]), nn.GELU(), nn.Dropout(p=0.1),
                nn.Linear(channels[2], channels[3]),
            )
        elif layers == 2:
            channels = [embed_dim, embed_dim // 4, num_classes]
            self.classifier = nn.Sequential(
                nn.Linear(channels[0], channels[1]), nn.GELU(), nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]),
            )
        else:
            channels = [embed_dim, num_classes]
            self.classifier = nn.Sequential(nn.Linear(channels[0], channels[1]))
        self.channel_bn = nn.BatchNorm2d(self.embed_dim, eps=1e-6, momentum=0.99)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)  # [B, C] -> [B, C, 1, 1]
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _strip(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def load_pretrained(model, weights_path, device, key="teacher"):
    """加载 encoder 权重. key='teacher' for iBOT pretrained, 'visionfm_state_dict' for fine-tuned."""
    print(f"[Load] reading weights: {weights_path} ({weights_path.stat().st_size/1e9:.2f} GB)")
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = ckpt[key] if key in ckpt else ckpt
    state_dict = _strip(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[Load] encoder missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"[Load]   missing sample: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"[Load]   unexpected sample: {msg.unexpected_keys[:5]}")
    assert len(msg.missing_keys) == 0, f"encoder 权重有缺失 key: {msg.missing_keys[:5]}"
    model = model.to(device)
    model.eval()
    print("[Load] encoder weights loaded, model in eval mode")
    return model


def load_classifier(cls_weights, num_classes, device):
    """加载 PAPILA 微调 ClsHead (classifier_state_dict)."""
    print(f"[Load] reading classifier weights: {cls_weights}")
    ckpt = torch.load(str(cls_weights), map_location="cpu", weights_only=False)
    csd = _strip(ckpt["classifier_state_dict"])
    head = ClsHead(embed_dim=768 * 4, num_classes=num_classes, layers=3)
    msg = head.load_state_dict(csd, strict=False)
    print(f"[Load] classifier missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"[Load]   missing sample: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"[Load]   unexpected sample: {msg.unexpected_keys[:5]}")
    assert len(msg.missing_keys) == 0, f"classifier 权重有缺失 key: {msg.missing_keys[:5]}"
    head = head.to(device)
    head.eval()
    print("[Load] classifier weights loaded, head in eval mode")
    return head


def preprocess(image_path, device):
    """PIL RGB -> resize 224 -> ToTensor -> Normalize(Fundus)."""
    img = Image.open(image_path).convert("RGB")
    print(f"[Input] {Path(image_path).name}: original size={img.size}, mode={img.mode}")
    img = img.resize((224, 224), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(FUNDUS_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(FUNDUS_STD, dtype=torch.float32).view(3, 1, 1)
    arr = (arr - mean) / std
    return arr.unsqueeze(0).to(device)


@torch.no_grad()
def embed(model, inp, name=""):
    """前向, 返回 768 维 CLS token embedding."""
    out = model(inp)  # [1, 768]
    assert out.dim() == 2 and out.shape[1] == 768, f"unexpected output shape {tuple(out.shape)}"
    vec = out.cpu().float().numpy().reshape(-1)
    print(f"[Run] Output: embedding for {name} shape={tuple(out.shape)}")
    print(f"[Run]   L2 norm={np.linalg.norm(vec):.6f}  mean={vec.mean():.6f}  std={vec.std():.6f}")
    print(f"[Run]   min={vec.min():.6f}  max={vec.max():.6f}")
    print(f"[Run]   first 12 dims: {np.array2string(vec[:12], precision=5, max_line_width=120)}")
    return vec


@torch.no_grad()
def classify(model, head, inp, n_last_blocks=4, classes=PAPILA_CLASSES):
    """前向分类: 取最后 n block 的 CLS token 拼接 -> ClsHead -> 3 类概率."""
    feats = model.get_intermediate_layers(inp, n=n_last_blocks)  # list of [1,197,768]
    output = [x[:, 0] for x in feats]  # 取 CLS token, 4 x [1, 768]
    output = torch.cat(output, dim=-1)  # [1, 3072]
    print(f"[Run] intermediate features: {len(feats)} blocks x CLS, concat shape={tuple(output.shape)}")
    logits = head(output)  # [1, 3]
    probs = F.softmax(logits, dim=-1)
    logits = logits.cpu().float().numpy().reshape(-1)
    probs = probs.cpu().float().numpy().reshape(-1)
    pred = int(np.argmax(probs))
    print(f"[Run] logits:  {np.array2string(logits, precision=4)}")
    print(f"[Run] probs:   {np.array2string(probs, precision=4)}")
    print(f"[Diagnosis] 预测 = {classes[pred]}  (置信度 {probs[pred]*100:.2f}%)")
    for i, c in enumerate(classes):
        print(f"            - {c}: {probs[i]*100:5.2f}%")
    return probs, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="不加载权重, 随机初始化模型, 仅验证 NPU 架构兼容性")
    ap.add_argument("--classify", action="store_true",
                    help="青光眼诊断模式: 加载 PAPILA 微调权重, 输出 3 类概率")
    ap.add_argument("--image", default=str(SAMPLE_DIR / "fundus_01.png"))
    ap.add_argument("--image2", default=str(SAMPLE_DIR / "fundus_02.png"))
    ap.add_argument("--weights", default=str(WEIGHTS_PATH), help="encoder 权重 (特征模式)")
    ap.add_argument("--cls-weights", default=str(CLS_WEIGHTS_PATH), help="微调权重 (分类模式)")
    ap.add_argument("--num-labels", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()[0]
    if device == "cpu" and not args.dry_run:
        print("[Device] FATAL: 不允许 CPU 回退做真实推理")
        sys.exit(1)
    print(f"[Device] Using device: {device}")

    torch.manual_seed(0)
    model = build_model(device)

    if args.dry_run:
        print("[Load] dry-run: 随机初始化 (不加载权重)")
        model = model.to(device); model.eval()
        if args.classify:
            head = ClsHead(768 * 4, args.num_labels, layers=3).to(device); head.eval()
            inp = preprocess(args.image, device)
            classify(model, head, inp)
        else:
            inp = preprocess(args.image, device)
            embed(model, inp, name=Path(args.image).name)
        print("[Success] dry-run forward completed on accelerator")
        return

    if args.classify:
        # 分类模式: encoder 用微调 ckpt 里的 visionfm_state_dict, head 用 classifier_state_dict
        model = load_pretrained(model, Path(args.cls_weights), device, key="visionfm_state_dict")
        head = load_classifier(Path(args.cls_weights), args.num_labels, device)
        print(f"[Task] 青光眼诊断 (PAPILA, {args.num_labels} 类)")
        probs_all = {}
        for img_path in [args.image, args.image2]:
            print("-" * 60)
            inp = preprocess(img_path, device)
            print(f"[Input] tensor shape={tuple(inp.shape)} on {device}")
            probs, _ = classify(model, head, inp)
            probs_all[Path(img_path).name] = probs.tolist()
        np.save(ADAPT_DIR / "classification_probs.npy", probs_all)
        print(f"[Save] classification_probs.npy written")
        print("[Success] 青光眼诊断完成, 真实权重推理 on NPU")
    else:
        # 特征模式
        model = load_pretrained(model, Path(args.weights), device, key="teacher")
        inp1 = preprocess(args.image, device)
        vec1 = embed(model, inp1, name=Path(args.image).name)
        if Path(args.image2).exists():
            inp2 = preprocess(args.image2, device)
            vec2 = embed(model, inp2, name=Path(args.image2).name)
            cos = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9))
            print(f"[Sim] cosine(fundus_01, fundus_02) = {cos:.6f}")
        np.save(ADAPT_DIR / "embedding_fundus_01.npy", vec1)
        print(f"[Save] embedding_fundus_01.npy (768-dim) written")
        print("[Success] VisionFM fundus encoder forward completed on accelerator")


if __name__ == "__main__":
    main()
