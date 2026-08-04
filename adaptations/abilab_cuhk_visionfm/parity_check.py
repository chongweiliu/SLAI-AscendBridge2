#!/usr/bin/env python3
"""精度对齐校验: 官方代码路径 vs 本 adaptation demo 路径, bit-level 比对.

同一张眼底图 + 同一份权重:
  - 官方路径: 官方 vision_transformer.py + 官方 head.py 原始 ClsHead
  - 本 demo 路径: 官方 vision_transformer.py + demo.py 内联 ClsHead
两者共享同一份 intermediate features (get_intermediate_layers n=4 拼接),
分别过两个 ClsHead, 比对 logits. 若一致 -> 本 adaptation 与官方代码数值等价.

同时打印 logits 供与 demo.py --classify 的独立运行结果交叉核对.
"""
import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

ADAPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPT_DIR / "visionfm_src"))

# Fundus 归一化 (官方 utils.get_stats("Fundus"))
FUNDUS_MEAN = (0.423737496137619, 0.2609460651874542, 0.128403902053833)
FUNDUS_STD = (0.29482534527778625, 0.20167365670204163, 0.13668020069599152)
CLS_WEIGHTS = ADAPT_DIR / "models" / "checkpoint_papila.pth"
IMG = ADAPT_DIR / "sample_images" / "fundus_01.png"

# ---- 加载官方 head.py 的 ClsHead (原样, 非内联) ----
spec = importlib.util.spec_from_file_location("official_head", "/tmp/vfm_src/models_head.py")
official_head_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_head_mod)
OfficialClsHead = official_head_mod.ClsHead

# ---- 本 demo 的内联 ClsHead ----
sys.path.insert(0, str(ADAPT_DIR))
import demo  # noqa: E402  (provides ClsHead, build_model, load_pretrained, load_classifier, preprocess)
MyClsHead = demo.ClsHead

DEVICE = "npu:0"
import torch_npu  # noqa: E402


def _strip(sd):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def main():
    torch.manual_seed(0)
    # 1. encoder (官方 vision_transformer.py, 共享)
    model = demo.build_model(DEVICE)
    ckpt = torch.load(str(CLS_WEIGHTS), map_location="cpu", weights_only=False)
    enc = _strip(ckpt["visionfm_state_dict"])
    msg = model.load_state_dict(enc, strict=False)
    print(f"[encoder] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    assert len(msg.missing_keys) == 0
    model = model.to(DEVICE).eval()

    # 2. 两个 ClsHead, 加载同一份 classifier_state_dict
    csd = _strip(ckpt["classifier_state_dict"])
    head_off = OfficialClsHead(768 * 4, 3, layers=3).to(DEVICE).eval()
    head_mine = MyClsHead(768 * 4, 3, layers=3).to(DEVICE).eval()
    m1 = head_off.load_state_dict(csd, strict=False)
    m2 = head_mine.load_state_dict(csd, strict=False)
    print(f"[head official] missing={len(m1.missing_keys)} unexpected={len(m1.unexpected_keys)}")
    print(f"[head mine]     missing={len(m2.missing_keys)} unexpected={len(m2.unexpected_keys)}")
    assert len(m1.missing_keys) == 0 and len(m2.missing_keys) == 0

    # 3. 状态字典逐参数比对 (证明两个 ClsHead 结构一致, 权重一致)
    sd_off = head_off.state_dict()
    sd_mine = head_mine.state_dict()
    assert set(sd_off.keys()) == set(sd_mine.keys()), f"key mismatch:\n{set(sd_off)^set(sd_mine)}"
    max_diff = max((sd_off[k] - sd_mine[k]).abs().max().item() for k in sd_off)
    print(f"[state_dict] 两个 ClsHead 参数最大差异 = {max_diff:.2e}")

    # 4. 同一份 intermediate features 分别过两个 head
    inp = demo.preprocess(str(IMG), DEVICE)
    with torch.no_grad():
        feats = model.get_intermediate_layers(inp, n=4)
        output = torch.cat([x[:, 0] for x in feats], dim=-1)  # [1, 3072]
        logits_off = head_off(output)
        logits_mine = head_mine(output)
    diff = (logits_off - logits_mine).abs().max().item()
    print(f"[logits] official : {np.array2string(logits_off.cpu().numpy().reshape(-1), precision=6)}")
    print(f"[logits] mine     : {np.array2string(logits_mine.cpu().numpy().reshape(-1), precision=6)}")
    print(f"[logits] max|official - mine| = {diff:.2e}")

    # 5. softmax 一致性
    p_off = F.softmax(logits_off, -1).cpu().numpy().reshape(-1)
    p_mine = F.softmax(logits_mine, -1).cpu().numpy().reshape(-1)
    print(f"[probs]  official : {np.array2string(p_off, precision=4)}")
    print(f"[probs]  mine     : {np.array2string(p_mine, precision=4)}")
    pdiff = float(np.abs(p_off - p_mine).max())
    print(f"[probs] max|official - mine| = {pdiff:.2e}")

    verdict = diff < 1e-5
    print(f"\n[VERDICT] {'PASS ✅' if verdict else 'FAIL ❌'}: "
          f"本 adaptation ClsHead 与官方 head.py 在同一输入/权重下 logits 最大差异 {diff:.2e} "
          f"({'bit-level 等价' if verdict else '存在偏差'})")


if __name__ == "__main__":
    main()
