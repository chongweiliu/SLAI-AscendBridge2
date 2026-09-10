#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""progress_table.py — 通用长任务进度表（issue #3：进度百分比 + 预计剩余时间）

任何多阶段长任务（批量适配编排 / benchmark / 优化 / CPT / 下载）都能用：
阶段动态添加，状态落 JSON，屏幕实时重印 Markdown 表——每阶段进度% +
整体进度% + 剩余 ETA + 预计完成时刻。

用法:
  progress_table.py --file t.json init                          # 初始化（可选，add 自动建）
  progress_table.py --file t.json add "下载模型权重" 600          # 添加阶段（名称, 预计秒）
  progress_table.py --file t.json overall 3600 "~1h"             # 整体预估秒（可选；缺省用各阶段之和）
  progress_table.py --file t.json doing 1 "3/10 分片"            # 标记进行中（note 含 N/M 自动算阶段%）
  progress_table.py --file t.json doing 1 --pct 30 "下载中"       # 显式阶段百分比
  progress_table.py --file t.json done 1 540 "完成"              # 完成（实际秒）
  progress_table.py --file t.json note 1 "改用镜像"               # 只改说明
  progress_table.py --file t.json show                           # 打印表（默认动作）

进度口径:
  阶段%   = done→100%；doing→note 中 "N/M" 自动解析（如 "50/200步"、"3/10 模型"）或 --pct 显式
  整体%   = (已完成阶段预计之和 + doing阶段预计×阶段%) / 整体预估
  剩余ETA = 整体预估 × (1−整体%)；预计完成时刻 = 当前时间 + 剩余
"""
import argparse
import datetime
import json
import os
import re
import sys


def load(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    return {"overall_estimate_s": None, "overall_note": "", "phases": [], "next_id": 0}


def save(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def fmt(s):
    if s is None:
        return "—"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}min"
    return f"{s/3600:.1f}h"


def parse_pct(note):
    """note 中 "N/M" 模式 → 百分比。"""
    if not note:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", note)
    if m:
        n, t = float(m.group(1)), float(m.group(2))
        if t > 0 and n <= t * 1.05:
            return min(100.0, n / t * 100.0)
    return None


def render(d):
    ico = {"done": "✅done", "doing": "⏳doing", "pending": "⏳pending"}
    lines = ["| # | 阶段 | 预计 | 实际 | 进度 | 状态 | 说明 |", "|---|---|---|---|---|---|---|"]
    sum_est = sum_act = done_est = doing_s = 0.0
    for p in d["phases"]:
        est, act = p.get("est_s"), p.get("actual_s")
        sum_est += est or 0
        if act is not None:
            sum_act += act
        st = p.get("status", "pending")
        pct = p.get("pct")
        if st == "done":
            pct_txt = "100%"
            done_est += est or act or 0
        elif st == "doing":
            if pct is None:
                pct = parse_pct(p.get("note", ""))
            pct_txt = f"{pct:.0f}%" if pct is not None else "…"
            if pct is not None and est:
                doing_s += est * pct / 100.0
        else:
            pct_txt = "—"
        lines.append(f"| {p['id']} | {p['name']} | {fmt(est)} | {fmt(act)} | {pct_txt} | {ico.get(st, st)} | {p.get('note','')} |")
    oest = d.get("overall_estimate_s") or (sum_est or None)
    denom = oest
    overall_pct = min(100.0, (done_est + doing_s) / denom * 100.0) if denom else None
    rem = oest * (1 - overall_pct / 100.0) if (oest and overall_pct is not None) else None
    pct_txt = f" **{overall_pct:.0f}%**" if overall_pct is not None else ""
    rem_txt = f" 剩余 ~{fmt(rem)}" if rem is not None else ""
    eta_txt = ""
    if rem is not None and rem > 0:
        eta_txt = f" 预计完成 {(datetime.datetime.now() + datetime.timedelta(seconds=rem)).strftime('%H:%M')}"
    st_all = ("✅完成" if d["phases"] and all(p.get("status") == "done" for p in d["phases"])
              else "进行中" if any(p.get("status") == "doing" for p in d["phases"]) else "待开始")
    lines.append(f"| **合计** | **{len(d['phases'])} 阶段** | **{fmt(oest) or d.get('overall_note') or '—'}** "
                 f"| **{fmt(sum_act) if sum_act else '—'}** |{pct_txt} | **{st_all}** |{rem_txt}{eta_txt} |")
    return "\n".join(lines)


def find(d, pid):
    for p in d["phases"]:
        if p["id"] == pid:
            return p
    sys.exit(f"错误: 阶段 #{pid} 不存在（现有: {[p['id'] for p in d['phases']]}）")


def main():
    ap = argparse.ArgumentParser(description="通用长任务进度表（百分比+ETA）")
    ap.add_argument("--file", required=True, help="状态 JSON 路径")
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("init")
    sp = sub.add_parser("add"); sp.add_argument("name"); sp.add_argument("est_s", type=float, nargs="?", default=None)
    sp = sub.add_parser("overall"); sp.add_argument("est_s", type=float); sp.add_argument("note", nargs="?", default="")
    sp = sub.add_parser("doing"); sp.add_argument("pid", type=int); sp.add_argument("--pct", type=float, default=None)
    sp.add_argument("note", nargs="*", default=[])
    sp = sub.add_parser("done"); sp.add_argument("pid", type=int); sp.add_argument("actual_s", type=float, nargs="?", default=None)
    sp.add_argument("note", nargs="*", default=[])
    sp = sub.add_parser("note"); sp.add_argument("pid", type=int); sp.add_argument("note", nargs="*")
    sub.add_parser("show")
    a = ap.parse_args()
    d = load(a.file)
    cmd = a.cmd or "show"
    if cmd == "init":
        save(a.file, d)
    elif cmd == "add":
        d["phases"].append({"id": d.get("next_id", len(d["phases"])), "name": a.name,
                            "est_s": a.est_s, "actual_s": None, "status": "pending", "note": ""})
        d["next_id"] = d.get("next_id", len(d["phases"]) - 1) + 1
        save(a.file, d)
    elif cmd == "overall":
        d["overall_estimate_s"] = a.est_s
        d["overall_note"] = a.note
        save(a.file, d)
    elif cmd == "doing":
        p = find(d, a.pid)
        p["status"] = "doing"
        p["note"] = " ".join(a.note) or p.get("note", "")
        if a.pct is not None:
            p["pct"] = a.pct
        save(a.file, d)
    elif cmd == "done":
        p = find(d, a.pid)
        p["status"] = "done"
        if a.actual_s is not None:
            p["actual_s"] = a.actual_s
        if a.note:
            p["note"] = " ".join(a.note)
        p.pop("pct", None)
        save(a.file, d)
    elif cmd == "note":
        p = find(d, a.pid)
        p["note"] = " ".join(a.note) or p.get("note", "")
        save(a.file, d)
    print(render(d), flush=True)


if __name__ == "__main__":
    main()
