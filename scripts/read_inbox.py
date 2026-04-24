#!/usr/bin/env python3
"""
收件箱读取工具 - 读取指定 agent 的消息收件箱。

用法：
    python read_inbox.py --team optimization-team-v7 --agent team-lead  # 读取 team-lead 最近 30 分钟消息（推荐）
    python read_inbox.py --team optimization-team-v7 --agent adapter-1  # 读取指定 agent 的收件箱
    python read_inbox.py --team optimization-team-v7 --agent team-lead --all              # 显示所有消息
    python read_inbox.py --team optimization-team-v7 --agent team-lead --since 15         # 最近 N 分钟的消息
    python read_inbox.py --team optimization-team-v7 --agent team-lead --since 0 --all    # 全部消息（不按时间过滤）
    python read_inbox.py --team optimization-team-v7 --agent team-lead --mark-read        # 将目标收件箱所有消息标记为已读
    python read_inbox.py --team optimization-team-v7 --agent team-lead --clean            # 清理目标收件箱 24 小时前旧消息

关键规则：
- 收件箱是 JSONL 格式（多个 JSON 对象拼接），不能用 json.load() 整体读
- 消息内容在 `text` 字段，不在 `content` 字段
- **不要依赖 read 标记**：有时 read=true 的消息 teammate-message 没收到，有时 read=false 的消息 teammate-message 已收到
- **按时间过滤**：用 --since N 只看最近 N 分钟的消息，这是判断"是否有新消息"的可靠方式
- **必须显式指定团队**：禁止依赖自动推断团队名，避免跨团队误读收件箱
- **必须显式指定 agent**：禁止依赖默认 recipient，避免误读 `team-lead` 或其他成员的收件箱
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_inbox(inbox_path: Path) -> list[dict]:
    """解析 JSONL 收件箱文件，返回消息列表。"""
    if not inbox_path.exists():
        return []

    content = inbox_path.read_text(encoding="utf-8")
    msgs = []
    depth = 0
    start = None
    for i, c in enumerate(content):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(content[start : i + 1])
                    msgs.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return msgs


def extract_message_content(msg: dict) -> dict:
    """从消息中提取真正的内容（text 字段优先于 content 字段）。"""
    raw = msg.get("text") or msg.get("content", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def print_messages(msgs: list[dict], show_all: bool = False, max_count: int = 50):
    """打印消息列表。"""
    if show_all:
        print(f"共 {len(msgs)} 条消息")
    else:
        unread = [m for m in msgs if not m.get("read", False)]
        print(f"共 {len(msgs)} 条消息，其中 {len(unread)} 条未读")

    # 按时间倒序
    msgs_sorted = sorted(msgs, key=lambda m: m.get("timestamp", ""), reverse=True)
    shown = 0
    for m in msgs_sorted:
        if shown >= max_count:
            print(f"\n... 还有 {len(msgs_sorted) - shown} 条消息")
            break
        ts = m.get("timestamp", "")[:19]
        sender = m.get("from", "")
        is_read = m.get("read", False)
        mark = "   " if is_read else "NEW"

        content = extract_message_content(m)
        result = content.get("result", "")
        model_id = content.get("model_id", "")
        progress = content.get("progress", "")
        status = content.get("status", "")
        failure_reason = content.get("failure_reason", "")

        if show_all:
            print(f"[{mark}] {ts} {sender}:")
            if result:
                print(f"  result={result}, model_id={model_id}")
            elif progress:
                print(f"  progress={progress}, status={status}")
            elif failure_reason:
                print(f"  failure_reason={failure_reason[:100]}")
            else:
                print(f"  {json.dumps(content, ensure_ascii=False)[:200]}")
        else:
            # 只显示未读
            if not is_read:
                if result:
                    print(f"[{mark}] {ts} {sender}: result={result}, model_id={model_id}")
                elif progress:
                    print(f"[{mark}] {ts} {sender}: progress={progress}, status={status}")
                elif status == "idle":
                    print(f"[{mark}] {ts} {sender}: status=idle")
                elif failure_reason:
                    print(f"[{mark}] {ts} {sender}: failed={failure_reason[:100]}")
                else:
                    print(f"[{mark}] {ts} {sender}: {json.dumps(content, ensure_ascii=False)[:200]}")

        shown += 1


def mark_all_read(inbox_path: Path):
    """将所有消息标记为已读。"""
    msgs = parse_inbox(inbox_path)
    for m in msgs:
        m["read"] = True
    write_inbox(inbox_path, msgs)
    print(f"已将 {len(msgs)} 条消息全部标记为已读")


def clean_old_messages(inbox_path: Path, hours: int = 24):
    """删除指定小时数之前的旧消息。"""
    msgs = parse_inbox(inbox_path)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    kept = []
    removed = 0
    for m in msgs:
        ts_str = m.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts > cutoff:
                kept.append(m)
            else:
                removed += 1
        except (ValueError, TypeError):
            kept.append(m)  # 解析失败则保留

    write_inbox(inbox_path, kept)
    print(f"清理完成：删除 {removed} 条旧消息，保留 {len(kept)} 条")


def write_inbox(inbox_path: Path, msgs: list[dict]):
    """写回 JSONL 格式收件箱（每行一个 JSON 对象）。"""
    with inbox_path.open("w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def get_inbox_path(team_name: str, agent_name: str) -> Path:
    """根据团队名与 agent 名称定位收件箱文件。"""
    return Path.home() / ".claude" / "teams" / team_name / "inboxes" / f"{agent_name}.json"


def main():
    parser = argparse.ArgumentParser(description="读取指定 agent 收件箱")
    parser.add_argument("--team", "-t", required=True, help="团队名称（必填，如 optimization-team-v7）")
    parser.add_argument(
        "--agent",
        "--recipient",
        dest="agent",
        required=True,
        help="收件箱所属 agent 名称（必填，如 team-lead、adapter-1）",
    )
    parser.add_argument("--all", "-a", action="store_true", help="显示所有消息（不按时间过滤）")
    parser.add_argument("--since", "-s", type=int, default=30, help="只显示最近 N 分钟的消息（默认 30），设为 0 则显示全部")
    parser.add_argument("--mark-read", "-m", action="store_true", help="将所有消息标记为已读")
    parser.add_argument("--clean", "-c", action="store_true", help="清理 24 小时前的旧消息")
    parser.add_argument("--count", "-n", type=int, default=100, help="最多显示消息数（默认 100）")
    args = parser.parse_args()

    inbox_path = get_inbox_path(args.team, args.agent)

    if not inbox_path.exists():
        print(f"错误：收件箱不存在: {inbox_path}", file=sys.stderr)
        print(f"可用团队：{[d.name for d in (Path.home() / '.claude' / 'teams').iterdir() if d.is_dir()]}", file=sys.stderr)
        sys.exit(1)

    if args.mark_read:
        mark_all_read(inbox_path)
        return

    if args.clean:
        clean_old_messages(inbox_path)
        return

    msgs = parse_inbox(inbox_path)

    # 按时间过滤（不依赖 read 标记）
    if args.since > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=args.since)
        filtered = []
        for m in msgs:
            ts_str = m.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts > cutoff:
                    filtered.append(m)
            except (ValueError, TypeError):
                pass  # 解析失败则保留
        msgs = filtered
        print(f"{args.agent} 最近 {args.since} 分钟内消息（共 {len(msgs)} 条）：")

    # --since 时默认显示所有过滤后的消息（不按 read 过滤），--all 时也显示全部
    print_messages(msgs, show_all=args.all or args.since > 0, max_count=args.count)


if __name__ == "__main__":
    main()
