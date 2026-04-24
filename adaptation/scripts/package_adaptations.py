#!/usr/bin/env python3

import os
import sqlite3
import zipfile
from pathlib import Path


def _max_mtime_for_dir(adaptation_dir: Path) -> float:
    max_mtime = 0.0
    for root, _dirs, files in os.walk(adaptation_dir):
        for file_name in files:
            path = Path(root) / file_name
            try:
                max_mtime = max(max_mtime, path.stat().st_mtime)
            except OSError:
                pass
    return max_mtime


def package_adaptations(board_db_path="board.db", output_dir="dashboard/artifacts", incremental=None):
    """
    将 board.db 中 adaptation_status=completed 的模型适配目录打包成 zip。
    """
    if incremental is None:
        incremental = os.environ.get("PACKAGE_ARTIFACTS_INCREMENTAL", "1") == "1"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adaptations_dir = Path("adaptations").resolve()
    db_path = Path(board_db_path)
    if not db_path.exists():
        print(f"数据库文件不存在: {board_db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT model_id, adaptation_path
        FROM models
        WHERE adaptation_status = 'completed'
          AND adaptation_path IS NOT NULL
          AND adaptation_path != ''
        """
    )
    rows = cursor.fetchall()
    conn.close()

    print(f"找到 {len(rows)} 个已完成的模型待打包...", flush=True)
    for model_id, path_raw in rows:
        resolved_path = Path(path_raw.strip())
        if not resolved_path.is_absolute():
            resolved_path = Path.cwd() / resolved_path
        if not resolved_path.exists():
            print(f"警告: 模型 {model_id} 的目录不存在: {resolved_path}")
            continue
        try:
            resolved_path = resolved_path.resolve()
            resolved_path.relative_to(adaptations_dir)
        except (OSError, RuntimeError, ValueError):
            print(f"警告: 模型 {model_id} 的目录 {resolved_path} 不在 adaptations/ 下")
            continue

        zip_path = output_path / f"{resolved_path.name}.zip"
        if incremental and zip_path.exists():
            try:
                if _max_mtime_for_dir(resolved_path) <= zip_path.stat().st_mtime:
                    continue
            except OSError:
                pass

        print(f"正在打包 {resolved_path} -> {zip_path}")
        skipped_count = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(resolved_path):
                dirs[:] = [directory for directory in dirs if directory != ".git"]
                for file_name in files:
                    file_path = Path(root) / file_name
                    try:
                        file_path.stat()
                        arcname = file_path.relative_to(resolved_path)
                        zipf.write(file_path, arcname)
                    except (FileNotFoundError, OSError):
                        skipped_count += 1
        if skipped_count > 0:
            print(f"  已跳过 {skipped_count} 个无法访问的文件", flush=True)


if __name__ == "__main__":
    package_adaptations()
