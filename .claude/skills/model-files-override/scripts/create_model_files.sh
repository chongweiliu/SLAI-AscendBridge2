#!/usr/bin/env bash
# 从 HF 缓存创建 model_files 目录，用于性能优化测试。
# 用法: ./create_model_files.sh <adaptation_dir>
# 示例: ./create_model_files.sh adaptations/dutir_bionlp_taiyi_llm

set -e
ADAPT_DIR="${1:?Usage: $0 <adaptation_dir>}"
SNAPSHOT=$(find "$ADAPT_DIR/models" -path "*/snapshots/*" -type d 2>/dev/null | head -1)
if [ -z "$SNAPSHOT" ]; then
  echo "Error: No snapshot found in $ADAPT_DIR/models"
  exit 1
fi
TARGET="$ADAPT_DIR/model_files"
mkdir -p "$TARGET"
echo "[model_files] Snapshot: $SNAPSHOT"
echo "[model_files] Target: $TARGET"

# 1. 权重与词表用符号链接（避免重复占用空间）
for pat in model-*.safetensors pytorch_model-*.bin *.tiktoken *.model; do
  for f in "$SNAPSHOT"/$pat; do
    [ -e "$f" ] && ln -sf "$(realpath "$f")" "$TARGET/$(basename "$f")"
  done
done

# 2. 其余文件复制（config、*.py、*.json 等）
for f in "$SNAPSHOT"/*; do
  [ -d "$f" ] && continue
  base=$(basename "$f")
  [ -e "$TARGET/$base" ] && continue
  cp "$f" "$TARGET/"
done

echo "[model_files] Done. Edit $TARGET/modeling_*.py for optimization, then run accuracy_run_perf.py"
