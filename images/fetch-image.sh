#!/usr/bin/env bash
# 在具备条件的机器上生成 vLLM-Ascend 离线镜像 tar。
#
# 用法:
#   bash images/fetch-image.sh <tag> [variant]
#   例: bash images/fetch-image.sh v0.23.0rc1 a3
#       bash images/fetch-image.sh v0.19.1rc1 a3
#
# 优先用 crane 直接从 registry 拉成 docker-load 兼容 tar（绕开 docker daemon，
# 单步、按 blob 重试）。crane 不可用时回退 docker pull + docker save（两步，
# 需 daemon，且大镜像可能受 socket 流传输限制）。
# 可通过 CRANE_BIN 指定 crane 命令或可执行文件路径。
#
# 产物: images/vllm-ascend-<tag>-<variant>.tar + .sha256
# 环境要求: 能稳定访问 quay.io（大 blob 不 EOF），或本地 docker daemon 能 save 16GB。

set -euo pipefail

TAG="${1:?用法: $0 <tag> [variant]，例: v0.23.0rc1 a3}"
VARIANT="${2:-a3}"
IMAGE="quay.io/ascend/vllm-ascend:${TAG}-${VARIANT}"
OUT="images/vllm-ascend-${TAG}-${VARIANT}.tar"
cd "$(dirname "$0")/.."   # 回到项目根

echo "[fetch] image=$IMAGE"
echo "[fetch] out=$OUT"

# --- 方式 1: crane 直接拉成 tar（推荐）---
CRANE="${CRANE_BIN:-}"
if [ -z "$CRANE" ] && command -v crane >/dev/null 2>&1; then
  CRANE="$(command -v crane)"
fi

if [ -n "$CRANE" ]; then
  if ! command -v "$CRANE" >/dev/null 2>&1 && [ ! -x "$CRANE" ]; then
    echo "[fetch] CRANE_BIN 不可执行: $CRANE" >&2
    exit 2
  fi
  echo "[fetch] 方式1: $CRANE pull $IMAGE $OUT --format=legacy"
  if "$CRANE" pull "$IMAGE" "$OUT" --format=legacy; then
    sha256sum "$OUT" > "${OUT}.sha256"
    echo "[fetch] OK (crane): $(ls -lh "$OUT" | awk '{print $5}')"
    exit 0
  fi
  echo "[fetch] crane 失败，回退 docker pull+save"
fi

# --- 方式 2: docker pull + docker save（兜底，需 daemon）---
echo "[fetch] 方式2: docker pull + docker save"
docker pull "$IMAGE"
docker save -o "$OUT" "$IMAGE"
sha256sum "$OUT" > "${OUT}.sha256"
echo "[fetch] OK (docker): $(ls -lh "$OUT" | awk '{print $5}')"

# --- 校验 ---
echo "[fetch] 校验: docker load -i $OUT"
docker load -i "$OUT"
docker images | grep "${TAG}-${VARIANT}"
echo "[fetch] 完成。tar=$OUT"
