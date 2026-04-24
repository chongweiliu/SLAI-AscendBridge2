#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [ -e ".agents" ] && [ ! -L ".agents" ]; then
  echo "[error] ${ROOT_DIR}/.agents exists and is not a symlink" >&2
  exit 1
fi

if [ -e "AGENTS.md" ] && [ ! -L "AGENTS.md" ]; then
  echo "[error] ${ROOT_DIR}/AGENTS.md exists and is not a symlink" >&2
  exit 1
fi

ln -sfn .claude .agents
ln -sfn CLAUDE.md AGENTS.md

echo "[ok] ensured .agents -> .claude"
echo "[ok] ensured AGENTS.md -> CLAUDE.md"
