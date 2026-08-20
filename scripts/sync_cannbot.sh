#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CACHE_PARENT="${PROJECT_ROOT}/.cache/cannbot"
CANNBOT_ROOT="${CACHE_PARENT}/cannbot-skills"
UPSTREAM_URL="https://gitcode.com/cann/cannbot-skills.git"
UPSTREAM_BRANCH="master"
VERSION_FILE="${CANNBOT_ROOT}/.slai-upstream-version"
LOCK_DIR="${CACHE_PARENT}/.sync-lock"

usage() {
  printf '%s\n' \
    "Usage: scripts/sync_cannbot.sh [--print-path|--status]" \
    "" \
    "Without --status, checks upstream and synchronizes the latest master." \
    "The cache never modifies Claude user settings or project identity files."
}

mode="sync"
case "${1:-}" in
  ""|--print-path) mode="sync" ;;
  --status) mode="status" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

if [ "${mode}" = "status" ]; then
  if [ ! -d "${CANNBOT_ROOT}/.git" ]; then
    printf 'not installed\n'
    exit 1
  fi
  git -C "${CANNBOT_ROOT}" rev-parse HEAD
  exit 0
fi

mkdir -p "${CACHE_PARENT}"

acquire_lock() {
  attempts=0
  while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    attempts=$((attempts + 1))
    if [ "${attempts}" -ge 120 ]; then
      printf '[error] timed out waiting for CANNBot project-cache lock: %s\n' "${LOCK_DIR}" >&2
      exit 1
    fi
    sleep 0.25
  done
  printf '%s\n' "$$" > "${LOCK_DIR}/pid"
}

release_lock() {
  if [ -d "${LOCK_DIR}" ]; then
    rm -f "${LOCK_DIR}/pid"
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
}

acquire_lock
trap release_lock EXIT INT TERM

if [ ! -d "${CANNBOT_ROOT}/.git" ]; then
  if [ -e "${CANNBOT_ROOT}" ]; then
    printf '[error] cache path exists but is not a Git checkout: %s\n' "${CANNBOT_ROOT}" >&2
    exit 1
  fi
  git clone --branch "${UPSTREAM_BRANCH}" --single-branch "${UPSTREAM_URL}" "${CANNBOT_ROOT}" >&2
else
  current_url="$(git -C "${CANNBOT_ROOT}" remote get-url origin)"
  if [ "${current_url}" != "${UPSTREAM_URL}" ]; then
    printf '[error] unexpected CANNBot cache origin: %s\n' "${current_url}" >&2
    exit 1
  fi
  if [ -n "$(git -C "${CANNBOT_ROOT}" status --porcelain --untracked-files=no)" ]; then
    printf '[error] CANNBot cache contains local changes; refusing to overwrite: %s\n' "${CANNBOT_ROOT}" >&2
    exit 1
  fi
  git -C "${CANNBOT_ROOT}" fetch --prune origin "${UPSTREAM_BRANCH}" >&2
  git -C "${CANNBOT_ROOT}" checkout --detach "origin/${UPSTREAM_BRANCH}" >&2
fi

required_paths=(
  "plugins-official/ops-direct-invoke/agents/ascendc-kernel-architect.md"
  "plugins-official/ops-direct-invoke/agents/ascendc-kernel-design-reviewer.md"
  "plugins-official/ops-direct-invoke/agents/ascendc-kernel-developer.md"
  "plugins-official/ops-direct-invoke/agents/ascendc-kernel-reviewer.md"
  "plugins-official/ops-direct-invoke/workflows/task-prompts.md"
  "ops/ascendc-env-check/SKILL.md"
)
for required_path in "${required_paths[@]}"; do
  if [ ! -f "${CANNBOT_ROOT}/${required_path}" ]; then
    printf '[error] latest CANNBot is missing required asset: %s\n' "${required_path}" >&2
    exit 1
  fi
done

commit="$(git -C "${CANNBOT_ROOT}" rev-parse HEAD)"
subject="$(git -C "${CANNBOT_ROOT}" log -1 --format=%s)"
printf 'source=%s\nbranch=%s\ncommit=%s\nsubject=%s\n' \
  "${UPSTREAM_URL}" "${UPSTREAM_BRANCH}" "${commit}" "${subject}" > "${VERSION_FILE}"

printf '%s\n' "${CANNBOT_ROOT}"
