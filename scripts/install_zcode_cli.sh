#!/bin/bash
# 与 install_zcode.sh 共用安装流程，保留原 zcode-cli 命令名。
set -euo pipefail
export ZCODE_CLI_COMMAND="${ZCODE_CLI_COMMAND:-zcode-cli}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/install_zcode.sh" "$@"
