#!/bin/bash

# Claude Code 容器内一键安装脚本
# 用法: bash install_claude_code.sh [模型名称] [API_KEY] [BASE_URL]
# 交互模式 (不传参): 先弹提供方菜单(智谱/百炼/幻方/火山/MiniMax/Kimi/阶跃/Anthropic/自定义),
#                   选完后填模型名(带默认) → 填 API_KEY, URL 由所选提供方自动给出
# 非交互示例:
#       bash install_claude_code.sh glm-4.6 sk-xxx                              # 前缀自动匹配智谱
#       bash install_claude_code.sh glm-4.6 sk-xxx https://open.bigmodel.cn/api/anthropic
#       bash install_claude_code.sh Qwen3-30B-A3B "" http://localhost:8000/v1
#       bash install_claude_code.sh MiniMax-M1 sk-xxx                           # 前缀自动匹配 minimax
#       CLAUDE_PROVIDER=bailian bash install_claude_code.sh deepseek-v3 sk-xxx  # 百炼托管的任意模型
#       CLAUDE_PROVIDER=moonshot bash install_claude_code.sh kimi-k2 sk-xxx
# URL 解析优先级: 显式 BASE_URL > CLAUDE_PROVIDER > 模型名前缀 > 交互菜单
#   CLAUDE_PROVIDER 取值: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun
#   适配"一个 API 支持多种模型"的平台 (百炼/火山/MiniMax 等), URL 与模型名无关时用它最稳
# 仿照智谱官方脚本 https://cdn.bigmodel.cn/install/claude_code_env.sh

set -euo pipefail

# ========================
#       常量定义
# ========================
NODE_MIN_VERSION=22
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-22}"
CLAUDE_PACKAGE="@anthropic-ai/claude-code"
API_TIMEOUT_MS=3000000

# ========================
#       工具函数
# ========================
log_info()    { echo "🔹 $*"; }
log_success() { echo "✅ $*"; }
log_error()   { echo "❌ $*" >&2; }

# ========================
#    参数解析
# ========================
# ========================
#  平台清单 (展示名 | provider key | 默认模型 | Anthropic 兼容 URL)
# ========================
PROVIDER_NAMES=(
    "智谱 AI (GLM)"
    "阿里百炼 (Qwen)"
    "幻方 DeepSeek"
    "火山方舟 (豆包)"
    "MiniMax (M1/abab)"
    "月之暗面 (Kimi)"
    "阶跃星辰 (Step)"
    "Anthropic 官方"
    "自定义 (手动填 URL)"
)
PROVIDER_KEYS=(  "zhipu"      "bailian"      "deepseek"   "volcano"   "minimax"   "moonshot"   "stepfun"     "anthropic"      "custom" )
PROVIDER_MODELS=("glm-4.6"    "qwen3-coder-plus" "deepseek-v3" "doubao-seed-1-6-250615" "MiniMax-M1" "kimi-k2"   "step-2"      "claude-sonnet-4-5" "" )
PROVIDER_URLS=(
    "https://open.bigmodel.cn/api/anthropic"
    "https://dashscope.aliyuncs.com/apps/anthropic"
    "https://api.deepseek.com/anthropic"
    "https://ark.cn-beijing.volces.com/api/coding"
    "https://api.minimaxi.com/anthropic"
    "https://api.moonshot.cn/anthropic"
    "https://api.stepfun.com/step_plan"
    "https://api.anthropic.com"
    ""
)

# ========================
#  平台(provider) → Anthropic 兼容 URL
# ========================
url_for_provider() {
    case "$1" in
        anthropic|claude)          echo "https://api.anthropic.com" ;;
        zhipu|glm)                 echo "https://open.bigmodel.cn/api/anthropic" ;;
        deepseek)                  echo "https://api.deepseek.com/anthropic" ;;
        bailian|dashscope|qwen)    echo "https://dashscope.aliyuncs.com/apps/anthropic" ;;
        volcano|ark|doubao)        echo "https://ark.cn-beijing.volces.com/api/coding" ;;
        minimax)                   echo "https://api.minimaxi.com/anthropic" ;;
        moonshot|kimi)             echo "https://api.moonshot.cn/anthropic" ;;
        stepfun|step)              echo "https://api.stepfun.com/step_plan" ;;
        *)                         echo "" ;;
    esac
}

# ========================
#  模型名前缀 → Anthropic 兼容 URL (无 provider 时的回退)
# ========================
get_provider_url() {
    local model="${1,,}"   # 小写化后匹配, 兼容 Qwen3 / GLM-4.6 等大写写法
    case "$model" in
        claude-*)                  echo "https://api.anthropic.com" ;;
        deepseek-*)                echo "https://api.deepseek.com/anthropic" ;;
        glm-*|chatglm-*)           echo "https://open.bigmodel.cn/api/anthropic" ;;
        doubao-*)                  echo "https://ark.cn-beijing.volces.com/api/coding" ;;
        qwen*|qwq*)                echo "https://dashscope.aliyuncs.com/apps/anthropic" ;;
        minimax-*|MiniMax-*|abab*) echo "https://api.minimaxi.com/anthropic" ;;
        kimi-*|moonshot-*)         echo "https://api.moonshot.cn/anthropic" ;;
        step-*)                    echo "https://api.stepfun.com/step_plan" ;;
        *)                         echo "" ;;
    esac
}

# ========================
#  交互式选择提供方菜单
# ========================
select_provider_menu() {
    echo "请选择模型提供方:"
    local i=1
    for name in "${PROVIDER_NAMES[@]}"; do
        printf "  [%d] %s\n" "$i" "$name"
        i=$((i+1))
    done
    echo "  [0] 退出"
    echo ""
    local choice
    read -e -p "请选择 [1-${#PROVIDER_NAMES[@]}]: " choice
    if [ -z "$choice" ] || [ "$choice" = "0" ]; then
        log_error "未选择提供方"; exit 1
    fi
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#PROVIDER_NAMES[@]}" ]; then
        log_error "无效选择: $choice"; exit 1
    fi
    local idx=$((choice-1))
    SELECTED_KEY="${PROVIDER_KEYS[$idx]}"
    SELECTED_MODEL="${PROVIDER_MODELS[$idx]}"
    SELECTED_URL="${PROVIDER_URLS[$idx]}"
}

# ========================
#  参数与提供方选择
#  URL 解析优先级: 显式 BASE_URL > CLAUDE_PROVIDER > 模型前缀 > 交互菜单
# ========================
CUSTOM_URL="${3:-}"
PROVIDER="${CLAUDE_PROVIDER:-}"
MODEL="${1:-}"
API_KEY="${2:-}"

# 交互模式 (未传模型名且未指定 provider/url): 先选提供方, 再填模型与 key
if [ -z "$MODEL" ] && [ -z "$CUSTOM_URL" ] && [ -z "$PROVIDER" ]; then
    select_provider_menu
    PROVIDER="$SELECTED_KEY"
    if [ "$PROVIDER" = "custom" ]; then
        read -e -p "请输入 API BASE_URL (如 http://192.168.1.100:8000/v1): " PROVIDER_URL
        [ -z "$PROVIDER_URL" ] && { log_error "API 地址不能为空"; exit 1; }
    else
        PROVIDER_URL="$SELECTED_URL"
    fi
    read -e -p "请输入模型名称 (默认 ${SELECTED_MODEL}): " MODEL
    MODEL="${MODEL:-$SELECTED_MODEL}"
    read -e -p "请输入 API_KEY (默认 sk-local): " API_KEY
    API_KEY="${API_KEY:-sk-local}"
else
    # 非交互: 位置参数 + 环境变量
    [ -z "$MODEL" ] && { read -e -p "请输入模型名称 (默认 deepseek-v4-pro): " MODEL; MODEL="${MODEL:-deepseek-v4-pro}"; }
    if [ -z "$API_KEY" ]; then
        read -e -p "请输入 API_KEY (默认 sk-local): " API_KEY
        API_KEY="${API_KEY:-sk-local}"
    fi
    if [ -n "$CUSTOM_URL" ]; then
        PROVIDER_URL="$CUSTOM_URL"
    elif [ -n "$PROVIDER" ]; then
        PROVIDER_URL=$(url_for_provider "$PROVIDER")
        if [ -z "$PROVIDER_URL" ]; then
            log_error "未知 CLAUDE_PROVIDER: $PROVIDER"
            echo "支持: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun" >&2
            exit 1
        fi
    else
        PROVIDER_URL=$(get_provider_url "$MODEL")
    fi
    if [ -z "$PROVIDER_URL" ]; then
        echo ""
        echo "模型 '${MODEL}' 不在已知前缀列表中。"
        echo "可选: 1) 显式传 BASE_URL;  2) 设 CLAUDE_PROVIDER=<平台>;  3) 下方手动输入"
        echo "已知前缀: claude-*, deepseek-*, glm-*, doubao-*, qwen*, minimax-*, abab*, kimi-*, moonshot-*, step-*"
        echo "已知 provider: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun"
        echo ""
        read -e -p "请输入 API BASE_URL (如 http://192.168.1.100:8000/v1): " PROVIDER_URL
        [ -z "$PROVIDER_URL" ] && { log_error "API 地址不能为空"; exit 1; }
    fi
fi

echo ""
echo "========================================"
echo "  模型:    ${MODEL}"
[ -n "$PROVIDER" ] && echo "  平台:    ${PROVIDER}"
echo "  API URL: ${PROVIDER_URL}"
echo "  API_KEY: ${API_KEY}"
echo "========================================"
echo ""

# ========================
#    Node.js 安装
# ========================
check_nodejs() {
    if command -v node &>/dev/null; then
        current_version=$(node -v | sed 's/v//')
        major_version=$(echo "$current_version" | cut -d. -f1)
        if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
            log_success "Node.js is already installed: v$current_version"
            return 0
        fi
        log_info "Node.js v$current_version < $NODE_MIN_VERSION, upgrading..."
    fi

    log_info "Installing Node.js..."

    local machine_arch node_arch node_full_ver node_tar node_archive checksums_file
    local dist_base index_data expected_checksum actual_checksum install_prefix version_dir
    machine_arch=$(uname -m)
    case "$machine_arch" in
        aarch64|arm64) node_arch="linux-arm64" ;;
        x86_64|amd64) node_arch="linux-x64" ;;
        *) log_error "Unsupported architecture: $machine_arch"; return 1 ;;
    esac

    # 可用 NODE_VERSION=v22.x.y 锁定版本；未指定时解析 Node 22 的最新补丁版本。
    if [ -n "${NODE_VERSION:-}" ]; then
        node_full_ver="$NODE_VERSION"
        [[ "$node_full_ver" == v* ]] || node_full_ver="v${node_full_ver}"
    else
        index_data=""
        for dist_base in "https://nodejs.org/dist" "https://npmmirror.com/mirrors/node"; do
            log_info "Resolving latest Node.js ${NODE_INSTALL_VERSION}.x from ${dist_base}..."
            if index_data=$(curl --fail --show-error --silent --location --retry 3 --connect-timeout 15 "${dist_base}/index.tab"); then
                node_full_ver=$(printf '%s\n' "$index_data" | awk -v prefix="v${NODE_INSTALL_VERSION}." 'NR > 1 && index($1, prefix) == 1 { print $1; exit }')
                [ -n "$node_full_ver" ] && break
            fi
        done
        if [ -z "${node_full_ver:-}" ]; then
            log_error "Unable to resolve the latest Node.js ${NODE_INSTALL_VERSION}.x version. Set NODE_VERSION=v${NODE_INSTALL_VERSION}.x.y and retry."
            return 1
        fi
    fi

    # 使用 gzip 包，避免 minimal 容器缺少 xz 时静默退出。
    node_tar="node-${node_full_ver}-${node_arch}.tar.gz"
    node_archive=$(mktemp "/tmp/${node_tar}.XXXXXX")
    checksums_file=$(mktemp "/tmp/node-shasums.XXXXXX")

    dist_base=""
    local candidate_base
    for candidate_base in "https://nodejs.org/dist" "https://npmmirror.com/mirrors/node"; do
        log_info "Downloading Node.js ${node_full_ver} (${node_arch}) from ${candidate_base}..."
        if curl --fail --show-error --location --retry 3 --connect-timeout 15 \
            "${candidate_base}/${node_full_ver}/${node_tar}" -o "$node_archive"; then
            dist_base="$candidate_base"
            break
        fi
        log_error "Download failed from ${candidate_base}; trying the next mirror."
    done
    if [ -z "$dist_base" ]; then
        rm -f "$node_archive" "$checksums_file"
        log_error "Failed to download ${node_tar} from all configured mirrors."
        return 1
    fi

    log_info "Verifying Node.js archive checksum..."
    if ! curl --fail --show-error --silent --location --retry 3 --connect-timeout 15 \
        "${dist_base}/${node_full_ver}/SHASUMS256.txt" -o "$checksums_file"; then
        rm -f "$node_archive" "$checksums_file"
        log_error "Failed to download SHASUMS256.txt from ${dist_base}."
        return 1
    fi
    expected_checksum=$(awk -v filename="$node_tar" '$2 == filename { print $1; exit }' "$checksums_file")
    if command -v sha256sum >/dev/null 2>&1; then
        actual_checksum=$(sha256sum "$node_archive" | awk '{ print $1 }')
    elif command -v shasum >/dev/null 2>&1; then
        actual_checksum=$(shasum -a 256 "$node_archive" | awk '{ print $1 }')
    else
        rm -f "$node_archive" "$checksums_file"
        log_error "Neither sha256sum nor shasum is available; refusing to install an unverified archive."
        return 1
    fi
    if [ -z "$expected_checksum" ] || [ "$actual_checksum" != "$expected_checksum" ]; then
        rm -f "$node_archive" "$checksums_file"
        log_error "Checksum verification failed for ${node_tar}."
        return 1
    fi
    rm -f "$checksums_file"

    # root 或 /usr/local 可写时使用系统目录，否则完整安装到当前用户目录。
    if [ "$(id -u)" -eq 0 ] || [ -w /usr/local ]; then
        install_prefix="/usr/local"
        mkdir -p "$install_prefix"
        log_info "Installing Node.js into ${install_prefix}..."
        tar -xzf "$node_archive" -C "$install_prefix" --strip-components=1
        export PATH="/usr/local/bin:$PATH"
    else
        install_prefix="$HOME/.local"
        version_dir="${install_prefix}/lib/nodejs/node-${node_full_ver}-${node_arch}"
        log_info "/usr/local is not writable; installing Node.js into ${version_dir}..."
        mkdir -p "$version_dir" "${install_prefix}/bin"
        tar -xzf "$node_archive" -C "$version_dir" --strip-components=1
        local executable
        for executable in node npm npx corepack; do
            if [ -e "${version_dir}/bin/${executable}" ]; then
                ln -sfn "${version_dir}/bin/${executable}" "${install_prefix}/bin/${executable}"
            fi
        done
        export PATH="${install_prefix}/bin:$PATH"
        export NPM_CONFIG_PREFIX="$install_prefix"
        npm config set prefix "$install_prefix"
        touch "$HOME/.bashrc"
        grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    rm -f "$node_archive"

    command -v node >/dev/null 2>&1 || { log_error "Node.js installation completed but node is not on PATH."; return 1; }
    command -v npm >/dev/null 2>&1 || { log_error "Node.js installation completed but npm is not on PATH."; return 1; }
    log_success "Node.js installed: $(node -v)"
    log_success "npm version: $(npm -v)"
}

# ========================
#    Claude Code 安装
# ========================
install_claude_code() {
    if command -v claude &>/dev/null; then
        log_info "Claude Code is already installed: $(claude --version 2>/dev/null || echo 'unknown version')"
        log_info "Updating Claude Code to the latest version..."
    else
        log_info "Installing Claude Code..."
    fi

    if ! npm install -g "${CLAUDE_PACKAGE}@latest"; then
        log_error "npm installation from the default registry failed; retrying with npmmirror."
        npm install -g "${CLAUDE_PACKAGE}@latest" --registry=https://registry.npmmirror.com
    fi
    log_success "Claude Code is ready: $(claude --version 2>/dev/null || echo 'latest version installed')"
}

# ========================
#    跳过 Onboarding
# ========================
configure_claude_json() {
    node --eval '
        const os = require("os");
        const fs = require("fs");
        const path = require("path");
        const homeDir = os.homedir();
        const filePath = path.join(homeDir, ".claude.json");
        if (fs.existsSync(filePath)) {
            const content = JSON.parse(fs.readFileSync(filePath, "utf-8"));
            fs.writeFileSync(filePath, JSON.stringify({ ...content, hasCompletedOnboarding: true }, null, 2), "utf-8");
        } else {
            fs.writeFileSync(filePath, JSON.stringify({ hasCompletedOnboarding: true }, null, 2), "utf-8");
        }
    '
}

# ========================
#    写入 settings.json
# ========================
configure_claude() {
    log_info "Configuring Claude Code..."

    mkdir -p "$HOME/.claude"

    node --eval '
        const os = require("os");
        const fs = require("fs");
        const path = require("path");
        const homeDir = os.homedir();
        const filePath = path.join(homeDir, ".claude", "settings.json");
        const content = fs.existsSync(filePath)
            ? JSON.parse(fs.readFileSync(filePath, "utf-8"))
            : {};
        fs.writeFileSync(filePath, JSON.stringify({
            ...content,
            env: {
                ...content.env,
                ANTHROPIC_AUTH_TOKEN: "'"$API_KEY"'",
                ANTHROPIC_BASE_URL: "'"$PROVIDER_URL"'",
                ANTHROPIC_MODEL: "'"$MODEL"'",
                ANTHROPIC_DEFAULT_HAIKU_MODEL: "'"$MODEL"'",
                ANTHROPIC_DEFAULT_SONNET_MODEL: "'"$MODEL"'",
                ANTHROPIC_DEFAULT_OPUS_MODEL: "'"$MODEL"'",
                API_TIMEOUT_MS: "'"$API_TIMEOUT_MS"'",
                CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: 1,
                CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: 1
            }
        }, null, 2), "utf-8");
    ' || { log_error "Failed to write settings.json"; exit 1; }

    log_success "Claude Code configured successfully"
    log_info "Agent Teams 已启用（CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1）"
}

# ========================
#        主流程
# ========================
main() {
    check_nodejs
    install_claude_code
    configure_claude_json
    configure_claude

    echo ""
    log_success "Installation completed successfully!"
    echo ""
    echo "🚀 使用方式: claude"
}

main "$@"
