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
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
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
    ORIG_PWD=$(pwd)
    # 检测系统类型 + 包管理器（兼容 Debian/openEuler/centos/rhel/fedora/anolis/kylin/suse 等）
    if [ -f /etc/os-release ]; then
        . /etc/os-release
    fi
    SYS_IDS="${ID:-} ${ID_LIKE:-}"
    if command -v dnf &>/dev/null; then PM="dnf install -y"
    elif command -v yum &>/dev/null; then PM="yum install -y"
    elif command -v apt-get &>/dev/null; then PM="apt-get install -y"
    else PM=""; fi

    case "$SYS_IDS" in
        *openEuler*|*centos*|*rhel*|*fedora*|*anolis*|*kylin*|*suse*|*sles*)
            # 非 Debian 系，从 nodejs.org 装 tar 包（不依赖 nodesource deb setup）
            ARCH=$(uname -m)
            case "$ARCH" in
                aarch64) NODE_ARCH="linux-arm64" ;;
                x86_64)  NODE_ARCH="linux-x64" ;;
                *) log_error "Unsupported arch: $ARCH"; exit 1 ;;
            esac
            NODE_FULL_VER="v${NODE_INSTALL_VERSION}.14.0"
            NODE_TAR="node-${NODE_FULL_VER}-${NODE_ARCH}.tar.xz"
            log_info "Downloading Node.js ${NODE_FULL_VER} (${NODE_ARCH}) from nodejs.org..."
            curl -fsSL "https://nodejs.org/dist/${NODE_FULL_VER}/${NODE_TAR}" -o "/tmp/${NODE_TAR}" || {
                log_error "Failed to download Node.js from nodejs.org"; exit 1
            }
            # tar 解压 .tar.xz 需要 xz，minimal 系统可能没装
            command -v xz >/dev/null 2>&1 || { [ -n "$PM" ] && $PM xz 2>/dev/null; }
            cd /tmp && tar -xf "${NODE_TAR}" && cp -r "node-${NODE_FULL_VER}-${NODE_ARCH}/"* /usr/local/ && rm -rf "node-${NODE_FULL_VER}-${NODE_ARCH}" "${NODE_TAR}"
            cd "$ORIG_PWD"
            # 确保 /usr/local/bin 在 PATH（装到 /usr/local/，当前 shell 可能未 reload）
            export PATH="/usr/local/bin:$PATH"
            # 持久化到 ~/.bashrc（新 shell 能找到 /usr/local/bin 下的 node/npm/claude）
            grep -q '/usr/local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
            ;;
        *)
            # Debian 系，用 nodesource
            curl -fsSL https://deb.nodesource.com/setup_${NODE_INSTALL_VERSION}.x | bash - 2>/dev/null || {
                curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/nodesource/deb_${NODE_INSTALL_VERSION}.x/setup_${NODE_INSTALL_VERSION}.x | bash - 2>/dev/null
            }
            ${PM:-apt-get install -y} nodejs 2>/dev/null
            ;;
    esac
    log_success "Node.js installed: $(node -v 2>/dev/null || echo '/usr/local/bin/node')"
    log_success "npm version: $(npm -v 2>/dev/null || echo '/usr/local/bin/npm')"
}

# ========================
#    Claude Code 安装
# ========================
install_claude_code() {
    if command -v claude &>/dev/null; then
        log_success "Claude Code is already installed: $(claude --version 2>/dev/null || echo 'ok')"
    else
        log_info "Installing Claude Code..."
        npm install -g "$CLAUDE_PACKAGE" 2>/dev/null || \
            npm install -g "$CLAUDE_PACKAGE" --registry=https://registry.npmmirror.com
        log_success "Claude Code installed successfully"
    fi
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
            },
            enabledPlugins: {
                ...content.enabledPlugins,
                "ops-direct-invoke-skills@cannbot": true,
                "infra-skills@cannbot": true,
                "ops-direct-invoke@cannbot": true
            },
            extraKnownMarketplaces: {
                ...content.extraKnownMarketplaces,
                "cannbot": {
                    "source": { "source": "git", "url": "https://gitcode.com/cann/skills.git" }
                }
            }
        }, null, 2), "utf-8");
    ' || { log_error "Failed to write settings.json"; exit 1; }

    log_success "Claude Code configured successfully"
    log_info "cannbot 插件（ops-direct-invoke + skills + infra）已写入 settings，首次启动 claude 时自动从 gitcode.com/cann/skills 拉取安装"
    log_info "Agent Teams 已启用（CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1），cannbot 协同适配需要 team 模式"
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
