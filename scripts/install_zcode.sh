#!/bin/bash

# ZCode 一键安装与推理服务配置脚本（Linux）
# 用法: bash install_zcode.sh [模型名称] [API_KEY] [BASE_URL]
# 交互模式 (不传参): 先弹提供方菜单，再填写模型和 API Key。
#                   选完后填模型名(带默认) → 填 API_KEY, URL 由所选提供方自动给出
# 非交互示例:
#       bash install_zcode.sh glm-4.6 sk-xxx                              # 前缀自动匹配智谱
#       bash install_zcode.sh glm-4.6 sk-xxx https://open.bigmodel.cn/api/anthropic
#       bash install_zcode.sh Qwen3-30B-A3B "" http://localhost:8000/v1
#       bash install_zcode.sh MiniMax-M1 sk-xxx                           # 前缀自动匹配 minimax
#       ZCODE_PROVIDER=bailian bash install_zcode.sh deepseek-v3 sk-xxx  # 百炼托管的任意模型
#       ZCODE_PROVIDER=moonshot bash install_zcode.sh kimi-k2 sk-xxx
# URL 解析优先级: 显式 BASE_URL > ZCODE_PROVIDER > 模型名前缀 > 交互菜单。
# API 协议可通过 ZCODE_API_PROTOCOL 指定: openai-completions / openai-responses / anthropic-messages
#   ZCODE_PROVIDER 取值: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun
#   适配"一个 API 支持多种模型"的平台 (百炼/火山/MiniMax 等), URL 与模型名无关时用它最稳
# 仿照智谱官方脚本 https://cdn.bigmodel.cn/install/claude_code_env.sh

set -euo pipefail

# ========================
#       常量定义
# ========================
NODE_MIN_VERSION=22
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-22}"
ZCODE_INSTALL_PAGE="${ZCODE_INSTALL_PAGE:-https://zcode.z.ai/cn/docs/install}"
ZCODE_CDN_ROOT="${ZCODE_CDN_ROOT:-https://cdn-zcode.z.ai/zcode/electron/releases}"
ZCODE_INSTALL_DIR="${ZCODE_INSTALL_DIR:-$HOME/.local/opt/zcode}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

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
#  平台清单 (展示名 | provider key | 默认模型 | API URL | ZCode 协议)
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
    "https://open.bigmodel.cn/api/paas/v4"
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
    "https://api.deepseek.com/v1"
    "https://ark.cn-beijing.volces.com/api/v3"
    "https://api.minimaxi.com/v1"
    "https://api.moonshot.cn/v1"
    "https://api.stepfun.com/v1"
    "https://api.anthropic.com"
    ""
)
PROVIDER_PROTOCOLS=(
    "openai-completions" "openai-completions" "openai-completions"
    "openai-completions" "openai-completions" "openai-completions"
    "openai-completions" "anthropic-messages" ""
)

# ========================
#  平台(provider) → API URL
# ========================
url_for_provider() {
    case "$1" in
        anthropic|claude)          echo "https://api.anthropic.com" ;;
        zhipu|glm)                 echo "https://open.bigmodel.cn/api/paas/v4" ;;
        deepseek)                  echo "https://api.deepseek.com/v1" ;;
        bailian|dashscope|qwen)    echo "https://dashscope.aliyuncs.com/compatible-mode/v1" ;;
        volcano|ark|doubao)        echo "https://ark.cn-beijing.volces.com/api/v3" ;;
        minimax)                   echo "https://api.minimaxi.com/v1" ;;
        moonshot|kimi)             echo "https://api.moonshot.cn/v1" ;;
        stepfun|step)              echo "https://api.stepfun.com/v1" ;;
        *)                         echo "" ;;
    esac
}

# ========================
#  模型名前缀 → API URL (无 provider 时的回退)
# ========================
get_provider_url() {
    local model="${1,,}"   # 小写化后匹配, 兼容 Qwen3 / GLM-4.6 等大写写法
    case "$model" in
        claude-*)                  echo "https://api.anthropic.com" ;;
        deepseek-*)                echo "https://api.deepseek.com/v1" ;;
        glm-*|chatglm-*)           echo "https://open.bigmodel.cn/api/paas/v4" ;;
        doubao-*)                  echo "https://ark.cn-beijing.volces.com/api/v3" ;;
        qwen*|qwq*)                echo "https://dashscope.aliyuncs.com/compatible-mode/v1" ;;
        minimax-*|MiniMax-*|abab*) echo "https://api.minimaxi.com/v1" ;;
        kimi-*|moonshot-*)         echo "https://api.moonshot.cn/v1" ;;
        step-*)                    echo "https://api.stepfun.com/v1" ;;
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
    SELECTED_PROTOCOL="${PROVIDER_PROTOCOLS[$idx]}"
}

# ========================
#  参数与提供方选择
#  URL 解析优先级: 显式 BASE_URL > ZCODE_PROVIDER > 模型前缀 > 交互菜单
# ========================
CUSTOM_URL="${3:-}"
PROVIDER="${ZCODE_PROVIDER:-}"
API_PROTOCOL="${ZCODE_API_PROTOCOL:-}"
MODEL="${1:-}"
API_KEY="${2:-}"

# 交互模式 (未传模型名且未指定 provider/url): 先选提供方, 再填模型与 key
if [ -z "$MODEL" ] && [ -z "$CUSTOM_URL" ] && [ -z "$PROVIDER" ]; then
    select_provider_menu
    PROVIDER="$SELECTED_KEY"
    API_PROTOCOL="${API_PROTOCOL:-$SELECTED_PROTOCOL}"
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
            log_error "未知 ZCODE_PROVIDER: $PROVIDER"
            echo "支持: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun" >&2
            exit 1
        fi
    else
        PROVIDER_URL=$(get_provider_url "$MODEL")
    fi
    if [ -z "$PROVIDER_URL" ]; then
        echo ""
        echo "模型 '${MODEL}' 不在已知前缀列表中。"
        echo "可选: 1) 显式传 BASE_URL;  2) 设 ZCODE_PROVIDER=<平台>;  3) 下方手动输入"
        echo "已知前缀: claude-*, deepseek-*, glm-*, doubao-*, qwen*, minimax-*, abab*, kimi-*, moonshot-*, step-*"
        echo "已知 provider: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun"
        echo ""
        read -e -p "请输入 API BASE_URL (如 http://192.168.1.100:8000/v1): " PROVIDER_URL
        [ -z "$PROVIDER_URL" ] && { log_error "API 地址不能为空"; exit 1; }
    fi
fi

if [ -z "$API_PROTOCOL" ]; then
    case "$PROVIDER_URL" in
        *anthropic*|https://api.anthropic.com*) API_PROTOCOL="anthropic-messages" ;;
        *) API_PROTOCOL="openai-completions" ;;
    esac
fi
case "$API_PROTOCOL" in
    openai-completions|openai-responses|anthropic-messages) ;;
    *) log_error "不支持的 ZCODE_API_PROTOCOL: $API_PROTOCOL"; exit 1 ;;
esac

echo ""
echo "========================================"
echo "  模型:    ${MODEL}"
[ -n "$PROVIDER" ] && echo "  平台:    ${PROVIDER}"
echo "  API URL: ${PROVIDER_URL}"
echo "  协议:     ${API_PROTOCOL}"
echo "  API_KEY: $(printf '%*s' "${#API_KEY}" '' | tr ' ' '*')"
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
                # 使用 here-string，避免 awk 提前退出时上游 printf 收到 SIGPIPE；
                # 在 set -euo pipefail 下，该 SIGPIPE 会让脚本以 141 静默退出。
                node_full_ver=$(awk -v prefix="v${NODE_INSTALL_VERSION}." 'NR > 1 && index($1, prefix) == 1 { print $1; exit }' <<< "$index_data")
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
#    ZCode 安装
# ========================
install_zcode() {
    [ "$(uname -s)" = Linux ] || { log_error "此脚本仅支持 Linux。"; return 1; }
    local machine_arch zcode_arch version page file_name url temp_file target_file
    machine_arch="$(uname -m)"
    case "$machine_arch" in
        x86_64|amd64) zcode_arch=x64 ;;
        aarch64|arm64) zcode_arch=arm64 ;;
        *) log_error "ZCode 不支持的 CPU 架构: $machine_arch"; return 1 ;;
    esac
    if [ -n "${ZCODE_VERSION:-}" ]; then
        version="$ZCODE_VERSION"
    else
        page="$(curl -fsSL --retry 3 --connect-timeout 15 "$ZCODE_INSTALL_PAGE")"
        version="$(sed -nE 's#.*ZCode-([0-9]+\.[0-9]+\.[0-9]+)-linux-x64\.AppImage.*#\1#p' <<<"$page" | head -n 1)"
    fi
    [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { log_error "非法 ZCode 版本: $version"; return 1; }
    file_name="ZCode-${version}-linux-${zcode_arch}.AppImage"
    url="${ZCODE_CDN_ROOT}/${version}/linux-${zcode_arch}/${file_name}"
    temp_file="$(mktemp "/tmp/${file_name}.XXXXXX")"
    trap 'rm -f "${temp_file:-}"' RETURN
    log_info "Downloading ZCode $version ($zcode_arch)..."
    curl -fSL --retry 3 --connect-timeout 15 "$url" -o "$temp_file"
    [ "$(od -An -tx1 -N4 "$temp_file" | tr -d ' \n')" = 7f454c46 ] || {
        log_error "下载内容不是有效的 Linux AppImage。"; return 1;
    }
    mkdir -p "$ZCODE_INSTALL_DIR" "$BIN_DIR"
    target_file="${ZCODE_INSTALL_DIR}/$file_name"
    mv "$temp_file" "$target_file"
    chmod 0755 "$target_file"
    ln -sfn "$target_file" "${BIN_DIR}/zcode"
    export PATH="${BIN_DIR}:$PATH"
    log_success "ZCode $version is ready: ${BIN_DIR}/zcode"
}

# ========================
#    写入 ZCode provider 配置
# ========================
configure_zcode() {
    log_info "Configuring ZCode..."
    local route_id zcode_protocol
    route_id="$(printf '%s' "${PROVIDER:-slai-custom}" | tr '[:upper:]_' '[:lower:]-' | sed 's/[^a-z0-9-]/-/g')"
    [ -n "$route_id" ] || route_id="slai-custom"
    case "$API_PROTOCOL" in
        openai-completions) zcode_protocol="openai-chat-completions" ;;
        openai-responses) zcode_protocol="openai-responses" ;;
        anthropic-messages) zcode_protocol="anthropic-messages" ;;
    esac
    INSTALL_ROUTE_ID="$route_id" INSTALL_MODEL="$MODEL" INSTALL_API_KEY="$API_KEY" \
    INSTALL_BASE_URL="$PROVIDER_URL" INSTALL_PROTOCOL="$zcode_protocol" node <<'NODE'
const fs = require("fs");
const os = require("os");
const path = require("path");
const filePath = process.env.ZCODE_PERSONAL_PROVIDER_CONFIG_FILE ||
  path.join(os.homedir(), ".zcode", "v2", "provider_config.json");
fs.mkdirSync(path.dirname(filePath), { recursive: true });
let doc = { schemaVersion: 1, config: {} };
if (fs.existsSync(filePath)) doc = JSON.parse(fs.readFileSync(filePath, "utf8"));
doc.schemaVersion = 1;
doc.config ||= {};
doc.config.providerConfigRules ||= {};
doc.config.providerConfigRules.providerRules ||= [];
doc.config.modelConfigRules ||= { providerModelRules: [], manualProviderModelRules: [] };
const providerId = process.env.INSTALL_ROUTE_ID;
const rule = {
  providerId,
  providerName: providerId,
  config: {
    group: "standard-personal",
    access: { type: "api-key", apiKey: process.env.INSTALL_API_KEY },
    api: { type: process.env.INSTALL_PROTOCOL, baseUrl: process.env.INSTALL_BASE_URL },
    personalModelIds: [process.env.INSTALL_MODEL],
  },
};
const rules = doc.config.providerConfigRules.providerRules;
const index = rules.findIndex((item) => item && item.providerId === providerId);
if (index >= 0) rules[index] = rule;
else rules.push(rule);
doc.config.defaultModelSelection = { providerId, modelId: process.env.INSTALL_MODEL };
fs.writeFileSync(filePath, JSON.stringify(doc, null, 2) + "\n", { mode: 0o600 });
fs.chmodSync(filePath, 0o600);
NODE
    log_success "ZCode configured: provider=$route_id model=$MODEL protocol=$zcode_protocol"
}

# ========================
#        主流程
# ========================
main() {
    check_nodejs
    install_zcode
    configure_zcode

    echo ""
    log_success "Installation completed successfully!"
    echo ""
    echo "🚀 使用方式: zcode"
}

main "$@"
