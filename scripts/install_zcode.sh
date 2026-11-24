#!/bin/bash

# ZCode 官方 CLI 一键构建、安装与推理服务配置脚本（Linux / 容器 / macOS）
# 用法: bash install_zcode.sh [模型名称] [API_KEY] [BASE_URL]
# 交互模式 (不传参): 先弹提供方菜单，再填写模型和 API Key。
#                   选完后填模型名(带默认) → 填 API_KEY, URL 由所选提供方自动给出
# 非交互示例:
#       bash install_zcode.sh glm-4.6 sk-xxx
#       bash install_zcode.sh Qwen3-30B-A3B "" http://localhost:8000/v1
#       ZCODE_PROVIDER=bailian bash install_zcode.sh deepseek-v3 sk-xxx
#       ZCODE_API_PROTOCOL=openai-responses bash install_zcode.sh my-model sk-xxx http://localhost:8000/v1
# URL 解析优先级: 显式 BASE_URL > ZCODE_PROVIDER > 模型名前缀 > 交互菜单。
# API 协议可通过 ZCODE_API_PROTOCOL 指定: openai-completions / openai-responses / anthropic-messages
#   ZCODE_PROVIDER 取值: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun
#   适配"一个 API 支持多种模型"的平台 (百炼/火山/MiniMax 等), URL 与模型名无关时用它最稳
# 仿照智谱官方脚本 https://cdn.bigmodel.cn/install/claude_code_env.sh

set -euo pipefail

# ========================
#       常量定义
# ========================
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-24}"
ZCODE_CLI_REPOSITORY="${ZCODE_CLI_REPOSITORY:-}"
ZCODE_CLI_BINARY="${ZCODE_CLI_BINARY:-}"
ZCODE_CLI_NODE_DIR="$HOME/.cache/zcode-cli/node"
ZCODE_CLI_REF="${ZCODE_CLI_REF:-main}"
ZCODE_CLI_SOURCE_DIR="${ZCODE_CLI_SOURCE_DIR:-$HOME/.cache/zcode-cli/source}"
if [ "$(id -u)" -eq 0 ]; then
    ZCODE_CLI_INSTALL_DIR="${ZCODE_CLI_INSTALL_DIR:-/usr/local/lib/zcode-cli}"
    BIN_DIR="${BIN_DIR:-/usr/local/bin}"
else
    ZCODE_CLI_INSTALL_DIR="${ZCODE_CLI_INSTALL_DIR:-$HOME/.local/opt/zcode-cli}"
    BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
fi
ZCODE_CLI_COMMAND="${ZCODE_CLI_COMMAND:-zcode}"
ZCODE_CLI_PNPM_VERSION="${ZCODE_CLI_PNPM_VERSION:-10.33.2}"

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

# ========================
#  平台(provider) → API URL
# ========================
url_for_provider() {
    case "${API_PROTOCOL:-}" in
        openai-completions|openai-responses)
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
            return ;;
    esac
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
#  模型名前缀 → API URL (无 provider 时的回退)
# ========================
get_provider_url() {
    local model
    model=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')   # 小写化后匹配, 兼容 Qwen3 / GLM-4.6 等大写写法
    case "$model" in
        claude-*)                  url_for_provider anthropic ;;
        deepseek-*)                url_for_provider deepseek ;;
        glm-*|chatglm-*)           url_for_provider zhipu ;;
        doubao-*)                  url_for_provider volcano ;;
        qwen*|qwq*)                url_for_provider bailian ;;
        minimax-*|abab*)            url_for_provider minimax ;;
        kimi-*|moonshot-*)         url_for_provider moonshot ;;
        step-*)                    url_for_provider stepfun ;;
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
    IFS= read -r -p "请选择 [1-${#PROVIDER_NAMES[@]}]: " choice
    if [ -z "$choice" ] || [ "$choice" = "0" ]; then
        log_error "未选择提供方"; exit 1
    fi
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#PROVIDER_NAMES[@]}" ]; then
        log_error "无效选择: $choice"; exit 1
    fi
    local idx=$((10#$choice-1))
    SELECTED_KEY="${PROVIDER_KEYS[$idx]}"
    SELECTED_MODEL="${PROVIDER_MODELS[$idx]}"
    SELECTED_URL=$(url_for_provider "$SELECTED_KEY")
}

# ========================
#  参数与提供方选择
#  URL 解析优先级: 显式 BASE_URL > ZCODE_PROVIDER > 模型前缀 > 交互菜单
# ========================
CUSTOM_URL="${3:-}"
PROVIDER="${ZCODE_PROVIDER:-}"
API_PROTOCOL="${ZCODE_API_PROTOCOL:-}"
MODEL="${1:-}"
API_KEY="${2-${ZCODE_API_KEY:-}}"
[ "$#" -lt 2 ] || API_KEY="${API_KEY:-sk-local}"

# 交互模式 (未传模型名且未指定 provider/url): 先选提供方, 再填模型与 key
if [ -z "$MODEL" ] && [ -z "$CUSTOM_URL" ] && [ -z "$PROVIDER" ]; then
    select_provider_menu
    PROVIDER="$SELECTED_KEY"
    if [ "$PROVIDER" = "custom" ]; then
        IFS= read -r -p "请输入 API BASE_URL (如 http://192.168.1.100:8000/v1): " PROVIDER_URL
        [ -z "$PROVIDER_URL" ] && { log_error "API 地址不能为空"; exit 1; }
    else
        PROVIDER_URL="$SELECTED_URL"
    fi
    IFS= read -r -p "请输入模型名称 (默认 ${SELECTED_MODEL}): " MODEL
    MODEL="${MODEL:-$SELECTED_MODEL}"
    if [ -z "$API_KEY" ]; then
        IFS= read -r -s -p "请输入 API_KEY (默认 sk-local): " API_KEY
        [ ! -t 0 ] || echo
        API_KEY="${API_KEY:-sk-local}"
    fi
else
    # 非交互: 位置参数 + 环境变量
    [ -z "$MODEL" ] && { IFS= read -r -p "请输入模型名称 (默认 deepseek-v4-pro): " MODEL; MODEL="${MODEL:-deepseek-v4-pro}"; }
    if [ -z "$API_KEY" ]; then
        IFS= read -r -s -p "请输入 API_KEY (默认 sk-local): " API_KEY
        [ ! -t 0 ] || echo
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
        IFS= read -r -p "请输入 API BASE_URL (如 http://192.168.1.100:8000/v1): " PROVIDER_URL
        [ -z "$PROVIDER_URL" ] && { log_error "API 地址不能为空"; exit 1; }
    fi
fi

[ -n "$MODEL" ] || { log_error "模型名称不能为空"; exit 1; }

if [ -z "$API_PROTOCOL" ]; then
    case "$PROVIDER_URL" in
        *anthropic*|https://ark.cn-beijing.volces.com/api/coding*|https://api.stepfun.com/step_plan*) API_PROTOCOL="anthropic-messages" ;;
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
echo "  API_KEY: 已设置"
echo "========================================"
echo ""

# ========================
#    Node.js 安装
# ========================
check_nodejs() {
    local requested_version="${NODE_VERSION:-}"
    if [ -z "$requested_version" ] && [ -x "$ZCODE_CLI_NODE_DIR/bin/node" ] &&
        [ -x "$ZCODE_CLI_NODE_DIR/bin/npm" ] && "$ZCODE_CLI_NODE_DIR/bin/node" -e '
            const [major, minor] = process.versions.node.split(".").map(Number);
            process.exit(major > 24 || major === 24 && minor >= 14 ? 0 : 1);
        '; then
        export PATH="$ZCODE_CLI_NODE_DIR/bin:$PATH"
        log_info "使用缓存的官方 Node.js $(node -v)"
        return 0
    fi

    log_info "Installing Node.js..."

    local node_os machine_arch node_arch node_full_ver node_tar node_archive checksums_file
    local dist_base index_data expected_checksum actual_checksum
    case "$(uname -s)" in
        Linux) node_os=linux ;;
        Darwin) node_os=darwin ;;
        *) log_error "不支持的操作系统: $(uname -s)"; return 1 ;;
    esac
    machine_arch=$(uname -m)
    case "$machine_arch" in
        aarch64|arm64) node_arch="$node_os-arm64" ;;
        x86_64|amd64) node_arch="$node_os-x64" ;;
        *) log_error "Unsupported architecture: $machine_arch"; return 1 ;;
    esac

    # 可用 NODE_VERSION=v24.x.y 锁定版本；未指定时解析 Node 24 的最新补丁版本。
    if [ -n "${NODE_VERSION:-}" ]; then
        node_full_ver="$NODE_VERSION"
        [[ "$node_full_ver" == v* ]] || node_full_ver="v${node_full_ver}"
    else
        index_data=""
        for dist_base in "https://nodejs.org/dist" "https://npmmirror.com/mirrors/node"; do
            log_info "Resolving latest Node.js ${NODE_INSTALL_VERSION}.x from ${dist_base}..."
            if index_data=$(curl --fail --show-error --silent --location --retry 1 --connect-timeout 10 --max-time 30 "${dist_base}/index.tab"); then
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
        if curl --fail --show-error --location --retry 1 --connect-timeout 10 --max-time 180 \
            "${candidate_base}/${node_full_ver}/${node_tar}" -o "$node_archive" &&
            curl --fail --show-error --silent --location --retry 1 --connect-timeout 10 --max-time 30 \
                "${candidate_base}/${node_full_ver}/SHASUMS256.txt" -o "$checksums_file"; then
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

    # SEA copies this executable. Use the official distribution, including on
    # hosts where a system Node links to an external libnode.
    mkdir -p "$ZCODE_CLI_NODE_DIR"
    tar -xzf "$node_archive" -C "$ZCODE_CLI_NODE_DIR" --strip-components=1
    export PATH="$ZCODE_CLI_NODE_DIR/bin:$PATH"
    rm -f "$node_archive"

    command -v node >/dev/null 2>&1 || { log_error "Node.js installation completed but node is not on PATH."; return 1; }
    command -v npm >/dev/null 2>&1 || { log_error "Node.js installation completed but npm is not on PATH."; return 1; }
    node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 24 || major === 24 && minor >= 14 ? 0 : 1)' ||
        { log_error "ZCode CLI 需要 Node.js 24.14 或更新版本"; return 1; }
    log_success "Node.js installed: $(node -v)"
    log_success "npm version: $(npm -v)"
}

# ========================
#    ZCode 官方 CLI 源码构建与安装
# ========================
npm_install() {
    if ! npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 "$@"; then
        log_info "默认 npm 源安装失败，改用 npmmirror..."
        npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 \
            --registry=https://registry.npmmirror.com "$@"
    fi
}

add_bin_to_path() {
    local bin_dir=$1 rc path_line
    export PATH="$bin_dir:$PATH"
    path_line="export PATH=$(printf '%q' "$bin_dir"):\$PATH"
    for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if ! grep -Fxq "$path_line" "$rc" 2>/dev/null; then
            printf '\n%s\n' "$path_line" >> "$rc"
        fi
    done
}

# GitCode is the default for networks that cannot reach GitHub.
prepare_zcode_source() {
    local repo remote dirty
    local repositories=("${ZCODE_CLI_REPOSITORY:-https://gitcode.com/zai-org/ZCode.git}")
    [ -n "$ZCODE_CLI_REPOSITORY" ] || repositories+=("https://github.com/zai-org/ZCode.git")
    local git_options=(-c http.lowSpeedLimit=512 -c http.lowSpeedTime=20)
    if [ ! -d "$ZCODE_CLI_SOURCE_DIR/.git" ]; then
        if [ -e "$ZCODE_CLI_SOURCE_DIR" ]; then
            log_error "$ZCODE_CLI_SOURCE_DIR 已存在但不是 Git 仓库，请换用 ZCODE_CLI_SOURCE_DIR。"
            return 1
        fi
        mkdir -p "$(dirname "$ZCODE_CLI_SOURCE_DIR")"
        for repo in "${repositories[@]}"; do
            log_info "从 $repo 获取 ZCode 源码 ($ZCODE_CLI_REF)..."
            if GIT_TERMINAL_PROMPT=0 git "${git_options[@]}" clone --depth 1 --branch "$ZCODE_CLI_REF" -- "$repo" "$ZCODE_CLI_SOURCE_DIR"; then
                return 0
            fi
        done
    else
        dirty=$(git -C "$ZCODE_CLI_SOURCE_DIR" status --porcelain --untracked-files=no) || return 1
        if [ -n "$dirty" ]; then
            log_error "源码缓存有未提交修改: $ZCODE_CLI_SOURCE_DIR"
            return 1
        fi
        for remote in "${repositories[@]}"; do
            if GIT_TERMINAL_PROMPT=0 git "${git_options[@]}" -C "$ZCODE_CLI_SOURCE_DIR" fetch --depth 1 -- "$remote" "$ZCODE_CLI_REF"; then
                git -C "$ZCODE_CLI_SOURCE_DIR" checkout --detach FETCH_HEAD
                return $?
            fi
        done
    fi
    log_error "无法获取 ZCode 源码。可设置 ZCODE_CLI_REPOSITORY 或 ZCODE_CLI_BINARY 后重试。"
    return 1
}

build_zcode_cli() {
    local zcode_target=$1 pnpm_root pnpm_cmd
    command -v git >/dev/null || { log_error "构建 ZCode CLI 需要 Git"; return 1; }
    prepare_zcode_source
    pnpm_root="$HOME/.cache/zcode-cli/build-tools"
    npm_install --prefix "$pnpm_root" "pnpm@$ZCODE_CLI_PNPM_VERSION"
    pnpm_cmd="$pnpm_root/node_modules/.bin/pnpm"
    (
        cd "$ZCODE_CLI_SOURCE_DIR"
        export COREPACK_ENABLE_PROJECT_SPEC=0
        export PATH="$(dirname "$pnpm_cmd"):$PATH"
        log_info "正在安装 ZCode 构建依赖，初次构建可能需要数分钟..."
        if ! "$pnpm_cmd" install --ignore-scripts --frozen-lockfile --fetch-retries=1 --fetch-timeout=60000; then
            "$pnpm_cmd" install --ignore-scripts --frozen-lockfile --fetch-retries=1 --fetch-timeout=60000 --registry=https://registry.npmmirror.com
        fi
        "$pnpm_cmd" run build:sea:prepare
        # Reuse the host Node binary so upstream's SEA packager never fetches it again.
        "$pnpm_cmd" --filter @zcode/cli sea --target "$zcode_target" \
            --node-binary "$zcode_target=$(node -p 'process.execPath')"
    )
}

install_zcode_cli() {
    local zcode_os zcode_arch zcode_target binary_path target_file staged
    case "$ZCODE_CLI_COMMAND" in
        ""|.|..|*[!a-zA-Z0-9._-]*)
            log_error "ZCODE_CLI_COMMAND 只能包含字母、数字、点、下划线或连字符。"
            return 1 ;;
    esac
    case "$(uname -s)" in
        Linux) zcode_os=linux ;;
        Darwin) zcode_os=darwin ;;
        *) log_error "不支持的操作系统: $(uname -s)"; return 1 ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64) zcode_arch=x64 ;;
        aarch64|arm64) zcode_arch=arm64 ;;
        *) log_error "ZCode CLI 不支持的 CPU 架构: $(uname -m)"; return 1 ;;
    esac
    zcode_target="$zcode_os-$zcode_arch"
    binary_path="$ZCODE_CLI_BINARY"
    if [ -z "$binary_path" ]; then
        build_zcode_cli "$zcode_target"
        binary_path="$ZCODE_CLI_SOURCE_DIR/apps/zcode-cli/packages/cli/dist/zcode-$zcode_target"
    fi
    mkdir -p "$ZCODE_CLI_INSTALL_DIR" "$BIN_DIR"
    target_file="$ZCODE_CLI_INSTALL_DIR/zcode-$zcode_target"
    staged=$(mktemp "$ZCODE_CLI_INSTALL_DIR/.zcode.XXXXXX")
    if ! cp "$binary_path" "$staged" || ! chmod 0755 "$staged" || ! "$staged" --version; then
        rm -f "$staged"
        log_error "ZCode CLI 文件不可用: $binary_path"
        return 1
    fi
    mv -f "$staged" "$target_file"
    ln -sfn "$(cd "$ZCODE_CLI_INSTALL_DIR" && pwd)/zcode-$zcode_target" "$BIN_DIR/$ZCODE_CLI_COMMAND"
    add_bin_to_path "$BIN_DIR"
    log_success "ZCode CLI is ready: $BIN_DIR/$ZCODE_CLI_COMMAND"
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
  path.join(process.env.ZCODE_DATA_BASE_DIR || os.homedir(), ".zcode", "v2", "provider_config.json");
fs.mkdirSync(path.dirname(filePath), { recursive: true });
let doc = { schemaVersion: 1, config: {} };
if (fs.existsSync(filePath)) doc = JSON.parse(fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
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
const tmp = filePath + "." + require("node:crypto").randomUUID();
try {
  fs.writeFileSync(tmp, JSON.stringify(doc, null, 2) + "\n", { mode: 0o600, flag: "wx" });
  fs.renameSync(tmp, filePath);
} finally {
  fs.rmSync(tmp, { force: true });
}
NODE
    log_success "ZCode configured: provider=$route_id model=$MODEL protocol=$zcode_protocol"
}

# ========================
#        主流程
# ========================
main() {
    if [ -z "$ZCODE_CLI_BINARY" ] || ! node -e 'process.exit(+process.versions.node.split(".")[0] >= 16 ? 0 : 1)' 2>/dev/null; then
        check_nodejs
    fi
    install_zcode_cli
    configure_zcode

    echo ""
    log_success "Installation completed successfully!"
    echo ""
    echo "🚀 使用方式: $ZCODE_CLI_COMMAND"
    echo "              $ZCODE_CLI_COMMAND --prompt \"检查当前项目\" --mode plan"
}

main "$@"
