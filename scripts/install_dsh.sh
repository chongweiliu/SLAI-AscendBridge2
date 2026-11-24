#!/bin/bash

# DSH 一键安装与推理服务配置脚本（Linux / 容器）
# 用法: bash install_dsh.sh [模型名称] [API_KEY] [BASE_URL]
# 交互模式 (不传参): 先弹提供方菜单，再填写模型和 API Key。
#                   选完后填模型名(带默认) → 填 API_KEY, URL 由所选提供方自动给出
# 非交互示例:
#       bash install_dsh.sh glm-4.6 sk-xxx                              # 前缀自动匹配智谱
#       bash install_dsh.sh glm-4.6 sk-xxx https://open.bigmodel.cn/api/anthropic
#       bash install_dsh.sh Qwen3-30B-A3B "" http://localhost:8000/v1
#       bash install_dsh.sh MiniMax-M1 sk-xxx                           # 前缀自动匹配 minimax
#       DSH_PROVIDER=bailian bash install_dsh.sh deepseek-v3 sk-xxx  # 百炼托管的任意模型
#       DSH_PROVIDER=moonshot bash install_dsh.sh kimi-k2 sk-xxx
# URL 解析优先级: 显式 BASE_URL > DSH_PROVIDER > 模型名前缀 > 交互菜单。
# API 协议可通过 DSH_API_PROTOCOL 指定: openai-completions / openai-responses / anthropic-messages
#   DSH_PROVIDER 取值: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun
#   适配"一个 API 支持多种模型"的平台 (百炼/火山/MiniMax 等), URL 与模型名无关时用它最稳
# 仿照智谱官方脚本 https://cdn.bigmodel.cn/install/claude_code_env.sh

set -euo pipefail

# ========================
#       常量定义
# ========================
NODE_INSTALL_VERSION="${NODE_INSTALL_VERSION:-24}"
DSH_PACKAGE="@deepseek-ai/dsh"

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
#  平台清单 (展示名 | provider key | 默认模型 | API URL | DSH 协议)
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
#  URL 解析优先级: 显式 BASE_URL > DSH_PROVIDER > 模型前缀 > 交互菜单
# ========================
CUSTOM_URL="${3:-}"
PROVIDER="${DSH_PROVIDER:-}"
API_PROTOCOL="${DSH_API_PROTOCOL:-}"
MODEL="${1:-}"
API_KEY="${2-${DSH_API_KEY:-}}"
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
            log_error "未知 DSH_PROVIDER: $PROVIDER"
            echo "支持: anthropic zhipu deepseek bailian volcano minimax moonshot stepfun" >&2
            exit 1
        fi
    else
        PROVIDER_URL=$(get_provider_url "$MODEL")
    fi
    if [ -z "$PROVIDER_URL" ]; then
        echo ""
        echo "模型 '${MODEL}' 不在已知前缀列表中。"
        echo "可选: 1) 显式传 BASE_URL;  2) 设 DSH_PROVIDER=<平台>;  3) 下方手动输入"
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
    *) log_error "不支持的 DSH_API_PROTOCOL: $API_PROTOCOL"; exit 1 ;;
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
    if command -v node &>/dev/null && command -v npm &>/dev/null; then
        current_version=$(node -v | sed 's/v//')
        major_version=$(echo "$current_version" | cut -d. -f1)
        minor_version=$(echo "$current_version" | cut -d. -f2)
        if { [ "$major_version" -eq 24 ] && [ "$minor_version" -ge 9 ]; } || [ "$major_version" -gt 24 ]; then
            log_success "Node.js is already installed: v$current_version"
            return 0
        fi
        log_info "DSH requires Node.js 24.9 or newer; current version is v$current_version. Installing a compatible release..."
    fi

    log_info "Installing Node.js..."

    local node_os machine_arch node_arch node_full_ver node_tar node_archive checksums_file
    local dist_base index_data expected_checksum actual_checksum install_prefix version_dir
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
        touch "$HOME/.bashrc"
        grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" || \
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    rm -f "$node_archive"

    command -v node >/dev/null 2>&1 || { log_error "Node.js installation completed but node is not on PATH."; return 1; }
    command -v npm >/dev/null 2>&1 || { log_error "Node.js installation completed but npm is not on PATH."; return 1; }
    node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 24 || major === 24 && minor >= 9 ? 0 : 1)' ||
        { log_error "DSH 需要 Node.js 24.9 或更新版本"; return 1; }
    log_success "Node.js installed: $(node -v)"
    log_success "npm version: $(npm -v)"
}

# ========================
#    DSH 安装
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

install_dsh() {
    if [ "$(id -u)" -eq 0 ]; then
        DSH_NPM_PREFIX=/usr/local
    else
        DSH_NPM_PREFIX="$HOME/.local"
    fi
    npm_install --global --prefix "$DSH_NPM_PREFIX" "${DSH_PACKAGE}@latest" yaml@2 pnpm@10
    add_bin_to_path "$DSH_NPM_PREFIX/bin"
    "$DSH_NPM_PREFIX/bin/dsh" --version
}

# ========================
#    写入 DSH provider 配置
# ========================
configure_dsh() {
    log_info "Configuring DSH..."
    local dsh_home route_id
    dsh_home="${DSH_HOME:-$HOME/.dsh}"
    route_id="slai-$(printf '%s' "${PROVIDER:-custom}" | tr '[:upper:]_' '[:lower:]-' | sed 's/[^a-z0-9-]/-/g')"
    mkdir -p "$dsh_home"

    INSTALL_DSH_HOME="$dsh_home" INSTALL_ROUTE_ID="$route_id" INSTALL_MODEL="$MODEL" \
    INSTALL_BASE_URL="$PROVIDER_URL" INSTALL_PROTOCOL="$API_PROTOCOL" INSTALL_API_KEY="$API_KEY" \
    NODE_PATH="$DSH_NPM_PREFIX/lib/node_modules" node <<'NODE'
const fs = require("node:fs");
const path = require("node:path");
const { parseEnv } = require("node:util");
const YAML = require("yaml");
const home = process.env.INSTALL_DSH_HOME;
const route = process.env.INSTALL_ROUTE_ID;
const model = process.env.INSTALL_MODEL;
const ref = "SLAI_DSH_" + route.toUpperCase().replace(/-/g, "_") + "_API_KEY";
const patchFile = path.join(home, "cordis.patch.yml");
const envFile = path.join(home, ".env");
const rows = fs.existsSync(patchFile) ? YAML.parse(fs.readFileSync(patchFile, "utf8")) ?? [] : [];
if (!Array.isArray(rows)) throw Error("DSH cordis.patch.yml 必须是 YAML 列表");

function upsert(id, name, update) {
  let row = rows.find((item) => item?.id === id);
  if (!row) rows.push(row = { id, name });
  row.config = { ...row.config, ...update };
}

const providers = rows.find((row) => row?.id === "llm-pi-ai")?.config?.providers ?? {};
upsert("llm-pi-ai", "@deepseek-ai/dsh-llm-pi-ai", {
  providers: { ...providers, [route]: {
    ...providers[route],
    displayName: route,
    apiKeyEnv: ref,
    api: process.env.INSTALL_PROTOCOL,
    baseURL: process.env.INSTALL_BASE_URL,
    models: [{ id: model }],
  }},
});
upsert("agent-default-model", "@deepseek-ai/dsh-agent-default-model", { provider: route, model });

const key = process.env.INSTALL_API_KEY;
const assignment = ['"', "'", "`"].map((quote) => ref + "=" + quote + key + quote)
  .find((line) => !/[\r\n]/.test(key) && parseEnv(line)[ref] === key);
if (!assignment) throw Error("API Key 含无法写入 DSH .env 的字符");
const existing = fs.existsSync(envFile) ? fs.readFileSync(envFile, "utf8") : "";
const lines = existing.split(/\r?\n/).filter((line) => !new RegExp("^\\s*(?:export\\s+)?" + ref + "\\s*=").test(line));
const envConfig = lines.join("\n").replace(/\n*$/, "") + "\n" + assignment + "\n";
for (const [file, value] of [[envFile, envConfig], [patchFile, YAML.stringify(rows)]]) {
  const tmp = file + "." + require("node:crypto").randomUUID();
  try {
    fs.writeFileSync(tmp, value, { mode: 0o600, flag: "wx" });
    fs.renameSync(tmp, file);
  } finally {
    fs.rmSync(tmp, { force: true });
  }
}
NODE

    log_success "DSH configured: provider=$route_id model=$MODEL protocol=$API_PROTOCOL"
}

# ========================
#        主流程
# ========================
main() {
    check_nodejs
    install_dsh
    configure_dsh

    echo ""
    log_success "Installation completed successfully!"
    echo ""
    echo "🚀 使用方式: dsh web"
    echo "              dsh headless \"检查当前项目\""
}

main "$@"
