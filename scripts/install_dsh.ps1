[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Model,

    [Parameter(Position = 1)]
    [string]$ApiKey = $env:DSH_API_KEY,

    [Parameter(Position = 2)]
    [string]$BaseUrl,

    [string]$Provider = $env:DSH_PROVIDER,

    [ValidateSet("openai-completions", "openai-responses", "anthropic-messages")]
    [string]$ApiProtocol = $env:DSH_API_PROTOCOL
)

# DSH Windows 原生一键安装脚本。
# 用法：powershell -ExecutionPolicy Bypass -File .\scripts\install_dsh.ps1 [模型名称] [API_KEY] [BASE_URL]

if ($PSBoundParameters.ContainsKey("ApiKey") -and -not $ApiKey) { $ApiKey = "sk-local" }

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$NodeInstallVersion = if ($env:NODE_INSTALL_VERSION) { $env:NODE_INSTALL_VERSION } else { "24" }
$DshPackage = "@deepseek-ai/dsh"

function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Failure([string]$Message) { Write-Host "[ERROR] $Message" -ForegroundColor Red }

$Providers = @(
    [pscustomobject]@{ Name = "智谱 AI (GLM)"; Key = "zhipu"; Model = "glm-4.6" },
    [pscustomobject]@{ Name = "阿里百炼 (Qwen)"; Key = "bailian"; Model = "qwen3-coder-plus" },
    [pscustomobject]@{ Name = "幻方 DeepSeek"; Key = "deepseek"; Model = "deepseek-v3" },
    [pscustomobject]@{ Name = "火山方舟 (豆包)"; Key = "volcano"; Model = "doubao-seed-1-6-250615" },
    [pscustomobject]@{ Name = "MiniMax (M1/abab)"; Key = "minimax"; Model = "MiniMax-M1" },
    [pscustomobject]@{ Name = "月之暗面 (Kimi)"; Key = "moonshot"; Model = "kimi-k2" },
    [pscustomobject]@{ Name = "阶跃星辰 (Step)"; Key = "stepfun"; Model = "step-2" },
    [pscustomobject]@{ Name = "Anthropic 官方"; Key = "anthropic"; Model = "claude-sonnet-4-5" },
    [pscustomobject]@{ Name = "自定义 (手动填 URL)"; Key = "custom"; Model = "" }
)

function Get-ProviderUrl([string]$ProviderKey) {
    $useOpenAi = $script:ApiProtocol -in @("openai-completions", "openai-responses")
    switch ($ProviderKey.ToLowerInvariant()) {
        { $_ -in @("anthropic", "claude") } { return "https://api.anthropic.com" }
        { $_ -in @("zhipu", "glm") } { return $(if ($useOpenAi) { "https://open.bigmodel.cn/api/paas/v4" } else { "https://open.bigmodel.cn/api/anthropic" }) }
        "deepseek" { return $(if ($useOpenAi) { "https://api.deepseek.com/v1" } else { "https://api.deepseek.com/anthropic" }) }
        { $_ -in @("bailian", "dashscope", "qwen") } { return $(if ($useOpenAi) { "https://dashscope.aliyuncs.com/compatible-mode/v1" } else { "https://dashscope.aliyuncs.com/apps/anthropic" }) }
        { $_ -in @("volcano", "ark", "doubao") } { return $(if ($useOpenAi) { "https://ark.cn-beijing.volces.com/api/v3" } else { "https://ark.cn-beijing.volces.com/api/coding" }) }
        "minimax" { return $(if ($useOpenAi) { "https://api.minimaxi.com/v1" } else { "https://api.minimaxi.com/anthropic" }) }
        { $_ -in @("moonshot", "kimi") } { return $(if ($useOpenAi) { "https://api.moonshot.cn/v1" } else { "https://api.moonshot.cn/anthropic" }) }
        { $_ -in @("stepfun", "step") } { return $(if ($useOpenAi) { "https://api.stepfun.com/v1" } else { "https://api.stepfun.com/step_plan" }) }
        default { return "" }
    }
}

function Get-ModelUrl([string]$ModelName) {
    $normalizedModel = $ModelName.ToLowerInvariant()
    if ($normalizedModel -match "^claude-") { return (Get-ProviderUrl "anthropic") }
    if ($normalizedModel -match "^deepseek-") { return (Get-ProviderUrl "deepseek") }
    if ($normalizedModel -match "^(glm-|chatglm-)") { return (Get-ProviderUrl "zhipu") }
    if ($normalizedModel -match "^doubao-") { return (Get-ProviderUrl "volcano") }
    if ($normalizedModel -match "^(qwen|qwq)") { return (Get-ProviderUrl "bailian") }
    if ($normalizedModel -match "^(minimax-|abab)") { return (Get-ProviderUrl "minimax") }
    if ($normalizedModel -match "^(kimi-|moonshot-)") { return (Get-ProviderUrl "moonshot") }
    if ($normalizedModel -match "^step-") { return (Get-ProviderUrl "stepfun") }
    return ""
}

function Read-RequiredValue([string]$Prompt, [string]$DefaultValue) {
    $value = Read-Host "$Prompt (默认 $DefaultValue)"
    if ([string]::IsNullOrWhiteSpace($value)) {
        if ([string]::IsNullOrWhiteSpace($DefaultValue)) { throw "$Prompt 不能为空" }
        return $DefaultValue
    }
    return $value
}

function Read-ApiKey {
    $secret = Read-Host "请输入 API_KEY (默认 sk-local)" -AsSecureString
    $value = [Net.NetworkCredential]::new("", $secret).Password
    if ($value) { return $value }
    return "sk-local"
}

function Resolve-Configuration {
    if ([string]::IsNullOrWhiteSpace($script:Model) -and
        [string]::IsNullOrWhiteSpace($script:BaseUrl) -and
        [string]::IsNullOrWhiteSpace($script:Provider)) {
        Write-Host "请选择模型提供方:"
        for ($i = 0; $i -lt $Providers.Count; $i++) {
            Write-Host ("  [{0}] {1}" -f ($i + 1), $Providers[$i].Name)
        }
        Write-Host "  [0] 退出"
        $choiceText = Read-Host "请选择 [1-$($Providers.Count)]"
        $choice = 0
        if (-not [int]::TryParse($choiceText, [ref]$choice) -or $choice -lt 1 -or $choice -gt $Providers.Count) {
            throw "无效选择: $choiceText"
        }
        $selected = $Providers[$choice - 1]
        $script:Provider = $selected.Key
        $script:BaseUrl = if ($selected.Key -eq "custom") {
            Read-Host "请输入 API BASE_URL (如 http://192.168.1.100:8000/v1)"
        } else {
            Get-ProviderUrl $selected.Key
        }
        if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) { throw "API 地址不能为空" }
        $script:Model = Read-RequiredValue "请输入模型名称" $selected.Model
        if (-not $script:ApiKey) { $script:ApiKey = Read-ApiKey }
        Resolve-ApiProtocol
        return
    }

    if ([string]::IsNullOrWhiteSpace($script:Model)) {
        $script:Model = Read-RequiredValue "请输入模型名称" "deepseek-v4-pro"
    }
    if ([string]::IsNullOrWhiteSpace($script:ApiKey)) {
        $script:ApiKey = Read-ApiKey
    }
    if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
        if (-not [string]::IsNullOrWhiteSpace($script:Provider)) {
            $script:BaseUrl = Get-ProviderUrl $script:Provider
            if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
                throw "未知 DSH_PROVIDER: $($script:Provider)"
            }
        } else {
            $script:BaseUrl = Get-ModelUrl $script:Model
        }
        if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) {
            $script:BaseUrl = Read-Host "模型 '$($script:Model)' 未匹配已知平台，请输入 API BASE_URL"
            if ([string]::IsNullOrWhiteSpace($script:BaseUrl)) { throw "API 地址不能为空" }
        }
    }
    Resolve-ApiProtocol
}

function Resolve-ApiProtocol {
    if ([string]::IsNullOrWhiteSpace($script:ApiProtocol)) {
        $script:ApiProtocol = if ($script:BaseUrl -match "anthropic|^https://ark\.cn-beijing\.volces\.com/api/coding|^https://api\.stepfun\.com/step_plan") {
            "anthropic-messages"
        } else {
            "openai-completions"
        }
    }
    if ($script:ApiProtocol -notin @("openai-completions", "openai-responses", "anthropic-messages")) {
        throw "不支持的 API 协议: $script:ApiProtocol"
    }
}

function Add-UserPath([string]$Directory) {
    $normalized = $Directory.TrimEnd("\")
    $currentEntries = @($env:Path -split ";" | Where-Object { $_ })
    if (-not ($currentEntries | Where-Object { $_.TrimEnd("\") -ieq $normalized })) {
        $env:Path = "$Directory;$($env:Path)"
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $userEntries = @($userPath -split ";" | Where-Object { $_ })
    if (-not ($userEntries | Where-Object { $_.TrimEnd("\") -ieq $normalized })) {
        [Environment]::SetEnvironmentVariable("Path", ((@($Directory) + $userEntries) -join ";"), "User")
    }
}

function Get-LatestNodeVersion {
    if ($env:NODE_VERSION) {
        if ($env:NODE_VERSION.StartsWith("v")) { return $env:NODE_VERSION }
        return "v$($env:NODE_VERSION)"
    }

    foreach ($distBase in @("https://nodejs.org/dist", "https://npmmirror.com/mirrors/node")) {
        try {
            Write-Info "正在从 $distBase 查询 Node.js $NodeInstallVersion.x 最新版本..."
            $index = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$distBase/index.tab").Content
            $line = $index -split "`n" | Where-Object { $_ -match "^v$([regex]::Escape($NodeInstallVersion))\." } | Select-Object -First 1
            if ($line) { return ($line -split "`t")[0].Trim() }
        } catch {
            Write-Info "无法访问 $distBase，尝试下一个镜像。"
        }
    }
    throw "无法解析 Node.js $NodeInstallVersion.x 版本；可设置 NODE_VERSION=v$NodeInstallVersion.x.y 后重试。"
}

function Install-NodeJs {
    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCommand) {
        $currentVersion = (& node --version).Trim().TrimStart("v")
        $versionParts = $currentVersion.Split(".")
        $majorVersion = [int]$versionParts[0]
        $minorVersion = if ($versionParts.Count -gt 1) { [int]$versionParts[1] } else { 0 }
        $isSupported = ($majorVersion -eq 24 -and $minorVersion -ge 9) -or $majorVersion -gt 24
        # PowerShell 可能优先解析到未签名的 npm.ps1；始终检查并调用 npm.cmd。
        if ($isSupported -and (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
            Write-Success "Node.js 已安装: v$currentVersion"
            return
        }
        Write-Info "DSH 需要 Node.js 24.9 或更高版本；当前为 v$currentVersion，开始安装兼容版本。"
    }

    $architecture = $env:PROCESSOR_ARCHITECTURE.ToUpperInvariant()
    $nodeArch = switch ($architecture) {
        "AMD64" { "win-x64" }
        "ARM64" { "win-arm64" }
        default { throw "不支持的 Windows 架构: $architecture" }
    }
    $nodeVersion = Get-LatestNodeVersion
    $archiveName = "node-$nodeVersion-$nodeArch.zip"
    $tempDir = Join-Path ([IO.Path]::GetTempPath()) ("dsh-node-" + [guid]::NewGuid().ToString("N"))
    $archivePath = Join-Path $tempDir $archiveName
    $checksumsPath = Join-Path $tempDir "SHASUMS256.txt"
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    try {
        $downloaded = $false
        foreach ($distBase in @("https://nodejs.org/dist", "https://npmmirror.com/mirrors/node")) {
            try {
                Write-Info "正在从 $distBase 下载 Node.js $nodeVersion ($nodeArch)..."
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$distBase/$nodeVersion/$archiveName" -OutFile $archivePath
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Uri "$distBase/$nodeVersion/SHASUMS256.txt" -OutFile $checksumsPath
                $downloaded = $true
                break
            } catch {
                Write-Info "从 $distBase 下载失败，尝试下一个镜像。"
            }
        }
        if (-not $downloaded) { throw "所有镜像均下载失败: $archiveName" }

        $checksumLine = Get-Content $checksumsPath | Where-Object { $_ -match "\s+$([regex]::Escape($archiveName))$" } | Select-Object -First 1
        if (-not $checksumLine) { throw "校验文件中找不到 $archiveName" }
        $expectedChecksum = ($checksumLine -split "\s+")[0].ToLowerInvariant()
        $actualChecksum = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
        if ($actualChecksum -ne $expectedChecksum) { throw "Node.js 安装包 SHA-256 校验失败" }

        $installRoot = Join-Path $env:LOCALAPPDATA "Programs\nodejs"
        $versionDir = Join-Path $installRoot "node-$nodeVersion-$nodeArch"
        New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
        if (Test-Path $versionDir) { Remove-Item -LiteralPath $versionDir -Recurse -Force }
        Write-Info "正在安装 Node.js 到 $versionDir..."
        Expand-Archive -LiteralPath $archivePath -DestinationPath $installRoot -Force
        Add-UserPath $versionDir
    } finally {
        if (Test-Path $tempDir) { Remove-Item -LiteralPath $tempDir -Recurse -Force }
    }

    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Node.js 安装完成，但 node 不在 PATH 中。" }
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw "Node.js 安装完成，但 npm.cmd 不在 PATH 中。" }
    & node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 24 || major === 24 && minor >= 9 ? 0 : 1)'
    if ($LASTEXITCODE -ne 0) { throw "DSH 需要 Node.js 24.9 或更新版本" }
    Write-Success "Node.js 已安装: $(& node --version)"
    $npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
    Write-Success "npm 版本: $(& $npmCommand --version)"
}

function Install-NpmPackages([string[]]$NpmArguments) {
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    & $npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 @NpmArguments
    if ($LASTEXITCODE -ne 0) {
        Write-Info "默认 npm 源安装失败，改用 npmmirror..."
        & $npm install --no-audit --no-fund --fetch-retries=1 --fetch-timeout=60000 --registry=https://registry.npmmirror.com @NpmArguments
    }
    if ($LASTEXITCODE -ne 0) { throw "npm 安装失败，退出码: $LASTEXITCODE" }
}

function Install-Dsh {
    $script:DshNpmPrefix = Join-Path $env:LOCALAPPDATA "Programs\dsh"
    Install-NpmPackages @("--global", "--prefix", $DshNpmPrefix, "$DshPackage@latest", "yaml@2", "pnpm@10")
    Add-UserPath $DshNpmPrefix
    & (Join-Path $DshNpmPrefix "dsh.cmd") --version
    if ($LASTEXITCODE -ne 0) { throw "DSH CLI 启动失败" }
}

function Configure-Dsh {
    $previousNodePath = $env:NODE_PATH
    $env:NODE_PATH = Join-Path $DshNpmPrefix "node_modules"
    $routeId = "slai-" + $(if ($script:Provider) { $script:Provider.ToLowerInvariant() } else { "custom" })
    $routeId = $routeId -replace "[^a-z0-9-]", "-"
    $dshHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $HOME ".dsh" }
    New-Item -ItemType Directory -Force -Path $dshHome | Out-Null
    $env:INSTALL_API_KEY = $script:ApiKey
    $env:INSTALL_DSH_HOME = $dshHome
    $env:INSTALL_ROUTE_ID = $routeId
    $env:INSTALL_BASE_URL = $script:BaseUrl
    $env:INSTALL_MODEL = $script:Model
    $env:INSTALL_PROTOCOL = $script:ApiProtocol
    $configScriptPath = Join-Path ([IO.Path]::GetTempPath()) ("dsh-config-" + [guid]::NewGuid().ToString("N") + ".js")
    try {
        $configScript = @'
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
'@
        [IO.File]::WriteAllText($configScriptPath, $configScript, (New-Object Text.UTF8Encoding($false)))
        & node $configScriptPath
        if ($LASTEXITCODE -ne 0) { throw "写入 DSH 配置失败" }
    } finally {
        Remove-Item -LiteralPath $configScriptPath -Force -ErrorAction SilentlyContinue
        $env:NODE_PATH = $previousNodePath
        Remove-Item Env:INSTALL_API_KEY -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_DSH_HOME -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_ROUTE_ID -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_BASE_URL -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:INSTALL_PROTOCOL -ErrorAction SilentlyContinue
    }
    Write-Success "DSH 配置完成: provider=$routeId model=$($script:Model) protocol=$($script:ApiProtocol)"
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Resolve-Configuration
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  模型:    $Model"
    if ($Provider) { Write-Host "  平台:    $Provider" }
    Write-Host "  API URL: $BaseUrl"
    Write-Host "  协议:    $ApiProtocol"
    Write-Host "========================================"
    Write-Host ""

    Install-NodeJs
    Install-Dsh
    Configure-Dsh
    Write-Host ""
    Write-Success "安装完成！重新打开终端后运行: dsh headless `"检查当前项目`" 或 dsh web"
    # 安装脚本是子进程，改不了"已经开着"的那个终端的环境变量，只能在这里给出可直接粘贴的刷新命令。
    Write-Info "注意：当前已打开的终端不会自动加载新的 PATH。重开一个全新的终端窗口，"
    Write-Info "      或先在当前窗口执行下面这行刷新，即可直接用 dsh:"
    Write-Host '      $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")' -ForegroundColor Yellow
} catch {
    Write-Failure $_.Exception.Message
    exit 1
}
