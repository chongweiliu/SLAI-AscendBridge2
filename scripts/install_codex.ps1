[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Model = $env:CODEX_MODEL,
    [Parameter(Position = 1)][string]$ApiKey = $env:CODEX_API_KEY,
    [Parameter(Position = 2)][string]$BaseUrl = $env:CODEX_BASE_URL,
    [string]$Provider = $env:CODEX_PROVIDER
)

# 安装或更新 Windows 原生 Codex CLI，支持第三方 OpenAI Responses API。
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$NodeMinVersion = 22
$NodeInstallVersion = if ($env:NODE_INSTALL_VERSION) { $env:NODE_INSTALL_VERSION } else { "22" }

function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }

$Providers = @(
    [pscustomobject]@{ Key = "zhipu"; Name = "智谱 GLM Coding Plan"; Model = "glm-5.4"; Url = "https://open.bigmodel.cn/api/v1" }
    [pscustomobject]@{ Key = "bailian"; Name = "阿里百炼 Token Plan"; Model = "auto"; Url = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" }
    [pscustomobject]@{ Key = "deepseek"; Name = "DeepSeek"; Model = "deepseek-v4-pro"; Url = "https://api.deepseek.com" }
    [pscustomobject]@{ Key = "volcano"; Name = "火山方舟 Coding Plan"; Model = "ark-code-latest"; Url = "https://ark.cn-beijing.volces.com/api/coding/v3" }
    [pscustomobject]@{ Key = "minimax"; Name = "MiniMax M Plan（中国）"; Model = "MiniMax-M3.1-Flash"; Url = "https://api.minimax.cn/v1" }
    [pscustomobject]@{ Key = "moonshot"; Name = "Kimi Code（中国）"; Model = "kimi-for-coding"; Url = "https://api.kimi.com/coding/v1" }
    [pscustomobject]@{ Key = "stepfun"; Name = "阶跃星辰 Step Plan"; Model = "step-3.7-flash"; Url = "https://api.stepfun.com/step_plan/v1" }
    [pscustomobject]@{ Key = "custom"; Name = "自定义 Responses 网关"; Model = ""; Url = "" }
)

function Get-ProviderForModel([string]$ModelName) {
    if ($ModelName -match '^(glm-|chatglm-)') { return "zhipu" }
    if ($ModelName -match '^(qwen|qwq)') { return "bailian" }
    if ($ModelName -match '^deepseek-') { return "deepseek" }
    if ($ModelName -match '^(doubao-|ark-code-)') { return "volcano" }
    if ($ModelName -match '^minimax-') { return "minimax" }
    if ($ModelName -match '^(kimi-|k3$|k3-|k3\.|moonshot-)') { return "moonshot" }
    if ($ModelName -match '^step-') { return "stepfun" }
    return "custom"
}

function Select-CodexProvider {
    Write-Host "请选择模型提供方:"
    for ($i = 0; $i -lt $Providers.Count; $i++) {
        Write-Host ("  [{0}] {1}" -f ($i + 1), $Providers[$i].Name)
    }
    Write-Host "  [0] 退出"
    $choice = 0
    $inputChoice = Read-Host "请选择 [1-$($Providers.Count)]"
    if (-not [int]::TryParse($inputChoice, [ref]$choice) -or $choice -lt 0 -or $choice -gt $Providers.Count) {
        throw "无效选择: $inputChoice"
    }
    if ($choice -eq 0) { exit 0 }
    return $Providers[$choice - 1].Key
}

function Get-CodexProvider([string]$Key) {
    $aliases = @{ glm = "zhipu"; qwen = "bailian"; dashscope = "bailian"; ark = "volcano";
        doubao = "volcano"; kimi = "moonshot"; step = "stepfun" }
    if ($aliases.ContainsKey($Key)) { $Key = $aliases[$Key] }
    $selected = $Providers | Where-Object { $_.Key -eq $Key } | Select-Object -First 1
    if (-not $selected) { throw "未知 CODEX_PROVIDER: $Key" }
    return $selected
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
        $majorVersion = [int]($currentVersion.Split(".")[0])
        # PowerShell 可能优先解析到未签名的 npm.ps1；始终检查并调用 npm.cmd。
        if ($majorVersion -ge $NodeMinVersion -and (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
            Write-Success "Node.js 已安装: v$currentVersion"
            return
        }
        Write-Info "Node.js v$currentVersion 低于 v$NodeMinVersion，开始升级。"
    }

    $architecture = $env:PROCESSOR_ARCHITECTURE.ToUpperInvariant()
    $nodeArch = switch ($architecture) {
        "AMD64" { "win-x64" }
        "ARM64" { "win-arm64" }
        default { throw "不支持的 Windows 架构: $architecture" }
    }
    $nodeVersion = Get-LatestNodeVersion
    $archiveName = "node-$nodeVersion-$nodeArch.zip"
    $tempDir = Join-Path ([IO.Path]::GetTempPath()) ("codex-node-" + [guid]::NewGuid().ToString("N"))
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
    $actualVersion = [version]((& node --version).Trim().TrimStart("v"))
    if ($actualVersion.Major -lt $NodeMinVersion) { throw "需要 Node.js 22 或更新版本" }
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

function Write-CodexProfile {
    $OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $env:INSTALL_MODEL = $Model
    $env:INSTALL_BASE_URL = $BaseUrl
    $env:INSTALL_API_KEY = $ApiKey
    $env:INSTALL_PROVIDER_NAME = $providerName
    try {
        @'
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const home = process.env.CODEX_HOME || path.join(os.homedir(), ".codex");
const profile = path.join(home, "slai-ascendbridge.config.toml");
const marker = "# Managed by SLAI-AscendBridge2 install_codex";
const tomlString = (value) => JSON.stringify(value).replace(/\x7f/g, "\\u007f");
const lines = [
  marker,
  "model = " + tomlString(process.env.INSTALL_MODEL),
  'model_provider = "slai-ascendbridge"',
  'forced_login_method = "api"',
  "",
  "[model_providers.slai-ascendbridge]",
  "name = " + tomlString(process.env.INSTALL_PROVIDER_NAME),
  "base_url = " + tomlString(process.env.INSTALL_BASE_URL),
  'wire_api = "responses"',
  "experimental_bearer_token = " + tomlString(process.env.INSTALL_API_KEY),
  "",
];
fs.mkdirSync(home, { recursive: true, mode: 0o700 });
if (fs.existsSync(profile) && !fs.readFileSync(profile, "utf8").startsWith(marker + "\n")) {
  throw Error("已有同名 Codex 配置，请先将其改名: " + profile);
}
const tmp = profile + "." + require("node:crypto").randomUUID();
try {
  fs.writeFileSync(tmp, lines.join("\n"), { mode: 0o600, flag: "wx" });
  fs.renameSync(tmp, profile);
} finally {
  fs.rmSync(tmp, { force: true });
}
'@ | node -
        if ($LASTEXITCODE -ne 0) { throw "写入 Codex 配置失败" }
    } finally {
        Remove-Item Env:INSTALL_MODEL, Env:INSTALL_BASE_URL, Env:INSTALL_API_KEY, Env:INSTALL_PROVIDER_NAME -ErrorAction SilentlyContinue
    }
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    $configureProvider = [bool]($Model -or $ApiKey -or $BaseUrl -or $Provider -or $interactive)
    $providerName = "自定义 Responses 网关"
    if ($configureProvider) {
        if (-not $BaseUrl) {
            if (-not $Provider) {
                $Provider = if (-not $Model -and $interactive) { Select-CodexProvider } else { Get-ProviderForModel $Model }
            }
            $selected = Get-CodexProvider $Provider
            $BaseUrl = $selected.Url
            $providerName = $selected.Name
            if (-not $Model -and $selected.Model) {
                if ($interactive) { $Model = Read-Host "请输入模型名称 (默认 $($selected.Model))" }
                if (-not $Model) { $Model = $selected.Model }
            }
            if (-not $BaseUrl -and $interactive) {
                $BaseUrl = Read-Host "请输入第三方 OpenAI Responses BASE_URL"
            }
        }
        if ($BaseUrl -notmatch '^https?://.+') { throw "请通过 -BaseUrl 或 CODEX_BASE_URL 指定第三方 HTTP(S) 地址" }
        $BaseUrl = $BaseUrl.TrimEnd("/")
        if ($BaseUrl -match '/(chat/completions|messages)$') {
            throw "请填写支持 OpenAI Responses API 的 BASE_URL"
        }
        $BaseUrl = $BaseUrl -replace '/responses$', ''
        if (-not $Model -and $interactive) { $Model = Read-Host "请输入模型名称" }
        if (-not $Model) { throw "请指定模型名称" }
        if (-not $ApiKey -and $PSBoundParameters.ContainsKey("ApiKey")) { $ApiKey = "sk-local" }
        if (-not $ApiKey -and $interactive) {
            $ApiKey = Read-Host "请输入 API_KEY（输入可见）"
        }
        if (-not $ApiKey) { throw "请传入 API_KEY 或设置 CODEX_API_KEY" }
        Write-Info "提供方: $providerName | 模型: $Model | BASE_URL: $BaseUrl"
    }

    Install-NodeJs
    $npmPrefix = Join-Path $env:LOCALAPPDATA "Programs\codex"
    Install-NpmPackages @("--global", "--prefix", $npmPrefix, "--include=optional", "@openai/codex@latest")
    & (Join-Path $npmPrefix "codex.cmd") --version
    if ($LASTEXITCODE -ne 0) { throw "Codex CLI 启动失败" }
    Add-UserPath $npmPrefix
    if ($configureProvider) {
        Write-CodexProfile
        Write-Success "安装完成。重新打开终端后运行: codex --profile slai-ascendbridge"
    } else {
        Write-Success "安装完成。重新打开终端后运行: codex"
    }
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
