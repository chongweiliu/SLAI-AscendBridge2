[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Model = $env:CODEX_MODEL,
    [Parameter(Position = 1)][string]$ApiKey = $env:CODEX_API_KEY,
    [Parameter(Position = 2)][string]$BaseUrl = $env:CODEX_BASE_URL
)

# 安装或更新 Windows 原生 Codex CLI，支持 ChatGPT 登录和 Responses API。
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$NodeMinVersion = 22
$NodeInstallVersion = if ($env:NODE_INSTALL_VERSION) { $env:NODE_INSTALL_VERSION } else { "22" }

function Write-Info([string]$Message) { Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }

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
    $env:INSTALL_MODEL = $Model
    $env:INSTALL_BASE_URL = $BaseUrl
    $env:INSTALL_API_KEY = $ApiKey
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
  "",
  "[model_providers.slai-ascendbridge]",
  'name = "SLAI Responses API"',
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
        Remove-Item Env:INSTALL_MODEL, Env:INSTALL_BASE_URL, Env:INSTALL_API_KEY -ErrorAction SilentlyContinue
    }
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $configureProvider = [bool]($Model -or $ApiKey -or $BaseUrl)
    $interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected
    if (-not $configureProvider -and $interactive) {
        Write-Host "  [1] 安装 Codex，使用 ChatGPT 登录（默认）"
        Write-Host "  [2] 安装并配置 Responses API"
        $choice = Read-Host "请选择 [1-2]"
        switch ($choice) {
            "" { }
            "1" { }
            "2" { $configureProvider = $true }
            default { throw "无效选择: $choice" }
        }
    }
    if ($configureProvider) {
        if (-not $BaseUrl -and $interactive) {
            $BaseUrl = Read-Host "请输入 Responses API BASE_URL (默认 https://api.openai.com/v1)"
        }
        if (-not $BaseUrl) { $BaseUrl = "https://api.openai.com/v1" }
        $BaseUrl = $BaseUrl.TrimEnd("/")
        if ($BaseUrl -notmatch '^https?://.+') { throw "BASE_URL 必须是 HTTP(S) 地址" }
        if (-not $Model) {
            if ($BaseUrl -eq "https://api.openai.com/v1") { $Model = "gpt-5.4" }
            elseif ($interactive) { $Model = Read-Host "请输入模型名称" }
        }
        if (-not $Model) { throw "请指定模型名称" }
        if (-not $ApiKey -and $PSBoundParameters.ContainsKey("ApiKey") -and $BaseUrl -ne "https://api.openai.com/v1") {
            $ApiKey = "sk-local"
        }
        if (-not $ApiKey -and $interactive) {
            $secret = Read-Host "请输入 API_KEY" -AsSecureString
            $ApiKey = [Net.NetworkCredential]::new("", $secret).Password
        }
        if (-not $ApiKey) { throw "请传入 API_KEY 或设置 CODEX_API_KEY" }
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
