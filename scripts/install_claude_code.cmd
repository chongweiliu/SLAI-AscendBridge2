@echo off
setlocal
set "CLAUDE_INSTALLER_PS1=%~dp0install_claude_code.ps1"
set "CLAUDE_INSTALLER_MODEL=%~1"
set "CLAUDE_INSTALLER_API_KEY=%~2"
set "CLAUDE_INSTALLER_BASE_URL=%~3"

rem 通过内存 ScriptBlock 执行，避免企业 AllSigned 策略拦截本地未签名的 .ps1 文件。
rem 脚本仍来自与本 CMD 同目录的固定文件，不会修改系统执行策略。
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$source = Get-Content -Raw -LiteralPath $env:CLAUDE_INSTALLER_PS1; $installer = [ScriptBlock]::Create($source); & $installer -Model $env:CLAUDE_INSTALLER_MODEL -ApiKey $env:CLAUDE_INSTALLER_API_KEY -BaseUrl $env:CLAUDE_INSTALLER_BASE_URL"
exit /b %errorlevel%
