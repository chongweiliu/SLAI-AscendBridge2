@echo off
setlocal
set "CLAUDE_INSTALLER_PS1=%~dp0install_claude_code.ps1"
set "CLAUDE_INSTALLER_MODEL=%~1"
set "CLAUDE_INSTALLER_API_KEY=%~2"
set "CLAUDE_INSTALLER_BASE_URL=%~3"

rem ASCII-only on purpose: cmd.exe decodes .cmd using the console OEM codepage,
rem so non-ASCII comments can be mis-decoded and break batch parsing.
rem Run the in-memory ScriptBlock to avoid enterprise AllSigned policy blocking the
rem local unsigned .ps1. The script still comes from the fixed file next to this CMD
rem and the system execution policy is not modified.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$source = Get-Content -Raw -LiteralPath $env:CLAUDE_INSTALLER_PS1; $installer = [ScriptBlock]::Create($source); . $installer -Model $env:CLAUDE_INSTALLER_MODEL -ApiKey $env:CLAUDE_INSTALLER_API_KEY -BaseUrl $env:CLAUDE_INSTALLER_BASE_URL"
exit /b %errorlevel%
