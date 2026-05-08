param(
    [string]$TaskPrefix = "TradingAgent"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "daily_digest_runner.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project venv python not found: $Python"
}

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Daily digest runner not found: $Runner"
}

function Get-ConfigValue {
    param([string]$Expression)

    $Value = & $Python -c "from config import Config; print($Expression)"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Value)) {
        throw "Could not read $Expression from config"
    }
    return $Value.Trim()
}

function Register-DigestTask {
    param(
        [string]$TaskName,
        [string]$RunTime,
        [string]$Arguments,
        [string]$Description
    )

    $At = [datetime]::ParseExact($RunTime, "HH:mm", $null)
    $Action = New-ScheduledTaskAction `
        -Execute $Python `
        -Argument "`"$Runner`" $Arguments" `
        -WorkingDirectory $RepoRoot
    $Trigger = New-ScheduledTaskTrigger -Daily -At $At
    $Settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -WakeToRun `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries
    $Principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType Interactive `
        -RunLevel Limited

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description $Description `
        -Force | Out-Null

    Write-Host "Installed scheduled task '$TaskName' at $RunTime daily."
}

$DigestTime = Get-ConfigValue "Config.TOP_RECOMMENDATIONS_TIME"
$OpenTime = Get-ConfigValue "Config.SIMULATION_OPEN_TIME"
$MiddayTime = Get-ConfigValue "Config.SIMULATION_MIDDAY_TIME"
$EodTime = Get-ConfigValue "Config.SIMULATION_EOD_TIME"

Register-DigestTask `
    -TaskName "$($TaskPrefix)DailyDigest" `
    -RunTime $DigestTime `
    -Arguments "--run-once" `
    -Description "Send the trading council WhatsApp digest every day."

Register-DigestTask `
    -TaskName "$($TaskPrefix)OpenSimulation" `
    -RunTime $OpenTime `
    -Arguments "--capture-open" `
    -Description "Capture simulated open-entry trades for the top recommendations."

Register-DigestTask `
    -TaskName "$($TaskPrefix)MiddaySimulationSummary" `
    -RunTime $MiddayTime `
    -Arguments "--send-midday-summary" `
    -Description "Send simulated midday P&L snapshot for top recommendations."

Register-DigestTask `
    -TaskName "$($TaskPrefix)EodSimulationSummary" `
    -RunTime $EodTime `
    -Arguments "--send-eod-summary" `
    -Description "Send simulated end-of-day P&L summary for top recommendations."

Write-Host "Wake-to-run is enabled, but Windows power settings and laptop sleep/hibernate behavior still control whether it can wake."
