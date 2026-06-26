param(
    [string]$TaskPrefix = "TradingAgent"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"
$Runner = Join-Path $RepoRoot "daily_digest_runner.py"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Project venv pythonw not found: $PythonPath"
}

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Daily digest runner not found: $Runner"
}

Set-Location -LiteralPath $RepoRoot

function Get-ConfigValue {
    param([string]$Expression)

    $Defaults = @{
        "Config.TOP_RECOMMENDATIONS_TIME" = "06:15"
        "Config.SIMULATION_OPEN_TIME" = "06:35"
        "Config.SIMULATION_MIDDAY_TIME" = "09:00"
        "Config.SIMULATION_EOD_TIME" = "13:10"
    }

    $EnvName = $Expression -replace "^Config\.", ""
    $EnvFiles = @(
        (Join-Path $RepoRoot ".env"),
        (Join-Path $RepoRoot ".env.example")
    )

    foreach ($EnvFile in $EnvFiles) {
        if (-not (Test-Path -LiteralPath $EnvFile)) {
            continue
        }

        foreach ($Line in Get-Content -LiteralPath $EnvFile) {
            if ($Line -match "^\s*$EnvName\s*=\s*(.+?)\s*(?:#.*)?$") {
                return $Matches[1].Trim().Trim('"').Trim("'")
            }
        }
    }

    $Value = $Defaults[$Expression]
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "Could not read $Expression from config"
    }
    return $Value
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
        -Execute $PythonPath `
        -Argument "`"$Runner`" $Arguments" `
        -WorkingDirectory $RepoRoot
    $Trigger = New-ScheduledTaskTrigger `
        -Weekly `
        -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
        -At $At
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

    Write-Host "Installed scheduled task '$TaskName' at $RunTime on weekdays."
}

$DigestTime = Get-ConfigValue "Config.TOP_RECOMMENDATIONS_TIME"
$OpenTime = Get-ConfigValue "Config.SIMULATION_OPEN_TIME"
$MiddayTime = Get-ConfigValue "Config.SIMULATION_MIDDAY_TIME"
$EodTime = Get-ConfigValue "Config.SIMULATION_EOD_TIME"

Register-DigestTask `
    -TaskName "$($TaskPrefix)DailyDigest" `
    -RunTime $DigestTime `
    -Arguments "--run-once" `
    -Description "Send the trading council WhatsApp digest on market weekdays."

Register-DigestTask `
    -TaskName "$($TaskPrefix)OpenSimulation" `
    -RunTime $OpenTime `
    -Arguments "--capture-open" `
    -Description "Capture simulated open-entry trades for top recommendations on market weekdays."

Register-DigestTask `
    -TaskName "$($TaskPrefix)MiddaySimulationSummary" `
    -RunTime $MiddayTime `
    -Arguments "--send-midday-summary" `
    -Description "Send simulated midday P&L snapshot for top recommendations on market weekdays."

Register-DigestTask `
    -TaskName "$($TaskPrefix)EodSimulationSummary" `
    -RunTime $EodTime `
    -Arguments "--send-eod-summary" `
    -Description "Send simulated end-of-day P&L summary for top recommendations on market weekdays."

Write-Host "Wake-to-run is enabled, but Windows power settings and laptop sleep/hibernate behavior still control whether it can wake."
