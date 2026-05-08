param(
    [string]$TaskPrefix = "TradingAgent"
)

$ErrorActionPreference = "Stop"

$TaskNames = @(
    "$($TaskPrefix)DailyDigest",
    "$($TaskPrefix)OpenSimulation",
    "$($TaskPrefix)MiddaySimulationSummary",
    "$($TaskPrefix)EodSimulationSummary"
)

foreach ($TaskName in $TaskNames) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "Scheduled task '$TaskName' was not installed."
    }
}
