[CmdletBinding()]
param([switch]$StopServices)

$ErrorActionPreference = 'Stop'
$taskName = 'PageGlance Development Services'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed the '$taskName' logon task."
} else {
    Write-Host 'The PageGlance automatic startup task is not installed.'
}
if ($StopServices) {
    & (Join-Path $PSScriptRoot 'stop-pageglance.ps1')
}
Write-Host 'Logs and service state files were kept for diagnosis.'
