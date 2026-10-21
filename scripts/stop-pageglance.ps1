[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$stateDir = Join-Path $env:LOCALAPPDATA 'PageGlance'
$logPath = Join-Path $stateDir 'logs\startup.log'

function Stop-TrackedProcess([string]$Name, [string]$PidPath, [string]$ExpectedExe) {
    if (-not (Test-Path -LiteralPath $PidPath -PathType Leaf)) {
        Write-Output "$Name was not started by this script (no tracked PID). Leaving any existing listener untouched."
        return
    }
    $rawPid = (Get-Content -LiteralPath $PidPath -Raw).Trim()
    $processId = 0
    if (-not [int]::TryParse($rawPid, [ref]$processId)) {
        Remove-Item -LiteralPath $PidPath -Force
        Write-Output "$Name has a stale PID record."
        return
    }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) {
        Remove-Item -LiteralPath $PidPath -Force
        Write-Output "$Name is already stopped."
        return
    }
    try { $actualExe = $process.Path } catch { $actualExe = '' }
    if (-not $actualExe -or -not [string]::Equals($actualExe, $ExpectedExe, [StringComparison]::OrdinalIgnoreCase)) {
        Write-Output "$Name PID no longer matches the expected executable; leaving it untouched."
        Remove-Item -LiteralPath $PidPath -Force
        return
    }
    & taskkill.exe /PID $processId /T /F | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Remove-Item -LiteralPath $PidPath -Force
        Write-Output "Stopped PageGlance $Name process tree."
    } else {
        Write-Output "Could not stop PageGlance $Name process tree (taskkill exit $LASTEXITCODE)."
    }
}

if (-not (Test-Path -LiteralPath $stateDir)) {
    Write-Output 'PageGlance has no tracked services to stop.'
    exit 0
}
$backendPythonFile = Join-Path $stateDir 'backend.python'
$backendPython = if (Test-Path -LiteralPath $backendPythonFile -PathType Leaf) {
    (Get-Content -LiteralPath $backendPythonFile -Raw).Trim()
} else { $null }
$node = (Get-Command node.exe -ErrorAction SilentlyContinue).Source
if ($node) {
    Stop-TrackedProcess 'frontend' (Join-Path $stateDir 'frontend.pid') $node
} else {
    Write-Output 'Node.js is not on PATH; the frontend process was not stopped.'
}
if ($backendPython) {
    Stop-TrackedProcess 'backend' (Join-Path $stateDir 'backend.pid') $backendPython
    Remove-Item -LiteralPath $backendPythonFile -Force -ErrorAction SilentlyContinue
} else {
    Write-Output 'No backend executable record is available; the backend process was not stopped.'
}
if (Test-Path -LiteralPath $logPath) {
    Add-Content -LiteralPath $logPath -Value ('{0} Manual stop requested.' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss zzz')) -Encoding utf8
}
