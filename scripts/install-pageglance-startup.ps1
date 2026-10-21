[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendDir = Join-Path $repoRoot 'backend'
$extensionDir = Join-Path $repoRoot 'extension'
$startupScript = Join-Path $PSScriptRoot 'start-pageglance.ps1'
$taskName = 'PageGlance Development Services'
$venvPython = Join-Path $backendDir '.venv\Scripts\python.exe'
$taskUser = "$env:USERDOMAIN\$env:USERNAME"

function Test-PythonRuntime([string]$PythonPath) {
    $probeId = [Guid]::NewGuid().ToString('N')
    $stdout = Join-Path $env:TEMP "pageglance-python-check-$probeId.out"
    $stderr = Join-Path $env:TEMP "pageglance-python-check-$probeId.err"
    try {
        $probe = Start-Process -FilePath $PythonPath `
            -ArgumentList '-c "import fastapi, uvicorn, pydantic, groq, dotenv"' `
            -PassThru -Wait -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        return ($probe.ExitCode -eq 0)
    } finally {
        Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Get-Command python.exe -ErrorAction SilentlyContinue)) {
    throw 'Python 3 was not found on PATH. Install Python 3.10 or newer and ensure python.exe is on PATH.'
}
if (-not (Get-Command node.exe -ErrorAction SilentlyContinue) -or -not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw 'Node.js and npm were not found on PATH. Install a supported Node.js version and retry.'
}

if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    if (Test-PythonRuntime $venvPython) { $python = $venvPython }
}
if (-not $python) {
    $systemPython = (Get-Command python.exe).Source
    if (Test-PythonRuntime $systemPython) { $python = $systemPython }
}
if (-not $python) {
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        throw 'backend\.venv exists but lacks runtime packages, and the system Python does too. Repair that environment or remove it yourself, then rerun this installer.'
    }
    Write-Host 'Creating a backend virtual environment and installing dependencies from backend\pyproject.toml...'
    & (Get-Command python.exe).Source -m venv (Join-Path $backendDir '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create backend\.venv.' }
    Push-Location $backendDir
    try {
        & $venvPython -m pip install -e .
        if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed; no startup task was registered.' }
    } finally {
        Pop-Location
    }
    $python = $venvPython
}
Write-Host "Using backend Python at $python."

if (-not (Test-Path -LiteralPath (Join-Path $extensionDir 'node_modules\vite\bin\vite.js'))) {
    Write-Host 'Installing frontend dependencies with npm ci...'
    Push-Location $extensionDir
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed; no startup task was registered.' }
    } finally {
        Pop-Location
    }
}

$action = New-ScheduledTaskAction -Execute (Join-Path $PSHOME 'powershell.exe') `
    -Argument "-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$startupScript`"" `
    -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 3)
$task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
    -Description 'Starts the local PageGlance FastAPI backend, waits for its health endpoint, then starts Vite after user logon.'
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
Write-Host "Registered per-user logon task '$taskName'."
Write-Host 'Starting it now so startup can be verified without waiting for a restart...'
Start-ScheduledTask -TaskName $taskName
Write-Host "Task started. Logs will be written under $env:LOCALAPPDATA\PageGlance\logs."
