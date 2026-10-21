[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendDir = Join-Path $repoRoot 'backend'
$extensionDir = Join-Path $repoRoot 'extension'
$stateDir = Join-Path $env:LOCALAPPDATA 'PageGlance'
$logDir = Join-Path $stateDir 'logs'
$startupLog = Join-Path $logDir 'startup.log'
$backendStdout = Join-Path $logDir 'backend.log'
$backendStderr = Join-Path $logDir 'backend-error.log'
$frontendStdout = Join-Path $logDir 'frontend.log'
$frontendStderr = Join-Path $logDir 'frontend-error.log'
$backendPidFile = Join-Path $stateDir 'backend.pid'
$backendPythonFile = Join-Path $stateDir 'backend.python'
$frontendPidFile = Join-Path $stateDir 'frontend.pid'
$frontendUrl = 'http://127.0.0.1:5173/'
$frontendPort = 5173
$healthTimeoutSeconds = 45
$frontendTimeoutSeconds = 30
$mutex = $null

function Write-StartupLog([string]$Message) {
    $line = '{0} {1}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss zzz'), $Message
    Add-Content -LiteralPath $startupLog -Value $line -Encoding utf8
}

function Read-EnvValue([string]$Path, [string]$Name, [string]$DefaultValue) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $DefaultValue }
    $pattern = '^\s*(?:export\s+)?' + [Regex]::Escape($Name) + '\s*=\s*(.*?)\s*$'
    $found = Get-Content -LiteralPath $Path | Where-Object { $_ -match $pattern } | Select-Object -Last 1
    if (-not $found) { return $DefaultValue }
    $value = [Regex]::Match($found, $pattern).Groups[1].Value.Trim()
    if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    return $value
}

function Test-TcpListener([string]$HostName, [int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connect = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $connect.AsyncWaitHandle.WaitOne(700)) { return $false }
        $client.EndConnect($connect)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Test-BackendHealth {
    try {
        $response = Invoke-RestMethod -Uri "$script:apiBase/health" -TimeoutSec 2
        return ($response.status -eq 'ok')
    } catch {
        return $false
    }
}

function Test-ViteReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $frontendUrl -TimeoutSec 2
        return ($response.StatusCode -eq 200 -and $response.Content.Contains('/@vite/client') -and $response.Content.Contains('PageGlance'))
    } catch {
        return $false
    }
}

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

function Wait-ForService([string]$Name, [int]$TimeoutSeconds, [System.Diagnostics.Process]$StartedProcess, [scriptblock]$Probe) {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if (& $Probe) { return $true }
        if ($StartedProcess -and $StartedProcess.HasExited) {
            Write-StartupLog "$Name process exited before becoming ready (exit code $($StartedProcess.ExitCode)). See its service log."
            return $false
        }
        Start-Sleep -Seconds 1
    }
    Write-StartupLog "$Name did not become ready within $TimeoutSeconds seconds."
    return $false
}

try {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $mutex = [Threading.Mutex]::new($false, 'Local\PageGlanceStartup')
    if (-not $mutex.WaitOne(0)) {
        Write-StartupLog 'Another PageGlance startup check is already running; exiting without launching duplicates.'
        exit 0
    }
    Write-StartupLog 'Startup check started.'

    $rootEnv = Join-Path $repoRoot '.env'
    $backendHost = Read-EnvValue $rootEnv 'PAGEGLANCE_HOST' '127.0.0.1'
    $backendPortValue = Read-EnvValue $rootEnv 'PAGEGLANCE_PORT' '8000'
    $backendPort = 0
    if (-not [int]::TryParse($backendPortValue, [ref]$backendPort) -or $backendPort -lt 1 -or $backendPort -gt 65535) {
        throw 'PAGEGLANCE_PORT in the root .env must be an integer from 1 to 65535.'
    }
    if ($backendHost -notin @('127.0.0.1', 'localhost')) {
        throw 'PAGEGLANCE_HOST must be a loopback address; automatic startup does not expose the development API to the network.'
    }
    $script:apiBase = "http://$backendHost`:$backendPort"

    $configuredApiUrl = [Environment]::GetEnvironmentVariable('VITE_PAGEGLANCE_API_URL', 'Process')
    if (-not $configuredApiUrl) {
        $viteEnvFiles = @('.env', '.env.local', '.env.development', '.env.development.local')
        foreach ($envName in $viteEnvFiles) {
            $envPath = Join-Path $extensionDir $envName
            $fileApiUrl = Read-EnvValue $envPath 'VITE_PAGEGLANCE_API_URL' ''
            if ($fileApiUrl) { $configuredApiUrl = $fileApiUrl }
        }
    }
    if ($configuredApiUrl) {
        $parsedApiUrl = $null
        if (-not [Uri]::TryCreate($configuredApiUrl.TrimEnd('/'), [UriKind]::Absolute, [ref]$parsedApiUrl) -or
            $parsedApiUrl.Scheme -ne 'http' -or
            $parsedApiUrl.Host -notin @('127.0.0.1', 'localhost') -or
            $parsedApiUrl.Port -ne $backendPort -or
            $parsedApiUrl.AbsolutePath -ne '/' -or
            $parsedApiUrl.Query -or $parsedApiUrl.Fragment) {
            throw "VITE_PAGEGLANCE_API_URL does not match the local backend at $script:apiBase. Update the setting only if that is intentional."
        }
        $env:VITE_PAGEGLANCE_API_URL = $configuredApiUrl.TrimEnd('/')
    } else {
        $env:VITE_PAGEGLANCE_API_URL = $script:apiBase
    }
    Write-StartupLog "Using local backend address $script:apiBase and the matching Vite API URL."

    $venvPython = Join-Path $backendDir '.venv\Scripts\python.exe'
    $systemPythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    $python = $null
    foreach ($candidate in @($venvPython, $systemPythonCommand.Source)) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        if (Test-PythonRuntime $candidate) { $python = $candidate; break }
    }
    if (-not $python) {
        throw 'FastAPI runtime dependencies are unavailable. Run scripts\install-pageglance-startup.ps1 once to set up the backend environment.'
    }
    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    if (-not $nodeCommand) { throw 'Node.js was not found on PATH. Install Node.js, then run the startup installer again.' }
    $viteEntry = Join-Path $extensionDir 'node_modules\vite\bin\vite.js'
    if (-not (Test-Path -LiteralPath $viteEntry -PathType Leaf)) {
        throw 'Vite dependencies are missing. Run npm ci from the extension directory, then retry startup.'
    }

    $backendStarted = $null
    if (-not (Test-BackendHealth)) {
        if (Test-TcpListener $backendHost $backendPort) {
            throw "Port $backendPort is occupied, but GET $script:apiBase/health did not return status=ok. No backend process was launched."
        }
        Write-StartupLog "Starting FastAPI backend with $python."
        $backendStarted = Start-Process -FilePath $python -ArgumentList @(
            '-m', 'uvicorn', 'app.main:app', '--host', $backendHost, '--port', "$backendPort", '--no-access-log'
        ) -WorkingDirectory $backendDir -WindowStyle Hidden -PassThru -RedirectStandardOutput $backendStdout -RedirectStandardError $backendStderr
        Set-Content -LiteralPath $backendPidFile -Value $backendStarted.Id -NoNewline
        Set-Content -LiteralPath $backendPythonFile -Value $python -NoNewline
    } else {
        Write-StartupLog 'An existing healthy FastAPI service is responding; reusing it.'
    }
    if (-not (Wait-ForService 'FastAPI backend' $healthTimeoutSeconds $backendStarted { Test-BackendHealth })) {
        throw "Backend health check failed. Review $backendStderr and $backendStdout."
    }
    Write-StartupLog 'FastAPI GET /health returned status=ok.'

    $frontendStarted = $null
    if (-not (Test-ViteReady)) {
        if (Test-TcpListener '127.0.0.1' $frontendPort) {
            throw "Port $frontendPort is occupied by a service that did not identify as this Vite frontend. No frontend process was launched."
        }
        Write-StartupLog 'Starting Vite frontend after backend health passed.'
        $frontendStarted = Start-Process -FilePath $nodeCommand.Source -ArgumentList @(
            $viteEntry, '--host', '127.0.0.1', '--port', "$frontendPort", '--strictPort'
        ) -WorkingDirectory $extensionDir -WindowStyle Hidden -PassThru -RedirectStandardOutput $frontendStdout -RedirectStandardError $frontendStderr
        Set-Content -LiteralPath $frontendPidFile -Value $frontendStarted.Id -NoNewline
    } else {
        Write-StartupLog 'An existing Vite frontend is responding; reusing it.'
    }
    if (-not (Wait-ForService 'Vite frontend' $frontendTimeoutSeconds $frontendStarted { Test-ViteReady })) {
        throw "Frontend readiness check failed. Review $frontendStderr and $frontendStdout."
    }
    Write-StartupLog 'Vite frontend is ready.'
    if ($frontendStarted) {
        Start-Process -FilePath $frontendUrl | Out-Null
        Write-StartupLog 'Opened the Vite frontend in the default browser.'
    } else {
        Write-StartupLog 'Reused the existing Vite frontend; did not open another browser window.'
    }
} catch {
    try { Write-StartupLog "STARTUP FAILED: $($_.Exception.Message)" } catch { }
    exit 1
} finally {
    if ($mutex) {
        try { $mutex.ReleaseMutex() } catch { }
        $mutex.Dispose()
    }
}
