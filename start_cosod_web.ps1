$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$hostName = '127.0.0.1'
$port = 8765
$url = "http://${hostName}:$port/code.html"
$healthUrl = "http://${hostName}:$port/api/health"
$python = 'D:\Anaconda\Anaconda\envs\scosparc\python.exe'
$pidFile = Join-Path $root 'web_backend.pid'
$runLogDir = Join-Path $root 'web_logs'
$stdoutLog = Join-Path $runLogDir 'backend_server_stdout.log'
$stderrLog = Join-Path $runLogDir 'backend_server_stderr.log'

New-Item -ItemType Directory -Force -Path $runLogDir | Out-Null

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python not found: $python"
}

function Test-BackendHealth {
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3
        return ($resp.StatusCode -eq 200 -and $resp.Content -match '"ok"\s*:\s*true')
    } catch {
        return $false
    }
}

if (-not (Test-BackendHealth)) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' } | Select-Object -First 1
    if ($conn) {
        throw "Port $port is already in use by process $($conn.OwningProcess), but /api/health is not responding."
    }

    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList @('cosod_backend.py', '--host', $hostName, '--port', [string]$port) `
        -WorkingDirectory $root `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru

    Set-Content -LiteralPath $pidFile -Value $proc.Id -Encoding ASCII

    $ready = $false
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-BackendHealth) {
            $ready = $true
            break
        }
    }

    if (-not $ready) {
        throw "Backend did not become ready. See logs: $stdoutLog ; $stderrLog"
    }
}

Start-Process $url
Write-Host "CoSOD web app is ready: $url"
