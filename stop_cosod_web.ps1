$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root 'web_backend.pid'
$port = 8765

$stopped = $false
if (Test-Path -LiteralPath $pidFile) {
    $pidText = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    $pidValue = 0
    if ([int]::TryParse($pidText, [ref]$pidValue)) {
        $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $pidValue -Force
            $stopped = $true
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
    foreach ($conn in $conns) {
        $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -like 'python*') {
            Stop-Process -Id $conn.OwningProcess -Force
            $stopped = $true
        }
    }
}

if ($stopped) {
    Write-Host "CoSOD backend stopped."
} else {
    Write-Host "No CoSOD backend process found."
}
