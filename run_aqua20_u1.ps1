$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
if (-not $root) {
    $root = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    throw "python not found on PATH. Please activate the scosparc environment before running this script."
}
$py = $pyCmd.Source
$checkpoint = 'baseline运行出的checkpoints\model_combo_base8-136_0.7291838924090067.pt'
$modelFolder = 'ours_aqua20_u1_k20'
$aquaRoot = Join-Path $root 'datasets\AQUA20_grouped'
$coCAPath = Join-Path $root 'datasets\CoCA'
$backupPath = Join-Path $root ('datasets\CoCA__bak_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$logDir = Join-Path $root ('aqua20_logs\' + (Get-Date -Format 'yyyy-MM-dd'))
$logPath = Join-Path $logDir ('job_' + $modelFolder + '.log')

if (-not (Test-Path -LiteralPath $aquaRoot)) {
    throw "AQUA20 dataset not found: $aquaRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path (Join-Path $root 'checkpoints') $checkpoint))) {
    throw "Checkpoint not found: checkpoints\$checkpoint"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$restored = $false
try {
    if (Test-Path -LiteralPath $coCAPath) {
        Move-Item -LiteralPath $coCAPath -Destination $backupPath
    }

    New-Item -ItemType Junction -Path $coCAPath -Target $aquaRoot | Out-Null

    Push-Location $root
    try {
        & $py test.py `
            --model_folder $modelFolder `
            --checkpoint_name $checkpoint `
            --datasets CoCA `
            --size 224 `
            --test_num_workers 0 `
            --max_group_images 20 `
            --stage2_proto acre `
            --topk_mode rtg `
            --topk_ratio 0.04 `
            --topk_res_alpha 0.10 `
            --topk_conf_gate 0.58 `
            --topk_mass_min 0.45 `
            --topk_delta_th 0.045 `
            --rpf_rounds 2 `
            --rpf_soft_lambda 0 `
            --tau2_mode fixed `
            --tau2_delta 0.005 `
            --baseline_legacy 0 `
            --tau1_sim 0.76 2>&1 | Tee-Object -FilePath $logPath
    }
    finally {
        Pop-Location
    }
}
finally {
    if (Test-Path -LiteralPath $coCAPath) {
        $coCAItem = Get-Item -LiteralPath $coCAPath -Force
        if (($coCAItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            [System.IO.Directory]::Delete($coCAPath)
        } else {
            throw "Refusing to delete non-junction path: $coCAPath"
        }
    }
    if (Test-Path -LiteralPath $backupPath) {
        Move-Item -LiteralPath $backupPath -Destination $coCAPath
    }
    $restored = $true
}

if ($restored) {
    Write-Host "LOG=$logPath"
}
