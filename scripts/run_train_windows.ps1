param (
    [string]$DataDir = "$PSScriptRoot/../data/subject_89335547",
    [int]$MaxEpochs = 40,
    [int]$NumWorkers = 1,
    [string]$Decoder = "ctc_greedy",
    [switch]$DryRun
)

# Enforce num_workers > 0
if ($NumWorkers -le 0) {
    Write-Host "Warning: NumWorkers must be > 0. Setting to 1."
    $NumWorkers = 1
}

$env:HYDRA_FULL_ERROR = "1"

# Convert Windows backslashes to forward slashes for Hydra
$DataDirForward = $DataDir.Replace("\", "/")

# Find existing run directories before launch
$LogsDir = Join-Path $PSScriptRoot "..\logs"
$BeforeRunDirs = @()
if (Test-Path $LogsDir) {
    $BeforeRunDirs = Get-ChildItem -Path "$LogsDir\*" -Directory -ErrorAction SilentlyContinue | Get-ChildItem -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
}

# The python command
$PyCmd = "import torch, runpy; _load = torch.load; torch.load = lambda *a, **k: _load(*a, **dict(k, weights_only=False)); runpy.run_module('emg2qwerty.train', run_name='__main__')"

$Args = @(
    "-c", $PyCmd,
    "user=single_user",
    "dataset.root=$DataDirForward",
    "trainer.max_epochs=$MaxEpochs",
    "num_workers=$NumWorkers",
    "train=True",
    "decoder=$Decoder",
    "trainer.accelerator=gpu",
    "trainer.devices=1"
)

if ($DryRun) {
    Write-Host "DRY RUN. Command to execute:"
    Write-Host "python $Args"
    exit
}

# Run it
& python $Args

# Wait a moment for file system to sync
Start-Sleep -Seconds 2

# Find the newly created run directory
$AfterRunDirs = @()
if (Test-Path $LogsDir) {
    $AfterRunDirs = Get-ChildItem -Path "$LogsDir\*" -Directory -ErrorAction SilentlyContinue | Get-ChildItem -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
}

$NewDirs = Compare-Object -ReferenceObject $BeforeRunDirs -DifferenceObject $AfterRunDirs | Where-Object { $_.SideIndicator -eq "=>" } | Select-Object -ExpandProperty InputObject

$LatestRunDir = $null
if ($NewDirs) {
    $LatestRunDir = $NewDirs | Sort-Object { (Get-Item $_).LastWriteTime } -Descending | Select-Object -First 1
} elseif ($AfterRunDirs) {
    $LatestRunDir = $AfterRunDirs | Sort-Object { (Get-Item $_).LastWriteTime } -Descending | Select-Object -First 1
}

if ($LatestRunDir) {
    Write-Host "`nRun completed. Latest run directory discovered:"
    Write-Host "RUN_DIR=$LatestRunDir"
    
    # Try to extract best checkpoint
    $CkptsDir = Join-Path $LatestRunDir "checkpoints"
    if (Test-Path $CkptsDir) {
        $Ckpts = Get-ChildItem -Path $CkptsDir -Filter "*.ckpt" | Sort-Object LastWriteTime -Descending
        if ($Ckpts) {
            Write-Host "Checkpoint: $($Ckpts[0].FullName)"
        }
    }
} else {
    Write-Host "`nCould not discover the new run directory under logs/."
}
