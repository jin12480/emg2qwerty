param (
    [Parameter(Mandatory=$true)]
    [string]$CkptPath,
    [string]$DataDir = "$PSScriptRoot/../data/subject_89335547",
    [int]$NumWorkers = 1,
    [string]$Decoder = "ctc_greedy"
)

# Enforce num_workers > 0
if ($NumWorkers -le 0) {
    Write-Host "Warning: NumWorkers must be > 0. Setting to 1."
    $NumWorkers = 1
}

$env:HYDRA_FULL_ERROR = "1"

# Convert Windows backslashes to forward slashes for Hydra
$DataDirForward = $DataDir.Replace("\", "/")
$CkptPathForward = $CkptPath.Replace("\", "/")

if (-not (Test-Path $CkptPath)) {
    Write-Host "Error: Checkpoint path does not exist ($CkptPath)"
    exit 1
}

# The python command
$PyCmd = "import torch, runpy; _load = torch.load; torch.load = lambda *a, **k: _load(*a, **dict(k, weights_only=False)); runpy.run_module('emg2qwerty.train', run_name='__main__')"

$Args = @(
    "-c", $PyCmd,
    "user=single_user",
    "dataset.root=$DataDirForward",
    "num_workers=$NumWorkers",
    "train=False",
    "decoder=$Decoder",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "ckpt_path=$CkptPathForward"
)

Write-Host "Evaluating checkpoint: $CkptPath"

# Run it and capture output
$Output = & python $Args *>&1 | Tee-Object -Variable "CapturedOutput"

$TestCerLine = $CapturedOutput | Where-Object { $_ -match "test/cer" -or $_ -match "test_cer" }
Write-Host "`n=== Evaluation complete ==="
if ($TestCerLine) {
    Write-Host "Found Test CER metric:"
    Write-Host $TestCerLine
} else {
    Write-Host "Test CER metric not found in output."
}

# Write results/evals json
$EvalsDir = Join-Path $PSScriptRoot "..\results\evals"
if (-not (Test-Path $EvalsDir)) {
    New-Item -ItemType Directory -Force $EvalsDir | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$EvalJsonPath = Join-Path $EvalsDir "${Timestamp}_eval.json"

$Result = @{
    timestamp = (Get-Date -Format "o")
    ckpt_path = $CkptPath
    data_dir = $DataDir
    decoder = $Decoder
    test_cer = $null
}

# simplistic parse to find the number, if possible
foreach ($line in $CapturedOutput) {
    if ($line -match "(?:test/cer|test_cer).*?([0-9]+\.[0-9]+)") {
        $Result.test_cer = [float]$matches[1]
    }
}

$Result | ConvertTo-Json | Set-Content $EvalJsonPath
Write-Host "Evaluation details saved to $EvalJsonPath"
