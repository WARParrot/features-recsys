[CmdletBinding()]
param(
    [string]$Checkpoint = "",
    [string]$DataDir = "",
    [string]$Output = "",
    [switch]$Inspect
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResumeScript = Join-Path $PSScriptRoot "resume_kion_v7.py"

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Join-Path $ProjectRoot "results\kion_10k_v7_corrected.json"
}
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $ProjectRoot "datasets\kion"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ProjectRoot "results\kion_10k_v7_resumed.json"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw "Python was not found. Install Python 3.11 and create .venv, or put python.exe on PATH."
    }
    $Python = $PythonCommand.Source
}

foreach ($RequiredPath in @($ResumeScript, $Checkpoint, $DataDir)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path does not exist: $RequiredPath"
    }
}

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$Arguments = @(
    $ResumeScript,
    "--checkpoint", (Resolve-Path -LiteralPath $Checkpoint).Path,
    "--data-dir", (Resolve-Path -LiteralPath $DataDir).Path,
    "--output", ([IO.Path]::GetFullPath($Output))
)
if ($Inspect) {
    $Arguments += "--inspect"
}

Write-Host "Python:     $Python"
Write-Host "Checkpoint: $Checkpoint"
Write-Host "Data:       $DataDir"
Write-Host "Output:     $Output"
if ($Inspect) {
    Write-Host "Mode:       inspect only"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
