#Requires -Version 5.1
<#
.SYNOPSIS
  Run Bruker TIFF+.log reconstruction with the algotom-gpu conda env.
.PARAMETER ScanDir
  Folder containing projection TIFFs and a .log file.
.PARAMETER OutDir
  Optional output folder.
.PARAMETER Config
  Optional path to YAML config (default: config\default.yaml).
.PARAMETER Preview
  Fast single-slice mode for tuning ring removal (recommended first).
.PARAMETER PreviewRow
  Optional detector row index for -Preview (default: middle row).
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ScanDir,

    [Parameter(Mandatory = $false)]
    [string]$OutDir = "",

    [Parameter(Mandatory = $false)]
    [string]$Config = "",

    [Parameter(Mandatory = $false)]
    [switch]$Preview,

    [Parameter(Mandatory = $false)]
    [int]$PreviewRow = -1
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Get-CondaExe {
    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\Miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        return (Get-Command conda).Source
    }
    return $null
}

$condaExe = Get-CondaExe
if (-not $condaExe) {
    throw "conda not found. Run .\setup.ps1 first."
}

if (-not (Test-Path $ScanDir)) {
    throw "ScanDir not found: $ScanDir"
}

if (-not $Config) {
    $Config = Join-Path $Root "config\default.yaml"
}
if (-not (Test-Path $Config)) {
    throw "Config not found: $Config"
}

$script = Join-Path $Root "scripts\reconstruct_bruker.py"
$argList = @(
    "run", "-n", "algotom-gpu", "--no-capture-output",
    "python", $script,
    "--scan-dir", (Resolve-Path $ScanDir).Path,
    "--config", (Resolve-Path $Config).Path
)
if ($OutDir) {
    $argList += @("--out-dir", $OutDir)
}
if ($Preview) {
    $argList += "--preview"
    if ($PreviewRow -ge 0) {
        $argList += @("--preview-row", "$PreviewRow")
    }
}

if ($Preview) {
    Write-Host "Running PREVIEW (single slice)..." -ForegroundColor Cyan
} else {
    Write-Host "Running FULL reconstruction..." -ForegroundColor Cyan
}
Write-Host "  ScanDir: $ScanDir"
Write-Host "  Config:  $Config"
if ($OutDir) { Write-Host "  OutDir:  $OutDir" }
if ($Preview -and $PreviewRow -ge 0) { Write-Host "  PreviewRow: $PreviewRow" }

& $condaExe @argList
exit $LASTEXITCODE
