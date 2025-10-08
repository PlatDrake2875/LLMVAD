# Kaggle dataset downloader (aria2 with live progress; WebRequest fallback with Write-Progress)
# - Loads KAGGLE_USERNAME/KAGGLE_KEY from .env file in project root
# - Uses aria2c if available for faster multi-connection downloads
# - Falls back to PowerShell WebRequest with progress bar if aria2c not found
# - Extracts dataset to datasets/<DatasetName>/ folder

[CmdletBinding()]
param(
  [switch]$Force,
  [string]$DatasetSlug = 'adrianpatrascu/xd-violence-1-1004',
  [string]$DatasetName = 'XD_Violence_1-1004',
  [string]$ZipName = 'xd-violence.zip'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'Continue'
try { [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12 } catch {}

# --- Paths ---
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectRoot = Split-Path -Parent $ScriptDir
$DatasetsDir = Join-Path $ProjectRoot 'datasets'
$DatasetPath = Join-Path $DatasetsDir $DatasetName
$ZipPath     = Join-Path $DatasetsDir $ZipName
$DotEnvPath  = Join-Path $ProjectRoot '.env'

# Kaggle dataset
$DatasetUrl  = "https://www.kaggle.com/api/v1/datasets/download/$DatasetSlug"

Write-Host "Kaggle Dataset Downloader (aria2 + progress)"

# --- .env loader ---
function Import-DotEnv {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return }
  $lines = [System.IO.File]::ReadAllLines($Path)
  foreach ($raw in $lines) {
    $line = $raw
    if ($line -match '^\s*(#|;|$)') { continue }
    $line = $line -replace '^\s*export\s+', ''
    $m = [regex]::Match($line, '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$')
    if (-not $m.Success) { continue }

    $key = $m.Groups[1].Value
    $val = $m.Groups[2].Value.Trim()

    $isQuoted = ($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))
    if (-not $isQuoted) { $val = ($val -replace '\s+#.*$', '').Trim() }
    if ($val.StartsWith('"') -and $val.EndsWith('"')) {
      $val = $val.Substring(1, $val.Length - 2)
      $val = $val.Replace('\"','"').Replace('\n',[Environment]::NewLine).Replace('\r','')
    } elseif ($val.StartsWith("'") -and $val.EndsWith("'")) {
      $val = $val.Substring(1, $val.Length - 2)
    }
    Set-Item -Path ("Env:{0}" -f $key) -Value $val
  }
}

Import-DotEnv -Path $DotEnvPath

# --- Require env creds ---
function Get-KaggleCredentials {
  if ($env:KAGGLE_USERNAME -and $env:KAGGLE_KEY) { return @($env:KAGGLE_USERNAME, $env:KAGGLE_KEY) }
  throw "KAGGLE_USERNAME / KAGGLE_KEY not found in environment (.env)."
}

# --- Skip if already present ---
if ((Test-Path $DatasetPath) -and -not $Force) {
  $count = (Get-ChildItem $DatasetPath -Force -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($count -gt 0) {
    Write-Host "Dataset already exists with $count items. Use -Force to re-download."
    exit 0
  }
}

# --- Ensure folder & cleanup ---
if (-not (Test-Path $DatasetsDir)) { New-Item -ItemType Directory -Path $DatasetsDir -Force | Out-Null }
if (Test-Path $ZipPath)     { Remove-Item $ZipPath -Force }
if (Test-Path $DatasetPath) { Remove-Item $DatasetPath -Recurse -Force }

$username, $apiKey = Get-KaggleCredentials
$basic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("$username`:$apiKey"))
$authHeader = "Authorization: Basic $basic"

# --- Try aria2c with live progress (direct invocation) ---
$aria = Get-Command aria2c -ErrorAction SilentlyContinue
if ($aria) {
  Write-Host "Downloading via Kaggle API (aria2c)..."
  $dir  = Split-Path -Parent $ZipPath
  $name = Split-Path -Leaf $ZipPath

  # direct invocation -> progress shows in the same console
  $args = @(
    "--header=$authHeader",
    "--header=Accept: application/zip",
    "--max-connection-per-server=16",
    "--split=16",
    "--min-split-size=1M",
    "--continue=true",
    "--allow-overwrite=true",
    "--auto-file-renaming=false",
    "--file-allocation=none",
    "--summary-interval=0",
    "--console-log-level=warn",
    "--show-console-readout=true",
    "--enable-color=true",
    "--dir=$dir",
    "--out=$name",
    $DatasetUrl
  )
  & $aria.Path @args
  if ($LASTEXITCODE -ne 0) { throw "aria2c failed with exit code $LASTEXITCODE." }
}
else {
  # --- Fallback: manual streaming download with HttpWebRequest + Write-Progress ---
  Write-Host "aria2c not found. Falling back to streaming download with progress..."
  $req = [System.Net.HttpWebRequest]::Create($DatasetUrl)
  $req.AllowAutoRedirect = $true
  $req.Headers['Authorization'] = "Basic $basic"
  $req.Accept = 'application/zip'
  $resp = $req.GetResponse()

  $totalBytes = $resp.ContentLength
  $inStream   = $resp.GetResponseStream()
  $outStream  = [System.IO.File]::Open($ZipPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
  $buffer     = New-Object byte[] (1MB)
  $readTotal  = 0L
  $sw         = [System.Diagnostics.Stopwatch]::StartNew()

  try {
    while (($read = $inStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
      $outStream.Write($buffer, 0, $read)
      $readTotal += $read

      $pct = if ($totalBytes -gt 0) { [int](($readTotal * 100) / $totalBytes) } else { 0 }
      $mb  = [math]::Round($readTotal / 1MB, 1)
      $spd = if ($sw.Elapsed.TotalSeconds -gt 0) { [math]::Round(($readTotal/1MB)/$sw.Elapsed.TotalSeconds, 2) } else { 0 }
      $eta = if ($totalBytes -gt 0 -and $spd -gt 0) {
        [TimeSpan]::FromSeconds( ((($totalBytes - $readTotal)/1MB) / $spd) )
      } else { [TimeSpan]::Zero }

      $status = "Downloaded: $mb MB | Speed: $spd MB/s"
      if ($eta -ne [TimeSpan]::Zero) { $status += " | ETA: " + $eta.ToString('mm\:ss') }
      Write-Progress -Activity "Downloading Kaggle Dataset" -Status $status -PercentComplete $pct
    }
  } finally {
    $outStream.Dispose(); $inStream.Dispose(); $resp.Close()
    Write-Progress -Activity "Downloading Kaggle Dataset" -Completed
  }
}

if (-not (Test-Path $ZipPath)) { throw "Download failed - file not created." }

# --- Sanity: ZIP signature "PK" ---
$fs = [System.IO.File]::OpenRead($ZipPath)
try {
  $sig = New-Object byte[] 2
  [void]$fs.Read($sig,0,2)
  $isZip = ($sig[0] -eq 0x50 -and $sig[1] -eq 0x4B)
} finally { $fs.Dispose() }
if (-not $isZip) {
  $head = Get-Content $ZipPath -TotalCount 200 -Raw
  throw "Expected ZIP but got something else. First bytes/text:`n$([string]$head)"
}

# --- Extract ---
Write-Host "Extracting..."
if (-not (Test-Path $DatasetPath)) { New-Item -ItemType Directory -Path $DatasetPath | Out-Null }
Expand-Archive -LiteralPath $ZipPath -DestinationPath $DatasetPath -Force
Remove-Item $ZipPath -Force

$extracted = Get-ChildItem $DatasetPath -Force -ErrorAction SilentlyContinue
if ($extracted.Count -gt 0) {
  Write-Host "Extraction completed."
  Write-Host "Dataset ready at: $DatasetPath"
  Write-Host "Items: $($extracted.Count)"
} else {
  throw "Extraction failed - no files found."
}
