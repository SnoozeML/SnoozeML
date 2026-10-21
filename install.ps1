# SnoozeML Zero-Admin Installer for Windows (PowerShell)
# Usage:
#   irm https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.ps1 | iex
#   $env:SNOOZE_VARIANT="cpu"; irm https://raw.githubusercontent.com/SnoozeML/SnoozeML/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo = if ($env:SNOOZE_REPO) { $env:SNOOZE_REPO } else { "SnoozeML/SnoozeML" }
$Version = if ($env:SNOOZE_VERSION) { $env:SNOOZE_VERSION } else { "latest" }
$Variant = if ($env:SNOOZE_VARIANT) { $env:SNOOZE_VARIANT.ToLower() } else { "standard" }
$InstallRoot = if ($env:SNOOZE_INSTALL_DIR) { $env:SNOOZE_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Programs\SnoozeML" }
$BinDir = Join-Path $InstallRoot "bin"
$ModelsDir = if ($env:SNOOZE_MODELS_DIR) { $env:SNOOZE_MODELS_DIR } else { Join-Path $InstallRoot "models" }
$ArtifactName = if ($Variant -eq "cpu") { "snooze-windows-x86_64-cpu" } else { "snooze-windows-x86_64" }

if ($Version -eq "latest") {
    $BaseUrl = "https://github.com/$Repo/releases/latest/download"
} else {
    $BaseUrl = "https://github.com/$Repo/releases/download/$Version"
}

$ZipUrl = "$BaseUrl/$ArtifactName.zip"
$ChecksumUrl = "$BaseUrl/$ArtifactName.zip.sha256"

$TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("snoozeml-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

try {
    Write-Host "==> Downloading SnoozeML ($ArtifactName, version: $Version)..." -ForegroundColor Cyan
    $ZipPath = Join-Path $TmpDir "$ArtifactName.zip"
    $ChecksumPath = Join-Path $TmpDir "$ArtifactName.zip.sha256"

    Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath -UseBasicParsing

    try {
        Invoke-WebRequest -Uri $ChecksumUrl -OutFile $ChecksumPath -UseBasicParsing
        $ExpectedHash = ((Get-Content $ChecksumPath -Raw).Trim() -split '\s+')[0].ToLower()
        $ActualHash = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash.ToLower()
        if ($ExpectedHash -and ($ExpectedHash -ne $ActualHash)) {
            throw "SHA-256 checksum mismatch! Expected $ExpectedHash, got $ActualHash"
        }
        Write-Host "[OK] Verified SHA-256 checksum ($ActualHash)" -ForegroundColor Green
    } catch {
        if ($_.Exception.Message -like "*checksum mismatch*") {
            throw $_
        }
        Write-Warning "Checksum file not found at release URL; proceeding with extraction."
    }

    Write-Host "==> Extracting to $InstallRoot (no Administrator privileges required)..." -ForegroundColor Cyan
    Expand-Archive -Path $ZipPath -DestinationPath $TmpDir -Force

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

    Copy-Item -Path (Join-Path $TmpDir "$ArtifactName\bin\*") -Destination $BinDir -Force

    # Add to User PATH if not already present
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $PathEntries = ($UserPath -split ';') | Where-Object { $_ -ne "" }
    if ($PathEntries -notcontains $BinDir) {
        $NewUserPath = ($PathEntries + $BinDir) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
        $env:Path = "$BinDir;$env:Path"
        Write-Host "[OK] Added $BinDir to User PATH" -ForegroundColor Green
    }

    $SnoozeExe = Join-Path $BinDir "snooze.exe"
    $ModelsToml = Join-Path $ModelsDir "models.toml"
    if (-not (Test-Path $ModelsToml)) {
        & $SnoozeExe init $ModelsDir | Out-Null
        Write-Host "[OK] Initialized models directory at $ModelsDir" -ForegroundColor Green
    }

    Write-Host "`nSnoozeML is installed! Quick commands to get started:" -ForegroundColor Green
    Write-Host "  snooze serve --models-dir `"$ModelsDir`""
    Write-Host "  snooze download BAAI/bge-small-en-v1.5 -o `"$ModelsDir`""
    Write-Host "  snooze monitor"
} finally {
    Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
}
