param(
    [switch]$KeepEnvFile,
    [switch]$KeepCerts
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$tempBootstrapDir = Join-Path $env:TEMP "assistant-v2-web-portable-bootstrap"

$targets = @(
    @{ Path = Join-Path $scriptDir ".venv"; Description = "virtual environment" },
    @{ Path = Join-Path $scriptDir ".cache"; Description = "local package/model caches" },
    @{ Path = Join-Path $scriptDir "output"; Description = "generated output" },
    @{ Path = $tempBootstrapDir; Description = "temporary Python installer download" }
)

if (-not $KeepEnvFile) {
    $targets += @{ Path = Join-Path $scriptDir ".env"; Description = "local env file" }
}

if (-not $KeepCerts) {
    $targets += @{ Path = Join-Path $scriptDir "certs\cert.pem"; Description = "generated HTTPS certificate" }
    $targets += @{ Path = Join-Path $scriptDir "certs\key.pem"; Description = "generated HTTPS private key" }
}

foreach ($target in $targets) {
    $path = $target.Path
    if (-not (Test-Path $path)) {
        continue
    }
    Remove-Item -LiteralPath $path -Recurse -Force
    Write-Host "[uninstall] Removed $($target.Description): $path"
}

$pycacheDirs = Get-ChildItem -Path $scriptDir -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue
foreach ($dir in $pycacheDirs) {
    Remove-Item -LiteralPath $dir.FullName -Recurse -Force
    Write-Host "[uninstall] Removed bytecode cache: $($dir.FullName)"
}

Write-Host ""
Write-Host "[uninstall] Complete."
Write-Host "[uninstall] Removed bundle-local runtime artifacts only."
Write-Host "[uninstall] Not removed:"
Write-Host "  - LM Studio"
Write-Host "  - any globally installed Python"
Write-Host "  - caches outside this bundle folder"
