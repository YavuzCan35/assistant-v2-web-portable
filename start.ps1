param(
    [string]$ListenHost = "0.0.0.0",
    [int]$Port = 8765,
    [switch]$NoTts
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
$setupHelper = Join-Path $scriptDir "setup_runtime.py"
$appScript = Join-Path $scriptDir "assistant_v2_web.py"
$certPath = Join-Path $scriptDir "certs\cert.pem"
$keyPath = Join-Path $scriptDir "certs\key.pem"
$cacheRoot = Join-Path $scriptDir ".cache"

function Set-ProjectRuntimeEnvironment {
    $hfHome = Join-Path $cacheRoot "huggingface"
    $hfHubCache = Join-Path $hfHome "hub"
    $torchHome = Join-Path $cacheRoot "torch"
    $pipCacheDir = Join-Path $cacheRoot "pip"

    foreach ($path in @($cacheRoot, $hfHome, $hfHubCache, $torchHome, $pipCacheDir)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }

    $env:XDG_CACHE_HOME = $cacheRoot
    $env:HF_HOME = $hfHome
    $env:HF_HUB_CACHE = $hfHubCache
    $env:HUGGINGFACE_HUB_CACHE = $hfHubCache
    $env:TRANSFORMERS_CACHE = $hfHubCache
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    $env:TORCH_HOME = $torchHome
    $env:PIP_CACHE_DIR = $pipCacheDir
}

Set-ProjectRuntimeEnvironment

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

if (-not ((Test-Path $certPath) -and (Test-Path $keyPath))) {
    Write-Host "[start] Local HTTPS cert files are missing. Generating them now."
    & $venvPython $setupHelper --generate-certs
    if ($LASTEXITCODE -ne 0) { throw "certificate generation failed." }
}

Write-Host "[start] Make sure LM Studio is running and the model in .env is loaded."
Write-Host "[start] Local cache root: $cacheRoot"
Write-Host "[start] Launching web server..."
Write-Host "[start] Keep this terminal open while using the assistant."

$args = @($appScript, "--host", $ListenHost, "--port", "$Port")
if ($NoTts) {
    $args += "--no-tts"
}

& $venvPython @args
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "[start] Server exited with code $exitCode."
    Write-Host "[start] If no 'Web UI:' / 'LAN UI:' lines were printed, startup did not complete."
}
exit $exitCode
