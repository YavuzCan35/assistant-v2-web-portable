param(
    [string]$Host = "0.0.0.0",
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

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}

if (-not ((Test-Path $certPath) -and (Test-Path $keyPath))) {
    Write-Host "[start] Local HTTPS cert files are missing. Generating them now."
    & $venvPython $setupHelper --generate-certs
    if ($LASTEXITCODE -ne 0) { throw "certificate generation failed." }
}

Write-Host "[start] Make sure LM Studio is running and the model in .env is loaded."

$args = @($appScript, "--host", $Host, "--port", "$Port")
if ($NoTts) {
    $args += "--no-tts"
}

& $venvPython @args
exit $LASTEXITCODE
