param(
    [switch]$SkipModelPreload,
    [switch]$ForceCertRegenerate
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$envFile = Join-Path $scriptDir ".env"
$envExampleFile = Join-Path $scriptDir ".env.example"
$requirements = Join-Path $scriptDir "requirements.txt"
$setupHelper = Join-Path $scriptDir "setup_runtime.py"

function Resolve-BootstrapPython {
    $condaPython = Join-Path $env:USERPROFILE "anaconda3\python.exe"
    if (Test-Path $condaPython) { return @($condaPython) }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) { return @($pyCmd.Source, "-3") }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) { return @($pythonCmd.Source) }

    throw "Python executable not found. Install Python 3.11+ first."
}

function Invoke-PythonCommand([string[]]$command, [string[]]$args) {
    $prefixArgs = @()
    if ($command.Length -gt 1) {
        $prefixArgs = $command[1..($command.Length - 1)]
    }
    & $command[0] @prefixArgs @args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $($command -join ' ') $($args -join ' ')"
    }
}

if (-not (Test-Path $envFile)) {
    if (-not (Test-Path $envExampleFile)) {
        throw ".env.example not found."
    }
    Copy-Item -LiteralPath $envExampleFile -Destination $envFile -Force
    Write-Host "[setup] Created local .env from .env.example"
}

if (-not (Test-Path $venvPython)) {
    $bootstrap = Resolve-BootstrapPython
    Write-Host "[setup] Creating virtual environment at $venvDir"
    Invoke-PythonCommand $bootstrap @("-m", "venv", $venvDir)
}

Write-Host "[setup] Using venv python: $venvPython"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

Write-Host "[setup] Installing Python packages"
& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "requirements installation failed." }

$certArgs = @($setupHelper, "--generate-certs")
if ($ForceCertRegenerate) {
    $certArgs += "--force-cert"
}
Write-Host "[setup] Ensuring local HTTPS certificate exists"
& $venvPython @certArgs
if ($LASTEXITCODE -ne 0) { throw "certificate generation failed." }

if (-not $SkipModelPreload) {
    Write-Host "[setup] Preloading Whisper and Kokoro assets"
    & $venvPython $setupHelper --preload-models
    if ($LASTEXITCODE -ne 0) { throw "model preload failed." }
}

Write-Host ""
Write-Host "[setup] Complete."
Write-Host "[setup] Next run command: .\start.ps1"
Write-Host "[setup] Phone/browser note:"
Write-Host "  1. Open the HTTPS URL printed by start.ps1."
Write-Host "  2. If microphone permission is blocked, trust the generated local certificate on that device."
Write-Host "  3. On Android, installing/trusting the cert is usually enough."
Write-Host "  4. On iPhone/iPad, you may need to install the cert and enable full trust in Settings."
Write-Host "  5. The cert files are local only and are not meant to be committed to Git."
