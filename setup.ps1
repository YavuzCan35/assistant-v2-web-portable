param(
    [switch]$SkipModelPreload,
    [switch]$ForceCertRegenerate
)

$ErrorActionPreference = "Stop"
$MinimumPythonVersion = [Version]"3.10.0"
$MaximumPythonVersionExclusive = [Version]"3.13.0"
$PreferredPythonVersion = "3.12.10"
$PythonRequirementText = ">= 3.10 and < 3.13"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$envFile = Join-Path $scriptDir ".env"
$envExampleFile = Join-Path $scriptDir ".env.example"
$requirements = Join-Path $scriptDir "requirements.txt"
$setupHelper = Join-Path $scriptDir "setup_runtime.py"

function Get-PythonCandidates {
    $candidates = New-Object System.Collections.Generic.List[object]
    $seen = New-Object System.Collections.Generic.HashSet[string]

    function Add-Candidate([string[]]$command) {
        if (-not $command -or $command.Length -eq 0) { return }
        $key = $command -join " "
        if ($seen.Add($key)) {
            [void]$candidates.Add($command)
        }
    }

    $localPythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path $localPythonRoot) {
        Get-ChildItem -Path $localPythonRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object {
                $pythonExe = Join-Path $_.FullName "python.exe"
                if (Test-Path $pythonExe) {
                    Add-Candidate @($pythonExe)
                }
            }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        Add-Candidate @($pyCmd.Source, "-3.12")
        Add-Candidate @($pyCmd.Source, "-3.11")
        Add-Candidate @($pyCmd.Source, "-3.10")
        Add-Candidate @($pyCmd.Source, "-3")
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) { Add-Candidate @($pythonCmd.Source) }

    $condaPython = Join-Path $env:USERPROFILE "anaconda3\python.exe"
    if (Test-Path $condaPython) { Add-Candidate @($condaPython) }

    return $candidates
}

function Resolve-BootstrapPython {
    $compatibleCandidates = New-Object System.Collections.Generic.List[object]

    foreach ($candidate in (Get-PythonCandidates)) {
        $version = Get-PythonVersion $candidate
        if ($null -eq $version) {
            continue
        }
        if (Test-CompatiblePythonVersion $version) {
            Write-Host "[setup] Found compatible bootstrap Python $version via $($candidate -join ' ')"
            [void]$compatibleCandidates.Add($candidate)
            continue
        }
        Write-Host "[setup] Skipping incompatible Python $version via $($candidate -join ' ')"
    }

    if ($compatibleCandidates.Count -gt 0) {
        return $compatibleCandidates
    }

    Install-CompatiblePython

    $maxWaitSeconds = 45
    $deadline = (Get-Date).AddSeconds($maxWaitSeconds)
    do {
        $compatibleCandidates = New-Object System.Collections.Generic.List[object]
        foreach ($candidate in (Get-PythonCandidates)) {
            $version = Get-PythonVersion $candidate
            if ($null -ne $version -and (Test-CompatiblePythonVersion $version)) {
                Write-Host "[setup] Found installed Python $version via $($candidate -join ' ')"
                [void]$compatibleCandidates.Add($candidate)
            }
        }
        if ($compatibleCandidates.Count -gt 0) {
            return $compatibleCandidates
        }
        Start-Sleep -Seconds 2
    }
    while ((Get-Date) -lt $deadline)

    throw "Compatible Python not found after install. Required version range is $PythonRequirementText."
}

function Get-PythonVersion([string[]]$command) {
    try {
        $prefixArgs = @()
        if ($command.Length -gt 1) {
            $prefixArgs = $command[1..($command.Length - 1)]
        }
        $output = & $command[0] @prefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $output) {
            return $null
        }
        return [Version]($output | Select-Object -First 1)
    } catch {
        return $null
    }
}

function Test-CompatiblePythonVersion([Version]$version) {
    if ($null -eq $version) {
        return $false
    }
    return ($version -ge $MinimumPythonVersion) -and ($version -lt $MaximumPythonVersionExclusive)
}

function Install-CompatiblePython {
    $arch = "amd64"
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64" -or $env:PROCESSOR_ARCHITEW6432 -eq "ARM64") {
        $arch = "arm64"
    }
    if ($arch -notin @("amd64", "arm64")) {
        throw "Unsupported Windows architecture for automatic Python install: $arch"
    }

    $installerFile = "python-$PreferredPythonVersion-$arch.exe"
    $installerUrl = "https://www.python.org/ftp/python/$PreferredPythonVersion/$installerFile"
    $downloadDir = Join-Path $env:TEMP "assistant-v2-web-portable-bootstrap"
    $installerPath = Join-Path $downloadDir $installerFile

    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    Write-Host "[setup] No compatible Python found. Downloading Python $PreferredPythonVersion from python.org"
    try {
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath
    } catch {
        throw "Failed to download Python $PreferredPythonVersion from $installerUrl"
    }

    $installerArgs = @(
        "/quiet",
        "InstallAllUsers=0",
        "Include_launcher=1",
        "InstallLauncherAllUsers=0",
        "Include_pip=1",
        "Include_test=0",
        "AssociateFiles=0",
        "Shortcuts=0",
        "PrependPath=1",
        "SimpleInstall=1"
    )

    Write-Host "[setup] Running Python installer"
    $process = Start-Process -FilePath $installerPath -ArgumentList $installerArgs -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }
}

function New-ProjectVenv([string[]]$command) {
    $prefixArgs = @()
    if ($command.Length -gt 1) {
        $prefixArgs = $command[1..($command.Length - 1)]
    }

    if (Test-Path $venvDir) {
        Remove-Item -Recurse -Force $venvDir
    }

    Write-Host "[setup] Creating virtual environment at $venvDir using $($command -join ' ')"
    & $command[0] @prefixArgs -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] Venv creation failed via $($command -join ' ')"
        return $false
    }

    if (-not (Test-Path $venvPython)) {
        Write-Host "[setup] Venv creation finished but $venvPython is missing"
        return $false
    }

    $venvVersion = Get-PythonVersion @($venvPython)
    if (-not (Test-CompatiblePythonVersion $venvVersion)) {
        Write-Host "[setup] Created venv uses incompatible Python version $venvVersion"
        return $false
    }

    Write-Host "[setup] Created venv with Python $venvVersion"
    return $true
}

if (-not (Test-Path $envFile)) {
    if (-not (Test-Path $envExampleFile)) {
        throw ".env.example not found."
    }
    Copy-Item -LiteralPath $envExampleFile -Destination $envFile -Force
    Write-Host "[setup] Created local .env from .env.example"
}

if (Test-Path $venvPython) {
    $existingVenvVersion = Get-PythonVersion @($venvPython)
    if (-not (Test-CompatiblePythonVersion $existingVenvVersion)) {
        Write-Host "[setup] Existing .venv uses incompatible Python version $existingVenvVersion. Recreating it."
        Remove-Item -Recurse -Force $venvDir
    }
}

if (-not (Test-Path $venvPython)) {
    $bootstrapCandidates = Resolve-BootstrapPython
    $venvCreated = $false
    foreach ($bootstrap in $bootstrapCandidates) {
        if (New-ProjectVenv $bootstrap) {
            $venvCreated = $true
            break
        }
    }
    if (-not $venvCreated) {
        throw "Unable to create a working virtual environment with any compatible Python interpreter."
    }
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
Write-Host "[setup] Python requirement: $PythonRequirementText"
Write-Host "[setup] Phone/browser note:"
Write-Host "  1. Open the HTTPS URL printed by start.ps1."
Write-Host "  2. If microphone permission is blocked, trust the generated local certificate on that device."
Write-Host "  3. On Android, installing/trusting the cert is usually enough."
Write-Host "  4. On iPhone/iPad, you may need to install the cert and enable full trust in Settings."
Write-Host "  5. The cert files are local only and are not meant to be committed to Git."
