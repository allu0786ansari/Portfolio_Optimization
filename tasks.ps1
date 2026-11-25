<#
.SYNOPSIS
  Windows task runner — the PowerShell equivalent of the Makefile.

.DESCRIPTION
  Run from the repository root in a VS Code terminal (PowerShell):
      .\tasks.ps1 help
      .\tasks.ps1 install
      .\tasks.ps1 check          # lint + typecheck + leakage gate + tests
  Or use VS Code: Terminal > Run Task... (the tasks call this script).

  If PowerShell blocks the script ("running scripts is disabled"), run once:
      Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  or call it as:  powershell -ExecutionPolicy Bypass -File .\tasks.ps1 <task>

  Works with Windows PowerShell 5.1 and PowerShell 7+.
#>
param(
    [Parameter(Position = 0)] [string] $Task = "help",
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)] [string[]] $Rest
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = "1"                  # UTF-8 everywhere (Windows default is cp1252)
$env:PYTHONIOENCODING = "utf-8"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$Extras = @("--extra", "dev", "--extra", "rl", "--extra", "ml", "--extra", "serving", "--extra", "data")

function Invoke-Step([string] $Exe, [string[]] $Arguments) {
    Write-Host ">> $Exe $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Uv([string[]] $Arguments) { Invoke-Step "uv" $Arguments }
function UvRun([string[]] $Arguments) { Invoke-Step "uv" (@("run") + $Arguments) }

function Assert-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "uv is not installed. Install it with:" -ForegroundColor Yellow
        Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
        Write-Host "then open a NEW terminal and re-run this task."
        exit 1
    }
}

$Tasks = [ordered]@{
    "help"            = "list tasks"
    "setup"           = "check uv, create .env from .env.example if missing"
    "install"         = "create .venv from uv.lock (all extras) + pre-commit hooks"
    "lint"            = "ruff"
    "typecheck"       = "mypy on src/pfo"
    "test"            = "full test suite with coverage (synthetic data, no downloads)"
    "leakage"         = "Phase 0 exit gate on synthetic data"
    "check"           = "lint + typecheck + leakage + test (what CI runs)"
    "leakage-real"    = "Phase 0 exit gate on downloaded data (run 'data' first)"
    "phase0-smoke"    = "corrected-baseline pipeline on synthetic data (~3 min)"
    "phase0-baseline" = "corrected v1 baseline on real data (run 'data' first)"
    "data"            = "download prices + benchmarks, build v1 + macro features"
    "train"           = "train PPO + SAC (needs MLflow at MLFLOW_TRACKING_URI)"
    "serve"           = "run the API on http://localhost:8000"
    "dashboard"       = "run the Streamlit dashboard on http://localhost:8501"
    "mlflow"          = "start only the MLflow server container (Docker Desktop)"
    "up"              = "docker compose up --build (Docker Desktop)"
    "down"            = "docker compose down"
    "logs"            = "docker compose logs -f"
    "clean"           = "docker compose down -v"
}

if (-not $Tasks.Contains($Task)) {
    Write-Host "Unknown task '$Task'." -ForegroundColor Red
    $Task = "help"
}
if ($Task -notin @("help", "up", "down", "logs", "clean", "mlflow")) { Assert-Uv }

switch ($Task) {
    "help" {
        Write-Host "Usage: .\tasks.ps1 <task> [extra args]`n"
        foreach ($k in $Tasks.Keys) { Write-Host ("  {0,-16} {1}" -f $k, $Tasks[$k]) }
    }
    "setup" {
        if (-not (Test-Path ".env")) {
            Copy-Item ".env.example" ".env"
            Write-Host "Created .env from .env.example - edit it and set API_KEY." -ForegroundColor Yellow
        } else { Write-Host ".env already exists." }
        Invoke-Step "uv" @("--version")
    }
    "install" {
        Uv (@("sync") + $Extras)
        UvRun @("pre-commit", "install")
    }
    "lint"      { UvRun @("ruff", "check", ".") }
    "typecheck" { UvRun @("mypy") }
    "test"      { UvRun (@("pytest", "tests", "--cov") + $Rest) }
    "leakage" {
        UvRun @("pytest", "tests/leakage", "-m", "leakage", "-v")
        UvRun @("pfo-leakage-scan", "--source", "synthetic")
    }
    "check" {
        UvRun @("ruff", "check", ".")
        UvRun @("mypy")
        UvRun @("pytest", "tests/leakage", "-m", "leakage", "-v")
        UvRun @("pytest", "tests", "--cov")
    }
    "leakage-real" {
        UvRun @("pfo-leakage-scan", "--source", "v1", "--universe", "both", "--out", "reports/phase0/leakage_scan_real.json")
    }
    "phase0-smoke" {
        UvRun (@("pfo-v1-baseline", "--source", "synthetic", "--rl-timesteps", "20000", "--out", "reports/phase0/synthetic") + $Rest)
    }
    "phase0-baseline" {
        UvRun (@("pfo-v1-baseline", "--source", "v1", "--universe", "both", "--out", "reports/phase0/real") + $Rest)
    }
    "data" {
        UvRun @("python", "-m", "data.ingestion.fetch_data")
        UvRun @("python", "-m", "data.ingestion.feature_engineering")
        UvRun @("python", "-m", "data.ingestion.macro_features")
    }
    "train" {
        UvRun @("python", "-m", "models.rl_agent.train_agent", "--algo", "ppo")
        UvRun @("python", "-m", "models.rl_agent.train_agent", "--algo", "sac")
    }
    "serve"     { UvRun @("uvicorn", "serving.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload") }
    "dashboard" { UvRun @("streamlit", "run", "dashboard/app.py") }
    "mlflow"    { Invoke-Step "docker" @("compose", "up", "-d", "--build", "mlflow") }
    "up"        { Invoke-Step "docker" @("compose", "up", "--build") }
    "down"      { Invoke-Step "docker" @("compose", "down") }
    "logs"      { Invoke-Step "docker" @("compose", "logs", "-f") }
    "clean"     { Invoke-Step "docker" @("compose", "down", "-v") }
}
