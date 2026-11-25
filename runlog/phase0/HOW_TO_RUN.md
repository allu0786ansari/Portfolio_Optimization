# Phase 0 — How to run

Phase 0 delivers the leakage-safe core (`src/pfo`), fixes the v1 defects and
produces the **corrected v1 baseline**. Everything except the real-data steps
runs offline on synthetic data.

**Primary setup: Windows 10/11 + VS Code + PowerShell.** Every step has a VS Code
way (Command Palette / Run Task / Testing / Run and Debug) and a terminal way
(`.\tasks.ps1 <task>`). Linux/macOS users run the same tasks with `make <task>`.
CI runs the Windows path on a real Windows runner on every push, so these
instructions are tested, not just written.

---

## 0. One-time prerequisites (Windows)

| Tool | Install | Check (new PowerShell window) |
|---|---|---|
| Git for Windows | https://git-scm.com/download/win | `git --version` |
| VS Code | https://code.visualstudio.com | — |
| Python 3.12 (64-bit) | https://www.python.org/downloads/ — tick **"Add python.exe to PATH"** | `py -3.12 --version` |
| uv | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `uv --version` (≥ 0.11) |
| Docker Desktop *(only for MLflow server / full stack)* | https://www.docker.com/products/docker-desktop — use the **WSL 2 backend** | `docker version` |

Allow local scripts once (lets `tasks.ps1` run):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

(The VS Code tasks pass `-ExecutionPolicy Bypass` themselves, so this is only needed
for typing `.\tasks.ps1` in a terminal.)

## 1. Get the code and open it in VS Code

```powershell
git clone https://github.com/allu0786ansari/Portfolio_Optimization.git
cd Portfolio_Optimization
git checkout phase0
code .
```

VS Code will offer the **recommended extensions** (Python, Pylance, Ruff, Mypy, Docker, …) —
click *Install All*. The repository ships `.vscode/settings.json`, `tasks.json` and
`launch.json`, and `.gitattributes` keeps line endings correct on Windows.

## 2. Install (≈3–6 min the first time)

| VS Code | Terminal (PowerShell) | Linux/macOS |
|---|---|---|
| *Terminal → Run Task… → "Install: create .venv from uv.lock"* | `.\tasks.ps1 setup` then `.\tasks.ps1 install` | `cp .env.example .env && make install` |

`setup` creates `.env` from `.env.example`. **Open `.env` and set `API_KEY`** to any
long random string. Then select the interpreter: *Ctrl+Shift+P → "Python: Select
Interpreter" → `.venv\Scripts\python.exe`*.

On Windows, PyPI's PyTorch wheel is CPU-only, so nothing extra is needed.

## 3. Quality gates (offline, ≈1–2 min)

| What | VS Code | Terminal |
|---|---|---|
| Everything CI runs | *Ctrl+Shift+B* (default build task = "Check") or *Run Task → "Check: …"* | `.\tasks.ps1 check` |
| Lint only | *Run Task → "Lint (ruff)"* | `.\tasks.ps1 lint` |
| Type check | *Run Task → "Typecheck (mypy)"* | `.\tasks.ps1 typecheck` |
| Leakage gate (exit gate, synthetic) | *Run Task → "Phase 0: leakage gate (synthetic)"* or *Run and Debug → "Pytest: leakage suite"* | `.\tasks.ps1 leakage` |
| Tests | *Testing* panel (beaker icon) → ▶ Run All | `.\tasks.ps1 test` |

Expected: ruff "All checks passed!", mypy "Success", **11 leakage tests pass**,
**134 passed, 15 skipped**. The 15 skipped tests need downloaded data (step 5).

## 4. Synthetic smoke run of the corrected baseline (≈3 min)

| VS Code | Terminal |
|---|---|
| *Run Task → "Phase 0: corrected baseline (synthetic, ~3 min)"*, or *Run and Debug (F5) → "Phase 0: corrected baseline (synthetic)"* to step through it | `.\tasks.ps1 phase0-smoke` |

Output: `reports\phase0\synthetic\report.md` (open it with *Ctrl+Shift+V* for the
Markdown preview), `summary.json`, `<universe>\equity.csv`, `fold_metrics.csv`.
Synthetic prices have no exploitable signal, so no strategy should beat equal
weight with significance — if one does, suspect leakage.

## 5. Real data: exit gate + corrected v1 baseline

Needs internet access to Yahoo Finance.

| Step | VS Code task | Terminal |
|---|---|---|
| Download 40 tickers + 2 benchmarks (START_DATE..today) and build features | "Data: download prices + build features" | `.\tasks.ps1 data` |
| **Exit gate on real data** (exit code 0 = pass) | "Phase 0: leakage gate (real data)" | `.\tasks.ps1 leakage-real` |
| Corrected baseline (~10–25 min; PPO re-trained per fold) | "Phase 0: corrected baseline (real data)" | `.\tasks.ps1 phase0-baseline` |
| Re-run tests — the 15 data tests now run too | Testing panel → Run All | `.\tasks.ps1 test` |

Extra options are passed straight through, e.g.:

```powershell
.\tasks.ps1 phase0-baseline --universe sp500 --rl-timesteps 50000 --cost-bps 20
uv run pfo-v1-baseline --help
```

| Flag | Default | Meaning |
|---|---|---|
| `--universe nifty\|sp500\|both` | both | Each market on its own exchange calendar (no NSE/NYSE intersection, no FX mixing) |
| `--train-days / --test-days` | 504 / 126 | Walk-forward windows; test windows never overlap |
| `--rebalance` | 21 | Trading days between decisions |
| `--execution-lag` | 1 | Decide at close *t*, trade at close *t+1* |
| `--cost-bps` | 10 | Cost per unit of one-way turnover |
| `--rl-timesteps` | 30000 | PPO steps per fold (0 disables RL) |

**Commit the real-data outputs for the run log:**

```powershell
New-Item -ItemType Directory -Force runlog\phase0\artifacts\real | Out-Null
Copy-Item reports\phase0\real\report.md, reports\phase0\real\summary.json, reports\phase0\leakage_scan_real.json `
          -Destination runlog\phase0\artifacts\real\
git add runlog/phase0/artifacts/real
git commit -m "phase0: real-data corrected baseline"
git push
```

## 6. (Optional) MLflow training + API with the fixed code

Start Docker Desktop first.

```powershell
.\tasks.ps1 mlflow                                    # pinned MLflow 3.18 server on http://localhost:5000
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
uv run python -m models.rl_agent.train_agent --algo ppo --universe sp500 --timesteps 50000
.\tasks.ps1 serve                                     # API on http://127.0.0.1:8000 (or F5 → "API (uvicorn)")
```

In a second terminal:

```powershell
$key = (Select-String -Path .env -Pattern '^API_KEY=(.*)').Matches[0].Groups[1].Value
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -Headers @{ "X-API-Key" = $key } -ContentType "application/json" `
  -Body '{"tickers":["AAPL","MSFT","NVDA"],"date":"2024-06-28"}'
```

Or open http://127.0.0.1:8000/docs and use *Authorize*. The v1 model zips in
`models\rl_agent\saved_models\` were trained in the leaky v1 environment and have
no `serving_spec`; the API deliberately refuses to serve them — train a new model as above.

## 7. (Optional) Full container stack

Fill in `.env`: `API_KEY`, `TRAINING_SERVICE_API_KEY` (same value as `API_KEY`),
`AIRFLOW_FERNET_KEY`, `AIRFLOW_WEBSERVER_SECRET_KEY`, `AIRFLOW_ADMIN_PASSWORD`,
`GRAFANA_ADMIN_PASSWORD`. Generate a Fernet key with:

```powershell
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Then `.\tasks.ps1 up` (or *Run Task → "Docker: full stack up"*). Airflow runs only
inside Docker on Windows — that is expected; it does not support native Windows.

## Troubleshooting (Windows)

| Symptom | Fix |
|---|---|
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use the VS Code tasks. |
| `uv : The term 'uv' is not recognized` | Install uv (step 0), then open a **new** terminal / restart VS Code. |
| `UnicodeEncodeError: 'charmap' codec …` | Run through `tasks.ps1` or a VS Code terminal (both set `PYTHONUTF8=1`), or `$env:PYTHONUTF8 = "1"`. |
| Tests not shown in the Testing panel | Select `.venv\Scripts\python.exe` as interpreter, then *Testing → Refresh*. |
| `FileNotFoundError: No price files for nifty` | Run `.\tasks.ps1 data` (needs internet). |
| `/predict` → 503 "API_KEY is not configured" | Set `API_KEY` in `.env` and restart the API. |
| `/ready` → 503 | No champion in MLflow and no local model with a `.spec.json`: train one (step 6). |
| `docker compose` → `set AIRFLOW_FERNET_KEY in .env` | Fill the Airflow/Grafana secrets (step 7). |
| A container fails with `$'\r': command not found` | Your checkout predates `.gitattributes`: `git rm --cached -r . ; git reset --hard` (after committing your work). |
| `uv lock --check` fails in CI | You edited `pyproject.toml`: run `uv lock` and commit `uv.lock`. |
| Long-path errors during install | Enable long paths: `git config --global core.longpaths true` (and the Windows "Enable Win32 long paths" policy). |
