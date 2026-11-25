# Dependencies

Since Phase 0 all Python dependencies are declared in `pyproject.toml` (extras:
`data`, `ml`, `rl`, `serving`, `dashboard`, `dev`) and pinned in `uv.lock`.

* Local development: `uv sync --extra dev --extra rl --extra ml --extra serving --extra data`
* Docker / CI: `scripts/install_deps.sh "<extras>"` exports the locked set and installs
  CPU-only PyTorch at the locked version.

The old hand-maintained `api.txt`, `dashboard.txt` and `airflow.txt` were removed:
they drifted from the code (e.g. `cvxpy`, `tqdm`, `rich` were missing).
