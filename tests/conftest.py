"""Shared fixtures. Synthetic data keeps every test runnable without downloads."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@pytest.fixture(scope="session")
def synth_market():
    from pfo.data.market import make_synthetic_market

    return make_synthetic_market(n_assets=8, n_days=900, seed=7)
