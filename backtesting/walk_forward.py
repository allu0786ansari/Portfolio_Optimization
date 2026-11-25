"""DEPRECATED v1 walk-forward backtest — kept only as a pointer.

The v1 implementation (preserved in git history before Phase 0) had two
defects that invalidate its results:

1. Look-ahead: the RL policy observed row ``t`` (including the return of day
   ``t``) and was credited with that same return.
2. Overlapping windows: 126-day test windows were stepped by 21 days and
   concatenated, so 735 unique days were counted as 3,780 and the reported
   p-value (4e-22) was meaningless.

The corrected evaluation lives in ``pfo.backtest`` and is run with::

    uv run pfo-v1-baseline --source v1 --universe both

This module forwards to it so ``make backtest`` keeps working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pfo.reports.v1_baseline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
