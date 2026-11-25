"""Walk-forward splits whose out-of-sample segments never overlap.

v1 stepped a 126-day test window forward by 21 days and concatenated the
results, so each day was counted up to six times. Here the default step equals
the test length; overlapping test windows are refused unless explicitly
requested (they are only valid for per-window statistics, never for stitching
an equity curve).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    @property
    def train(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def test(self) -> range:
        return range(self.test_start, self.test_end)


def walk_forward_splits(
    n_obs: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    expanding: bool = False,
    embargo: int = 0,
    allow_overlap: bool = False,
) -> list[Fold]:
    """Create walk-forward folds over ``n_obs`` time-ordered observations.

    Args:
        train_size: rows in each training window (initial window if expanding).
        test_size: rows in each test window.
        step: distance between consecutive test starts. Defaults to ``test_size``.
        expanding: if True the training window always starts at 0.
        embargo: rows skipped between the end of training and the test start.
        allow_overlap: permit ``step < test_size`` (per-window statistics only).
    """
    if min(train_size, test_size) <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    if step < test_size and not allow_overlap:
        raise ValueError(
            f"step={step} < test_size={test_size} creates overlapping test windows; "
            "pass allow_overlap=True only for per-window statistics"
        )
    folds: list[Fold] = []
    test_start = train_size + embargo
    k = 0
    while test_start + test_size <= n_obs:
        train_end = test_start - embargo
        train_start = 0 if expanding else train_end - train_size
        folds.append(Fold(k, train_start, train_end, test_start, test_start + test_size))
        k += 1
        test_start += step
    return folds


def assert_non_overlapping(folds: list[Fold]) -> None:
    """Raise if any two test windows overlap or any train window touches its test window."""
    for f in folds:
        if f.train_end > f.test_start:
            raise AssertionError(f"fold {f.fold_id}: training window overlaps its test window")
    ordered = sorted(folds, key=lambda f: f.test_start)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.test_start < a.test_end:
            raise AssertionError(f"folds {a.fold_id} and {b.fold_id} have overlapping test windows")
