"""PyTorch Dataset and DataLoader for time-series LSTM training.

Handles:
- Sequence construction with configurable lookback window
- Walk-forward train/validation/test splitting (no leakage)
- Per-window RobustScaler fitting (re-fit on train, applied to val/test)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from loguru import logger


FEATURE_COLS = [
    "log_return", "momentum_21d", "momentum_63d", "momentum_126d",
    "volatility_21d", "volatility_63d", "beta_63d", "rsi_14",
    "price_vs_ma50", "price_vs_ma200",
]
TARGET_COL = "log_return"


class ReturnSequenceDataset(Dataset):
    """Sliding window dataset for return forecasting."""

    def __init__(
        self,
        features: np.ndarray,   # shape (T, n_features) — already scaled
        targets: np.ndarray,    # shape (T,)
        seq_len: int = 30,
    ):
        self.seq_len = seq_len
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

        for i in range(seq_len, len(features)):
            self.X.append(features[i - seq_len : i])   # (seq_len, n_features)
            self.y.append(targets[i])                   # scalar: next-day return

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def make_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 30,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    batch_size: int = 32,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, RobustScaler]:
    """Split df into train/val/test, fit scaler on train, return DataLoaders.

    Split is strictly temporal — no shuffling across the split boundary.
    Returns: train_loader, val_loader, test_loader, fitted_scaler
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    feature_data = df[FEATURE_COLS].values.astype(np.float32)
    target_data = df[TARGET_COL].values.astype(np.float32)

    # Fit scaler ONLY on training data
    scaler = RobustScaler()
    train_features = scaler.fit_transform(feature_data[:train_end])
    val_features   = scaler.transform(feature_data[train_end:val_end])
    test_features  = scaler.transform(feature_data[val_end:])

    train_ds = ReturnSequenceDataset(train_features, target_data[:train_end], seq_len)
    val_ds   = ReturnSequenceDataset(val_features,   target_data[train_end:val_end], seq_len)
    test_ds  = ReturnSequenceDataset(test_features,  target_data[val_end:], seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    logger.debug(
        f"Dataset split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader, scaler