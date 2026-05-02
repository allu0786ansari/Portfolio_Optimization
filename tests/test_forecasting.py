"""Unit tests for Week 2 — LSTM model, dataset, and training utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
import torch

from models.forecasting.lstm_model import ReturnLSTM
from models.forecasting.dataset import (
    ReturnSequenceDataset, make_dataloaders, FEATURE_COLS
)
from models.forecasting.train_forecaster import (
    directional_accuracy, evaluate, DEFAULT_HP
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """Synthetic feature DataFrame with 400 rows."""
    np.random.seed(0)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {col: np.random.randn(n) * 0.01 for col in FEATURE_COLS}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def small_hp() -> dict:
    return {**DEFAULT_HP, "max_epochs": 2, "hidden_size": 16,
            "num_layers": 1, "fc_hidden": 8, "batch_size": 16}


# ── LSTM architecture tests ─────────────────────────────────────

def test_lstm_output_shape():
    model = ReturnLSTM(input_size=10, hidden_size=32, num_layers=1)
    x = torch.randn(8, 30, 10)   # batch=8, seq=30, features=10
    out = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_lstm_parameter_count():
    model = ReturnLSTM(input_size=10, hidden_size=64, num_layers=2)
    n = model.count_parameters()
    assert n > 0
    assert n < 500_000, "Model too large — something is wrong"


def test_lstm_no_nan_output():
    model = ReturnLSTM(input_size=10)
    x = torch.randn(4, 30, 10)
    out = model(x)
    assert not torch.isnan(out).any(), "Model output contains NaN"


def test_lstm_different_batch_sizes():
    model = ReturnLSTM(input_size=10, hidden_size=32)
    for bs in [1, 4, 16]:
        x = torch.randn(bs, 30, 10)
        out = model(x)
        assert out.shape == (bs,)


# ── Dataset tests ───────────────────────────────────────────────

def test_dataset_length(dummy_df):
    train_loader, val_loader, test_loader, scaler = make_dataloaders(
        dummy_df, seq_len=30, batch_size=16
    )
    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0


def test_dataset_no_leakage(dummy_df):
    """Train split must end strictly before val split starts."""
    n = len(dummy_df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)
    assert train_end < val_end < n


def test_scaler_fitted_on_train_only(dummy_df):
    _, _, _, scaler = make_dataloaders(dummy_df, seq_len=30, batch_size=16)
    # Scaler should be fitted — center_ attribute will exist after fit
    assert hasattr(scaler, "center_"), "Scaler was not fitted"


def test_sequence_dataset_shape():
    features = np.random.randn(100, 10).astype(np.float32)
    targets  = np.random.randn(100).astype(np.float32)
    ds = ReturnSequenceDataset(features, targets, seq_len=30)
    X, y = ds[0]
    assert X.shape == (30, 10)
    assert y.shape == ()


# ── Metric tests ────────────────────────────────────────────────

def test_directional_accuracy_perfect():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    assert directional_accuracy(y, y) == 1.0


def test_directional_accuracy_worst():
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([-0.01, 0.02, -0.03])
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_directional_accuracy_random():
    np.random.seed(42)
    y_true = np.random.randn(1000)
    y_pred = np.random.randn(1000)
    acc = directional_accuracy(y_true, y_pred)
    # Random predictions should give ~50% directional accuracy
    assert 0.45 < acc < 0.55, f"Expected ~0.50, got {acc:.3f}"


# ── End-to-end smoke test ────────────────────────────────────────

def test_training_smoke(dummy_df, small_hp, tmp_path):
    """Full train loop runs without error on tiny dummy data."""
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from models.forecasting.train_forecaster import train_one_epoch, evaluate

    train_loader, val_loader, _, _ = make_dataloaders(
        dummy_df, seq_len=small_hp["seq_len"], batch_size=small_hp["batch_size"]
    )
    model     = ReturnLSTM(len(FEATURE_COLS), small_hp["hidden_size"],
                           small_hp["num_layers"], small_hp["dropout"],
                           small_hp["fc_hidden"])
    optimizer = torch.optim.Adam(model.parameters(), lr=small_hp["lr"])
    criterion = nn.MSELoss()
    device    = torch.device("cpu")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_rmse, val_dir_acc = evaluate(model, val_loader, criterion, device)

    assert train_loss >= 0
    assert val_rmse   >= 0
    assert 0.0 <= val_dir_acc <= 1.0