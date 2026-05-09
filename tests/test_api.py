"""Integration tests for the FastAPI portfolio prediction API.

Uses FastAPI TestClient with mocked dependencies.
No running server or MLflow connection required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from serving.main import app

API_KEY        = "test-key"
HEADERS        = {"X-API-Key": API_KEY}
SAMPLE_TICKERS = ["RELIANCE.NS", "TCS.NS", "AAPL", "MSFT"]


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_registry():
    with patch("serving.model_loader.registry") as mock_reg:
        mock_reg.is_loaded    = True
        mock_reg.version      = "champion-v1"
        mock_reg.algo         = "PPO"
        mock_reg.load         = MagicMock()
        mock_reg.predict      = MagicMock(
            return_value=np.array([0.5, -0.3, 0.8, 0.2])
        )
        yield mock_reg


@pytest.fixture(scope="module")
def mock_predictor():
    with patch("serving.main.predict_weights") as mock_pw:
        mock_pw.return_value = {
            "weights":         {"RELIANCE.NS": 0.35, "TCS.NS": 0.25,
                                "AAPL": 0.25, "MSFT": 0.15},
            "weights_sum":     1.0,
            "model_version":   "champion-v1",
            "algo":            "PPO",
            "latency_ms":      42.3,
            "n_assets":        4,
            "timestamp":       "2024-12-31T10:00:00+00:00",
            "skipped_tickers": [],
        }
        yield mock_pw


@pytest.fixture(scope="module")
def client(mock_registry, mock_predictor):
    with TestClient(app) as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────

def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_no_auth_required(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_response_schema(client):
    data = client.get("/health").json()
    assert "status" in data
    assert data["status"] == "ok"


# ── Ready endpoint ────────────────────────────────────────────────

def test_ready_returns_200_when_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200


def test_ready_response_has_model_version(client):
    data = client.get("/ready").json()
    assert data["ready"] is True
    assert data["model_loaded"] is True
    assert "model_version" in data


# ── Auth tests ────────────────────────────────────────────────────

def test_predict_requires_api_key(client):
    r = client.post("/predict", json={"tickers": SAMPLE_TICKERS})
    assert r.status_code == 401


def test_predict_rejects_wrong_key(client):
    r = client.post(
        "/predict",
        json={"tickers": SAMPLE_TICKERS},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_predict_accepts_correct_key(client):
    r = client.post("/predict", json={"tickers": SAMPLE_TICKERS}, headers=HEADERS)
    assert r.status_code == 200


# ── Predict happy path ────────────────────────────────────────────

def test_predict_response_schema(client):
    data = client.post(
        "/predict", json={"tickers": SAMPLE_TICKERS}, headers=HEADERS
    ).json()
    for key in ["weights", "weights_sum", "model_version", "algo",
                "latency_ms", "n_assets", "timestamp"]:
        assert key in data, f"Missing key: {key}"


def test_predict_weights_sum_to_one(client):
    data = client.post(
        "/predict", json={"tickers": SAMPLE_TICKERS}, headers=HEADERS
    ).json()
    assert abs(data["weights_sum"] - 1.0) < 1e-4


def test_predict_weights_keys_match_tickers(client):
    data = client.post(
        "/predict", json={"tickers": SAMPLE_TICKERS}, headers=HEADERS
    ).json()
    assert set(data["weights"].keys()) == set(SAMPLE_TICKERS)


def test_predict_latency_positive(client):
    data = client.post(
        "/predict", json={"tickers": SAMPLE_TICKERS}, headers=HEADERS
    ).json()
    assert data["latency_ms"] > 0


# ── Validation errors ─────────────────────────────────────────────

def test_predict_rejects_single_ticker(client):
    r = client.post("/predict", json={"tickers": ["AAPL"]}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_empty_ticker_list(client):
    r = client.post("/predict", json={"tickers": []}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_rejects_missing_tickers_field(client):
    r = client.post("/predict", json={}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_uppercases_tickers(client):
    r = client.post(
        "/predict",
        json={"tickers": ["reliance.ns", "tcs.ns", "aapl", "msft"]},
        headers=HEADERS,
    )
    assert r.status_code == 200
    keys = list(r.json()["weights"].keys())
    assert all(t == t.upper() for t in keys)