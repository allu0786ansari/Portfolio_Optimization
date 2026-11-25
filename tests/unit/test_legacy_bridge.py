"""The v1 entry points (legacy env, serving predictor) now run on the corrected core.

Synthetic prices are written in the v1 on-disk layout (data/raw/<ticker>.parquet)
so these paths are exercised without downloading market data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from pfo.data.market import make_synthetic_market
from pfo.data.universes import SP500_V1


@pytest.fixture()
def v1_layout(tmp_path, monkeypatch):
    m = make_synthetic_market(n_assets=len(SP500_V1), n_days=1200, seed=3)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for syn, tic in zip(m.tickers, SP500_V1, strict=True):
        pd.DataFrame({"Close": m.prices[syn]}).rename_axis("date").to_parquet(raw / f"{tic}.parquet")
    pd.DataFrame({"Close": m.benchmark}).rename_axis("date").to_parquet(raw / "benchmark_sp500.parquet")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_legacy_env_is_the_corrected_env(v1_layout):
    from stable_baselines3.common.env_checker import check_env

    from models.rl_agent.portfolio_env import PortfolioEnv

    env = PortfolioEnv(universe="sp500", seed=0)
    check_env(env, warn=True)
    obs, info = env.reset(seed=1)
    assert "start_date" in info and obs.shape == env.observation_space.shape
    _, _, _, _, step_info = env.step(env.action_space.sample())
    assert step_info["reward_row"] == env._t - 1 + env.lag + 1
    assert step_info["reward_row"] < env.end  # never samples beyond the training rows
    spec = env.serving_spec()
    assert spec["tickers"] == list(SP500_V1) and len(spec["scaler_mean"]) == 10


def test_validation_env_only_uses_validation_rows(v1_layout):
    from models.rl_agent.portfolio_env import PortfolioEnv

    env = PortfolioEnv(universe="sp500", train_frac=0.7, start_frac=0.7, end_frac=0.85, episode_length=30, seed=0)
    T = len(env.dates)
    assert env.start == int(T * 0.7) and env.end == int(T * 0.85)


def test_predictor_serves_local_model_with_spec(v1_layout, monkeypatch):
    from stable_baselines3 import PPO

    import serving.predictor as pred
    from models.rl_agent.portfolio_env import PortfolioEnv

    env = PortfolioEnv(universe="sp500", seed=0)
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=64, seed=0, device="cpu", verbose=0)
    model.learn(256)
    out = v1_layout / "models" / "rl_agent" / "saved_models"
    out.mkdir(parents=True)
    model.save(out / "ppo_agent")
    (out / "ppo_agent.spec.json").write_text(json.dumps(env.serving_spec()), encoding="utf-8")

    monkeypatch.setattr(pred, "SAVED_MODELS_DIR", out)
    monkeypatch.setattr(pred, "_local", None)
    monkeypatch.setattr(pred, "_feature_cache", {})
    src = pred.serving_source()
    assert src["kind"] == "local-zip"
    res = pred.predict_weights(["AAPL", "MSFT", "NVDA", "NOT_A_TICKER"], as_of="2019-06-15")
    assert abs(sum(res["weights"].values()) - 1) < 1e-6
    assert res["skipped_tickers"] == ["NOT_A_TICKER"]
    assert res["as_of_date"] <= "2019-06-15"


def test_v1_zip_without_spec_is_not_served(tmp_path, monkeypatch):
    import serving.predictor as pred

    (tmp_path / "ppo_agent.zip").write_bytes(b"v1 model trained in the leaky env")
    monkeypatch.setattr(pred, "SAVED_MODELS_DIR", tmp_path)
    monkeypatch.setattr(pred, "_local", None)
    assert pred._local_model() is None
    np.testing.assert_equal(pred.registry.is_loaded, False)


@pytest.mark.slow
def test_train_registers_this_runs_version_and_serving_spec(v1_layout, monkeypatch):
    """train() logs the serving spec, registers its own version and sets the champion alias."""
    import mlflow

    uri = f"sqlite:///{(v1_layout / 'mlflow.db').as_posix()}"  # forward slashes: valid on Windows too
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    import models.rl_agent.train_agent as ta

    monkeypatch.setattr(ta, "MLFLOW_URI", uri)
    monkeypatch.setattr(ta, "PPO_PARAMS", {**ta.PPO_PARAMS, "n_steps": 128, "batch_size": 64, "verbose": 0})
    first = ta.train("ppo", universe="sp500", timesteps=256)
    assert first["promoted"] and first["version"] is not None

    client = mlflow.tracking.MlflowClient(uri)
    mv = client.get_model_version_by_alias("PortfolioAgent", "champion")
    assert str(mv.version) == str(first["version"]) and mv.run_id == first["run_id"]
    spec_path = mlflow.artifacts.download_artifacts(run_id=first["run_id"], artifact_path="serving_spec.json",
                                                    tracking_uri=uri)
    assert json.loads(open(spec_path, encoding="utf-8").read())["universe"] == "sp500"

    # the registry loader (serving) can load the champion and run policy.predict
    from serving.model_loader import ModelRegistry

    reg = ModelRegistry()
    reg._load_champion()
    assert reg.is_loaded and reg.version == f"champion-v{first['version']}"
    action = reg.predict(np.zeros(reg._policy.observation_space.shape, dtype=np.float32))
    assert action.shape == (20,) and np.all(np.abs(action) <= 1.0)

    # evaluate-champion must point the alias at the version of the best run
    from serving.retrain_router import evaluate_champion

    second = ta.train("ppo", universe="sp500", timesteps=256)
    res = evaluate_champion()
    best = max([first, second], key=lambda r: r["val_sharpe"])
    champion = client.get_model_version_by_alias("PortfolioAgent", "champion")
    assert res.status == "success"
    assert champion.run_id in (best["run_id"], first["run_id"])
