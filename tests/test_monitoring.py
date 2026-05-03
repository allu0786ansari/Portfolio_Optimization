"""Unit tests for Week 9 Prometheus metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.metrics import (
    record_request, record_portfolio_return,
    update_rolling_sharpe, _return_buffer,
    ROLLING_SHARPE, REQUEST_COUNT,
)


def test_record_request_increments_counter():
    before = REQUEST_COUNT.labels(status="success")._value.get()
    record_request(latency_ms=42.0, success=True)
    after = REQUEST_COUNT.labels(status="success")._value.get()
    assert after == before + 1


def test_record_request_error_status():
    before = REQUEST_COUNT.labels(status="error")._value.get()
    record_request(latency_ms=10.0, success=False)
    after = REQUEST_COUNT.labels(status="error")._value.get()
    assert after == before + 1


def test_record_request_does_not_raise():
    record_request(latency_ms=123.4, success=True)


def test_record_portfolio_return_updates_buffer():
    before = len(_return_buffer)
    record_portfolio_return(0.005)
    assert len(_return_buffer) == min(before + 1, 30)


def test_rolling_sharpe_positive_for_positive_returns():
    for _ in range(10):
        record_portfolio_return(0.001)
    assert ROLLING_SHARPE._value.get() > 0


def test_update_rolling_sharpe_directly():
    update_rolling_sharpe(1.42)
    assert abs(ROLLING_SHARPE._value.get() - 1.42) < 1e-6


def test_rolling_sharpe_negative_for_negative_returns():
    for _ in range(10):
        record_portfolio_return(-0.002)
    assert ROLLING_SHARPE._value.get() < 0


def test_return_buffer_max_size():
    for _ in range(50):
        record_portfolio_return(0.001)
    assert len(_return_buffer) <= 30