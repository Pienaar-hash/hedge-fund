import pytest

import execution.utils.metrics as metrics


class _DummyPnlTracker:
    def __init__(self, fees: float, pnl: float) -> None:
        self._fees = fees
        self._pnl = pnl

    def get_fees(self, symbol=None, window_days=7):
        return self._fees

    def get_gross_realized(self, symbol=None, window_days=7):
        return self._pnl


def test_fee_pnl_ratio_basic(monkeypatch):
    monkeypatch.setattr(metrics, "pnl_tracker", _DummyPnlTracker(fees=10.0, pnl=50.0))
    res = metrics.fee_pnl_ratio(symbol=None, window_days=7)
    assert res["fee_pnl_ratio"] == pytest.approx(0.2)
    assert res["fees"] == pytest.approx(10.0)
    assert res["pnl"] == pytest.approx(50.0)


def test_fee_pnl_ratio_handles_zero(monkeypatch):
    monkeypatch.setattr(metrics, "pnl_tracker", _DummyPnlTracker(fees=5.0, pnl=0.0))
    res = metrics.fee_pnl_ratio(symbol=None, window_days=7)
    assert res["fee_pnl_ratio"] is None
    assert res["fees"] == pytest.approx(5.0)
