import io
import json

import pytest

from execution import exchange_utils as exutil
from execution import signal_screener as ss


class _Snapshot:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def current_nav_usd(self) -> float:
        return 1000.0

    def current_gross_usd(self) -> float:
        return 0.0


@pytest.fixture(autouse=True)
def _no_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exutil, "get_positions", lambda: [])


def test_screener_respects_binance_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    strategies = [
        {
            "symbol": "BTCUSDT",
            "timeframe": "15m",
            "capital_per_trade": 5.0,
            "leverage": 1.0,
            "enabled": True,
            "entry": {"type": "always_on"},
        }
    ]
    fake_cfg = {
        "strategies": strategies,
        "sizing": {"min_gross_usd_per_order": 0.0},
        "ml": {"enabled": False},
    }

    def fake_open(path: str, *_args, **_kwargs):
        if path == "config/strategy_config.json":
            return io.StringIO(json.dumps(fake_cfg))
        if path == "config/risk_limits.json":
            return io.StringIO(json.dumps({"global": {"min_notional_usdt": 0.0}}))
        raise FileNotFoundError(path)

    monkeypatch.setattr(ss, "open", fake_open, raising=False)
    monkeypatch.setattr(ss, "resolve_allowed_symbols", lambda: (["BTCUSDT"], {"BTCUSDT": "A"}))
    monkeypatch.setattr(ss, "symbol_tier", lambda _sym: "A")
    monkeypatch.setattr(ss, "evaluate_entry_gate", lambda *_args, **_kwargs: (False, {}))
    monkeypatch.setattr(ss, "_zscore", lambda *_args, **_kwargs: -1.0)
    monkeypatch.setattr(ss, "_rsi", lambda *_args, **_kwargs: 30.0)
    monkeypatch.setattr(ss, "get_klines", lambda *_args, **_kwargs: [[0, 0, 0, 0, 10.0, 0.0]] * 200)
    monkeypatch.setattr(ss, "get_price", lambda *_args, **_kwargs: 50_000.0)
    monkeypatch.setattr(
        ss,
        "get_symbol_filters",
        lambda *_args, **_kwargs: {
            "MIN_NOTIONAL": {"filterType": "MIN_NOTIONAL", "minNotional": "100"},
            "LOT_SIZE": {"filterType": "LOT_SIZE", "minQty": "0.001"},
        },
    )
    monkeypatch.setattr(ss, "PortfolioSnapshot", _Snapshot)
    monkeypatch.setattr(ss, "check_order", lambda **_kwargs: (False, {}))

    signals = ss.generate_signals_from_config()
    orders = [row for row in signals if not row.get("reduceOnly")]
    assert orders, "expected at least one entry intent"

    order = orders[0]
    expected_floor = max(100.0, 0.001 * 50_000.0)
    assert order["gross_usd"] >= expected_floor - 1e-6
    assert order["capital_per_trade"] >= expected_floor - 1e-6
