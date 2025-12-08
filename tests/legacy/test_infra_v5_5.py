from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dashboard.router_health import load_router_health
from execution.risk_limits import RiskState
from utils import firestore_client

pytestmark = pytest.mark.legacy


def test_risk_state_persistence_roundtrip():
    now = time.time()
    state = RiskState()
    state.note_error(now - 5)
    state.note_attempt(now - 10)
    state.note_fill("BTCUSDT", now - 30)
    state.daily_pnl_pct = -2.5

    snapshot = state.snapshot()
    restored = RiskState(snapshot)

    assert restored.errors_in(60, now) == 1
    assert restored.attempts_in(60, now) == 1
    assert restored.last_fill_ts("BTCUSDT") > 0
    assert restored.daily_pnl_pct == pytest.approx(-2.5)


def test_firestore_noop_on_missing_client(monkeypatch):
    firestore_client._get_db_cached.cache_clear()
    monkeypatch.setenv("FIRESTORE_ENABLED", "1")
    monkeypatch.setattr(firestore_client, "firestore", None, raising=False)
    monkeypatch.setattr(firestore_client, "service_account", None, raising=False)

    db = firestore_client.get_db(strict=False)
    assert getattr(db, "_is_noop", False)

    firestore_client._get_db_cached.cache_clear()
    with pytest.raises(RuntimeError):
        firestore_client.get_db(strict=True)


@pytest.mark.skip(reason="load_router_health API changed in v6 - no longer accepts signal_path/order_path kwargs")
def test_router_health_aggregation(tmp_path: Path):
    exec_dir = tmp_path / "logs" / "execution"
    exec_dir.mkdir(parents=True)
    signal_path = exec_dir / "signal_metrics.jsonl"
    order_path = exec_dir / "order_metrics.jsonl"

    now = time.time()
    signals = [
        {
            "attempt_id": "sig1",
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "ts": now - 60,
            "doctor": {"ok": True, "confidence": 0.8},
        },
        {
            "attempt_id": "sig2",
            "symbol": "BTCUSDT",
            "signal": "SELL",
            "ts": now - 30,
            "doctor": {"ok": True, "confidence": 0.6},
        },
    ]
    orders = [
        {
            "attempt_id": "sig1",
            "symbol": "BTCUSDT",
            "event": "position_close",
            "pnl_at_close_usd": 12.5,
            "ts": now - 20,
        },
        {
            "attempt_id": "sig2",
            "symbol": "BTCUSDT",
            "event": "position_close",
            "pnl_at_close_usd": -7.0,
            "ts": now - 5,
        },
    ]

    with signal_path.open("w", encoding="utf-8") as handle:
        for row in signals:
            handle.write(json.dumps(row) + "\n")

    with order_path.open("w", encoding="utf-8") as handle:
        for row in orders:
            handle.write(json.dumps(row) + "\n")

    data = load_router_health(window=10, signal_path=signal_path, order_path=order_path)

    assert data.summary["count"] == 2
    assert data.summary["cum_pnl"] == pytest.approx(5.5)
    assert data.summary["win_rate"] == pytest.approx(50.0)

    assert not data.per_symbol.empty
    row = data.per_symbol.iloc[0]
    assert row["symbol"] == "BTCUSDT"
    assert row["count"] == 2
    assert row["cum_pnl"] == pytest.approx(5.5)
    assert row["win_rate"] == pytest.approx(50.0)
