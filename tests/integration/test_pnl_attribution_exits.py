from __future__ import annotations

import time
from datetime import datetime, timezone

from execution.pnl_tracker import build_pnl_attribution_snapshot


def _exit_trade(symbol: str, reason: str, pnl: float, rr: float | None = None, *, ts: float | None = None) -> dict:
    metadata = {
        "strategy": "vol_target_exit",
        "exit": {"reason": reason, "source_strategy": "vol_target"},
    }
    if rr is not None:
        metadata["vol_target"] = {"tp_sl": {"reward_risk": rr}}
    trade = {
        "symbol": symbol,
        "strategy": "vol_target",
        "realized_pnl": pnl,
        "metadata": metadata,
    }
    if ts is not None:
        trade["ts"] = ts
    return trade


def test_no_exits_block_when_no_exit_trades():
    snapshot = build_pnl_attribution_snapshot([], [])
    exits = snapshot.get("exits")
    assert exits is None or exits.get("summary", {}).get("total_exits", 0) == 0


def test_single_tp_exit_aggregates_summary_and_buckets():
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    trade = _exit_trade("BTCUSDT", "tp", 10.0, rr=2.0, ts=ts)
    snapshot = build_pnl_attribution_snapshot(
        [trade],
        [],
        regimes={"atr_regime": 1, "dd_regime": 2},
        risk_snapshot={"risk_mode": "OK"},
    )
    exits = snapshot.get("exits") or {}
    summary = exits["summary"]
    assert summary["total_exits"] == 1
    assert summary["tp_hits"] == 1
    assert summary["sl_hits"] == 0
    assert summary["avg_rr_tp"] == 2.0
    assert summary["avg_exit_pnl"] == 10.0
    assert summary["tp_ratio"] == 1.0
    by_strategy = exits["by_strategy"]["vol_target"]
    assert by_strategy["total_exits"] == 1
    by_symbol = exits["by_symbol"]["BTCUSDT"]
    assert by_symbol["tp_hits"] == 1


def test_single_sl_exit_tracks_rr_and_counts():
    trade = _exit_trade("ETHUSDT", "sl", -5.0, rr=0.5)
    snapshot = build_pnl_attribution_snapshot([trade], [], regimes={"atr_regime": 0, "dd_regime": 0}, risk_snapshot={"risk_mode": "DEFENSIVE"})
    exits = snapshot.get("exits") or {}
    summary = exits["summary"]
    assert summary["total_exits"] == 1
    assert summary["sl_hits"] == 1
    assert summary["avg_rr_sl"] == 0.5
    assert summary["avg_exit_pnl"] == -5.0


def test_mixed_tp_sl_rolls_up_by_strategy_and_symbol():
    trades = [
        _exit_trade("BTCUSDT", "tp", 4.0, rr=1.2),
        _exit_trade("ETHUSDT", "sl", -2.0, rr=0.8),
    ]
    snapshot = build_pnl_attribution_snapshot(trades, [], regimes={"atr_regime": 0, "dd_regime": 0}, risk_snapshot={"risk_mode": "OK"})
    exits = snapshot.get("exits") or {}
    summary = exits["summary"]
    assert summary["total_exits"] == 2
    assert summary["tp_hits"] == 1
    assert summary["sl_hits"] == 1
    assert exits["by_strategy"]["vol_target"]["total_exits"] == 2
    assert exits["by_symbol"]["BTCUSDT"]["tp_hits"] == 1
    assert exits["by_symbol"]["ETHUSDT"]["sl_hits"] == 1


def test_regime_and_day_buckets_increment():
    now = time.time()
    trade = _exit_trade("BTCUSDT", "tp", 3.0, rr=1.0, ts=now)
    snapshot = build_pnl_attribution_snapshot(
        [trade],
        [],
        regimes={"atr_regime": 3, "dd_regime": 1},
        risk_snapshot={"risk_mode": "WARN"},
    )
    exits = snapshot.get("exits") or {}
    regimes = exits["regimes"]
    assert regimes["atr"]["3"]["total_exits"] == 1
    assert regimes["dd"]["1"]["tp_hits"] == 1
    assert regimes["risk_mode"]["WARN"]["total_exits"] == 1
    day_key = datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
    assert regimes["day"][day_key]["total_exits"] == 1


def test_snapshot_handles_trades_without_exit_metadata():
    trade = {"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 1.0}
    snapshot = build_pnl_attribution_snapshot([trade], [])
    assert snapshot.get("exits") is None
