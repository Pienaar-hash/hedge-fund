from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from execution.pnl_tracker import build_pnl_attribution_snapshot, export_pnl_attribution_state


def _trade(symbol: str, pnl: float, ts: float) -> dict:
    return {"symbol": symbol, "strategy": "momentum", "realized_pnl": pnl, "ts": ts}


def test_atr_regime_bucket_assignment():
    trades = [_trade("BTCUSDT", 5.0, time.time())]
    regimes = {"atr_regime": 2, "dd_regime": 1}
    snapshot = build_pnl_attribution_snapshot(trades, [], regimes=regimes, risk_snapshot={"risk_mode": "OK"})
    assert "per_regime" in snapshot
    atr_bucket = snapshot["per_regime"]["atr"]["2"]
    assert atr_bucket["realized"] == 5.0


def test_dd_regime_bucket_assignment():
    trades = [_trade("BTCUSDT", -3.0, time.time())]
    regimes = {"atr_regime": 0, "dd_regime": 3}
    snapshot = build_pnl_attribution_snapshot(trades, [], regimes=regimes, risk_snapshot={"risk_mode": "OK"})
    dd_bucket = snapshot["per_regime"]["dd"]["3"]
    assert dd_bucket["realized"] == -3.0


def test_risk_mode_attribution():
    trades = [_trade("BTCUSDT", 2.0, time.time())]
    snapshot = build_pnl_attribution_snapshot(
        trades,
        [],
        regimes={"atr_regime": 0, "dd_regime": 0},
        risk_snapshot={"risk_mode": "DEFENSIVE"},
    )
    bucket = snapshot["per_risk_mode"]["DEFENSIVE"]
    assert bucket["realized"] == 2.0


def test_missing_regime_file_produces_empty_regime_blocks():
    trades = [_trade("BTCUSDT", 1.0, time.time())]
    snapshot = build_pnl_attribution_snapshot(trades, [], regimes={}, risk_snapshot={"risk_mode": "OK"})
    assert "per_regime" not in snapshot


def test_missing_risk_snapshot_defaults_to_ok():
    trades = [_trade("BTCUSDT", 4.0, time.time())]
    snapshot = build_pnl_attribution_snapshot(trades, [], regimes={"atr_regime": 1, "dd_regime": 2}, risk_snapshot={})
    bucket = snapshot["per_risk_mode"]["OK"]
    assert bucket["realized"] == 4.0


def test_per_day_aggregation_multiple_days():
    now = datetime(2025, 11, 25, tzinfo=timezone.utc).timestamp()
    later = now + 86400
    trades = [
        _trade("BTCUSDT", 1.0, now),
        _trade("ETHUSDT", 2.0, later),
    ]
    snapshot = build_pnl_attribution_snapshot(
        trades,
        [],
        regimes={"atr_regime": 0, "dd_regime": 0},
        risk_snapshot={"risk_mode": "OK"},
    )
    assert "2025-11-25" in snapshot["per_day"]
    assert "2025-11-26" in snapshot["per_day"]


def test_unrealized_pnl_added_to_regime_and_risk_mode():
    regimes = {"atr_regime": 1, "dd_regime": 2}
    risk_snapshot = {"risk_mode": "WARN"}
    positions = [{"symbol": "BTCUSDT", "unrealized_pnl": 3.5}]
    snapshot = build_pnl_attribution_snapshot(
        [],
        positions,
        regimes=regimes,
        risk_snapshot=risk_snapshot,
        now_ts=datetime(2025, 11, 25, tzinfo=timezone.utc).timestamp(),
    )
    assert snapshot["per_regime"]["atr"]["1"]["unrealized"] == 3.5
    assert snapshot["per_regime"]["dd"]["2"]["unrealized"] == 3.5
    assert snapshot["per_risk_mode"]["WARN"]["unrealized"] == 3.5


def test_unrealized_pnl_added_to_today_day_bucket():
    today = datetime(2025, 11, 25, tzinfo=timezone.utc).timestamp()
    positions = [{"symbol": "BTCUSDT", "unrealized_pnl": 2.0}]
    snapshot = build_pnl_attribution_snapshot(
        [],
        positions,
        regimes={"atr_regime": 0, "dd_regime": 0},
        risk_snapshot={"risk_mode": "OK"},
        now_ts=today,
    )
    assert snapshot["per_day"]["2025-11-25"]["unrealized"] == 2.0


def test_total_calculation_correct_for_each_bucket():
    trades = [_trade("BTCUSDT", 5.0, time.time())]
    positions = [{"symbol": "BTCUSDT", "unrealized_pnl": 1.0}]
    snapshot = build_pnl_attribution_snapshot(
        trades,
        positions,
        regimes={"atr_regime": 0, "dd_regime": 0},
        risk_snapshot={"risk_mode": "OK"},
    )
    bucket = snapshot["per_risk_mode"]["OK"]
    assert bucket["total"] == bucket["realized"] + bucket["unrealized"]


def test_zero_trades_yields_empty_but_valid_structures():
    snapshot = build_pnl_attribution_snapshot([], [], regimes={}, risk_snapshot={})
    assert snapshot["per_symbol"] == {}
    assert set(snapshot["per_risk_mode"].keys()) == {"OK", "WARN", "DEFENSIVE", "HALTED"}
    assert "per_day" not in snapshot


def test_decimal_consistency_no_nan():
    trades = [_trade("BTCUSDT", 0.0, time.time())]
    snapshot = build_pnl_attribution_snapshot(
        trades,
        [],
        regimes={"atr_regime": 0, "dd_regime": 0},
        risk_snapshot={"risk_mode": "OK"},
    )
    bucket = snapshot["per_regime"]["atr"]["0"]
    assert bucket["realized"] == 0.0


def test_valid_json_written_by_exporter(tmp_path: Path):
    trades_file = tmp_path / "trades.jsonl"
    trades_file.write_text(json.dumps({"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 1.0, "ts": time.time()}) + "\n")
    positions_file = tmp_path / "positions.json"
    positions_file.write_text(json.dumps({"rows": [{"symbol": "BTCUSDT", "unrealized_pnl": 0.5}]}))
    output = tmp_path / "attr.json"
    export_pnl_attribution_state(
        trades_path=trades_file,
        positions_path=positions_file,
        output_path=output,
    )
    data = json.loads(output.read_text())
    assert "per_symbol" in data


def test_symbol_strategy_blocks_unchanged_from_h1():
    trades = [_trade("BTCUSDT", 1.0, time.time())]
    snapshot = build_pnl_attribution_snapshot(
        trades, [], regimes={"atr_regime": 0, "dd_regime": 0}, risk_snapshot={"risk_mode": "OK"}
    )
    assert "BTCUSDT" in snapshot["per_symbol"]
    assert "momentum" in snapshot["per_strategy"]


def test_snapshot_stable_structure():
    snapshot = build_pnl_attribution_snapshot(
        [], [], regimes={"atr_regime": 0, "dd_regime": 1}, risk_snapshot={"risk_mode": "HALTED"}
    )
    assert set(snapshot["summary"].keys()) >= {"total_realized", "total_unrealized", "total_pnl", "ts"}


def test_json_atomic_write(tmp_path: Path):
    trades_file = tmp_path / "trades.jsonl"
    trades_file.write_text(json.dumps({"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 2.0, "ts": time.time()}) + "\n")
    output = tmp_path / "attr.json"
    export_pnl_attribution_state(
        trades_path=trades_file,
        positions_path=tmp_path / "positions.json",
        output_path=output,
    )
    assert output.exists()
