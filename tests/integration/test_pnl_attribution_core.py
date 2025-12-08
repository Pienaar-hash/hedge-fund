import json
from pathlib import Path

import pytest

from execution.pnl_tracker import (
    build_pnl_attribution_snapshot,
    export_pnl_attribution_state,
)


def test_empty_inputs_produce_zero_snapshot():
    snapshot = build_pnl_attribution_snapshot([], [])
    assert snapshot["per_symbol"] == {}
    assert snapshot["per_strategy"] == {}
    summary = snapshot["summary"]
    assert summary["total_realized"] == 0.0
    assert summary["total_unrealized"] == 0.0
    assert summary["total_pnl"] == 0.0
    assert summary["record_count"] == 0
    assert summary["win_rate"] == 0.0


def test_single_trade_attribution():
    trades = [{"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 10.0}]
    snapshot = build_pnl_attribution_snapshot(trades, [])
    sym = snapshot["per_symbol"]["BTCUSDT"]
    assert sym["realized_pnl"] == 10.0
    assert sym["unrealized_pnl"] == 0.0
    assert sym["total_pnl"] == 10.0
    assert sym["trade_count"] == 1
    strat = snapshot["per_strategy"]["momentum"]
    assert strat["realized_pnl"] == 10.0
    assert strat["trade_count"] == 1
    summary = snapshot["summary"]
    assert summary["total_realized"] == 10.0
    assert summary["total_unrealized"] == 0.0
    assert summary["total_pnl"] == 10.0
    assert summary["win_rate"] == 1.0


def test_multiple_trades_per_symbol():
    trades = [
        {"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 5.0},
        {"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": -2.0},
    ]
    snapshot = build_pnl_attribution_snapshot(trades, [])
    sym = snapshot["per_symbol"]["BTCUSDT"]
    assert sym["realized_pnl"] == 3.0
    assert sym["trade_count"] == 2


def test_unrealized_pnl_from_positions():
    trades = [{"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 5.0}]
    positions = [
        {"symbol": "BTCUSDT", "unrealized_pnl": 2.5},
        {"symbol": "ETHUSDT", "unrealized_pnl": 1.0},
    ]
    snapshot = build_pnl_attribution_snapshot(trades, positions)
    assert snapshot["per_symbol"]["BTCUSDT"]["unrealized_pnl"] == 2.5
    assert snapshot["per_symbol"]["BTCUSDT"]["total_pnl"] == 7.5
    assert snapshot["per_symbol"]["ETHUSDT"]["unrealized_pnl"] == 1.0
    summary = snapshot["summary"]
    assert summary["total_unrealized"] == 3.5
    assert summary["total_pnl"] == 8.5


def test_win_rate_calculation():
    trades = [
        {"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 5.0},
        {"symbol": "ETHUSDT", "strategy": "rv", "realized_pnl": -3.0},
        {"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 0.0},
    ]
    snapshot = build_pnl_attribution_snapshot(trades, [])
    summary = snapshot["summary"]
    assert summary["win_rate"] == pytest.approx(1 / 3)


def test_export_creates_file(tmp_path: Path):
    trades_path = tmp_path / "trades.jsonl"
    with trades_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 5.0}) + "\n")
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({"rows": [{"symbol": "BTCUSDT", "unrealized_pnl": 1.0}]}))
    output_path = tmp_path / "attr.json"
    result = export_pnl_attribution_state(
        trades_path=trades_path,
        positions_path=positions_path,
        output_path=output_path,
    )
    assert output_path.exists()
    loaded = json.loads(output_path.read_text())
    assert loaded["summary"]["total_pnl"] == result["summary"]["total_pnl"]


def test_missing_trades_file_fallback(tmp_path: Path):
    positions_path = tmp_path / "positions.json"
    positions_path.write_text(json.dumps({"rows": []}))
    output_path = tmp_path / "attr.json"
    result = export_pnl_attribution_state(
        trades_path=tmp_path / "missing_trades.jsonl",
        positions_path=positions_path,
        output_path=output_path,
    )
    assert result["summary"]["record_count"] == 0
    assert result["summary"]["win_rate"] == 0.0


def test_missing_positions_file_fallback(tmp_path: Path):
    trades_path = tmp_path / "trades.jsonl"
    trades_path.write_text(json.dumps({"symbol": "BTCUSDT", "strategy": "momentum", "realized_pnl": 3.0}) + "\n")
    output_path = tmp_path / "attr.json"
    result = export_pnl_attribution_state(
        trades_path=trades_path,
        positions_path=tmp_path / "missing_positions.json",
        output_path=output_path,
    )
    assert result["summary"]["total_unrealized"] == 0.0
