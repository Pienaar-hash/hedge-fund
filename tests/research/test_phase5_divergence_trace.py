from __future__ import annotations

import csv
import json
from pathlib import Path

from research.phase5_divergence_trace import find_first_divergence


def _write_orders(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_trades(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "side", "qty", "entry_price", "entry_ts", "entry_reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_find_first_divergence_reports_first_side_mismatch(tmp_path: Path) -> None:
    _write_orders(
        tmp_path / "logs" / "execution" / "orders_executed.jsonl",
        [
            {"symbol": "BTCUSDT", "side": "BUY", "timestamp": "2026-05-15T10:00:00Z"},
            {"symbol": "BTCUSDT", "side": "SELL", "timestamp": "2026-05-15T10:15:00Z"},
        ],
    )
    _write_trades(
        tmp_path / "replay" / "trades.csv",
        [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "qty": 1.0,
                "entry_price": 100.0,
                "entry_ts": "2026-05-15T10:00:00Z",
                "entry_reason": "x",
            },
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "qty": 1.0,
                "entry_price": 101.0,
                "entry_ts": "2026-05-15T10:15:00Z",
                "entry_reason": "x",
            },
        ],
    )

    result = find_first_divergence(tmp_path / "logs", tmp_path / "replay")

    assert result["divergence_found"] is True
    assert result["first_divergence"]["type"] == "side_mismatch"
    assert result["first_divergence"]["stream_index"] == 1
    assert result["first_divergence"]["symbol"] == "BTCUSDT"


def test_find_first_divergence_false_when_streams_match(tmp_path: Path) -> None:
    _write_orders(
        tmp_path / "logs" / "execution" / "orders_executed.jsonl",
        [
            {"symbol": "ETHUSDT", "side": "BUY", "timestamp": "2026-05-15T11:00:00Z"},
            {"symbol": "ETHUSDT", "side": "SELL", "timestamp": "2026-05-15T11:10:00Z"},
        ],
    )
    _write_trades(
        tmp_path / "replay" / "trades.csv",
        [
            {
                "symbol": "ETHUSDT",
                "side": "LONG",
                "qty": 1.0,
                "entry_price": 200.0,
                "entry_ts": "2026-05-15T11:00:00Z",
                "entry_reason": "x",
            },
            {
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "qty": 1.0,
                "entry_price": 199.0,
                "entry_ts": "2026-05-15T11:10:00Z",
                "entry_reason": "x",
            },
        ],
    )

    result = find_first_divergence(tmp_path / "logs", tmp_path / "replay")

    assert result["divergence_found"] is False
    assert result["first_divergence"] is None
