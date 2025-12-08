import json
import time
from pathlib import Path

from execution.mirror_builders import build_mirror_payloads


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_mirror_payloads_smoke(tmp_path):
    now = time.time()
    exec_dir = tmp_path / "logs" / "execution"
    attempts = [
        {"ts": now - 60, "symbol": "BTCUSDT", "signal": "BUY", "attempt_id": "a1", "doctor": {"confidence": 0.8}},
        {"ts": now - 300, "symbol": "ETHUSDT", "signal": "SELL", "attempt_id": "a2"},
    ]
    executed = [
        {
            "ts": now - 50,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "executedQty": 0.01,
            "avgPrice": 30000,
            "attempt_id": "a1",
        },
        {
            "ts": now - 200,
            "symbol": "ETHUSDT",
            "status": "FILLED",
            "executedQty": 0.5,
            "avgPrice": 2000,
            "attempt_id": "a2",
        },
    ]
    vetoes = [
        {"ts": now - 100, "symbol": "SOLUSDT", "reason": "risk_limit"},
    ]

    _write_jsonl(exec_dir / "orders_attempted.jsonl", attempts)
    _write_jsonl(exec_dir / "orders_executed.jsonl", executed)
    _write_jsonl(exec_dir / "risk_vetoes.jsonl", vetoes)

    payloads = build_mirror_payloads(tmp_path / "logs")

    assert payloads.router, "router payloads should not be empty"
    assert payloads.router[0].get("kind") == "summary"
    assert any(item.get("kind") != "summary" for item in payloads.router[1:])
    assert payloads.trades, "trade payloads should not be empty"
    assert payloads.signals, "signal payloads should not be empty"
    assert payloads.signals[0].get("kind") == "signals_summary"
    assert any(item.get("symbol") for item in payloads.signals[1:])


def test_build_mirror_payloads_signal_fallback(tmp_path):
    now = time.time()
    exec_dir = tmp_path / "logs" / "execution"
    signal_metrics = [
        {"ts": now - 120, "symbol": "XRPUSDT", "signal": "BUY", "attempt_id": "m1", "doctor": {"confidence": 0.6}},
        {"ts": now - 90, "symbol": "DOGEUSDT", "signal": "SELL", "attempt_id": "m2"},
    ]
    _write_jsonl(exec_dir / "signal_metrics.jsonl", signal_metrics)
    payloads = build_mirror_payloads(tmp_path / "logs")
    assert payloads.signals, "signals fallback should yield entries"
    assert payloads.signals[0].get("kind") == "signals_summary"
    assert len(payloads.signals) > 1
