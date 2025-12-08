import json
import time

import execution.router_metrics as rm


def test_get_recent_router_events_reads_jsonl_execution_hardening(tmp_path, monkeypatch):
    p = tmp_path / "order_metrics.jsonl"

    now = int(time.time())
    old = now - 10 * 24 * 3600

    rows = [
        {"symbol": "BTCUSDC", "ts": now, "is_maker_final": True},
        {"symbol": "ETHUSDC", "ts": now, "is_maker_final": False},
        {"symbol": "OLDUSDC", "ts": old, "is_maker_final": True},
    ]

    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    if hasattr(rm, "ROUTER_METRICS_PATH"):
        monkeypatch.setattr(rm, "ROUTER_METRICS_PATH", p)
    else:
        monkeypatch.setattr(rm, "ORDER_METRICS_PATH", p, raising=False)

    recent = rm.get_recent_router_events(symbol="BTCUSDC", window_days=7)
    assert len(recent) == 1
    assert recent[0]["symbol"] == "BTCUSDC"

    all_recent = rm.get_recent_router_events(symbol=None, window_days=7)
    assert {row["symbol"] for row in all_recent} == {"BTCUSDC", "ETHUSDC"}
