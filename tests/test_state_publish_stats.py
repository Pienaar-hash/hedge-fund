import json
from datetime import datetime, timezone, timedelta

import execution.state_publish as state_publish


def test_compute_exec_stats_uses_fill_events(tmp_path, monkeypatch) -> None:
    exec_dir = tmp_path
    monkeypatch.setattr(state_publish, "EXEC_LOG_DIR", exec_dir)
    monkeypatch.setattr(state_publish, "_EXEC_STATS_CACHE", {"ts": 0.0, "data": None})

    now = datetime.now(timezone.utc)
    attempts = [
        {"ts": now.isoformat(), "intent": "a"},
        {"ts": now.isoformat(), "intent": "b"},
    ]
    with (exec_dir / "orders_attempted.jsonl").open("w", encoding="utf-8") as handle:
        for row in attempts:
            handle.write(json.dumps(row) + "\n")

    ack_event = {
        "event_type": "order_ack",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "orderId": 1,
        "ts": now.isoformat(),
    }
    fill_event = {
        "event_type": "order_fill",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "orderId": 1,
        "executedQty": 0.1,
        "status": "FILLED",
        "ts": (now + timedelta(seconds=1)).isoformat(),
    }
    with (exec_dir / "orders_executed.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(ack_event) + "\n")
        handle.write(json.dumps(fill_event) + "\n")

    veto_event = {"ts": now.isoformat(), "reason": "test"}
    with (exec_dir / "risk_vetoes.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(veto_event) + "\n")

    stats = state_publish._compute_exec_stats()
    assert stats["attempted_24h"] == 2
    assert stats["executed_24h"] == 1
    assert stats["fill_rate"] == 0.5


def test_publish_positions_writes_state_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(state_publish, "LOG_DIR", tmp_path)
    monkeypatch.setattr(state_publish, "EXEC_LOG_DIR", tmp_path)
    monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state_publish, "_firestore_enabled", lambda: False)
    monkeypatch.setattr(state_publish, "_compute_exec_stats", lambda: {"attempted_24h": 0})

    rows = [{"symbol": "BTCUSDT", "positionSide": "LONG", "qty": 1.0, "entryPrice": 100.0, "leverage": 2.0, "uPnl": 0.0}]
    state_publish.publish_positions(rows)

    state_file = state_publish.STATE_DIR / "positions.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text())
    assert payload["rows"][0]["symbol"] == "BTCUSDT"
