from __future__ import annotations

from execution import executor_live


def test_pub_tick_writes_state(monkeypatch):
    nav_values = []
    positions_values = []
    synced_values = []
    monkeypatch.setattr(executor_live, "_compute_nav_snapshot", lambda: 123.45)
    monkeypatch.setattr(
        executor_live,
        "_collect_rows",
        lambda: [{"symbol": "BTCUSDT", "qty": 1.0, "positionSide": "LONG"}],
    )
    monkeypatch.setattr(executor_live, "_persist_positions_cache", lambda _rows: None)
    monkeypatch.setattr(executor_live, "_persist_nav_log", lambda *_args: None)
    monkeypatch.setattr(executor_live, "_persist_spot_state", lambda: None)
    monkeypatch.setattr(executor_live, "write_nav_state", lambda payload: nav_values.append(payload))
    monkeypatch.setattr(
        executor_live,
        "write_positions_state",
        lambda payload: positions_values.append(payload),
    )
    monkeypatch.setattr(
        executor_live,
        "write_synced_state",
        lambda payload: synced_values.append(payload),
    )
    executor_live._pub_tick()
    assert nav_values and nav_values[0]["nav"] == 123.45
    assert positions_values and positions_values[0]["rows"][0]["symbol"] == "BTCUSDT"
    assert synced_values and synced_values[0]["positions"][0]["symbol"] == "BTCUSDT"
    assert executor_live._LAST_NAV_STATE["nav"] == 123.45
    assert executor_live._LAST_POSITIONS_STATE["positions"][0]["symbol"] == "BTCUSDT"
