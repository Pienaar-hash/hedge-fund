from __future__ import annotations

from execution import executor_live


def test_pipeline_shadow_heartbeat_runs(monkeypatch):
    monkeypatch.setattr(executor_live, "PIPELINE_V6_SHADOW_ENABLED", True, raising=False)
    executor_live._LAST_NAV_STATE = {"nav": 0.0, "nav_usd": 0.0, "portfolio_gross_usd": 0.0}
    executor_live._LAST_POSITIONS_STATE = {"positions": [{"symbol": "BTCUSDT"}]}
    monkeypatch.setattr(executor_live, "_PIPELINE_V6_HEARTBEAT_INTERVAL_S", 0.0, raising=False)
    monkeypatch.setattr(executor_live, "_LAST_PIPELINE_V6_HEARTBEAT", 0.0, raising=False)
    monkeypatch.setattr(executor_live, "_get_risk_engine_v6", lambda: None)

    recorded = []

    def fake_run(symbol, *args, **kwargs):
        return {"symbol": symbol, "risk_decision": {"allowed": True}, "timestamp": 0.0}

    def fake_record(result):
        recorded.append(result)

    monkeypatch.setattr(executor_live.pipeline_v6_shadow, "run_pipeline_v6_shadow", fake_run)
    monkeypatch.setattr(executor_live, "_record_shadow_decision", fake_record)
    executor_live._maybe_run_pipeline_v6_shadow_heartbeat()
    assert recorded and recorded[0]["symbol"] == "BTCUSDT"
