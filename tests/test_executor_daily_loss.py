import json
from datetime import datetime, timezone

import pytest


def test_executor_blocks_when_daily_loss_limit_triggered(monkeypatch, tmp_path):
    from execution import executor_live as ex
    from execution import risk_limits as rl
    from execution import telegram_utils as tg

    nav_path = tmp_path / "nav_log.json"
    now = datetime.now(timezone.utc)
    open_row = {
        "timestamp": (now.replace(hour=0, minute=0, second=0, microsecond=0)).isoformat(),
        "nav": 1200.0,
    }
    later_row = {
        "timestamp": (now.replace(hour=1, minute=0, second=0, microsecond=0)).isoformat(),
        "nav": 1180.0,
    }
    nav_path.write_text(json.dumps([open_row, later_row]), encoding="utf-8")

    monkeypatch.setenv("NAV_LOG_PATH", str(nav_path))
    monkeypatch.setattr(ex, "NAV_LOG_PATH", str(nav_path), raising=False)
    ex._DAILY_OPEN_NAV_CACHE.clear()

    captured: dict[str, object] = {}

    def fake_save_json(path: str, payload: dict) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(ex, "save_json", fake_save_json)
    monkeypatch.setattr(ex, "publish_order_audit", lambda *a, **k: None)
    monkeypatch.setattr(ex, "publish_intent_audit", lambda *a, **k: None)
    monkeypatch.setattr(ex, "publish_close_audit", lambda *a, **k: None)
    monkeypatch.setattr(ex._PORTFOLIO_SNAPSHOT, "refresh", lambda: None)
    monkeypatch.setattr(ex, "get_positions", lambda: [])
    monkeypatch.setattr(ex._RISK_GATE, "allowed_gross_notional", lambda *a, **k: (True, ""))
    monkeypatch.setattr(ex, "build_order_payload", lambda **kw: ({}, {}))
    monkeypatch.setattr(ex, "send_order", lambda *a, **k: {})
    monkeypatch.setattr(ex, "load_json", lambda *a, **k: {})
    monkeypatch.setattr(ex, "_compute_nav", lambda: 1100.0)
    monkeypatch.setattr(tg, "send_telegram", lambda *a, **k: None)

    monkeypatch.setattr(ex, "_RISK_CFG", {"global": {"daily_loss_limit_pct": 5.0}})

    ex._RISK_STATE.daily_pnl_pct = None

    def wrapped_check_order(*args, **kwargs):
        ok, details = rl.check_order(*args, **kwargs)
        captured["details"] = details
        captured["state"] = kwargs.get("state")
        return ok, details

    monkeypatch.setattr(ex, "check_order", wrapped_check_order)

    intent = {
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "capital_per_trade": 10.0,
        "leverage": 1.0,
        "gross_usd": 10.0,
    }

    ex._send_order(intent)

    assert "payload" in captured
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload.get("reason") == "day_loss_limit"
    assert payload.get("reasons") and payload["reasons"][0] == "day_loss_limit"

    details = captured.get("details")
    assert isinstance(details, dict)
    assert details.get("daily_pnl_pct") == pytest.approx(-8.3333333333, rel=1e-6)
    assert details.get("limit_pct") == pytest.approx(5.0)

    state = captured.get("state")
    assert state is ex._RISK_STATE
    assert state.daily_pnl_pct == pytest.approx(-8.3333333333, rel=1e-6)
