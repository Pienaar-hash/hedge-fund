import time

import pytest

from execution import nav as navmod
from execution import risk_limits
from execution import sync_state
from execution.drawdown_tracker import load_peak_state
import execution.drawdown_tracker as drawdown_tracker


def test_enhanced_nav_mark_price_conversion(tmp_path, monkeypatch):
    cache_path = tmp_path / "nav_confirmed.json"
    monkeypatch.setattr(navmod, "_NAV_CACHE_PATH", str(cache_path))
    monkeypatch.setattr(navmod, "_NAV_LOG_PATH", str(tmp_path / "nav_log.json"))
    monkeypatch.setattr(navmod, "get_balances", lambda: [
        {"asset": "USDT", "balance": 100},
        {"asset": "USDC", "balance": 50},
        {"asset": "BTC", "balance": 0.1},
    ])
    monkeypatch.setattr(navmod, "get_positions", lambda: [{"unrealized": 10.0}])
    monkeypatch.setattr(navmod, "get_price", lambda symbol: 20000.0 if symbol == "BTCUSDT" else 0.0)

    nav_val, detail = navmod.compute_trading_nav(
        {"nav": {"include_assets": ["USDT", "USDC", "BTC"], "use_mark_price": True}}
    )

    assert nav_val == pytest.approx(2160.0, rel=1e-6)
    assert detail.get("nav_mode") == "enhanced"
    assert detail.get("mark_prices", {}).get("BTC") == 20000.0
    assert detail.get("freshness", {}).get("exchange_balances_fresh") is True


def test_drawdown_stale_state_does_not_veto(monkeypatch):
    now = time.time()
    monkeypatch.setattr(risk_limits, "PEAK_STATE_MAX_AGE_SEC", 5)
    monkeypatch.setattr(
        risk_limits,
        "load_peak_state",
        lambda: {"ts": now - 10, "peak": 5000, "nav": 1000, "dd_pct": 80.0},
    )
    monkeypatch.setattr(
        risk_limits,
        "get_confirmed_nav",
        lambda: {"nav": 1000.0, "ts": now - 1},
    )
    monkeypatch.setattr(
        risk_limits,
        "nav_health_snapshot",
        lambda *_a, **_k: {"age_s": 1.0, "sources_ok": True, "fresh": True, "stale_flags": {}},
    )
    monkeypatch.setattr(risk_limits, "_peak_from_nav_log", lambda *_a, **_k: 0.0)

    state = risk_limits.RiskState()
    cfg = {"global": {"max_nav_drawdown_pct": 5.0, "min_notional_usdt": 0.0, "peak_stale_seconds": 1}}
    veto, detail = risk_limits.check_order(
        "BTCUSDT",
        "BUY",
        requested_notional=50.0,
        price=25000.0,
        nav=1000.0,
        open_qty=0.0,
        now=now,
        cfg=cfg,
        state=state,
        current_gross_notional=0.0,
    )

    assert veto is False
    assert "drawdown_stale" in (detail.get("warnings") or [])


def test_testnet_peak_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    monkeypatch.setattr(drawdown_tracker, "PEAK_STATE_PATH", str(tmp_path / "cache" / "peak_state.json"))
    monkeypatch.setattr(sync_state, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(sync_state, "NAV_LOG", str(tmp_path / "nav_log.json"))
    monkeypatch.setattr(sync_state, "PEAK_STATE", str(tmp_path / "cache" / "peak_state.json"))
    monkeypatch.setattr(sync_state, "SYNCED_STATE", str(tmp_path / "state" / "synced_state.json"))
    monkeypatch.setattr(sync_state, "_export_dashboard_caches", lambda: None)
    monkeypatch.setattr(sync_state, "get_income_history", lambda *_a, **_k: [])
    monkeypatch.setattr(sync_state, "_runtime_risk_cfg", lambda: {"reset_peak_on_testnet": True})
    # Seed minimal nav log to satisfy tail guard
    (tmp_path / "nav_log.json").write_text('[{"t": 1, "nav": 500.0}]')

    nav_payload, _, _ = sync_state._sync_once_with_db(sync_state._noop_db())

    state = load_peak_state()
    assert state.get("peak_nav") == pytest.approx(nav_payload.get("series")[-1]["nav"])
    assert state.get("daily_peak") == state.get("peak_nav")


def test_peak_regeneration_when_stale(monkeypatch, tmp_path):
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    monkeypatch.setattr(drawdown_tracker, "PEAK_STATE_PATH", str(tmp_path / "cache" / "peak_state.json"))
    monkeypatch.setattr(sync_state, "LOGS_DIR", str(tmp_path))
    nav_log_path = tmp_path / "nav_log.json"
    nav_log_path.parent.mkdir(parents=True, exist_ok=True)
    nav_log_path.write_text('[{"t": 1, "nav": 100.0}, {"t": 2, "nav": 200.0}]')
    peak_path = tmp_path / "cache" / "peak_state.json"
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    peak_path.write_text('{"peak_nav": 10, "peak_ts": 0}')
    monkeypatch.setattr(sync_state, "NAV_LOG", str(nav_log_path))
    monkeypatch.setattr(sync_state, "PEAK_STATE", str(peak_path))
    monkeypatch.setattr(sync_state, "SYNCED_STATE", str(tmp_path / "state" / "synced_state.json"))
    monkeypatch.setattr(sync_state, "_export_dashboard_caches", lambda: None)
    monkeypatch.setattr(sync_state, "get_income_history", lambda *_a, **_k: [])
    monkeypatch.setattr(sync_state, "_runtime_risk_cfg", lambda: {"peak_stale_seconds": 1})

    nav_payload, _, _ = sync_state._sync_once_with_db(sync_state._noop_db())
    state = load_peak_state()
    assert state.get("peak_nav") == pytest.approx(200.0)
    assert nav_payload["series"][-1]["nav"] == pytest.approx(200.0)


def test_unified_drawdown_logic(monkeypatch):
    now = time.time()
    monkeypatch.setattr(
        risk_limits,
        "load_peak_state",
        lambda: {"peak_nav": 1000.0, "nav": 800.0, "peak_ts": now - 10, "ts": now - 10},
    )
    monkeypatch.setattr(
        risk_limits,
        "nav_health_snapshot",
        lambda *_a, **_k: {"age_s": 1.0, "sources_ok": True, "fresh": True},
    )
    monkeypatch.setattr(risk_limits, "get_confirmed_nav", lambda: {"nav": 800.0, "ts": now - 1})
    state = risk_limits.RiskState()
    cfg = {"global": {"max_nav_drawdown_pct": 10.0}}
    veto, detail = risk_limits.check_order(
        "ETHUSDT",
        "BUY",
        requested_notional=50.0,
        price=1500.0,
        nav=800.0,
        open_qty=0.0,
        now=now,
        cfg=cfg,
        state=state,
        current_gross_notional=0.0,
    )
    assert veto is True
    assert "nav_drawdown_limit" in detail.get("reasons", [])
    assert detail.get("drawdown", {}).get("pct") >= 20.0


def test_sync_state_uses_confirmed_nav_when_nav_log_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_state, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(sync_state, "NAV_LOG", str(tmp_path / "nav_log.json"))
    monkeypatch.setattr(sync_state, "PEAK_STATE", str(tmp_path / "cache" / "peak_state.json"))
    monkeypatch.setattr(sync_state, "SYNCED_STATE", str(tmp_path / "state" / "synced_state.json"))
    monkeypatch.setattr(sync_state, "_export_dashboard_caches", lambda: None)
    monkeypatch.setattr(sync_state, "_best_available_nav", lambda: 1234.56)

    nav_payload, pos_payload, _ = sync_state._sync_once_with_db(sync_state._noop_db())

    assert nav_payload
    assert nav_payload.get("series")
    assert nav_payload["series"][-1].get("nav") == pytest.approx(1234.56, rel=1e-6)
    assert pos_payload is not None
