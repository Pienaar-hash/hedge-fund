import os

from execution import risk_loader


def test_testnet_overrides_activate(monkeypatch):
    risk_loader.load_risk_config.cache_clear()
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    cfg = risk_loader.load_risk_config()
    global_cfg = cfg.get("global") or {}
    assert global_cfg.get("max_nav_drawdown_pct") == 95.0
    assert cfg.get("_meta", {}).get("testnet_overrides_active") is True
    risk_loader.load_risk_config.cache_clear()
