import os

from execution import risk_loader


def test_testnet_overrides_activate(monkeypatch):
    risk_loader.load_risk_config.cache_clear()
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    cfg = risk_loader.load_risk_config()
    global_cfg = cfg.get("global") or {}
    # v7: Config values are stored as fractions (0.95 = 95%)
    assert global_cfg.get("max_nav_drawdown_pct") == 0.95
    assert cfg.get("_meta", {}).get("testnet_overrides_active") is True
    risk_loader.load_risk_config.cache_clear()
