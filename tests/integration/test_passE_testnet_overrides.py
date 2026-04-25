
from execution import risk_loader


def test_testnet_overrides_activate(monkeypatch):
    risk_loader.load_risk_config.cache_clear()
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    monkeypatch.setenv("TESTNET_OVERRIDES_CONFIRM", "1")  # AUDIT-1.4c
    cfg = risk_loader.load_risk_config()
    global_cfg = cfg.get("global") or {}
    # v7: Config values are stored as fractions (0.95 = 95%)
    assert global_cfg.get("max_nav_drawdown_pct") == 0.95
    assert cfg.get("_meta", {}).get("testnet_overrides_active") is True
    risk_loader.load_risk_config.cache_clear()


def test_testnet_overrides_blocked_without_confirm(monkeypatch):
    """AUDIT-1.4c: overrides require TESTNET_OVERRIDES_CONFIRM=1."""
    risk_loader.load_risk_config.cache_clear()
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    monkeypatch.delenv("TESTNET_OVERRIDES_CONFIRM", raising=False)
    cfg = risk_loader.load_risk_config()
    # Overrides should NOT be applied
    assert cfg.get("_meta", {}).get("testnet_overrides_active") is not True
    risk_loader.load_risk_config.cache_clear()
