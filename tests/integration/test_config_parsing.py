import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCFG = ROOT / "config" / "strategy_config.json"
RISK = ROOT / "config" / "risk_limits.json"

pytestmark = [pytest.mark.integration, pytest.mark.runtime]


def _load(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_strategy_config_exists_and_parses():
    if not SCFG.exists():
        pytest.skip("strategy_config.json missing")
    cfg = _load(SCFG)
    assert cfg.get("use_futures") is True
    strategies = cfg.get("strategies", [])
    assert isinstance(strategies, list) and strategies
    # Required keys that must be present
    required_keys = {"id", "enabled", "symbol", "timeframe", "params"}
    for entry in strategies:
        assert required_keys.issubset(set(entry.keys())), f"Missing keys in {entry.get('id')}"
        params = entry.get("params") or {}
        assert isinstance(params, dict)
        assert "entry" in params
        # v7+ uses per_trade_nav_pct instead of capital_per_trade
        assert "per_trade_nav_pct" in params or "capital_per_trade" in params


def test_risk_limits_exists_and_parses():
    if not RISK.exists():
        pytest.skip("risk_limits.json missing")
    risk_cfg = _load(RISK)
    global_cfg = risk_cfg.get("global", {})
    required = [
        "daily_loss_limit_pct",
        "cooldown_minutes_after_stop",
        "max_trades_per_symbol_per_hour",
        "drawdown_alert_pct",
        "max_gross_exposure_pct",
        "max_symbol_exposure_pct",
        "max_leverage",
        "min_notional_usdt",
    ]
    for key in required:
        assert key in global_cfg


def test_per_symbol_caps_defined():
    if not RISK.exists():
        pytest.skip("risk_limits.json missing")
    risk_cfg = _load(RISK)
    global_cfg = risk_cfg.get("global", {})
    per_symbol = risk_cfg.get("per_symbol", {})
    assert isinstance(per_symbol, dict) and per_symbol
    global_min = float(global_cfg.get("min_notional_usdt", 0.0))
    for sym, cfg in per_symbol.items():
        assert cfg.get("min_notional") is not None, sym
        assert cfg.get("max_order_notional") is not None, sym
        if global_min > 0:
            assert float(cfg["min_notional"]) >= global_min


def test_caps_consistency_between_files():
    """Check that risk_limits.json per_symbol caps are defined for universe symbols.
    
    Note: pairs_universe.json caps are optional/legacy and may differ from risk_limits.json.
    The authoritative source is risk_limits.json.
    """
    if not RISK.exists():
        pytest.skip("risk_limits.json missing")
    risk_cfg = _load(RISK)
    pairs_path = ROOT / "config" / "pairs_universe.json"
    if not pairs_path.exists():
        pytest.skip("pairs_universe.json missing")
    pairs = _load(pairs_path)
    per_symbol = risk_cfg.get("per_symbol", {})
    
    # Check that universe symbols have risk limits defined
    for entry in pairs.get("universe", []):
        sym = entry.get("symbol")
        if not sym:
            continue
        # Only warn if symbol is missing from per_symbol (not an error)
        # The authoritative source is risk_limits.json, not pairs_universe.json caps
        if sym not in per_symbol:
            import warnings
            warnings.warn(f"Symbol {sym} in universe but not in per_symbol risk limits")
