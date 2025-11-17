import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCFG = ROOT / "config" / "strategy_config.json"
RISK = ROOT / "config" / "risk_limits.json"


def _load(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_strategy_config_exists_and_parses():
    assert SCFG.exists()
    cfg = _load(SCFG)
    assert cfg.get("use_futures") is True
    strategies = cfg.get("strategies", [])
    assert isinstance(strategies, list) and strategies
    allowed_keys = {"id", "label", "enabled", "symbol", "timeframe", "tags", "params"}
    for entry in strategies:
        assert set(entry.keys()).issubset(allowed_keys)
        params = entry.get("params") or {}
        assert isinstance(params, dict)
        assert "entry" in params
        assert "capital_per_trade" in params


def test_risk_limits_exists_and_parses():
    assert RISK.exists()
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
    risk_cfg = _load(RISK)
    pairs = _load(ROOT / "config" / "pairs_universe.json")
    per_symbol = risk_cfg.get("per_symbol", {})
    for entry in pairs.get("universe", []):
        sym = entry.get("symbol")
        caps = entry.get("caps", {})
        if not sym or sym not in per_symbol:
            continue
        sym_caps = per_symbol[sym]
        if "max_nav_pct" in caps:
            assert caps["max_nav_pct"] == sym_caps.get("max_nav_pct")
        if "max_order_notional" in caps:
            assert caps["max_order_notional"] == sym_caps.get("max_order_notional")
