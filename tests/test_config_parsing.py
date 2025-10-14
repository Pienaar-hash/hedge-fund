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
    universe = cfg.get("universe", [])
    assert isinstance(universe, list) and len(universe) >= 1
    sizing = cfg.get("sizing", {})
    assert sizing.get("capital_per_trade_usdt") is not None
    assert sizing.get("default_leverage") is not None
    risk = cfg.get("risk", {})
    assert risk.get("daily_loss_limit_pct") is not None


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
    ]
    for key in required:
        assert key in global_cfg


def test_caps_consistency_between_files():
    cfg = _load(SCFG)
    risk_cfg = _load(RISK)
    global_cfg = risk_cfg.get("global", {})
    sizing = cfg.get("sizing", {})
    risk = cfg.get("risk", {})
    assert global_cfg["max_gross_exposure_pct"] <= sizing["max_gross_exposure_pct"]
    assert global_cfg["max_symbol_exposure_pct"] <= sizing["max_symbol_exposure_pct"]
    assert global_cfg["daily_loss_limit_pct"] <= risk["daily_loss_limit_pct"]
