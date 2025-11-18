from execution.intel import router_autotune_apply_v6 as apply


def _policy(**overrides):
    base = {"maker_first": True, "taker_bias": "balanced", "quality": "good", "offset_bps": 2.0}
    base.update(overrides)
    return base


def _suggestion(**overrides):
    base = {"proposed_policy": {"maker_first": False, "taker_bias": 0.2, "offset_bps": 4.0}}
    base.update(overrides)
    return base


def test_apply_router_suggestion_flag_off(monkeypatch):
    monkeypatch.setattr(apply, "APPLY_ENABLED", False)
    policy, changed, offset = apply.apply_router_suggestion(_policy(), suggestion=_suggestion(), symbol="BTCUSDT", risk_mode="normal", current_offset_bps=2.0)
    assert changed is False
    assert policy["taker_bias"] == "balanced"
    assert offset == 2.0


def test_apply_router_suggestion_applies_within_bounds(monkeypatch):
    monkeypatch.setattr(apply, "APPLY_ENABLED", True)
    monkeypatch.setattr(apply, "SYMBOL_ALLOWLIST", {"BTCUSDT"})
    monkeypatch.setattr(apply, "MAX_BIAS_DELTA", 0.05)
    monkeypatch.setattr(apply, "MAX_OFFSET_STEP_BPS", 1.0)
    monkeypatch.setattr(apply, "MAX_OFFSET_ABS_BPS", 10.0)
    monkeypatch.setattr(apply, "ALLOW_MAKER_FLIP", True)
    policy, changed, offset = apply.apply_router_suggestion(
        _policy(),
        suggestion=_suggestion(),
        symbol="BTCUSDT",
        risk_mode="normal",
        current_offset_bps=2.0,
    )
    assert changed is True
    assert policy["taker_bias"] in {"prefer_maker", "balanced"}
    assert offset == 3.0
    assert policy["maker_first"] is False


def test_apply_router_suggestion_defensive_mode(monkeypatch):
    monkeypatch.setattr(apply, "APPLY_ENABLED", True)
    monkeypatch.setattr(apply, "SYMBOL_ALLOWLIST", {"BTCUSDT"})
    policy, changed, offset = apply.apply_router_suggestion(
        _policy(),
        suggestion=_suggestion(),
        symbol="BTCUSDT",
        risk_mode="defensive",
        current_offset_bps=2.0,
    )
    assert changed is False
    assert policy["maker_first"] is True
    assert offset == 2.0


def test_apply_router_suggestion_quality_filter(monkeypatch):
    monkeypatch.setattr(apply, "APPLY_ENABLED", True)
    monkeypatch.setattr(apply, "SYMBOL_ALLOWLIST", {"BTCUSDT"})
    monkeypatch.setattr(apply, "REQUIRE_QUALITY", {"good"})
    policy, changed, offset = apply.apply_router_suggestion(
        _policy(quality="broken"),
        suggestion=_suggestion(),
        symbol="BTCUSDT",
        risk_mode="normal",
        current_offset_bps=2.0,
    )
    assert changed is False
    assert offset == 2.0
