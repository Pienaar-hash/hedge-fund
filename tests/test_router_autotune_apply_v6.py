import json
import time
from datetime import datetime, timezone

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


def _write_allocator(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_missing_allocator_state_defaults_cautious(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(apply, "RISK_ALLOC_PATH", tmp_path / "risk_alloc_missing.json")
    monkeypatch.setattr(apply, "APPLY_ENABLED", True)
    monkeypatch.setattr(apply, "SYMBOL_ALLOWLIST", {"BTCUSDT"})
    caplog.set_level("WARNING")
    risk_mode = apply.get_current_risk_mode(now=1_000_000.0)
    assert risk_mode != "defensive"
    assert any("router_apply_no_allocator_state" in rec.getMessage() for rec in caplog.records)
    policy, changed, offset = apply.apply_router_suggestion(
        _policy(),
        suggestion=_suggestion(),
        symbol="BTCUSDT",
        risk_mode=risk_mode,
        current_offset_bps=2.0,
    )
    assert changed is True
    assert offset != 2.0


def test_stale_allocator_state_logs_and_allows(monkeypatch, tmp_path, caplog):
    alloc_path = tmp_path / "risk_alloc_stale.json"
    monkeypatch.setattr(apply, "RISK_ALLOC_PATH", alloc_path)
    monkeypatch.setattr(apply, "APPLY_ENABLED", True)
    monkeypatch.setattr(apply, "SYMBOL_ALLOWLIST", {"BTCUSDT"})
    monkeypatch.setattr(apply, "RISK_STATE_MAX_AGE_S", 10.0)
    stale_ts = time.time() - 30.0
    _write_allocator(
        alloc_path,
        {
            "generated_ts": datetime.fromtimestamp(stale_ts, tz=timezone.utc).isoformat(),
            "global": {"risk_mode": "defensive"},
        },
    )
    caplog.set_level("WARNING")
    risk_mode = apply.get_current_risk_mode(now=stale_ts + 31.0)
    assert risk_mode != "defensive"
    assert any("router_apply_stale_allocator_state" in rec.getMessage() for rec in caplog.records)
    policy, changed, _ = apply.apply_router_suggestion(
        _policy(),
        suggestion=_suggestion(),
        symbol="BTCUSDT",
        risk_mode=risk_mode,
        current_offset_bps=2.0,
    )
    assert changed is True


def test_fresh_defensive_allocator_blocks(monkeypatch, tmp_path):
    alloc_path = tmp_path / "risk_alloc_fresh.json"
    monkeypatch.setattr(apply, "RISK_ALLOC_PATH", alloc_path)
    monkeypatch.setattr(apply, "APPLY_ENABLED", True)
    monkeypatch.setattr(apply, "SYMBOL_ALLOWLIST", {"BTCUSDT"})
    now = time.time()
    _write_allocator(
        alloc_path,
        {
            "generated_ts": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "global": {"risk_mode": "defensive"},
        },
    )
    risk_mode = apply.get_current_risk_mode(now=now)
    assert risk_mode == "defensive"
    policy, changed, offset = apply.apply_router_suggestion(
        _policy(),
        suggestion=_suggestion(),
        symbol="BTCUSDT",
        risk_mode=risk_mode,
        current_offset_bps=2.0,
    )
    assert changed is False
    assert offset == 2.0
