import pytest

from execution.intel.router_autotune_shared import suggest_autotune_for_symbol
from execution.router_metrics import compute_maker_reliability


def _make_router_health(reliability: float) -> dict:
    return {
        "symbol_stats": {
            "BTCUSDT": {
                "maker_reliability": reliability,
            }
        }
    }


def _make_risk_snapshot(mode: str) -> dict:
    return {"risk_mode": mode}


def test_maker_reliability_computes_penalty():
    reliability = compute_maker_reliability(0.85, 0.1, 0.05)
    assert 0.0 <= reliability <= 1.0
    # Should subtract 0.3*fallback + 0.2*reject = 0.03 + 0.01 = 0.04
    assert pytest.approx(0.81, rel=1e-2) == reliability


def test_adaptive_offset_high_reliability():
    base = 5.0
    result = suggest_autotune_for_symbol(
        "BTCUSDT",
        base,
        router_health=_make_router_health(0.85),
        risk_snapshot=_make_risk_snapshot("OK"),
        min_offset_bps=0.5,
    )
    assert result["adaptive_offset_bps"] == pytest.approx(base)
    assert result["maker_first"] is True


def test_adaptive_offset_mid_reliability():
    base = 5.0
    result = suggest_autotune_for_symbol(
        "BTCUSDT",
        base,
        router_health=_make_router_health(0.65),
        risk_snapshot=_make_risk_snapshot("OK"),
        min_offset_bps=0.5,
    )
    assert result["adaptive_offset_bps"] == pytest.approx(base * 0.7)
    assert result["maker_first"] is True


def test_adaptive_offset_low_reliability_clamps():
    base = 5.0
    result = suggest_autotune_for_symbol(
        "BTCUSDT",
        base,
        router_health=_make_router_health(0.2),
        risk_snapshot=_make_risk_snapshot("OK"),
        min_offset_bps=0.5,
    )
    assert result["adaptive_offset_bps"] == pytest.approx(base * 0.4)
    assert result["maker_first"] is False


def test_risk_mode_warn_penalty():
    result = suggest_autotune_for_symbol(
        "BTCUSDT",
        4.0,
        router_health=_make_router_health(0.65),
        risk_snapshot=_make_risk_snapshot("WARN"),
        min_offset_bps=0.5,
    )
    assert result["maker_first"] is False  # effective reliability = 0.55


def test_risk_mode_defensive_requires_high_reliability():
    result = suggest_autotune_for_symbol(
        "BTCUSDT",
        4.0,
        router_health=_make_router_health(0.8),
        risk_snapshot=_make_risk_snapshot("DEFENSIVE"),
        min_offset_bps=0.5,
    )
    assert result["maker_first"] is True
    assert result["effective_reliability"] == pytest.approx(0.8)
    assert result["adaptive_offset_bps"] == pytest.approx(4.0 * 0.5)


def test_risk_mode_halted_disables_maker_first():
    result = suggest_autotune_for_symbol(
        "BTCUSDT",
        4.0,
        router_health=_make_router_health(0.9),
        risk_snapshot=_make_risk_snapshot("HALTED"),
        min_offset_bps=0.5,
    )
    assert result["maker_first"] is False


def test_missing_router_health_defaults_to_safe():
    result = suggest_autotune_for_symbol(
        "BTCUSDT",
        1.0,
        router_health={},
        risk_snapshot=_make_risk_snapshot("OK"),
        min_offset_bps=0.5,
    )
    assert result["maker_reliability"] == pytest.approx(0.0)
    assert result["maker_first"] is False
