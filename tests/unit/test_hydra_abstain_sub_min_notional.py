"""Unit tests for Hydra TREND min-notional abstain behavior."""

from execution.hydra_engine import HydraHeadConfig, generate_trend_intents


def _trend_cfg() -> HydraHeadConfig:
    return HydraHeadConfig(name="TREND", enabled=True, direction="both", max_nav_pct=0.50)


def test_abstains_when_projected_notional_below_floor() -> None:
    cfg = _trend_cfg()

    intents = generate_trend_intents(
        symbols=["SOLUSDT"],
        hybrid_scores={"SOLUSDT": 0.20},
        cerberus_multiplier=1.0,
        head_cfg=cfg,
        nav_usd=1000.0,
        base_nav_pct=0.02,
        min_notional_usd=10.0,
    )

    assert intents == []


def test_emits_when_projected_notional_meets_floor() -> None:
    cfg = _trend_cfg()

    intents = generate_trend_intents(
        symbols=["SOLUSDT"],
        hybrid_scores={"SOLUSDT": 0.60},
        cerberus_multiplier=1.0,
        head_cfg=cfg,
        nav_usd=1000.0,
        base_nav_pct=0.02,
        min_notional_usd=10.0,
    )

    assert len(intents) == 1
    assert intents[0].symbol == "SOLUSDT"
    assert intents[0].side == "long"


def test_emits_short_for_negative_score_on_non_inverted_symbol() -> None:
    cfg = _trend_cfg()

    intents = generate_trend_intents(
        symbols=["SOLUSDT"],
        hybrid_scores={"SOLUSDT": -0.60},
        cerberus_multiplier=1.0,
        head_cfg=cfg,
        nav_usd=1000.0,
        base_nav_pct=0.02,
        min_notional_usd=10.0,
    )

    assert len(intents) == 1
    assert intents[0].symbol == "SOLUSDT"
    assert intents[0].side == "short"


def test_floor_disabled_preserves_legacy_emission() -> None:
    cfg = _trend_cfg()

    intents = generate_trend_intents(
        symbols=["SOLUSDT"],
        hybrid_scores={"SOLUSDT": 0.20},
        cerberus_multiplier=1.0,
        head_cfg=cfg,
        nav_usd=1000.0,
        base_nav_pct=0.02,
        min_notional_usd=0.0,
    )

    assert len(intents) == 1
    assert intents[0].symbol == "SOLUSDT"
