from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from execution.alpha_decay import AlphaDecayState, SymbolDecayStats
from execution.doctrine_bridge import resolve_doctrine_entry_inputs
from execution.hydra_engine import HydraState


def _iso_now(offset_seconds: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


@pytest.mark.unit
def test_resolve_doctrine_inputs_uses_real_hydra_and_alpha(tmp_path) -> None:
    hydra_path = tmp_path / "hydra_state.json"
    alpha_path = tmp_path / "alpha_decay.json"
    hydra_state = HydraState(
        updated_ts=_iso_now(),
        head_budgets={"TREND": 0.9, "CARRY": 0.2},
        head_usage={"TREND": 0.2, "CARRY": 0.05},
    )
    alpha_state = AlphaDecayState(
        updated_ts=_iso_now(),
        avg_symbol_survival=0.55,
        overall_alpha_health=0.73,
        symbols={
            "BTCUSDT": SymbolDecayStats(
                symbol="BTCUSDT",
                decay_rate=-0.01,
                half_life=42.0,
                survival_prob=0.91,
                deterioration_prob=0.09,
                ema_edge_score=0.42,
                last_edge_score=0.18,
            )
        },
    )
    hydra_path.write_text(json.dumps(hydra_state.to_dict()), encoding="utf-8")
    alpha_path.write_text(json.dumps(alpha_state.to_dict()), encoding="utf-8")

    resolved = resolve_doctrine_entry_inputs(
        symbol="BTCUSDT",
        head="TREND",
        total_exposure_pct=0.31,
        drawdown_pct=0.04,
        risk_mode="OK",
        hydra_path=hydra_path,
        alpha_decay_path=alpha_path,
    )

    assert not resolved.degraded
    assert resolved.portfolio is not None
    assert resolved.alpha_health is not None
    assert resolved.portfolio.head_budget_remaining["TREND"] == pytest.approx(0.7)
    assert resolved.alpha_health.survival_probability == pytest.approx(0.91)
    assert resolved.telemetry["healthy"] is True
    assert resolved.telemetry["head_budget_remaining"] == pytest.approx(0.7)
    assert resolved.telemetry["alpha_survival_probability"] == pytest.approx(0.91)


@pytest.mark.unit
def test_resolve_doctrine_inputs_fails_closed_on_stale_hydra(tmp_path) -> None:
    hydra_path = tmp_path / "hydra_state.json"
    alpha_path = tmp_path / "alpha_decay.json"
    hydra_state = HydraState(
        updated_ts=_iso_now(offset_seconds=-1800),
        head_budgets={"TREND": 0.9},
        head_usage={"TREND": 0.1},
    )
    alpha_state = AlphaDecayState(updated_ts=_iso_now(), overall_alpha_health=0.8)
    hydra_path.write_text(json.dumps(hydra_state.to_dict()), encoding="utf-8")
    alpha_path.write_text(json.dumps(alpha_state.to_dict()), encoding="utf-8")

    resolved = resolve_doctrine_entry_inputs(
        symbol="BTCUSDT",
        head="TREND",
        total_exposure_pct=0.31,
        drawdown_pct=0.04,
        risk_mode="OK",
        hydra_path=hydra_path,
        alpha_decay_path=alpha_path,
    )

    assert resolved.degraded
    assert resolved.degraded_details is not None
    assert resolved.degraded_details["code"] == "HYDRA_STATE_STALE"
    assert resolved.degraded_details["dependency"] == "hydra"


@pytest.mark.unit
def test_resolve_doctrine_inputs_fails_closed_on_missing_alpha_state(tmp_path) -> None:
    hydra_path = tmp_path / "hydra_state.json"
    hydra_state = HydraState(
        updated_ts=_iso_now(),
        head_budgets={"TREND": 0.9},
        head_usage={"TREND": 0.1},
    )
    hydra_path.write_text(json.dumps(hydra_state.to_dict()), encoding="utf-8")

    resolved = resolve_doctrine_entry_inputs(
        symbol="BTCUSDT",
        head="TREND",
        total_exposure_pct=0.31,
        drawdown_pct=0.04,
        risk_mode="OK",
        hydra_path=hydra_path,
        alpha_decay_path=tmp_path / "missing_alpha_decay.json",
    )

    assert resolved.degraded
    assert resolved.degraded_details is not None
    assert resolved.degraded_details["code"] == "ALPHA_DECAY_STATE_MISSING"
    assert resolved.degraded_details["dependency"] == "alpha_decay"
