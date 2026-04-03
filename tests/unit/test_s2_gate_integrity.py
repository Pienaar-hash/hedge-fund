"""
S2 Gate Integrity — Edge threshold boundary tests.

Verifies that the eligibility gate's executable_edge check behaves correctly
at exact boundary conditions.  The contract:

    executable_edge <  min_edge_threshold  →  REJECTED  (NO_TRADE)
    executable_edge >= min_edge_threshold  →  PASS      (TRADE, if all other gates pass)

Gate 3 uses strict less-than: ``signal.executable_edge < min_edge``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from execution.binary_lab_s2_signals import (
    BinaryLabS2Signal,
    QUOTE_RECONSTRUCTION_MODE,
    _price_region,
    check_s2_eligibility,
)


# ---------------------------------------------------------------------------
# Shared limits (mirrors _TEST_LIMITS in smoke tests)
# ---------------------------------------------------------------------------
_TEST_LIMITS: Dict[str, Any] = {
    "_meta": {"sleeve_id": "binary_lab_s2"},
    "capital": {"sleeve_total_usd": 900, "per_round_usd": 30},
    "position_rules": {"max_concurrent": 3},
    "kill_conditions": {"kill_nav_usd": 650},
    "entry_gate": {
        "min_edge_threshold": 0.03,
        "max_spread_threshold": 0.04,
        "min_time_remaining_s": 120,
        "max_quote_age_s": 75,
        "signal_source": "probability_model",
    },
}

_COMMON_GATE_KWARGS = dict(
    current_nav_usd=900.0,
    open_positions=0,
    freeze_intact=True,
    time_remaining_s=600.0,
)


def _make_signal(
    *,
    p_yes_mid: float = 0.48,
    spread: float = 0.02,
    p_model_yes: float = 0.55,
    trade_side: str = "YES",
    skip_reason: Optional[str] = None,
    quote_age_s: float = 10.0,
    calibration_active: bool = False,
    calibration_confident: bool = False,
) -> BinaryLabS2Signal:
    """Deterministic signal builder — identical to smoke-test helper."""
    half = spread / 2.0
    p_yes_bid = p_yes_mid - half
    p_yes_ask = p_yes_mid + half
    p_no_bid = 1.0 - p_yes_ask
    p_no_ask = 1.0 - p_yes_bid

    edge_yes = p_model_yes - p_yes_mid

    if trade_side == "YES":
        entry_cost = p_yes_ask
        executable_edge = p_model_yes - entry_cost
    elif trade_side == "NO":
        entry_cost = p_no_ask
        executable_edge = entry_cost - p_model_yes
    else:
        entry_cost = 0.0
        executable_edge = 0.0

    return BinaryLabS2Signal(
        p_yes_bid=round(p_yes_bid, 6),
        p_yes_ask=round(p_yes_ask, 6),
        p_yes_mid=round(p_yes_mid, 6),
        p_no_bid=round(p_no_bid, 6),
        p_no_ask=round(p_no_ask, 6),
        spread=round(spread, 6),
        depth_score=0.5,
        quote_age_s=quote_age_s,
        quote_reconstruction_mode=QUOTE_RECONSTRUCTION_MODE,
        p_baseline_yes=round(p_yes_mid, 6),
        p_model_yes=round(p_model_yes, 6),
        edge_yes=round(edge_yes, 6),
        baseline_edge=0.0,
        entry_cost=round(entry_cost, 6),
        executable_edge=round(executable_edge, 6),
        trade_side=trade_side,
        skip_reason=skip_reason,
        expected_value_usd=round(executable_edge * 30.0, 4) if trade_side != "SKIP" else 0.0,
        calibration_active=calibration_active,
        calibration_confident=calibration_confident,
        price_region=_price_region(p_yes_mid),
        features={"p_yes_mid": p_yes_mid, "spread": spread},
        model_version="s2_naive_v1",
        ts=datetime.now(timezone.utc).isoformat(),
    )


# =========================================================================
# Edge threshold boundary tests
# =========================================================================

class TestEdgeThresholdBoundary:
    """Gate 3: executable_edge < min_edge_threshold → REJECTED."""

    # With p_yes_mid=0.48, spread=0.02 → entry_cost=0.49
    # executable_edge = p_model_yes - 0.49
    # For edge=0.03:  p_model_yes = 0.52
    # For edge=0.029: p_model_yes = 0.519
    # For edge=0.031: p_model_yes = 0.521

    def test_edge_exactly_at_threshold_passes(self) -> None:
        """executable_edge == 0.03 must PASS (gate uses strict <)."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.52)
        assert signal.executable_edge == pytest.approx(0.03, abs=1e-9)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible, f"Edge at threshold rejected: {result.deny_reason}"

    def test_edge_one_tick_below_threshold_rejected(self) -> None:
        """executable_edge = 0.029 must be REJECTED."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.519)
        assert signal.executable_edge == pytest.approx(0.029, abs=1e-9)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "executable_edge_below_min" in (result.deny_reason or "")

    def test_edge_one_tick_above_threshold_passes(self) -> None:
        """executable_edge = 0.031 must PASS."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.521)
        assert signal.executable_edge == pytest.approx(0.031, abs=1e-9)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible, f"Edge above threshold rejected: {result.deny_reason}"

    def test_zero_edge_rejected(self) -> None:
        """executable_edge = 0 must be REJECTED."""
        signal = _make_signal(p_yes_mid=0.50, spread=0.02, p_model_yes=0.51)
        assert signal.executable_edge == pytest.approx(0.0, abs=1e-9)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "executable_edge_below_min" in (result.deny_reason or "")

    def test_negative_edge_rejected(self) -> None:
        """Negative edge (model worse than market) must be REJECTED."""
        signal = _make_signal(p_yes_mid=0.50, spread=0.02, p_model_yes=0.48)
        assert signal.executable_edge < 0
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "executable_edge_below_min" in (result.deny_reason or "")

    def test_large_edge_passes(self) -> None:
        """Large edge (10pp) must PASS all gates."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.59)
        assert signal.executable_edge == pytest.approx(0.10, abs=1e-9)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible


class TestEdgeThresholdNOSide:
    """Same boundary contract for trade_side='NO'."""

    # For NO side: entry_cost = p_no_ask = 1 - p_yes_bid = 1 - (p_yes_mid - spread/2)
    # executable_edge = entry_cost - p_model_yes
    # With p_yes_mid=0.55, spread=0.02:
    #   p_yes_bid = 0.54 → p_no_ask = 0.46 (entry_cost for NO)
    #   executable_edge = 0.46 - p_model_yes
    #   For edge=0.03:  p_model_yes = 0.43
    #   For edge=0.029: p_model_yes = 0.431
    #   For edge=0.031: p_model_yes = 0.429

    def test_no_side_at_threshold_passes(self) -> None:
        signal = _make_signal(
            p_yes_mid=0.55, spread=0.02, p_model_yes=0.43, trade_side="NO",
        )
        assert signal.executable_edge == pytest.approx(0.03, abs=1e-9)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible

    def test_no_side_below_threshold_rejected(self) -> None:
        signal = _make_signal(
            p_yes_mid=0.55, spread=0.02, p_model_yes=0.431, trade_side="NO",
        )
        assert signal.executable_edge == pytest.approx(0.029, abs=1e-9)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "executable_edge_below_min" in (result.deny_reason or "")


class TestEdgeSkipSide:
    """trade_side='SKIP' produces zero edge and is rejected by gate 9."""

    def test_skip_side_zero_edge(self) -> None:
        signal = _make_signal(trade_side="SKIP", skip_reason="no_edge")
        assert signal.executable_edge == 0.0
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible

    def test_skip_side_reason_preserved(self) -> None:
        signal = _make_signal(trade_side="SKIP", skip_reason="no_edge")
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        # Gate 3 (edge) fires before gate 9 (SKIP), so reason is edge-based
        assert not result.eligible


class TestEdgeWithDifferentThresholds:
    """Verify the gate respects non-default threshold values."""

    def test_custom_threshold_1pp(self) -> None:
        limits = {**_TEST_LIMITS, "entry_gate": {**_TEST_LIMITS["entry_gate"], "min_edge_threshold": 0.01}}
        # edge=0.02 should pass with 1pp threshold
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.51)
        assert signal.executable_edge == pytest.approx(0.02, abs=1e-9)
        result = check_s2_eligibility(signal, limits, **_COMMON_GATE_KWARGS)
        assert result.eligible

    def test_custom_threshold_5pp(self) -> None:
        limits = {**_TEST_LIMITS, "entry_gate": {**_TEST_LIMITS["entry_gate"], "min_edge_threshold": 0.05}}
        # edge=0.03 should be rejected with 5pp threshold
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.52)
        assert signal.executable_edge == pytest.approx(0.03, abs=1e-9)
        result = check_s2_eligibility(signal, limits, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "executable_edge_below_min" in (result.deny_reason or "")


class TestGateOrderIndependence:
    """When multiple gates fail, the first gate in sequence wins."""

    def test_spread_checked_before_edge(self) -> None:
        """Spread gate (2) fires before edge gate (3)."""
        signal = _make_signal(spread=0.06, p_model_yes=0.49)
        # Both spread and edge should fail, but spread reason appears
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "spread_too_wide" in (result.deny_reason or "")

    def test_edge_checked_before_quote_age(self) -> None:
        """Edge gate (3) fires before quote_age gate (4)."""
        signal = _make_signal(p_model_yes=0.49, quote_age_s=200.0)
        # Both edge and quote_age should fail, but edge fires first
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "executable_edge_below_min" in (result.deny_reason or "")


# =========================================================================
# Ablation gate tests (Gate 10-11)
# =========================================================================

_ABL_LIMITS = {
    **_TEST_LIMITS,
    "ablation_gate": {
        "enabled": True,
        "min_edge_abs": 0.10,
        "side_filter": "ALL",
        "fallback_min_edge_abs": 0.07,
    },
}


class TestAblationEdgeGate:
    """Gate 10: |edge_yes| < ablation_min_edge_abs → REJECTED."""

    def test_edge_above_threshold_passes(self) -> None:
        """edge_yes=0.11 passes ablation gate at 0.10."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.59)
        assert abs(signal.edge_yes) == pytest.approx(0.11, abs=1e-6)
        result = check_s2_eligibility(signal, _ABL_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible, f"Unexpected rejection: {result.deny_reason}"

    def test_edge_below_threshold_rejected(self) -> None:
        """edge_yes=0.07 rejected by ablation gate at 0.10."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.55)
        assert abs(signal.edge_yes) == pytest.approx(0.07, abs=1e-6)
        result = check_s2_eligibility(signal, _ABL_LIMITS, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "ablation_edge_below_min" in (result.deny_reason or "")

    def test_edge_at_threshold_passes(self) -> None:
        """edge_yes=0.10 passes (gate uses strict <)."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.58)
        assert abs(signal.edge_yes) == pytest.approx(0.10, abs=1e-6)
        result = check_s2_eligibility(signal, _ABL_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible, f"Edge at threshold rejected: {result.deny_reason}"

    def test_no_side_edge_uses_abs(self) -> None:
        """NO-side: edge_yes=-0.12 → |edge_yes|=0.12 passes at 0.10."""
        signal = _make_signal(
            p_yes_mid=0.55, spread=0.02, p_model_yes=0.43, trade_side="NO",
        )
        assert signal.edge_yes < 0
        assert abs(signal.edge_yes) == pytest.approx(0.12, abs=1e-6)
        result = check_s2_eligibility(signal, _ABL_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible

    def test_disabled_skips_ablation_gate(self) -> None:
        """With enabled=False, low edge passes."""
        limits = {**_TEST_LIMITS, "ablation_gate": {"enabled": False, "min_edge_abs": 0.10}}
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.52)
        assert abs(signal.edge_yes) == pytest.approx(0.04, abs=1e-6)
        result = check_s2_eligibility(signal, limits, **_COMMON_GATE_KWARGS)
        assert result.eligible

    def test_no_ablation_config_skips_gate(self) -> None:
        """Without ablation_gate config, behaves as before."""
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.52)
        result = check_s2_eligibility(signal, _TEST_LIMITS, **_COMMON_GATE_KWARGS)
        assert result.eligible


class TestAblationSideFilter:
    """Gate 11: side_filter blocks unwanted trade sides."""

    def test_yes_only_allows_yes(self) -> None:
        limits = {**_ABL_LIMITS, "ablation_gate": {**_ABL_LIMITS["ablation_gate"], "side_filter": "YES_ONLY"}}
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.59)
        result = check_s2_eligibility(signal, limits, **_COMMON_GATE_KWARGS)
        assert result.eligible

    def test_yes_only_blocks_no(self) -> None:
        limits = {**_ABL_LIMITS, "ablation_gate": {**_ABL_LIMITS["ablation_gate"], "side_filter": "YES_ONLY"}}
        signal = _make_signal(
            p_yes_mid=0.55, spread=0.02, p_model_yes=0.43, trade_side="NO",
        )
        result = check_s2_eligibility(signal, limits, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "ablation_side_blocked" in (result.deny_reason or "")

    def test_no_only_blocks_yes(self) -> None:
        limits = {**_ABL_LIMITS, "ablation_gate": {**_ABL_LIMITS["ablation_gate"], "side_filter": "NO_ONLY"}}
        signal = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.59)
        result = check_s2_eligibility(signal, limits, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "ablation_side_blocked" in (result.deny_reason or "")

    def test_all_allows_both_sides(self) -> None:
        """side_filter=ALL allows both YES and NO."""
        signal_yes = _make_signal(p_yes_mid=0.48, spread=0.02, p_model_yes=0.59)
        signal_no = _make_signal(
            p_yes_mid=0.55, spread=0.02, p_model_yes=0.43, trade_side="NO",
        )
        r1 = check_s2_eligibility(signal_yes, _ABL_LIMITS, **_COMMON_GATE_KWARGS)
        r2 = check_s2_eligibility(signal_no, _ABL_LIMITS, **_COMMON_GATE_KWARGS)
        assert r1.eligible
        assert r2.eligible

    def test_side_filter_after_magnitude_gate(self) -> None:
        """Gate 10 (magnitude) fires before gate 11 (side filter)."""
        limits = {**_ABL_LIMITS, "ablation_gate": {**_ABL_LIMITS["ablation_gate"], "side_filter": "YES_ONLY"}}
        # NO-side with exec_edge ok (gate 3 passes) but |edge_yes| < 0.10 (gate 10 fires)
        signal = _make_signal(
            p_yes_mid=0.50, spread=0.02, p_model_yes=0.43, trade_side="NO",
        )
        assert abs(signal.edge_yes) == pytest.approx(0.07, abs=1e-6)
        assert signal.executable_edge >= 0.03  # gate 3 passes
        result = check_s2_eligibility(signal, limits, **_COMMON_GATE_KWARGS)
        assert not result.eligible
        assert "ablation_edge_below_min" in (result.deny_reason or "")
