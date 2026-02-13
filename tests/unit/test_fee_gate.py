"""
Tests for fee_gate module (v7.9-E2).

Tests the fee-aware edge gate that rejects entries where expected
edge does not beat estimated round-trip fee cost.
"""

import pytest
from execution.fee_gate import (
    FeeGateConfig,
    check_fee_edge,
    compute_round_trip_fee,
    compute_required_edge,
    load_fee_gate_config,
    DEFAULT_TAKER_FEE_RATE,
    DEFAULT_FEE_BUFFER_MULT,
)


# ── Config loading ─────────────────────────────────────────────────────────

class TestFeeGateConfig:
    def test_defaults(self):
        cfg = FeeGateConfig()
        assert cfg.taker_fee_rate == DEFAULT_TAKER_FEE_RATE
        assert cfg.fee_buffer_mult == DEFAULT_FEE_BUFFER_MULT
        assert cfg.enabled is True

    def test_load_from_dict(self):
        runtime = {
            "fee_gate": {
                "taker_fee_rate": 0.001,
                "fee_buffer_mult": 2.0,
                "enabled": False,
            }
        }
        cfg = load_fee_gate_config(runtime)
        assert cfg.taker_fee_rate == 0.001
        assert cfg.fee_buffer_mult == 2.0
        assert cfg.enabled is False

    def test_load_missing_section(self):
        cfg = load_fee_gate_config({})
        assert cfg.taker_fee_rate == DEFAULT_TAKER_FEE_RATE

    def test_load_none(self):
        cfg = load_fee_gate_config({"fee_gate": None})
        assert cfg.enabled is True


# ── Fee computation ────────────────────────────────────────────────────────

class TestFeeComputation:
    def test_round_trip_fee(self):
        """$100 trade @ 0.04% = $0.04 per leg, $0.08 round trip."""
        fee = compute_round_trip_fee(100.0, 0.0004)
        assert abs(fee - 0.08) < 1e-10

    def test_round_trip_fee_large_notional(self):
        """$10,000 trade @ 0.04% = $4 per leg, $8 round trip."""
        fee = compute_round_trip_fee(10_000.0, 0.0004)
        assert abs(fee - 8.0) < 1e-10

    def test_required_edge_1_5x(self):
        """$0.08 fee × 1.5 = $0.12 required edge."""
        edge = compute_required_edge(0.08, 1.5)
        assert abs(edge - 0.12) < 1e-10

    def test_required_edge_2x(self):
        edge = compute_required_edge(0.08, 2.0)
        assert abs(edge - 0.16) < 1e-10


# ── Gate check ─────────────────────────────────────────────────────────────

class TestFeeGateCheck:
    def test_pass_when_edge_exceeds_threshold(self):
        """Trade with 0.5% edge on $100 = $0.50 edge > $0.12 required."""
        cfg = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
        allowed, details = check_fee_edge(100.0, 0.005, config=cfg)
        assert allowed is True
        assert details["gate_status"] == "pass"

    def test_veto_when_edge_too_small(self):
        """Trade with 0.01% edge on $100 = $0.01 edge < $0.12 required."""
        cfg = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
        allowed, details = check_fee_edge(100.0, 0.0001, config=cfg)
        assert allowed is False
        assert details["gate_status"] == "veto"
        assert "shortfall_usd" in details

    def test_veto_when_zero_edge(self):
        """Zero edge should always be vetoed."""
        cfg = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
        allowed, details = check_fee_edge(100.0, 0.0, config=cfg)
        assert allowed is False

    def test_pass_when_disabled(self):
        """When disabled, always pass but still compute details."""
        cfg = FeeGateConfig(enabled=False)
        allowed, details = check_fee_edge(100.0, 0.0, config=cfg)
        assert allowed is True
        assert details["gate_status"] == "disabled"

    def test_details_contain_all_fields(self):
        cfg = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
        _, details = check_fee_edge(100.0, 0.005, config=cfg)
        assert "notional_usd" in details
        assert "taker_fee_rate" in details
        assert "round_trip_fee_usd" in details
        assert "fee_buffer_mult" in details
        assert "required_edge_usd" in details
        assert "expected_edge_pct" in details
        assert "expected_edge_usd" in details
        assert "gate_status" in details

    def test_negative_edge_treated_as_absolute(self):
        """Negative expected_edge_pct is handled via abs()."""
        cfg = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
        allowed, details = check_fee_edge(100.0, -0.01, config=cfg)
        # abs(-0.01) * 100 = $1.0 edge, vs $0.12 required → pass
        assert allowed is True

    def test_typical_churn_scenario(self):
        """$22 notional trade (ETHUSDT min) with zero edge → vetoed.
        
        Round-trip fee = $22 * 0.0004 * 2 = $0.0176
        Required edge  = $0.0176 * 1.5 = $0.0264
        With 0 edge → veto (this blocks the Feb 12 churn pattern)
        """
        cfg = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
        allowed, details = check_fee_edge(22.0, 0.0, config=cfg)
        assert allowed is False
        assert details["round_trip_fee_usd"] == pytest.approx(0.0176, abs=1e-4)

    def test_real_100usd_trade_needs_012pct_edge(self):
        """$100 trade needs 0.12% edge to pass 1.5x fee buffer.
        
        RT fee = $0.08, required = $0.12, so need $0.12/$100 = 0.12% edge.
        """
        cfg = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
        # Just below threshold
        allowed, _ = check_fee_edge(100.0, 0.0011, config=cfg)
        assert allowed is False  # $0.11 < $0.12
        # At threshold
        allowed, _ = check_fee_edge(100.0, 0.0012, config=cfg)
        assert allowed is True  # $0.12 >= $0.12
