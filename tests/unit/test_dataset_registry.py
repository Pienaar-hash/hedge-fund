"""
Unit tests for Dataset Registry and Rollback system.

These tests lock in the doctrine invariants from:
- DATASET_ADMISSION_GATE.md
- DATASET_ROLLBACK_CLAUSE.md

If these tests fail, doctrine has been violated.
"""

from datetime import datetime, timezone

import pytest

from execution.utils.dataset_registry import (
    DatasetState,
    DatasetTier,
    get_dataset_info,
    get_dataset_state,
    get_dataset_tier,
    is_observe_only,
    is_production_eligible,
    reload_registry,
    requires_cycle_boundary_rollback,
)
from execution.utils.dataset_rollback import (
    RollbackAction,
    RollbackTrigger,
    RollbackTriggerType,
    detect_latency_breach,
    detect_replay_divergence,
    detect_silent_gap,
    log_rollback_trigger,
    would_trigger_rollback,
)


# ============================================================
# DOCTRINE INVARIANT TESTS — Dataset Registry
# ============================================================


class TestDatasetStateInvariants:
    """Tests that verify dataset state classification matches doctrine."""

    def test_unknown_dataset_defaults_to_rejected(self):
        """Unknown datasets must be REJECTED by default (Admission Gate §II)."""
        state = get_dataset_state("completely_unknown_dataset_xyz")
        assert state == DatasetState.REJECTED

    def test_unknown_dataset_tier_is_unknown(self):
        """Unknown datasets must have UNKNOWN tier."""
        tier = get_dataset_tier("completely_unknown_dataset_xyz")
        assert tier == DatasetTier.UNKNOWN

    def test_unknown_dataset_not_production_eligible(self):
        """Unknown datasets must not be production eligible."""
        assert not is_production_eligible("completely_unknown_dataset_xyz")

    @pytest.mark.parametrize("dataset_id", [
        "binance_futures_klines",
        "binance_futures_positions",
        "binance_futures_balance",
        "binance_futures_fills",
    ])
    def test_existential_datasets_are_production_eligible(self, dataset_id):
        """EXISTENTIAL datasets must be PRODUCTION_ELIGIBLE (grandfathered)."""
        assert is_production_eligible(dataset_id)
        assert get_dataset_tier(dataset_id) == DatasetTier.EXISTENTIAL

    @pytest.mark.parametrize("dataset_id", [
        "binance_futures_klines",
        "binance_futures_positions",
        "binance_futures_balance",
        "binance_futures_fills",
        "sentinel_x_features",
    ])
    def test_existential_and_authoritative_require_cycle_boundary(self, dataset_id):
        """EXISTENTIAL and AUTHORITATIVE datasets require cycle boundary for rollback."""
        assert requires_cycle_boundary_rollback(dataset_id)

    @pytest.mark.parametrize("dataset_id", [
        "symbol_scores_v6",
        "expectancy_v6",
        "router_health",
        "binance_futures_orderbook",
    ])
    def test_advisory_datasets_can_rollback_immediately(self, dataset_id):
        """ADVISORY datasets can be rolled back immediately."""
        assert not requires_cycle_boundary_rollback(dataset_id)
        info = get_dataset_info(dataset_id)
        assert info.can_rollback_immediately

    @pytest.mark.parametrize("dataset_id", [
        "coingecko_prices",
        "regime_pressure",
        "factor_diagnostics",
    ])
    def test_observational_datasets_are_observe_only(self, dataset_id):
        """OBSERVATIONAL datasets must be OBSERVE_ONLY state."""
        assert is_observe_only(dataset_id)
        assert get_dataset_tier(dataset_id) == DatasetTier.OBSERVATIONAL

    def test_polymarket_snapshot_promoted_with_observational_tier(self):
        """
        Polymarket snapshot may be promoted while remaining observational tier.
        """
        assert is_production_eligible("polymarket_snapshot")
        assert get_dataset_tier("polymarket_snapshot") == DatasetTier.OBSERVATIONAL


class TestDatasetInfoProperties:
    """Tests for DatasetInfo computed properties."""

    def test_existential_requires_substitution(self):
        """EXISTENTIAL datasets require substitution for rollback."""
        info = get_dataset_info("binance_futures_klines")
        assert info.requires_substitution_for_rollback
        assert not info.can_rollback_immediately

    def test_observational_does_not_influence_decisions(self):
        """OBSERVATIONAL datasets must not influence decisions."""
        info = get_dataset_info("polymarket_snapshot")
        assert not info.influences_decisions
        assert not info.influences_regime
        assert not info.influences_signals
        assert not info.influences_sizing
        assert not info.influences_exits

    def test_sentinel_x_influences_regime(self):
        """sentinel_x_features must influence regime (it IS the regime)."""
        info = get_dataset_info("sentinel_x_features")
        assert info.influences_regime
        assert info.influences_decisions


# ============================================================
# DOCTRINE INVARIANT TESTS — Rollback Triggers
# ============================================================


class TestRollbackTriggerClassification:
    """Tests that verify rollback trigger classification matches doctrine."""

    @pytest.mark.parametrize("trigger_reason", [
        RollbackTrigger.TEMPORAL_VIOLATION,
        RollbackTrigger.REPLAY_DIVERGENCE,
        RollbackTrigger.REGIME_CORRUPTION,
        RollbackTrigger.LATENCY_BREACH,
        RollbackTrigger.SILENT_GAP,
    ])
    def test_mandatory_triggers_on_advisory(self, trigger_reason):
        """Mandatory triggers must be allowed on ADVISORY datasets."""
        would, trigger, reason = would_trigger_rollback(
            "symbol_scores_v6",
            trigger_reason,
            "Test details"
        )
        assert would is True
        assert trigger is not None
        assert trigger.trigger_type == RollbackTriggerType.MANDATORY

    @pytest.mark.parametrize("trigger_reason", [
        RollbackTrigger.QUALITY_DEGRADATION,
        RollbackTrigger.LATENCY_DRIFT,
        RollbackTrigger.PARTIAL_OUTAGE,
        RollbackTrigger.UPSTREAM_DEPRECATION,
    ])
    def test_discretionary_triggers_forbidden_on_existential(self, trigger_reason):
        """Discretionary triggers must be FORBIDDEN on EXISTENTIAL datasets."""
        would, trigger, reason = would_trigger_rollback(
            "binance_futures_klines",
            trigger_reason,
            "Test details"
        )
        assert would is False
        assert "forbidden" in reason.lower()
        assert "EXISTENTIAL" in reason

    @pytest.mark.parametrize("trigger_reason", [
        RollbackTrigger.QUALITY_DEGRADATION,
        RollbackTrigger.LATENCY_DRIFT,
    ])
    def test_discretionary_triggers_forbidden_on_authoritative(self, trigger_reason):
        """Discretionary triggers must be FORBIDDEN on AUTHORITATIVE datasets."""
        would, trigger, reason = would_trigger_rollback(
            "sentinel_x_features",
            trigger_reason,
            "Test details"
        )
        assert would is False
        assert "forbidden" in reason.lower()
        assert "AUTHORITATIVE" in reason

    def test_mandatory_trigger_on_existential_requires_cycle_boundary(self):
        """Mandatory triggers on EXISTENTIAL require cycle boundary."""
        would, trigger, reason = would_trigger_rollback(
            "binance_futures_klines",
            RollbackTrigger.TEMPORAL_VIOLATION,
            "Retroactive mutation detected"
        )
        assert would is True
        assert "cycle boundary" in reason.lower()

    def test_unknown_trigger_reason_rejected(self):
        """Unknown trigger reasons must be rejected."""
        would, trigger, reason = would_trigger_rollback(
            "symbol_scores_v6",
            "invented_trigger_reason",
            "Test details"
        )
        assert would is False
        assert "Unknown trigger reason" in reason


# ============================================================
# TRIGGER DETECTION HELPER TESTS
# ============================================================


class TestTriggerDetectionHelpers:
    """Tests for trigger detection helper functions."""

    def test_latency_breach_detected_above_threshold(self):
        """Latency breach must be detected when > 3x characterized."""
        trigger = detect_latency_breach(
            "symbol_scores_v6",
            observed_p99_ms=1600,
            characterized_p99_ms=500,
            threshold_multiplier=3.0
        )
        assert trigger is not None
        assert trigger.reason == RollbackTrigger.LATENCY_BREACH
        assert trigger.trigger_type == RollbackTriggerType.MANDATORY

    def test_latency_breach_not_detected_below_threshold(self):
        """Latency breach must NOT be detected when < 3x characterized."""
        trigger = detect_latency_breach(
            "symbol_scores_v6",
            observed_p99_ms=1400,
            characterized_p99_ms=500,
            threshold_multiplier=3.0
        )
        assert trigger is None

    def test_silent_gap_detected(self):
        """Silent gap must be detected when update exceeds max."""
        old_time = datetime(2026, 1, 24, 10, 0, 0, tzinfo=timezone.utc)
        trigger = detect_silent_gap(
            "symbol_scores_v6",
            last_update_ts=old_time,
            max_gap_seconds=60.0
        )
        # This will always trigger since old_time is in the past
        assert trigger is not None
        assert trigger.reason == RollbackTrigger.SILENT_GAP

    def test_replay_divergence_detected(self):
        """Replay divergence must be detected on hash mismatch."""
        trigger = detect_replay_divergence(
            "symbol_scores_v6",
            expected_hash="abc123def456",
            actual_hash="xyz789uvw012"
        )
        assert trigger is not None
        assert trigger.reason == RollbackTrigger.REPLAY_DIVERGENCE

    def test_replay_divergence_not_detected_on_match(self):
        """Replay divergence must NOT be detected on hash match."""
        trigger = detect_replay_divergence(
            "symbol_scores_v6",
            expected_hash="abc123def456",
            actual_hash="abc123def456"
        )
        assert trigger is None


# ============================================================
# ROLLBACK EVENT LOGGING TESTS
# ============================================================


class TestRollbackEventLogging:
    """Tests for rollback event logging behavior."""

    def test_log_rollback_trigger_creates_event(self, tmp_path):
        """log_rollback_trigger must create a valid event."""
        trigger = RollbackTrigger(
            trigger_type=RollbackTriggerType.MANDATORY,
            reason=RollbackTrigger.LATENCY_BREACH,
            details="Test latency breach"
        )
        
        event = log_rollback_trigger(
            "symbol_scores_v6",
            trigger,
            dry_run=True
        )
        
        assert event.dataset_id == "symbol_scores_v6"
        assert event.action == RollbackAction.DOWNGRADE
        assert event.from_state == DatasetState.PRODUCTION_ELIGIBLE
        assert event.to_state == DatasetState.RESEARCH_ONLY
        assert event.dry_run is True
        assert event.automatic is True

    def test_rollback_event_to_dict_format(self):
        """RollbackEvent.to_dict must produce valid JSON structure."""
        trigger = RollbackTrigger(
            trigger_type=RollbackTriggerType.MANDATORY,
            reason=RollbackTrigger.LATENCY_BREACH,
            details="Test"
        )
        
        event = log_rollback_trigger(
            "symbol_scores_v6",
            trigger,
            dry_run=True
        )
        
        d = event.to_dict()
        
        # Verify structure matches DATASET_ROLLBACK_CLAUSE.md §VII
        assert "ts" in d
        assert "dataset_id" in d
        assert "action" in d
        assert "from_state" in d
        assert "to_state" in d
        assert "trigger" in d
        assert "type" in d["trigger"]
        assert "reason" in d["trigger"]
        assert "details" in d["trigger"]
        assert "authority" in d
        assert "automatic" in d["authority"]
        assert "fallback" in d
        assert "dry_run" in d

    def test_downgrade_path_is_one_step(self):
        """Downgrade must step down one state at a time (normal path)."""
        trigger = RollbackTrigger(
            trigger_type=RollbackTriggerType.MANDATORY,
            reason=RollbackTrigger.LATENCY_BREACH,
            details="Test"
        )
        
        # PRODUCTION_ELIGIBLE -> RESEARCH_ONLY (one step)
        event = log_rollback_trigger(
            "symbol_scores_v6",
            trigger,
            proposed_action=RollbackAction.DOWNGRADE,
            dry_run=True
        )
        
        assert event.from_state == DatasetState.PRODUCTION_ELIGIBLE
        assert event.to_state == DatasetState.RESEARCH_ONLY

    def test_revoke_goes_directly_to_rejected(self):
        """REVOKE action must go directly to REJECTED."""
        trigger = RollbackTrigger(
            trigger_type=RollbackTriggerType.MANDATORY,
            reason=RollbackTrigger.TEMPORAL_VIOLATION,
            details="Test"
        )
        
        event = log_rollback_trigger(
            "symbol_scores_v6",
            trigger,
            proposed_action=RollbackAction.REVOKE,
            dry_run=True
        )
        
        assert event.to_state == DatasetState.REJECTED


# ============================================================
# TIER CONSTRAINT TESTS
# ============================================================


class TestTierConstraints:
    """Tests that verify tier-based rollback constraints."""

    def test_all_existential_datasets_classified(self):
        """All price/position/balance feeds must be EXISTENTIAL."""
        existential_expected = {
            "binance_futures_klines",
            "binance_futures_positions",
            "binance_futures_balance",
            "binance_futures_fills",
        }
        
        for ds in existential_expected:
            assert get_dataset_tier(ds) == DatasetTier.EXISTENTIAL, f"{ds} should be EXISTENTIAL"

    def test_sentinel_x_is_authoritative(self):
        """Sentinel-X features must be AUTHORITATIVE (defines regime)."""
        assert get_dataset_tier("sentinel_x_features") == DatasetTier.AUTHORITATIVE

    def test_advisory_cannot_influence_regime(self):
        """ADVISORY datasets must not influence regime."""
        advisory_datasets = [
            "symbol_scores_v6",
            "expectancy_v6",
            "router_health",
            "binance_futures_orderbook",
        ]
        
        for ds in advisory_datasets:
            info = get_dataset_info(ds)
            assert not info.influences_regime, f"{ds} should not influence regime"

    def test_observational_cannot_influence_anything(self):
        """OBSERVATIONAL datasets must not influence any decisions."""
        observational_datasets = [
            "coingecko_prices",
            "polymarket_snapshot",
            "regime_pressure",
            "factor_diagnostics",
        ]
        
        for ds in observational_datasets:
            info = get_dataset_info(ds)
            assert not info.influences_decisions, f"{ds} should not influence decisions"


# ============================================================
# REGISTRY RELOAD TESTS
# ============================================================


class TestRegistryReload:
    """Tests for registry reload behavior."""

    def test_reload_clears_cache(self):
        """reload_registry must clear cached data."""
        # First call caches
        _ = get_dataset_state("binance_futures_klines")
        
        # Reload should work without error
        reload_registry()
        
        # Should still return correct state after reload
        state = get_dataset_state("binance_futures_klines")
        assert state == DatasetState.PRODUCTION_ELIGIBLE
