# tests/unit/test_prediction_firewall.py
"""
Tests for prediction/firewall.py — influence firewall enforcement.

Covers:
    - Blacklisted consumers always denied (doctrine_kernel, sentinel_x)
    - Disabled layer → DENIED_DISABLED
    - Phase gating (P0/P1 = advisory-only, P2 requires PRODUCTION_ELIGIBLE)
    - Dataset state checks in production phase
    - Advisory-only flag set correctly
    - Denial logging to JSONL
    - is_consumer_allowed quick check
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from prediction.firewall import (
    AdvisoryPayload,
    FirewallResult,
    FirewallVerdict,
    is_consumer_allowed,
    request_advisory,
)


# ---------------------------------------------------------------------------
# Blacklist enforcement (Doctrine supremacy)
# ---------------------------------------------------------------------------

class TestBlacklist:
    def test_doctrine_kernel_always_denied(self):
        result = request_advisory(
            consumer="doctrine_kernel",
            question_id="Q_test",
            probs={"O_yes": 0.5, "O_no": 0.5},
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.DENIED_BLACKLISTED
        assert result.payload is None

    def test_sentinel_x_always_denied(self):
        result = request_advisory(
            consumer="sentinel_x",
            question_id="Q_test",
            probs={"O_yes": 0.7, "O_no": 0.3},
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.DENIED_BLACKLISTED

    def test_blacklist_overrides_everything(self):
        """Even with PRODUCTION_ELIGIBLE datasets and P2, blacklist wins."""
        result = request_advisory(
            consumer="doctrine_kernel",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={"poly": "PRODUCTION_ELIGIBLE"},
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.DENIED_BLACKLISTED

    def test_is_consumer_allowed_blacklisted(self):
        assert not is_consumer_allowed("doctrine_kernel")
        assert not is_consumer_allowed("sentinel_x")

    def test_is_consumer_allowed_normal(self):
        assert is_consumer_allowed("dashboard")
        assert is_consumer_allowed("research")
        assert is_consumer_allowed("some_new_module")


# ---------------------------------------------------------------------------
# Disabled layer
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_disabled_returns_denied(self):
        result = request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            enabled=False,
        )
        assert result.verdict == FirewallVerdict.DENIED_DISABLED
        assert result.payload is None


# ---------------------------------------------------------------------------
# Phase gating
# ---------------------------------------------------------------------------

class TestPhaseGating:
    def test_p0_observe_denies_all_consumers(self):
        """P0 = observe-only. No consumers allowed — not even dashboard."""
        result = request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.6, "O_no": 0.4},
            phase="P0_OBSERVE",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.DENIED_PHASE
        assert result.payload is None

    def test_p1_advisory_allowed_as_advisory(self):
        result = request_advisory(
            consumer="research",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.ALLOWED
        assert result.payload.advisory_only is True

    def test_p1_advisory_alert_ranking(self):
        result = request_advisory(
            consumer="alert_ranking",
            question_id="Q_test",
            probs={"O_yes": 0.7, "O_no": 0.3},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.ALLOWED
        assert result.payload.advisory_only is True

    def test_p2_production_non_advisory_consumer(self):
        result = request_advisory(
            consumer="some_new_execution_module",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={"poly": "PRODUCTION_ELIGIBLE"},
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.ALLOWED
        assert result.payload.advisory_only is False

    def test_p2_advisory_consumer_stays_advisory(self):
        """Even in P2, advisory-only consumers stay advisory."""
        result = request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={"poly": "PRODUCTION_ELIGIBLE"},
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.ALLOWED
        assert result.payload.advisory_only is True


# ---------------------------------------------------------------------------
# Dataset state enforcement in production
# ---------------------------------------------------------------------------

class TestDatasetStateEnforcement:
    def test_p2_rejected_dataset_denied(self):
        result = request_advisory(
            consumer="alert_ranking",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={"model_x": "REJECTED"},
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.DENIED_DATASET

    def test_p2_observe_only_dataset_denied(self):
        result = request_advisory(
            consumer="alert_ranking",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={"poly": "OBSERVE_ONLY"},
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.DENIED_DATASET

    def test_p2_all_production_eligible_allowed(self):
        result = request_advisory(
            consumer="alert_ranking",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={
                "poly": "PRODUCTION_ELIGIBLE",
                "human": "PRODUCTION_ELIGIBLE",
            },
            phase="P2_PRODUCTION",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.ALLOWED

    def test_p0_ignores_dataset_state(self):
        """In P0, dataset state doesn't matter — P0 blocks all consumers."""
        result = request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={"bad": "REJECTED"},
            phase="P0_OBSERVE",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.DENIED_PHASE

    def test_p1_ignores_dataset_state(self):
        """In P1, dataset state doesn't block — always advisory anyway."""
        result = request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            dataset_states={"bad": "REJECTED"},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.verdict == FirewallVerdict.ALLOWED
        assert result.payload.advisory_only is True


# ---------------------------------------------------------------------------
# Payload contents
# ---------------------------------------------------------------------------

class TestPayload:
    def test_payload_fields(self):
        result = request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.7, "O_no": 0.3},
            aggregate_hash="sha256:abc",
            aggregate_ts="2026-01-01T00:00:00Z",
            dataset_states={"poly": "OBSERVE_ONLY"},
            phase="P1_ADVISORY",
            enabled=True,
        )
        p = result.payload
        assert p.question_id == "Q_test"
        assert p.probs == {"O_yes": 0.7, "O_no": 0.3}
        assert p.aggregate_hash == "sha256:abc"
        assert p.dataset_states == {"poly": "OBSERVE_ONLY"}
        assert p.phase == "P1_ADVISORY"


# ---------------------------------------------------------------------------
# Denial logging
# ---------------------------------------------------------------------------

class TestDenialLogging:
    def test_denial_logged_to_jsonl(self, tmp_path):
        log_path = tmp_path / "denials.jsonl"
        request_advisory(
            consumer="doctrine_kernel",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            enabled=True,
            denial_log_path=log_path,
        )
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip())
        assert record["consumer"] == "doctrine_kernel"
        assert record["verdict"] == "DENIED_BLACKLISTED"

    def test_disabled_denial_logged(self, tmp_path):
        log_path = tmp_path / "denials.jsonl"
        request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            enabled=False,
            denial_log_path=log_path,
        )
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip())
        assert record["verdict"] == "DENIED_DISABLED"

    def test_allowed_not_logged(self, tmp_path):
        log_path = tmp_path / "denials.jsonl"
        request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            phase="P1_ADVISORY",
            enabled=True,
            denial_log_path=log_path,
        )
        # No denial = no log entry
        assert not log_path.exists()

    def test_p0_denial_logged(self, tmp_path):
        log_path = tmp_path / "denials.jsonl"
        request_advisory(
            consumer="dashboard",
            question_id="Q_test",
            probs={"O_yes": 0.5},
            phase="P0_OBSERVE",
            enabled=True,
            denial_log_path=log_path,
        )
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip())
        assert record["verdict"] == "DENIED_PHASE"
