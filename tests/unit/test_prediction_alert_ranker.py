# tests/unit/test_prediction_alert_ranker.py
"""
Tests for prediction/alert_ranker.py — P1 advisory alert ranking consumer.

Covers:
    1. Set equality — same alerts in, same alerts out
    2. Only order changes, never content
    3. Missing prediction → original order preserved
    4. Firewall gating — P0 denied, P1 allowed, blacklisted denied
    5. Uncertainty scoring (entropy)
    6. Shift scoring (Δprob)
    7. Severity preserved as primary sort
    8. Logging to prediction JSONL
    9. Empty edge cases
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
from prediction.alert_ranker import (
    PredictionSnapshot,
    RankingResult,
    _entropy,
    _max_shift,
    _normalized_entropy,
    _relevance_score,
    rank_alerts,
)
from prediction.firewall import FirewallVerdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _alerts() -> list[dict]:
    """Standard set of 4 alerts with mixed severities."""
    return [
        {"type": "router", "severity": "info", "msg": "Router health OK"},
        {"type": "drawdown", "severity": "warning", "msg": "DD at 3%"},
        {"type": "stale_nav", "severity": "critical", "msg": "NAV stale 120s"},
        {"type": "slippage", "severity": "info", "msg": "Slippage +2bps"},
    ]


def _snapshot(probs: Dict[str, float],
              prior: Dict[str, float] | None = None,
              q_id: str = "Q_test") -> PredictionSnapshot:
    return PredictionSnapshot(
        question_id=q_id,
        probs=probs,
        aggregate_hash="sha256:test",
        aggregate_ts="2026-02-06T00:00:00Z",
        dataset_states={"poly": "OBSERVE_ONLY"},
        prior_probs=prior,
    )


# ---------------------------------------------------------------------------
# Set equality — THE critical invariant
# ---------------------------------------------------------------------------

class TestSetEquality:
    def test_same_alerts_in_same_alerts_out(self):
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert set(a["msg"] for a in result.alerts) == set(a["msg"] for a in alerts)
        assert len(result.alerts) == len(alerts)

    def test_no_alerts_added_or_removed(self):
        alerts = [{"type": "x", "severity": "info", "msg": "only one"}]
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert len(result.alerts) == 1
        assert result.alerts[0]["msg"] == "only one"

    def test_alert_content_unchanged(self):
        """Alert dicts must not be mutated — same keys, same values."""
        alerts = _alerts()
        originals = [dict(a) for a in alerts]
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=True,
        )
        for ranked_alert in result.alerts:
            assert ranked_alert in originals


# ---------------------------------------------------------------------------
# Missing prediction → original order
# ---------------------------------------------------------------------------

class TestFallback:
    def test_no_snapshots_returns_original(self):
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots=None,
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.alerts == alerts
        assert result.rankings_applied is False

    def test_empty_snapshots_returns_original(self):
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.alerts == alerts
        assert result.rankings_applied is False

    def test_empty_alerts(self):
        result = rank_alerts(
            alerts=[],
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.alerts == []
        assert result.rankings_applied is False


# ---------------------------------------------------------------------------
# Firewall gating
# ---------------------------------------------------------------------------

class TestFirewallGating:
    def test_p0_denied_no_reorder(self):
        """P0 phase blocks all consumers — alerts returned as-is."""
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P0_OBSERVE",
            enabled=True,
        )
        assert result.rankings_applied is False
        assert "Q_test" in result.firewall_verdicts
        assert result.firewall_verdicts["Q_test"] == "DENIED_PHASE"

    def test_disabled_denied_no_reorder(self):
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=False,
        )
        assert result.rankings_applied is False
        assert result.firewall_verdicts["Q_test"] == "DENIED_DISABLED"

    def test_p1_advisory_allowed(self):
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.firewall_verdicts["Q_test"] == "ALLOWED"

    def test_firewall_denial_logged(self, tmp_path):
        denial_log = tmp_path / "denials.jsonl"
        rank_alerts(
            alerts=_alerts(),
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P0_OBSERVE",
            enabled=True,
            denial_log_path=denial_log,
        )
        assert denial_log.exists()
        record = json.loads(denial_log.read_text().strip())
        assert record["consumer"] == "alert_ranking"
        assert record["verdict"] == "DENIED_PHASE"


# ---------------------------------------------------------------------------
# Severity preserved as primary sort
# ---------------------------------------------------------------------------

class TestSeverityPrimary:
    def test_critical_always_first(self):
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=True,
        )
        assert result.alerts[0]["severity"] == "critical"

    def test_severity_order_preserved_with_prediction(self):
        alerts = _alerts()
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots={"Q_test": _snapshot(
                {"yes": 0.99, "no": 0.01},  # very certain — low entropy
                prior={"yes": 0.01, "no": 0.99},  # huge shift
            )},
            phase="P1_ADVISORY",
            enabled=True,
        )
        severities = [a["severity"] for a in result.alerts]
        severity_order = {"critical": 3, "warning": 2, "info": 1}
        severity_vals = [severity_order.get(s, 0) for s in severities]
        # Must be non-increasing (critical ≥ warning ≥ info)
        for i in range(len(severity_vals) - 1):
            assert severity_vals[i] >= severity_vals[i + 1]


# ---------------------------------------------------------------------------
# Entropy scoring
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_binary_uniform_max_entropy(self):
        h = _normalized_entropy({"yes": 0.5, "no": 0.5})
        assert abs(h - 1.0) < 1e-10

    def test_binary_certain_zero_entropy(self):
        h = _normalized_entropy({"yes": 1.0, "no": 0.0})
        assert abs(h) < 1e-10

    def test_ternary_uniform(self):
        h = _normalized_entropy({"a": 1/3, "b": 1/3, "c": 1/3})
        assert abs(h - 1.0) < 1e-10

    def test_single_outcome(self):
        h = _normalized_entropy({"only": 1.0})
        assert h == 0.0

    def test_empty(self):
        h = _normalized_entropy({})
        assert h == 0.0


# ---------------------------------------------------------------------------
# Shift scoring
# ---------------------------------------------------------------------------

class TestShift:
    def test_no_prior_zero_shift(self):
        assert _max_shift({"yes": 0.5, "no": 0.5}, None) == 0.0

    def test_identical_zero_shift(self):
        probs = {"yes": 0.7, "no": 0.3}
        assert _max_shift(probs, probs) == 0.0

    def test_large_shift(self):
        current = {"yes": 0.9, "no": 0.1}
        prior = {"yes": 0.1, "no": 0.9}
        assert abs(_max_shift(current, prior) - 0.8) < 1e-10

    def test_partial_shift(self):
        current = {"yes": 0.6, "no": 0.4}
        prior = {"yes": 0.5, "no": 0.5}
        assert abs(_max_shift(current, prior) - 0.1) < 1e-10


# ---------------------------------------------------------------------------
# Relevance score composition
# ---------------------------------------------------------------------------

class TestRelevanceScore:
    def test_max_uncertainty_no_shift(self):
        score = _relevance_score({"yes": 0.5, "no": 0.5}, None)
        # 0.5 * 1.0 (entropy) + 0.5 * 0.0 (shift) = 0.5
        assert abs(score - 0.5) < 1e-10

    def test_no_uncertainty_max_shift(self):
        score = _relevance_score(
            {"yes": 1.0, "no": 0.0},
            {"yes": 0.0, "no": 1.0},
        )
        # 0.5 * 0.0 (entropy) + 0.5 * 1.0 (shift) = 0.5
        assert abs(score - 0.5) < 1e-10

    def test_both_max(self):
        score = _relevance_score(
            {"yes": 0.5, "no": 0.5},
            {"yes": 0.0, "no": 1.0},
        )
        # 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        assert abs(score - 0.75) < 1e-10


# ---------------------------------------------------------------------------
# Ranking log
# ---------------------------------------------------------------------------

class TestRankingLog:
    def test_log_written_on_rank(self, tmp_path):
        log = tmp_path / "ranking.jsonl"
        rank_alerts(
            alerts=_alerts(),
            prediction_snapshots={"Q_test": _snapshot({"yes": 0.5, "no": 0.5})},
            phase="P1_ADVISORY",
            enabled=True,
            log_path=log,
        )
        assert log.exists()
        record = json.loads(log.read_text().strip())
        assert record["consumer"] == "alert_ranking"
        assert record["alerts_count"] == 4
        assert "rankings_applied" in record

    def test_log_written_on_no_snapshots(self, tmp_path):
        log = tmp_path / "ranking.jsonl"
        rank_alerts(
            alerts=_alerts(),
            prediction_snapshots=None,
            phase="P1_ADVISORY",
            enabled=True,
            log_path=log,
        )
        assert log.exists()
        record = json.loads(log.read_text().strip())
        assert record["rankings_applied"] is False


# ---------------------------------------------------------------------------
# Question-tagged alerts
# ---------------------------------------------------------------------------

class TestQuestionTagged:
    def test_tagged_alert_gets_specific_score(self):
        """Alert with prediction_question_id gets that question's score."""
        # Put low-uncertainty alert FIRST so prediction reorders them
        alerts = [
            {"type": "b", "severity": "info", "msg": "alert B",
             "prediction_question_id": "Q_low_uncertainty"},
            {"type": "a", "severity": "info", "msg": "alert A",
             "prediction_question_id": "Q_high_uncertainty"},
        ]
        snapshots = {
            "Q_high_uncertainty": _snapshot(
                {"yes": 0.5, "no": 0.5},  # max entropy
                q_id="Q_high_uncertainty",
            ),
            "Q_low_uncertainty": _snapshot(
                {"yes": 0.99, "no": 0.01},  # low entropy
                q_id="Q_low_uncertainty",
            ),
        }
        result = rank_alerts(
            alerts=alerts,
            prediction_snapshots=snapshots,
            phase="P1_ADVISORY",
            enabled=True,
        )
        # High uncertainty alert should rank first (reordered)
        assert result.alerts[0]["msg"] == "alert A"
        assert result.rankings_applied is True
