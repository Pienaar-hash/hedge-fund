# tests/unit/test_prediction_alert_wiring.py
"""
Tests for the alert ranker wiring into execution/telegram_utils.py.

Verifies:
    1. _maybe_rank_alerts is fail-open (import failure → unchanged alerts)
    2. _maybe_rank_alerts is no-op when prediction disabled
    3. _maybe_rank_alerts preserves set equality
    4. send_execution_alerts still works with prediction layer absent
    5. No execution dependency on prediction liveness
"""

from __future__ import annotations

from typing import Any, Dict, Sequence
from unittest.mock import patch, MagicMock

import pytest


def _sample_alerts() -> list[dict]:
    return [
        {"type": "router", "severity": "info", "msg": "Router health OK"},
        {"type": "drawdown", "severity": "warning", "msg": "DD at 3%"},
        {"type": "stale_nav", "severity": "critical", "msg": "NAV stale 120s"},
    ]


class TestMaybeRankAlerts:
    """Test the fail-open prediction ranking wrapper."""

    def test_returns_unchanged_when_disabled(self):
        """With PREDICTION_DLE_ENABLED=0 (default), alerts unchanged."""
        from execution.telegram_utils import _maybe_rank_alerts
        alerts = _sample_alerts()
        result = _maybe_rank_alerts(alerts)
        assert list(result) == alerts

    def test_returns_unchanged_on_import_error(self):
        """If prediction module doesn't exist, alerts unchanged (fail-open)."""
        from execution.telegram_utils import _maybe_rank_alerts
        alerts = _sample_alerts()
        with patch.dict("sys.modules", {"prediction.alert_ranker": None,
                                         "prediction.state_surface": None}):
            result = _maybe_rank_alerts(alerts)
        assert list(result) == alerts

    def test_returns_unchanged_on_exception(self):
        """If prediction layer raises, alerts unchanged (fail-open)."""
        from execution.telegram_utils import _maybe_rank_alerts
        alerts = _sample_alerts()

        with patch.dict("os.environ", {
            "PREDICTION_DLE_ENABLED": "1",
            "PREDICTION_PHASE": "P1_ADVISORY",
        }):
            with patch("prediction.alert_ranker.rank_alerts",
                        side_effect=RuntimeError("boom")):
                result = _maybe_rank_alerts(alerts)
        assert list(result) == alerts

    def test_set_equality_when_enabled(self):
        """When enabled, same alerts in, same alerts out."""
        from execution.telegram_utils import _maybe_rank_alerts
        alerts = _sample_alerts()

        with patch.dict("os.environ", {
            "PREDICTION_DLE_ENABLED": "1",
            "PREDICTION_PHASE": "P1_ADVISORY",
        }):
            result = _maybe_rank_alerts(alerts)

        assert set(a["msg"] for a in result) == set(a["msg"] for a in alerts)
        assert len(result) == len(alerts)

    def test_no_snapshots_returns_original_order(self):
        """With no prediction snapshots, order is unchanged."""
        from execution.telegram_utils import _maybe_rank_alerts
        alerts = _sample_alerts()

        with patch.dict("os.environ", {
            "PREDICTION_DLE_ENABLED": "1",
            "PREDICTION_PHASE": "P1_ADVISORY",
        }):
            result = _maybe_rank_alerts(alerts)

        # No snapshots → rank_alerts returns alerts unchanged
        assert list(result) == alerts


class TestSendExecutionAlertsWithPrediction:
    """Verify send_execution_alerts still works with prediction wired in."""

    def test_send_works_when_prediction_unavailable(self):
        """send_execution_alerts functions normally without prediction."""
        from execution.telegram_utils import send_execution_alerts

        with patch("execution.telegram_utils.send_telegram") as mock_send:
            with patch("execution.telegram_utils._last_alert_ts", {}):
                send_execution_alerts("BTCUSDT", _sample_alerts())
            # Should have been called (telegram sends)
            assert mock_send.called

    def test_send_preserves_severity_order(self):
        """Critical alerts always appear first regardless of prediction."""
        from execution.telegram_utils import send_execution_alerts

        sent_body = []
        def capture_send(msg, silent=False, parse_mode=None):
            sent_body.append(msg)
            return True

        with patch("execution.telegram_utils.send_telegram", side_effect=capture_send):
            with patch("execution.telegram_utils._last_alert_ts", {}):
                send_execution_alerts("BTCUSDT", _sample_alerts())

        assert sent_body
        body = sent_body[0]
        # Critical should appear before warning, warning before info
        crit_pos = body.find("CRITICAL")
        warn_pos = body.find("WARNING")
        info_pos = body.find("INFO")
        assert crit_pos < warn_pos < info_pos
