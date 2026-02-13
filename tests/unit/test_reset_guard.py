"""Tests for Testnet Reset Guard.

Validates:
- Detection requires ALL four conditions (multi-condition gate)
- Each condition checked in isolation (no false positives)
- Archive creates timestamped directory with state files
- Environment watermark written correctly
- Production balance anomaly logged (never triggers reset)
- Debounce prevents infinite reset loops
- Fail-closed: cycle halts on trigger
- Event logged to environment_events.jsonl
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from execution.reset_guard import (
    DEFAULT_TESTNET_BALANCE,
    BALANCE_MATCH_TOLERANCE,
    RESET_DEBOUNCE_S,
    ResetResult,
    check_for_testnet_reset,
    get_current_cycle_id,
    reset_debounce_state,
    _balance_matches_default,
    _delta_exceeds_threshold,
    _next_cycle_id,
    _read_last_logged_nav,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_debounce():
    """Reset debounce state between tests."""
    reset_debounce_state()
    yield
    reset_debounce_state()


@pytest.fixture
def mock_logger():
    """Intercept JSONL logger so no file I/O occurs."""
    captured: list[dict] = []

    def _capture(logger, event_type, payload):
        captured.append({"event_type": event_type, **dict(payload)})

    with patch("execution.reset_guard.log_event", side_effect=_capture):
        yield captured


@pytest.fixture
def mock_nav_state(tmp_path):
    """Provide a writable nav_state.json with known NAV."""
    nav_state = {
        "total_equity": 9500.0,
        "nav_usd": 9500.0,
        "series": [{"total_equity": 9500.0, "nav": 9500.0}],
    }
    nav_file = tmp_path / "nav_state.json"
    nav_file.write_text(json.dumps(nav_state))
    with patch("execution.reset_guard._NAV_STATE_PATH", nav_file):
        yield nav_file


@pytest.fixture
def mock_env_meta(tmp_path):
    """Provide a writable environment_meta.json."""
    meta_file = tmp_path / "environment_meta.json"
    with patch("execution.reset_guard._ENV_META_PATH", meta_file):
        yield meta_file


@pytest.fixture
def mock_archive(tmp_path):
    """Point archive base to tmp dir."""
    with patch("execution.reset_guard._ARCHIVE_BASE", tmp_path / "archive"):
        yield tmp_path / "archive"


@pytest.fixture
def mock_state_files(tmp_path):
    """Create fake state files for archival tests."""
    nav = tmp_path / "nav_state.json"
    ep = tmp_path / "episode_ledger.json"
    pos = tmp_path / "positions_state.json"
    nav.write_text('{"total_equity": 9500.0}')
    ep.write_text('{"episodes": []}')
    pos.write_text('[]')
    with patch("execution.reset_guard._NAV_STATE_PATH", nav), \
         patch("execution.reset_guard._EPISODE_LEDGER_PATH", ep), \
         patch("execution.reset_guard._POSITIONS_STATE_PATH", pos):
        yield {"nav": nav, "episode_ledger": ep, "positions": pos}


# ── Unit Tests: Helpers ───────────────────────────────────────────────────────


class TestBalanceMatchesDefault:
    def test_exact_match(self):
        assert _balance_matches_default(DEFAULT_TESTNET_BALANCE) is True

    def test_within_tolerance(self):
        assert _balance_matches_default(DEFAULT_TESTNET_BALANCE + 0.5) is True

    def test_outside_tolerance(self):
        assert _balance_matches_default(DEFAULT_TESTNET_BALANCE + 100) is False

    def test_zero_balance(self):
        assert _balance_matches_default(0.0) is False


class TestDeltaExceedsThreshold:
    def test_large_delta(self):
        # 9500 → 10000 = ~5.3% — below 50% threshold
        assert _delta_exceeds_threshold(10000.0, 9500.0) is False

    def test_massive_delta(self):
        # 5000 → 10000 = 100% — exceeds 50%
        assert _delta_exceeds_threshold(10000.0, 5000.0) is True

    def test_zero_prior_nav(self):
        assert _delta_exceeds_threshold(10000.0, 0.0) is False

    def test_negative_prior_nav(self):
        assert _delta_exceeds_threshold(10000.0, -1.0) is False


class TestReadLastLoggedNav:
    def test_reads_total_equity(self, mock_nav_state):
        assert _read_last_logged_nav() == 9500.0

    def test_returns_zero_when_missing(self):
        with patch("execution.reset_guard._NAV_STATE_PATH", Path("/nonexistent")):
            assert _read_last_logged_nav() == 0.0


class TestNextCycleId:
    def test_first_cycle(self, mock_env_meta):
        assert _next_cycle_id() == "CYCLE_TEST_001"

    def test_increments(self, mock_env_meta):
        mock_env_meta.write_text(json.dumps({"cycle_id": "CYCLE_TEST_003"}))
        assert _next_cycle_id() == "CYCLE_TEST_004"

    def test_handles_corrupt_meta(self, mock_env_meta):
        mock_env_meta.write_text("not json")
        assert _next_cycle_id() == "CYCLE_TEST_001"


# ── Core Detection: Multi-Condition Gate ──────────────────────────────────────


class TestResetDetectionGate:
    """Each condition must be met for trigger. Fail any one → no trigger."""

    def _check(self, balance=DEFAULT_TESTNET_BALANCE, positions=None,
               env="testnet", nav=5000.0):
        """Helper: call check with sensible defaults for a triggering scenario."""
        return check_for_testnet_reset(
            exchange_balance=balance,
            exchange_positions=positions or [],
            env=env,
            last_logged_nav=nav,
        )

    def test_all_conditions_met_triggers(self, mock_logger, mock_env_meta, mock_archive):
        result = self._check(
            balance=DEFAULT_TESTNET_BALANCE,
            positions=[],
            env="testnet",
            nav=5000.0,  # >50% delta from 10000
        )
        assert result.triggered is True
        assert result.reason == "testnet_reset_detected"
        assert result.cycle_id is not None
        assert result.pre_reset_nav == 5000.0
        assert result.post_reset_balance == DEFAULT_TESTNET_BALANCE

    def test_not_testnet_no_trigger(self, mock_logger):
        result = self._check(env="production", nav=5000.0)
        assert result.triggered is False
        assert result.reason == "not_testnet"

    def test_balance_not_default_no_trigger(self, mock_logger):
        result = self._check(balance=8500.0)
        assert result.triggered is False
        assert result.reason == "balance_not_default"

    def test_positions_open_no_trigger(self, mock_logger):
        positions = [{"symbol": "BTCUSDT", "positionAmt": "0.01"}]
        result = self._check(positions=positions)
        assert result.triggered is False
        assert result.reason == "positions_open"

    def test_no_prior_nav_no_trigger(self, mock_logger):
        result = self._check(nav=0.0)
        assert result.triggered is False
        assert result.reason == "no_prior_nav"

    def test_small_delta_no_trigger(self, mock_logger):
        # NAV close to default — not a reset, just normal trading
        result = self._check(nav=9800.0)
        assert result.triggered is False
        assert result.reason == "delta_below_threshold"

    def test_zero_qty_positions_ignored(self, mock_logger, mock_env_meta, mock_archive):
        """Positions with qty=0 should not block detection."""
        positions = [{"symbol": "BTCUSDT", "positionAmt": "0"}]
        result = self._check(positions=positions, nav=5000.0)
        assert result.triggered is True


# ── Production Safety ─────────────────────────────────────────────────────────


class TestProductionSafety:
    """Reset guard must NEVER trigger in production."""

    def test_production_never_triggers(self, mock_logger):
        result = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="production",
            last_logged_nav=5000.0,
        )
        assert result.triggered is False
        assert result.reason == "not_testnet"

    def test_production_anomaly_logged(self, mock_logger):
        """Balance anomaly in prod should log CRITICAL event, not trigger reset."""
        check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="production",
            last_logged_nav=5000.0,
        )
        anomaly_events = [e for e in mock_logger if e["event_type"] == "BALANCE_ANOMALY_PRODUCTION"]
        assert len(anomaly_events) == 1
        assert anomaly_events[0]["action"] == "CRITICAL_LOGGED"

    def test_production_normal_balance_no_log(self, mock_logger):
        """Normal balance in prod should not log anything."""
        check_for_testnet_reset(
            exchange_balance=9800.0,
            exchange_positions=[],
            env="production",
            last_logged_nav=9850.0,
        )
        assert len(mock_logger) == 0


# ── Debounce ──────────────────────────────────────────────────────────────────


class TestDebounce:
    def test_second_trigger_debounced(self, mock_logger, mock_env_meta, mock_archive):
        """Second reset within debounce window should be suppressed."""
        first = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert first.triggered is True

        second = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert second.triggered is False
        assert second.reason == "debounce"

    def test_trigger_after_debounce_expires(self, mock_logger, mock_env_meta, mock_archive):
        """After debounce window, should trigger again."""
        first = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert first.triggered is True

        # Fast-forward past debounce
        import execution.reset_guard as rg
        rg._last_reset_ts = time.time() - RESET_DEBOUNCE_S - 1

        third = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert third.triggered is True


# ── Event Logging ─────────────────────────────────────────────────────────────


class TestEventLogging:
    def test_reset_event_logged(self, mock_logger, mock_env_meta, mock_archive):
        check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        reset_events = [e for e in mock_logger if e["event_type"] == "TESTNET_RESET"]
        assert len(reset_events) == 1
        evt = reset_events[0]
        assert evt["event"] == "TESTNET_RESET"
        assert evt["pre_reset_nav"] == 5000.0
        assert evt["post_reset_balance"] == DEFAULT_TESTNET_BALANCE
        assert "cycle_id" in evt
        assert "archive_path" in evt

    def test_no_trigger_no_log(self, mock_logger):
        check_for_testnet_reset(
            exchange_balance=8500.0,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert len(mock_logger) == 0


# ── Archive ───────────────────────────────────────────────────────────────────


class TestArchive:
    def test_archive_creates_directory(self, mock_logger, mock_env_meta, mock_archive, mock_state_files):
        result = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert result.triggered is True
        archive_dir = Path(result.archive_path)
        assert archive_dir.exists()
        assert archive_dir.is_dir()

    def test_archive_copies_state_files(self, mock_logger, mock_env_meta, mock_archive, mock_state_files):
        result = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        archive_dir = Path(result.archive_path)
        assert (archive_dir / "nav_state.json").exists()
        assert (archive_dir / "episode_ledger.json").exists()
        assert (archive_dir / "positions_state.json").exists()

    def test_archive_preserves_content(self, mock_logger, mock_env_meta, mock_archive, mock_state_files):
        result = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        archive_dir = Path(result.archive_path)
        archived_nav = json.loads((archive_dir / "nav_state.json").read_text())
        assert archived_nav["total_equity"] == 9500.0


# ── Environment Watermark ─────────────────────────────────────────────────────


class TestEnvironmentMeta:
    def test_meta_written_on_trigger(self, mock_logger, mock_env_meta, mock_archive):
        result = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert mock_env_meta.exists()
        meta = json.loads(mock_env_meta.read_text())
        assert meta["cycle_id"] == result.cycle_id
        assert "last_testnet_reset_ts" in meta
        assert "last_testnet_reset_iso" in meta

    def test_meta_not_written_on_no_trigger(self, mock_logger, mock_env_meta):
        check_for_testnet_reset(
            exchange_balance=8500.0,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert not mock_env_meta.exists()

    def test_cycle_id_increments(self, mock_logger, mock_env_meta, mock_archive):
        check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        meta1 = json.loads(mock_env_meta.read_text())
        assert meta1["cycle_id"] == "CYCLE_TEST_001"

        # Expire debounce and trigger again
        import execution.reset_guard as rg
        rg._last_reset_ts = 0.0

        check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        meta2 = json.loads(mock_env_meta.read_text())
        assert meta2["cycle_id"] == "CYCLE_TEST_002"


# ── ResetResult Dataclass ────────────────────────────────────────────────────


class TestResetResult:
    def test_default_not_triggered(self):
        r = ResetResult()
        assert r.triggered is False
        assert r.reason is None
        assert r.cycle_id is None

    def test_fields_populated_on_trigger(self, mock_logger, mock_env_meta, mock_archive):
        r = check_for_testnet_reset(
            exchange_balance=DEFAULT_TESTNET_BALANCE,
            exchange_positions=[],
            env="testnet",
            last_logged_nav=5000.0,
        )
        assert r.triggered is True
        assert r.pre_reset_nav == 5000.0
        assert r.post_reset_balance == DEFAULT_TESTNET_BALANCE
        assert r.cycle_id.startswith("CYCLE_TEST_")
        assert r.archive_path is not None
