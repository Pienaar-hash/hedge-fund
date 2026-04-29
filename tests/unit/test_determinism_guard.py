"""Tests for execution.determinism_guard — runtime invariant checker."""

from __future__ import annotations

import os
import time
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from execution.determinism_guard import (
    CHECK_INTERVAL_S,
    HYSTERESIS_CLEAR_S,
    DeterminismSnapshot,
    _read_avail_mem_pct,
    _read_mem_psi_avg10,
    _read_proc_status,
    _read_system_swap,
    _reset_cache,
    check_determinism,
    compute_snapshot_hash,
    get_proc_read_failures,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level cache between tests."""
    _reset_cache()
    yield
    _reset_cache()


# ---------------------------------------------------------------------------
# Unit: /proc readers
# ---------------------------------------------------------------------------


class TestReadProcStatus:
    def test_reads_own_process(self):
        swap_kb, rss_kb = _read_proc_status(os.getpid())
        # Our test process should be readable
        assert rss_kb is not None
        assert rss_kb > 0
        # swap may be 0 or positive
        assert swap_kb is not None
        assert swap_kb >= 0

    def test_nonexistent_pid(self):
        swap_kb, rss_kb = _read_proc_status(999999999)
        assert swap_kb is None
        assert rss_kb is None


class TestReadSystemSwap:
    def test_returns_int(self):
        result = _read_system_swap()
        # Should work on any Linux system
        if result is not None:
            assert isinstance(result, int)
            assert result >= 0


class TestReadMemPsi:
    def test_returns_float_or_none(self):
        result = _read_mem_psi_avg10()
        # /proc/pressure/memory may not exist on all kernels
        if result is not None:
            assert isinstance(result, float)
            assert result >= 0.0


class TestReadAvailMemPct:
    def test_returns_percentage(self):
        result = _read_avail_mem_pct()
        if result is not None:
            assert isinstance(result, int)
            assert 0 <= result <= 100


# ---------------------------------------------------------------------------
# Unit: check_determinism
# ---------------------------------------------------------------------------


class TestCheckDeterminism:
    def test_returns_snapshot(self):
        snap = check_determinism(force=True)
        assert isinstance(snap, DeterminismSnapshot)
        assert isinstance(snap.degraded, bool)
        assert isinstance(snap.violations, list)
        assert snap.ts > 0

    def test_caching(self):
        snap1 = check_determinism(force=True)
        snap2 = check_determinism()  # should return cached
        assert snap1 is snap2

    def test_force_bypasses_cache(self):
        snap1 = check_determinism(force=True)
        snap2 = check_determinism(force=True)
        # Both should be valid snapshots (may or may not be same object)
        assert snap1.ts > 0
        assert snap2.ts > 0

    def test_snapshot_as_dict(self):
        snap = check_determinism(force=True)
        d = snap.as_dict()
        assert "ts" in d
        assert "degraded" in d
        assert "violations" in d
        assert "executor_swap_kb" in d

    def test_degraded_when_swap_exceeds_threshold(self):
        """Simulate high executor swap to verify detection."""
        with patch(
            "execution.determinism_guard._read_proc_status",
            return_value=(50000, 200000),  # 50MB swap
        ):
            snap = check_determinism(force=True)
            assert snap.degraded is True
            assert any("EXECUTOR_SWAP" in v for v in snap.violations)

    def test_healthy_when_no_swap(self):
        """Simulate zero swap across all checks."""
        with patch(
            "execution.determinism_guard._read_proc_status",
            return_value=(0, 200000),
        ), patch(
            "execution.determinism_guard._read_system_swap",
            return_value=0,
        ), patch(
            "execution.determinism_guard._read_mem_psi_avg10",
            return_value=0.0,
        ), patch(
            "execution.determinism_guard._read_avail_mem_pct",
            return_value=60,
        ):
            snap = check_determinism(force=True)
            assert snap.degraded is False
            assert snap.violations == []

    def test_degraded_on_low_memory(self):
        """Simulate low available memory."""
        with patch(
            "execution.determinism_guard._read_proc_status",
            return_value=(0, 200000),
        ), patch(
            "execution.determinism_guard._read_system_swap",
            return_value=0,
        ), patch(
            "execution.determinism_guard._read_mem_psi_avg10",
            return_value=0.0,
        ), patch(
            "execution.determinism_guard._read_avail_mem_pct",
            return_value=8,  # Below 15% floor
        ):
            snap = check_determinism(force=True)
            assert snap.degraded is True
            assert any("LOW_MEMORY" in v for v in snap.violations)

    def test_degraded_on_memory_psi(self):
        """Simulate memory pressure stalls."""
        with patch(
            "execution.determinism_guard._read_proc_status",
            return_value=(0, 200000),
        ), patch(
            "execution.determinism_guard._read_system_swap",
            return_value=0,
        ), patch(
            "execution.determinism_guard._read_mem_psi_avg10",
            return_value=5.5,  # Above 1% threshold
        ), patch(
            "execution.determinism_guard._read_avail_mem_pct",
            return_value=60,
        ):
            snap = check_determinism(force=True)
            assert snap.degraded is True
            assert any("MEMORY_PSI" in v for v in snap.violations)

    def test_failopen_on_all_none(self):
        """If /proc is completely unreadable, should NOT veto."""
        with patch(
            "execution.determinism_guard._read_proc_status",
            return_value=(None, None),
        ), patch(
            "execution.determinism_guard._read_system_swap",
            return_value=None,
        ), patch(
            "execution.determinism_guard._read_mem_psi_avg10",
            return_value=None,
        ), patch(
            "execution.determinism_guard._read_avail_mem_pct",
            return_value=None,
        ):
            snap = check_determinism(force=True)
            assert snap.degraded is False
            assert snap.violations == []


# ---------------------------------------------------------------------------
# Integration: DoctrineVerdict has the new enum member
# ---------------------------------------------------------------------------


class TestDoctrineVerdictExtension:
    def test_veto_environment_degraded_exists(self):
        from execution.doctrine_kernel import DoctrineVerdict
        assert hasattr(DoctrineVerdict, "VETO_ENVIRONMENT_DEGRADED")
        assert DoctrineVerdict.VETO_ENVIRONMENT_DEGRADED.value == "VETO_ENVIRONMENT_DEGRADED"


# ---------------------------------------------------------------------------
# Episode linkage: for_episode()
# ---------------------------------------------------------------------------


class TestForEpisode:
    def test_ok_snapshot(self):
        """for_episode() returns compact dict for healthy state."""
        with patch("execution.determinism_guard._read_proc_status", return_value=(0, 200000)), \
             patch("execution.determinism_guard._read_system_swap", return_value=50), \
             patch("execution.determinism_guard._read_mem_psi_avg10", return_value=0.1), \
             patch("execution.determinism_guard._read_avail_mem_pct", return_value=55):
            snap = check_determinism(force=True)
            ep = snap.for_episode()
            assert ep["determinism"] == "OK"
            assert ep["vm_swap_kb"] == 0
            assert ep["sys_swap_mb"] == 50
            assert ep["mem_psi_avg10"] == 0.1
            assert ep["mem_avail_pct"] == 55
            assert ep["violations"] == 0

    def test_degraded_snapshot(self):
        """for_episode() reflects DEGRADED with violation count."""
        with patch("execution.determinism_guard._read_proc_status", return_value=(50000, 200000)), \
             patch("execution.determinism_guard._read_system_swap", return_value=600), \
             patch("execution.determinism_guard._read_mem_psi_avg10", return_value=2.0), \
             patch("execution.determinism_guard._read_avail_mem_pct", return_value=10):
            snap = check_determinism(force=True)
            ep = snap.for_episode()
            assert ep["determinism"] == "DEGRADED"
            assert ep["violations"] >= 2  # Multiple violations


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:

    # Helpers to mock all /proc readers at once
    @staticmethod
    def _patch_all_clean():
        return (
            patch("execution.determinism_guard._read_proc_status", return_value=(0, 200000)),
            patch("execution.determinism_guard._read_system_swap", return_value=0),
            patch("execution.determinism_guard._read_mem_psi_avg10", return_value=0.0),
            patch("execution.determinism_guard._read_avail_mem_pct", return_value=60),
        )

    @staticmethod
    def _patch_all_degraded():
        return (
            patch("execution.determinism_guard._read_proc_status", return_value=(50000, 200000)),
            patch("execution.determinism_guard._read_system_swap", return_value=0),
            patch("execution.determinism_guard._read_mem_psi_avg10", return_value=0.0),
            patch("execution.determinism_guard._read_avail_mem_pct", return_value=60),
        )

    def test_clean_after_degraded_holds(self):
        """First clean reading after degradation should still report DEGRADED."""
        # Step 1: establish degraded state
        with self._patch_all_degraded()[0], self._patch_all_degraded()[1], \
             self._patch_all_degraded()[2], self._patch_all_degraded()[3]:
            snap1 = check_determinism(force=True)
            assert snap1.degraded is True

        # Step 2: clean reading — but hysteresis holds
        with self._patch_all_clean()[0], self._patch_all_clean()[1], \
             self._patch_all_clean()[2], self._patch_all_clean()[3]:
            snap2 = check_determinism(force=True)
            assert snap2.degraded is True
            assert snap2.held_degraded is True
            assert any("HYSTERESIS_HOLD" in v for v in snap2.violations)

    def test_clean_after_hysteresis_clears(self):
        """After hysteresis period, clean reading returns OK."""
        import execution.determinism_guard as dg

        # Step 1: degraded
        with self._patch_all_degraded()[0], self._patch_all_degraded()[1], \
             self._patch_all_degraded()[2], self._patch_all_degraded()[3]:
            check_determinism(force=True)

        # Step 2: first clean — starts hysteresis timer
        with self._patch_all_clean()[0], self._patch_all_clean()[1], \
             self._patch_all_clean()[2], self._patch_all_clean()[3]:
            check_determinism(force=True)

        # Step 3: simulate time passing beyond hysteresis window
        dg._first_clean_ts -= (HYSTERESIS_CLEAR_S + 1)

        with self._patch_all_clean()[0], self._patch_all_clean()[1], \
             self._patch_all_clean()[2], self._patch_all_clean()[3]:
            snap = check_determinism(force=True)
            assert snap.degraded is False
            assert snap.held_degraded is False

    def test_no_hysteresis_when_never_degraded(self):
        """First-ever clean reading should be OK immediately."""
        with self._patch_all_clean()[0], self._patch_all_clean()[1], \
             self._patch_all_clean()[2], self._patch_all_clean()[3]:
            snap = check_determinism(force=True)
            assert snap.degraded is False
            assert snap.held_degraded is False

    def test_re_degradation_resets_hysteresis(self):
        """If violations return during hysteresis hold, timer resets."""
        # Degraded → clean (hold) → degraded again
        with self._patch_all_degraded()[0], self._patch_all_degraded()[1], \
             self._patch_all_degraded()[2], self._patch_all_degraded()[3]:
            check_determinism(force=True)

        with self._patch_all_clean()[0], self._patch_all_clean()[1], \
             self._patch_all_clean()[2], self._patch_all_clean()[3]:
            snap_hold = check_determinism(force=True)
            assert snap_hold.held_degraded is True

        # Re-degrade — should be degraded with real violations, not hysteresis
        with self._patch_all_degraded()[0], self._patch_all_degraded()[1], \
             self._patch_all_degraded()[2], self._patch_all_degraded()[3]:
            snap_redegrade = check_determinism(force=True)
            assert snap_redegrade.degraded is True
            assert snap_redegrade.held_degraded is False
            assert any("EXECUTOR_SWAP" in v for v in snap_redegrade.violations)


# ---------------------------------------------------------------------------
# Proc-failure counter
# ---------------------------------------------------------------------------


class TestProcFailureCounter:
    def test_counter_increments_on_read_failure(self):
        """Proc read failures should be counted."""
        initial = get_proc_read_failures()
        _read_proc_status(999999999)  # nonexistent PID
        assert get_proc_read_failures() > initial

    def test_counter_accessible(self):
        """get_proc_read_failures() returns an integer."""
        count = get_proc_read_failures()
        assert isinstance(count, int)
        assert count >= 0


# ---------------------------------------------------------------------------
# Snapshot hash
# ---------------------------------------------------------------------------


class TestSnapshotHash:
    def test_deterministic(self):
        """Same state → same hash."""
        state = {"degraded": False, "ts": 1.0, "violations": []}
        assert compute_snapshot_hash(state) == compute_snapshot_hash(state)

    def test_different_state_different_hash(self):
        a = {"degraded": False, "ts": 1.0}
        b = {"degraded": True, "ts": 1.0}
        assert compute_snapshot_hash(a) != compute_snapshot_hash(b)

    def test_key_order_irrelevant(self):
        """Hash uses sort_keys — insertion order must not matter."""
        a = {"b": 2, "a": 1}
        b = {"a": 1, "b": 2}
        assert compute_snapshot_hash(a) == compute_snapshot_hash(b)

    def test_length_is_16(self):
        h = compute_snapshot_hash({"x": 1})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_snapshot_method_matches_standalone(self):
        """DeterminismSnapshot.snapshot_hash() must agree with compute_snapshot_hash(as_dict())."""
        snap = DeterminismSnapshot(
            ts=1700000000.0,
            degraded=False,
            violations=[],
            executor_swap_kb=0,
            executor_rss_kb=200000,
            system_swap_used_mb=50,
            mem_psi_avg10=0.1,
            avail_mem_pct=60,
        )
        assert snap.snapshot_hash() == compute_snapshot_hash(snap.as_dict())

    def test_degraded_snapshot_hash(self):
        """Hash works for degraded snapshots with violations."""
        snap = DeterminismSnapshot(
            ts=1700000000.0,
            degraded=True,
            violations=["EXECUTOR_SWAP: 50MB swapped"],
            executor_swap_kb=51200,
            held_degraded=False,
        )
        h = snap.snapshot_hash()
        assert len(h) == 16
        assert h == compute_snapshot_hash(snap.as_dict())
