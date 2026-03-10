"""Unit tests for execution.executor_state.ExecutorState (Phase 5)."""

from __future__ import annotations

import time

import pytest

from execution.executor_state import ExecutorState
from execution.position_cache import PositionCache


class TestExecutorStateConstruction:
    """Commit 1 acceptance: construction, defaults, identity."""

    def test_required_position_cache(self):
        """position_cache is mandatory — no mutable default footgun."""
        with pytest.raises(TypeError):
            ExecutorState()  # type: ignore[call-arg]

    def test_defaults_populated(self):
        cache = PositionCache()
        state = ExecutorState(position_cache=cache)
        assert state.position_cache is cache
        assert isinstance(state.run_id, str) and len(state.run_id) > 0
        assert isinstance(state.engine_version, str) and len(state.engine_version) > 0
        assert state.executor_start_ts > 0
        assert state.last_cycle_ts == 0.0
        assert state.last_disk_check_ts == 0.0
        assert state.last_disk_alert_ts == 0.0
        assert state.last_signal_pull == 0.0
        assert state.last_queue_depth == 0
        assert state.last_episode_ledger_rebuild_ts == 0.0
        assert state.last_doctrine_block_log_ts == 0.0

    def test_explicit_overrides(self):
        cache = PositionCache()
        state = ExecutorState(
            position_cache=cache,
            run_id="test-run-42",
            engine_version="v99.0",
            executor_start_ts=1000.0,
            last_cycle_ts=999.0,
        )
        assert state.run_id == "test-run-42"
        assert state.engine_version == "v99.0"
        assert state.executor_start_ts == 1000.0
        assert state.last_cycle_ts == 999.0

    def test_no_shared_cache_between_instances(self):
        """Each ExecutorState must use the cache it was given — no class-default sharing."""
        c1 = PositionCache()
        c2 = PositionCache()
        s1 = ExecutorState(position_cache=c1)
        s2 = ExecutorState(position_cache=c2)
        assert s1.position_cache is not s2.position_cache

    def test_touch_cycle(self):
        cache = PositionCache()
        state = ExecutorState(position_cache=cache, last_cycle_ts=0.0)
        assert state.last_cycle_ts == 0.0
        before = time.time()
        state.touch_cycle()
        after = time.time()
        assert before <= state.last_cycle_ts <= after

    def test_mutable_fields(self):
        """State fields are freely mutable (not frozen)."""
        cache = PositionCache()
        state = ExecutorState(position_cache=cache)
        state.last_disk_check_ts = 123.0
        assert state.last_disk_check_ts == 123.0
        state.last_disk_alert_ts = 456.0
        assert state.last_disk_alert_ts == 456.0
