"""
Tests for B.4 — Episode Schema Binding (authority chain)

Covers:
  - Shadow log index loading (LINK + DECISION)
  - Nearest-time binary search matching
  - Deterministic episode_uid
  - Authority attachment with flags
  - Regime binding from DECISION context_snapshot
  - V2 output format + stats
  - Edge cases: missing log, ambiguous matches, strategy fallback
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest import mock

import pytest

from execution.episode_ledger import (
    AuthorityFlags,
    AuthorityRef,
    Episode,
    EpisodeLedger,
    EpisodeV2,
    EPISODE_LEDGER_SCHEMA_V2,
    MATCH_WINDOW_NARROW_S,
    MATCH_WINDOW_WIDE_S,
    _bind_authority,
    _compute_episode_uid,
    _find_nearest_link,
    _iso_to_unix,
    _LinkRecord,
    _DecisionRecord,
    _load_shadow_indexes,
    _resolve_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(offset_s: float = 0.0) -> str:
    """Create ISO timestamp with UTC offset from a fixed base."""
    base = datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(seconds=offset_s)).isoformat()


def _make_link_record(
    offset_s: float = 0.0,
    symbol: str = "BTCUSDT",
    action: str = "ENTRY",
    strategy: str = "TREND",
    request_id: str = "REQ_001",
    decision_id: str = "DEC_001",
    permit_id: str | None = "PERM_001",
) -> _LinkRecord:
    ts_iso = _make_ts(offset_s)
    ts_unix = _iso_to_unix(ts_iso)
    assert ts_unix is not None
    return _LinkRecord(
        ts_unix=ts_unix,
        ts_iso=ts_iso,
        request_id=request_id,
        decision_id=decision_id,
        permit_id=permit_id,
        symbol=symbol,
        action=action,
        strategy=strategy,
    )


def _make_episode(
    offset_entry_s: float = 0.0,
    offset_exit_s: float = 3600.0,
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    strategy: str = "TREND",
    episode_id: str = "EP_0001",
) -> Episode:
    return Episode(
        episode_id=episode_id,
        symbol=symbol,
        side=side,
        entry_ts=_make_ts(offset_entry_s),
        exit_ts=_make_ts(offset_exit_s),
        duration_hours=round(offset_exit_s / 3600, 2),
        entry_fills=1,
        exit_fills=1,
        entry_notional=1000.0,
        exit_notional=1050.0,
        total_qty=0.01,
        avg_entry_price=50000.0,
        avg_exit_price=52500.0,
        gross_pnl=25.0,
        fees=2.0,
        net_pnl=23.0,
        regime_at_entry="unknown",
        regime_at_exit="unknown",
        exit_reason="TAKE_PROFIT",
        exit_reason_raw="tp_hit",
        strategy=strategy,
    )


def _write_shadow_log(path: Path, events: list[dict]) -> None:
    """Write shadow events to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for evt in events:
            f.write(json.dumps(evt, sort_keys=True) + "\n")


def _make_shadow_link(
    offset_s: float = 0.0,
    symbol: str = "BTCUSDT",
    action: str = "ENTRY",
    strategy: str = "TREND",
    request_id: str = "REQ_001",
    decision_id: str = "DEC_001",
    permit_id: str | None = "PERM_001",
) -> dict:
    ts = _make_ts(offset_s)
    return {
        "schema_version": "dle_shadow_v1",
        "event_type": "LINK",
        "ts": ts,
        "payload": {
            "ts": ts,
            "request_id": request_id,
            "decision_id": decision_id,
            "permit_id": permit_id,
            "symbol": symbol,
            "requested_action": action,
            "strategy": strategy,
        },
    }


def _make_shadow_decision(
    decision_id: str = "DEC_001",
    regime: str = "TREND_UP",
    offset_s: float = 0.0,
) -> dict:
    ts = _make_ts(offset_s)
    return {
        "schema_version": "dle_shadow_v2",
        "event_type": "DECISION",
        "ts": ts,
        "payload": {
            "decision_id": decision_id,
            "context_snapshot": {"regime": regime, "nav_usd": 10000.0},
            "provenance": {"engine_version": "v7.9", "git_sha": "abc123"},
        },
    }


# ===========================================================================
# Tests: Deterministic episode_uid
# ===========================================================================

class TestEpisodeUid:

    def test_uid_deterministic(self):
        uid1 = _compute_episode_uid("BTCUSDT", "LONG", "2026-01-01T00:00:00+00:00",
                                     "2026-01-01T01:00:00+00:00", 0.01, 50000.0, 52500.0)
        uid2 = _compute_episode_uid("BTCUSDT", "LONG", "2026-01-01T00:00:00+00:00",
                                     "2026-01-01T01:00:00+00:00", 0.01, 50000.0, 52500.0)
        assert uid1 == uid2
        assert uid1.startswith("EP_")
        assert len(uid1) == 3 + 12  # "EP_" + 12 hex chars

    def test_uid_changes_on_symbol(self):
        uid1 = _compute_episode_uid("BTCUSDT", "LONG", "ts1", "ts2", 0.01, 50000.0, 52500.0)
        uid2 = _compute_episode_uid("ETHUSDT", "LONG", "ts1", "ts2", 0.01, 50000.0, 52500.0)
        assert uid1 != uid2

    def test_uid_changes_on_side(self):
        uid1 = _compute_episode_uid("BTCUSDT", "LONG", "ts1", "ts2", 0.01, 50000.0, 52500.0)
        uid2 = _compute_episode_uid("BTCUSDT", "SHORT", "ts1", "ts2", 0.01, 50000.0, 52500.0)
        assert uid1 != uid2

    def test_uid_changes_on_entry_ts(self):
        uid1 = _compute_episode_uid("BTCUSDT", "LONG", "ts1", "ts2", 0.01, 50000.0, 52500.0)
        uid2 = _compute_episode_uid("BTCUSDT", "LONG", "ts1_diff", "ts2", 0.01, 50000.0, 52500.0)
        assert uid1 != uid2

    def test_uid_changes_on_qty(self):
        uid1 = _compute_episode_uid("BTCUSDT", "LONG", "ts1", "ts2", 0.01, 50000.0, 52500.0)
        uid2 = _compute_episode_uid("BTCUSDT", "LONG", "ts1", "ts2", 0.02, 50000.0, 52500.0)
        assert uid1 != uid2

    def test_uid_stable_under_float_jitter(self):
        """Rounding should absorb small float differences."""
        uid1 = _compute_episode_uid("BTCUSDT", "LONG", "ts1", "ts2", 0.010000, 50000.0001, 52500.0)
        uid2 = _compute_episode_uid("BTCUSDT", "LONG", "ts1", "ts2", 0.010000, 50000.0001, 52500.0)
        assert uid1 == uid2


# ===========================================================================
# Tests: Nearest-time matcher
# ===========================================================================

class TestFindNearestLink:

    def test_empty_candidates_returns_none(self):
        match, ambig = _find_nearest_link(1000.0, [], "TREND")
        assert match is None
        assert ambig is False

    def test_exact_match(self):
        rec = _make_link_record(offset_s=0.0)
        match, ambig = _find_nearest_link(rec.ts_unix, [rec], "TREND")
        assert match is rec
        assert ambig is False

    def test_match_within_narrow_window(self):
        rec = _make_link_record(offset_s=0.0)
        # Query 60s after the LINK
        match, ambig = _find_nearest_link(rec.ts_unix + 60, [rec], "TREND")
        assert match is rec
        assert ambig is False

    def test_no_match_outside_wide_window(self):
        rec = _make_link_record(offset_s=0.0)
        # Query 700s after — outside 600s wide window
        match, ambig = _find_nearest_link(rec.ts_unix + 700, [rec], "TREND")
        assert match is None
        assert ambig is False

    def test_match_in_wide_window_after_narrow_miss(self):
        rec = _make_link_record(offset_s=0.0)
        # Query 200s after — outside 120s narrow, inside 600s wide
        match, ambig = _find_nearest_link(rec.ts_unix + 200, [rec], "TREND")
        assert match is rec
        assert ambig is False

    def test_ambiguous_equidistant_different_strategy(self):
        """Two LINK records equidistant from query, different strategies → ambiguous."""
        rec1 = _make_link_record(offset_s=-10, strategy="TREND", request_id="R1")
        rec2 = _make_link_record(offset_s=10, strategy="MEAN_REVERT", request_id="R2")
        mid_ts = (rec1.ts_unix + rec2.ts_unix) / 2
        match, ambig = _find_nearest_link(mid_ts, [rec1, rec2], "VOL_HARVEST")
        assert match is None
        assert ambig is True

    def test_strategy_disambiguation(self):
        """Two equidistant, one matches strategy hint → resolved."""
        rec1 = _make_link_record(offset_s=-10, strategy="TREND", request_id="R1")
        rec2 = _make_link_record(offset_s=10, strategy="MEAN_REVERT", request_id="R2")
        mid_ts = (rec1.ts_unix + rec2.ts_unix) / 2
        match, ambig = _find_nearest_link(mid_ts, [rec1, rec2], "TREND")
        assert match is rec1
        assert ambig is False

    def test_nearest_when_multiple_in_window(self):
        """Multiple candidates, closest wins."""
        rec_far = _make_link_record(offset_s=0, request_id="FAR")
        rec_close = _make_link_record(offset_s=50, request_id="CLOSE")
        match, ambig = _find_nearest_link(rec_close.ts_unix - 5, [rec_far, rec_close], "TREND")
        assert match is rec_close
        assert ambig is False


# ===========================================================================
# Tests: Regime binding
# ===========================================================================

class TestResolveRegime:

    def test_missing_decision_id(self):
        ref = AuthorityRef(decision_id=None)
        assert _resolve_regime(ref, {}) == "unknown"

    def test_decision_not_in_index(self):
        ref = AuthorityRef(decision_id="DEC_MISSING")
        assert _resolve_regime(ref, {}) == "unknown"

    def test_decision_with_regime(self):
        ref = AuthorityRef(decision_id="DEC_001")
        idx = {"DEC_001": _DecisionRecord(
            decision_id="DEC_001",
            context_snapshot={"regime": "TREND_UP"},
            provenance={},
        )}
        assert _resolve_regime(ref, idx) == "TREND_UP"

    def test_decision_with_empty_snapshot(self):
        ref = AuthorityRef(decision_id="DEC_001")
        idx = {"DEC_001": _DecisionRecord(
            decision_id="DEC_001",
            context_snapshot={},
            provenance={},
        )}
        assert _resolve_regime(ref, idx) == "unknown"

    def test_decision_with_none_regime(self):
        ref = AuthorityRef(decision_id="DEC_001")
        idx = {"DEC_001": _DecisionRecord(
            decision_id="DEC_001",
            context_snapshot={"regime": None},
            provenance={},
        )}
        assert _resolve_regime(ref, idx) == "unknown"


# ===========================================================================
# Tests: Shadow log index loading
# ===========================================================================

class TestLoadShadowIndexes:

    def test_missing_file_returns_empty(self, tmp_path):
        with mock.patch("execution.episode_ledger.DLE_SHADOW_LOG_PATH", tmp_path / "nope.jsonl"):
            le, lx, di = _load_shadow_indexes()
            assert le == {}
            assert lx == {}
            assert di == {}

    def test_load_link_events(self, tmp_path):
        log_path = tmp_path / "shadow.jsonl"
        events = [
            _make_shadow_link(offset_s=0, action="ENTRY", decision_id="DEC_E"),
            _make_shadow_link(offset_s=3600, action="EXIT", decision_id="DEC_X"),
        ]
        _write_shadow_log(log_path, events)

        with mock.patch("execution.episode_ledger.DLE_SHADOW_LOG_PATH", log_path):
            le, lx, di = _load_shadow_indexes()
            # Exact strategy key + wildcard key
            assert ("BTCUSDT", "TREND") in le
            assert ("BTCUSDT", "*") in le
            assert ("BTCUSDT", "TREND") in lx
            assert ("BTCUSDT", "*") in lx
            assert len(le[("BTCUSDT", "TREND")]) == 1
            assert len(lx[("BTCUSDT", "TREND")]) == 1
            # DECISION not loaded for LINK events
            assert di == {}

    def test_load_decision_events(self, tmp_path):
        log_path = tmp_path / "shadow.jsonl"
        events = [
            _make_shadow_decision(decision_id="DEC_001", regime="TREND_UP"),
            _make_shadow_decision(decision_id="DEC_002", regime="CHOPPY"),
        ]
        _write_shadow_log(log_path, events)

        with mock.patch("execution.episode_ledger.DLE_SHADOW_LOG_PATH", log_path):
            le, lx, di = _load_shadow_indexes()
            assert len(di) == 2
            assert di["DEC_001"].context_snapshot["regime"] == "TREND_UP"
            assert di["DEC_002"].context_snapshot["regime"] == "CHOPPY"

    def test_malformed_lines_skipped(self, tmp_path):
        log_path = tmp_path / "shadow.jsonl"
        with open(log_path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps(_make_shadow_link(offset_s=0)) + "\n")
            f.write("{}\n")  # valid JSON but no event_type

        with mock.patch("execution.episode_ledger.DLE_SHADOW_LOG_PATH", log_path):
            le, lx, di = _load_shadow_indexes()
            assert ("BTCUSDT", "TREND") in le

    def test_link_events_sorted_by_ts(self, tmp_path):
        log_path = tmp_path / "shadow.jsonl"
        events = [
            _make_shadow_link(offset_s=100, request_id="R2"),
            _make_shadow_link(offset_s=0, request_id="R1"),
            _make_shadow_link(offset_s=50, request_id="R3"),
        ]
        _write_shadow_log(log_path, events)

        with mock.patch("execution.episode_ledger.DLE_SHADOW_LOG_PATH", log_path):
            le, lx, di = _load_shadow_indexes()
            bucket = le[("BTCUSDT", "TREND")]
            assert [r.request_id for r in bucket] == ["R1", "R3", "R2"]

    def test_link_without_payload_ts_falls_back_to_toplevel(self, tmp_path):
        """LINK events without payload.ts should fall back to top-level ts."""
        log_path = tmp_path / "shadow.jsonl"
        ts = _make_ts(0)
        event = {
            "schema_version": "dle_shadow_v1",
            "event_type": "LINK",
            "ts": ts,
            "payload": {
                # No "ts" in payload — old-format LINK
                "request_id": "REQ_OLD",
                "decision_id": "DEC_OLD",
                "permit_id": "PERM_OLD",
                "symbol": "BTCUSDT",
                "requested_action": "ENTRY",
                "strategy": "TREND",
            },
        }
        _write_shadow_log(log_path, [event])

        with mock.patch("execution.episode_ledger.DLE_SHADOW_LOG_PATH", log_path):
            le, lx, di = _load_shadow_indexes()
            assert ("BTCUSDT", "TREND") in le
            assert le[("BTCUSDT", "TREND")][0].request_id == "REQ_OLD"


# ===========================================================================
# Tests: Full authority binding
# ===========================================================================

class TestBindAuthority:

    def test_bind_with_matching_links(self):
        ep = _make_episode(offset_entry_s=0, offset_exit_s=3600)
        entry_link = _make_link_record(offset_s=5, action="ENTRY", decision_id="DEC_E", permit_id="PERM_E", request_id="REQ_E")
        exit_link = _make_link_record(offset_s=3610, action="EXIT", decision_id="DEC_X", permit_id="PERM_X", request_id="REQ_X")

        le = {("BTCUSDT", "TREND"): [entry_link]}
        lx = {("BTCUSDT", "TREND"): [exit_link]}
        di = {
            "DEC_E": _DecisionRecord("DEC_E", {"regime": "TREND_UP"}, {}),
            "DEC_X": _DecisionRecord("DEC_X", {"regime": "TREND_DOWN"}, {}),
        }

        v2s, stats = _bind_authority([ep], le, lx, di)
        assert len(v2s) == 1
        v2 = v2s[0]

        # Authority refs
        assert v2.authority_entry.decision_id == "DEC_E"
        assert v2.authority_entry.permit_id == "PERM_E"
        assert v2.authority_exit.decision_id == "DEC_X"
        assert v2.authority_exit.permit_id == "PERM_X"

        # Flags
        assert v2.authority_flags.entry_missing is False
        assert v2.authority_flags.exit_missing is False
        assert v2.authority_flags.entry_ambiguous is False
        assert v2.authority_flags.exit_ambiguous is False

        # Regime from DECISION
        assert v2.regime_at_entry == "TREND_UP"
        assert v2.regime_at_exit == "TREND_DOWN"
        assert v2.regime_bindable is True

        # UID
        assert v2.episode_uid.startswith("EP_")
        assert len(v2.episode_uid) == 15

        # Stats
        assert stats["entry_coverage_pct"] == 100.0
        assert stats["exit_coverage_pct"] == 100.0
        assert stats["missing_count"] == 0
        assert stats["ambiguous_count"] == 0

    def test_bind_with_no_links_flags_missing(self):
        ep = _make_episode()
        v2s, stats = _bind_authority([ep], {}, {}, {})
        v2 = v2s[0]
        assert v2.authority_flags.entry_missing is True
        assert v2.authority_flags.exit_missing is True
        assert v2.regime_at_entry == "unknown"
        assert v2.regime_at_exit == "unknown"
        assert v2.regime_bindable is False
        assert stats["entry_coverage_pct"] == 0.0
        assert stats["missing_count"] == 1

    def test_bind_with_null_permit_on_entry(self):
        """permit_id=None on executed entry should be counted as inconsistency."""
        ep = _make_episode(offset_entry_s=0, offset_exit_s=3600)
        entry_link = _make_link_record(offset_s=5, action="ENTRY", permit_id=None, decision_id="DEC_E")

        le = {("BTCUSDT", "TREND"): [entry_link]}
        lx = {}
        di = {}

        v2s, stats = _bind_authority([ep], le, lx, di)
        assert stats["permit_null_on_executed_entry_count"] == 1

    def test_wildcard_fallback(self):
        """When exact strategy key misses, fall back to wildcard bucket."""
        ep = _make_episode(strategy="TREND")
        # LINK has strategy "EMERGENT_ALPHA" — won't match exact key for TREND
        entry_link = _make_link_record(offset_s=5, action="ENTRY", strategy="EMERGENT_ALPHA", request_id="REQ_WC")

        # Only wildcard bucket exists
        le = {("BTCUSDT", "*"): [entry_link]}
        lx = {}
        di = {}

        v2s, stats = _bind_authority([ep], le, lx, di)
        v2 = v2s[0]
        assert v2.authority_entry.request_id == "REQ_WC"
        assert v2.authority_flags.entry_missing is False

    def test_bind_preserves_v1_fields(self):
        ep = _make_episode()
        v2s, _ = _bind_authority([ep], {}, {}, {})
        v2 = v2s[0]
        assert v2.symbol == ep.symbol
        assert v2.side == ep.side
        assert v2.net_pnl == ep.net_pnl
        assert v2.exit_reason == ep.exit_reason
        assert v2.exit_reason_raw == ep.exit_reason_raw
        assert v2.strategy == ep.strategy
        assert v2.episode_id == ep.episode_id

    def test_max_time_delta_tracked(self):
        ep = _make_episode(offset_entry_s=0, offset_exit_s=3600)
        # LINK 100s from entry
        entry_link = _make_link_record(offset_s=100, action="ENTRY")
        le = {("BTCUSDT", "TREND"): [entry_link]}
        lx = {}
        di = {}

        v2s, stats = _bind_authority([ep], le, lx, di)
        assert stats["max_time_delta_s_entry"] == pytest.approx(100.0, abs=1.0)

    def test_ambiguous_entry_flagged(self):
        """Two equidistant links with different strategies → ambiguous."""
        ep = _make_episode(offset_entry_s=0)
        rec1 = _make_link_record(offset_s=-10, strategy="A", request_id="R1")
        rec2 = _make_link_record(offset_s=10, strategy="B", request_id="R2")
        le = {("BTCUSDT", "TREND"): [rec1, rec2]}  # won't match
        # Wildcard bucket
        le[("BTCUSDT", "*")] = [rec1, rec2]
        lx = {}
        di = {}

        v2s, stats = _bind_authority([ep], le, lx, di)
        v2 = v2s[0]
        assert v2.authority_flags.entry_ambiguous is True
        assert stats["ambiguous_count"] == 1


# ===========================================================================
# Tests: V2 output format (EpisodeLedger.to_dict)
# ===========================================================================

class TestEpisodeLedgerV2Output:

    def test_v2_surface_present_when_episodes_v2_exist(self):
        ep = _make_episode()
        ep_v2 = EpisodeV2(
            episode_id="EP_0001",
            episode_uid="EP_abc123def456",
            symbol="BTCUSDT", side="LONG",
            entry_ts="ts1", exit_ts="ts2", duration_hours=1.0,
            entry_fills=1, exit_fills=1,
            entry_notional=1000.0, exit_notional=1050.0,
            total_qty=0.01, avg_entry_price=50000.0, avg_exit_price=52500.0,
            gross_pnl=25.0, fees=2.0, net_pnl=23.0,
            regime_at_entry="TREND_UP", regime_at_exit="TREND_DOWN",
            exit_reason="TAKE_PROFIT", exit_reason_raw="tp_hit",
            strategy="TREND",
            authority_entry=AuthorityRef("REQ_E", "DEC_E", "PERM_E"),
            authority_exit=AuthorityRef("REQ_X", "DEC_X", "PERM_X"),
            authority_flags=AuthorityFlags(),
            regime_bindable=True,
        )
        ledger = EpisodeLedger(
            episodes=[ep],
            episodes_v2=[ep_v2],
            last_rebuild_ts="2026-02-13T12:00:00+00:00",
            stats={},
        )
        d = ledger.to_dict()
        assert d["schema_version"] == EPISODE_LEDGER_SCHEMA_V2
        assert "episodes_v2" in d
        assert len(d["episodes_v2"]) == 1
        assert "episodes" in d  # V1 still present
        assert d["episode_count"] == 1

    def test_v1_only_when_no_v2_episodes(self):
        ep = _make_episode()
        ledger = EpisodeLedger(episodes=[ep], last_rebuild_ts="ts", stats={})
        d = ledger.to_dict()
        assert "schema_version" not in d
        assert "episodes_v2" not in d
        assert "episodes" in d

    def test_v2_authority_nested_shape(self):
        """V2 JSON shape should have authority.entry/exit nesting."""
        ep_v2 = EpisodeV2(
            episode_id="EP_0001", episode_uid="EP_abcdef123456",
            symbol="BTCUSDT", side="LONG",
            entry_ts="ts1", exit_ts="ts2", duration_hours=1.0,
            entry_fills=1, exit_fills=1,
            entry_notional=1000.0, exit_notional=1050.0,
            total_qty=0.01, avg_entry_price=50000.0, avg_exit_price=52500.0,
            gross_pnl=25.0, fees=2.0, net_pnl=23.0,
            regime_at_entry="TREND_UP", regime_at_exit="unknown",
            exit_reason="TAKE_PROFIT", exit_reason_raw="tp_hit",
            strategy="TREND",
            authority_entry=AuthorityRef("REQ_1", "DEC_1", "PERM_1"),
            authority_exit=AuthorityRef(),
            authority_flags=AuthorityFlags(exit_missing=True),
            regime_bindable=True,
        )
        d = ep_v2.to_dict()
        assert "authority" in d
        assert "entry" in d["authority"]
        assert "exit" in d["authority"]
        assert d["authority"]["entry"]["decision_id"] == "DEC_1"
        assert d["authority"]["exit"]["decision_id"] is None
        assert d["authority_flags"]["exit_missing"] is True
        # Flat field still present
        assert "authority_entry" not in d  # should be nested, not flat


# ===========================================================================
# Tests: Integration — build_episode_ledger with shadow log
# ===========================================================================

class TestBuildEpisodeLedgerWithShadow:

    def test_build_with_shadow_log(self, tmp_path):
        """Full integration: shadow log → authority binding in V2 episodes."""
        shadow_path = tmp_path / "shadow.jsonl"
        events = [
            _make_shadow_decision(decision_id="DEC_E", regime="TREND_UP", offset_s=5),
            _make_shadow_link(offset_s=5, action="ENTRY", decision_id="DEC_E", permit_id="PERM_E", request_id="REQ_E"),
            _make_shadow_decision(decision_id="DEC_X", regime="MEAN_REVERT", offset_s=3610),
            _make_shadow_link(offset_s=3610, action="EXIT", decision_id="DEC_X", permit_id="PERM_X", request_id="REQ_X"),
        ]
        _write_shadow_log(shadow_path, events)

        with mock.patch("execution.episode_ledger.DLE_SHADOW_LOG_PATH", shadow_path):
            le, lx, di = _load_shadow_indexes()
            ep = _make_episode(offset_entry_s=0, offset_exit_s=3600)
            v2s, stats = _bind_authority([ep], le, lx, di)

            assert len(v2s) == 1
            v2 = v2s[0]
            assert v2.authority_entry.decision_id == "DEC_E"
            assert v2.authority_exit.decision_id == "DEC_X"
            assert v2.regime_at_entry == "TREND_UP"
            assert v2.regime_at_exit == "MEAN_REVERT"
            assert v2.regime_bindable is True
            assert stats["entry_coverage_pct"] == 100.0
            assert stats["exit_coverage_pct"] == 100.0


# ===========================================================================
# Tests: Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_iso_to_unix_handles_none_string(self):
        assert _iso_to_unix("") is None
        assert _iso_to_unix("not-a-date") is None

    def test_iso_to_unix_valid(self):
        ts = "2026-02-13T12:00:00+00:00"
        result = _iso_to_unix(ts)
        assert result is not None
        assert isinstance(result, float)

    def test_empty_episodes_produces_empty_v2(self):
        v2s, stats = _bind_authority([], {}, {}, {})
        assert v2s == []
        assert stats["entry_coverage_pct"] == 0.0
        assert stats["exit_coverage_pct"] == 0.0

    def test_episode_with_unparseable_ts(self):
        """Episode with bad timestamps → authority flags missing, no crash."""
        ep = Episode(
            episode_id="EP_0001", symbol="BTCUSDT", side="LONG",
            entry_ts="garbage", exit_ts="garbage",
            duration_hours=0.0, entry_fills=0, exit_fills=0,
            entry_notional=0.0, exit_notional=0.0,
            total_qty=0.0, avg_entry_price=0.0, avg_exit_price=0.0,
            gross_pnl=0.0, fees=0.0, net_pnl=0.0,
            regime_at_entry="unknown", regime_at_exit="unknown",
            exit_reason="unknown", exit_reason_raw="",
        )
        v2s, stats = _bind_authority([ep], {}, {}, {})
        assert len(v2s) == 1
        assert v2s[0].authority_flags.entry_missing is True
        assert v2s[0].authority_flags.exit_missing is True

    def test_multiple_episodes_independent_binding(self):
        """Each episode binds independently to nearest LINK."""
        ep1 = _make_episode(offset_entry_s=0, offset_exit_s=3600, episode_id="EP_0001")
        ep2 = _make_episode(offset_entry_s=7200, offset_exit_s=10800, episode_id="EP_0002")
        
        entry1 = _make_link_record(offset_s=5, request_id="R1", decision_id="D1")
        entry2 = _make_link_record(offset_s=7205, request_id="R2", decision_id="D2")
        
        le = {("BTCUSDT", "TREND"): sorted([entry1, entry2], key=lambda r: r.ts_unix)}
        lx = {}
        di = {}
        
        v2s, stats = _bind_authority([ep1, ep2], le, lx, di)
        assert v2s[0].authority_entry.request_id == "R1"
        assert v2s[1].authority_entry.request_id == "R2"
        assert stats["entry_bound"] == 2

    def test_link_payload_ts_used_over_toplevel(self, tmp_path):
        """B.4 LINK payload.ts should be preferred for index building."""
        from execution.dle_shadow import shadow_build_chain, DLEShadowWriter
        
        log_path = tmp_path / "shadow.jsonl"
        w = DLEShadowWriter(str(log_path))
        shadow_build_chain(
            enabled=True, write_logs=True, writer=w,
            attempt_id="ATT_001",
            requested_action="ENTRY", symbol="BTCUSDT", side="BUY",
            strategy="TREND", qty_intent=100.0, context={},
            phase_id="CYCLE_001", action_class="ENTRY_ALLOW",
            policy_version="v7.9", scope={"symbol": "BTCUSDT"},
            constraints={"verdict": "PERMIT"}, risk={},
            verdict="PERMIT",
        )
        
        # Read back the LINK event
        with open(log_path, "r") as f:
            lines = f.readlines()
        link_line = [json.loads(l) for l in lines if json.loads(l).get("event_type") == "LINK"]
        assert len(link_line) == 1
        assert "ts" in link_line[0]["payload"]  # B.4: payload.ts present
        assert link_line[0]["payload"]["ts"] == link_line[0]["ts"]  # same value
