"""Tests for E1 Patch 1: Sizing Snapshot Ledger.

Validates:
- Snapshot written once per attempted order
- All required keys present
- final_qty matches computed qty from gross_target / price_hint
- Caps detection is best-effort correct
- No snapshot emitted on veto (by design — snapshot call site is post-veto)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from execution.sizing_snapshot import (
    REQUIRED_KEYS,
    emit_sizing_snapshot,
    _extract_adaptive,
    _extract_strategy,
    _extract_vol,
    _detect_caps,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _base_intent(**overrides: Any) -> Dict[str, Any]:
    """Minimal intent with all metadata paths populated."""
    intent: Dict[str, Any] = {
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "price": 50000.0,
        "gross_usd": 500.0,
        "per_trade_nav_pct": 0.05,
        "leverage": 1,
        "strategy": "TREND",
        "tier": "CORE",
        "sizing_notes": {
            "vol_regime": "normal",
            "vol_sizing_mult": 1.0,
            "effective_per_trade_nav_pct": 0.05,
            "floors": {
                "symbol_min_gross": 10.0,
                "symbol_min_notional": 5.0,
                "exchange_min_notional": 5.0,
                "min_qty_notional": 1.0,
            },
        },
        "metadata": {
            "strategy": "TREND",
            "adaptive": {
                "atr_factor": 1.0,
                "dd_factor": 0.8,
                "risk_factor": 1.0,
                "final_factor": 0.8,
            },
            "adaptive_weight": {
                "final_weight": 0.9,
            },
        },
        "symbol_caps": {
            "max_nav_pct": 0.30,
        },
    }
    intent.update(overrides)
    return intent


# ── Unit Tests: Extractors ────────────────────────────────────────────────────


class TestExtractStrategy:
    def test_from_top_level(self):
        assert _extract_strategy({"strategy": "MEAN_REVERT"}) == "MEAN_REVERT"

    def test_from_metadata(self):
        intent = {"metadata": {"strategy": "VOL_HARVEST"}}
        assert _extract_strategy(intent) == "VOL_HARVEST"

    def test_from_strategy_name(self):
        assert _extract_strategy({"strategy_name": "CATEGORY"}) == "CATEGORY"

    def test_unknown_fallback(self):
        assert _extract_strategy({}) == "UNKNOWN"


class TestExtractAdaptive:
    def test_full_metadata(self):
        intent = _base_intent()
        result = _extract_adaptive(intent)
        assert result["atr_factor"] == 1.0
        assert result["dd_factor"] == 0.8
        assert result["risk_factor"] == 1.0
        assert result["adaptive_factor"] == 0.8
        assert result["adaptive_weight"] == 0.9

    def test_missing_metadata_defaults_to_one(self):
        result = _extract_adaptive({})
        assert result["atr_factor"] == 1.0
        assert result["dd_factor"] == 1.0
        assert result["adaptive_factor"] == 1.0
        assert result["adaptive_weight"] == 1.0


class TestExtractVol:
    def test_normal_regime(self):
        intent = _base_intent()
        result = _extract_vol(intent)
        assert result["vol_regime"] == "normal"
        assert result["vol_multiplier"] == 1.0

    def test_missing_sizing_notes(self):
        result = _extract_vol({})
        assert result["vol_regime"] == "unknown"
        assert result["vol_multiplier"] == 1.0


class TestDetectCaps:
    def test_no_caps_when_gross_matches(self):
        intent = _base_intent()
        caps = _detect_caps(intent, 500.0, 10000.0)
        assert "PER_SYMBOL_CAP" not in caps

    def test_risk_adjusted_cap(self):
        intent = _base_intent(gross_usd=500.0)
        caps = _detect_caps(intent, 300.0, 10000.0)
        assert "RISK_ADJUSTED" in caps

    def test_min_notional_cap(self):
        intent = _base_intent()
        intent["sizing_notes"]["floors"]["symbol_min_gross"] = 500.0
        caps = _detect_caps(intent, 500.0, 10000.0)
        assert "MIN_NOTIONAL" in caps


# ── Integration Tests: emit_sizing_snapshot ───────────────────────────────────


class TestEmitSizingSnapshot:
    """Test the full emit path with mocked JSONL writer."""

    @pytest.fixture(autouse=True)
    def _mock_logger(self):
        """Intercept the JSONL logger so no file I/O occurs."""
        self.captured_events: list[dict] = []

        def _capture(logger, event_type, payload):
            self.captured_events.append({"event_type": event_type, **dict(payload)})

        with patch("execution.sizing_snapshot.log_event", side_effect=_capture):
            yield

    def _emit(self, **overrides) -> Dict[str, Any]:
        intent = _base_intent()
        kwargs: Dict[str, Any] = {
            "intent": intent,
            "attempt_id": "sig_test123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "pos_side": "LONG",
            "gross_target": 500.0,
            "nav_usd": 10000.0,
            "tier_name": "CORE",
            "price_hint": 50000.0,
            "reduce_only": False,
        }
        kwargs.update(overrides)
        return emit_sizing_snapshot(**kwargs)

    def test_snapshot_written_once(self):
        self._emit()
        assert len(self.captured_events) == 1
        assert self.captured_events[0]["event_type"] == "sizing_snapshot"

    def test_required_keys_present(self):
        snap = self._emit()
        missing = REQUIRED_KEYS - set(snap.keys())
        assert not missing, f"Missing required keys: {missing}"

    def test_final_qty_equals_gross_over_price(self):
        snap = self._emit(gross_target=600.0, price_hint=30000.0)
        expected_qty = 600.0 / 30000.0
        assert snap["final_qty"] == pytest.approx(expected_qty)

    def test_final_nav_pct_computed(self):
        snap = self._emit(gross_target=500.0, nav_usd=10000.0)
        assert snap["final_nav_pct"] == pytest.approx(0.05)

    def test_doctrine_multiplier_propagated(self):
        intent = _base_intent(doctrine_multiplier=0.7)
        snap = self._emit(intent=intent)
        assert snap["doctrine_multiplier"] == pytest.approx(0.7)

    def test_doctrine_multiplier_default_one(self):
        snap = self._emit()
        assert snap["doctrine_multiplier"] == pytest.approx(1.0)

    def test_strategy_extracted(self):
        snap = self._emit()
        assert snap["strategy"] == "TREND"

    def test_vol_regime_extracted(self):
        snap = self._emit()
        assert snap["vol_regime"] == "normal"
        assert snap["vol_multiplier"] == pytest.approx(1.0)

    def test_adaptive_factors_extracted(self):
        snap = self._emit()
        assert snap["dd_factor"] == pytest.approx(0.8)
        assert snap["adaptive_factor"] == pytest.approx(0.8)
        assert snap["adaptive_weight"] == pytest.approx(0.9)

    def test_zero_price_yields_zero_qty(self):
        snap = self._emit(price_hint=0.0)
        assert snap["final_qty"] == 0.0

    def test_zero_nav_yields_zero_nav_pct(self):
        snap = self._emit(nav_usd=0.0)
        assert snap["final_nav_pct"] == 0.0

    def test_reduce_only_flag(self):
        snap = self._emit(reduce_only=True)
        assert snap["reduce_only"] is True

    def test_snapshot_has_timestamp(self):
        before = time.time()
        snap = self._emit()
        after = time.time()
        assert before <= snap["ts"] <= after

    def test_caps_applied_is_list(self):
        snap = self._emit()
        assert isinstance(snap["caps_applied"], list)

    def test_crisis_vol_regime(self):
        intent = _base_intent()
        intent["sizing_notes"]["vol_regime"] = "crisis"
        intent["sizing_notes"]["vol_sizing_mult"] = 0.5
        snap = self._emit(intent=intent)
        assert snap["vol_regime"] == "crisis"
        assert snap["vol_multiplier"] == pytest.approx(0.5)

    def test_snapshot_json_serializable(self):
        snap = self._emit()
        # Must not raise
        json.dumps(snap)


# ── E1-P3: Conviction Shadow Mode Tests ───────────────────────────────────────


class TestConvictionShadowMode:
    """Validate that conviction shadow fields appear when mode=shadow,
    are absent when mode=off, and never mutate final_qty."""

    @pytest.fixture(autouse=True)
    def _mock_logger(self):
        self.captured_events: list[dict] = []

        def _capture(logger, event_type, payload):
            self.captured_events.append({"event_type": event_type, **dict(payload)})

        with patch("execution.sizing_snapshot.log_event", side_effect=_capture):
            yield

    def _emit_with_mode(self, mode: str, intent_overrides: dict | None = None) -> Dict[str, Any]:
        """Emit a snapshot with the conviction mode patched."""
        intent = _base_intent()
        if intent_overrides:
            intent.update(intent_overrides)
        kwargs: Dict[str, Any] = {
            "intent": intent,
            "attempt_id": "sig_conv_test",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "pos_side": "LONG",
            "gross_target": 500.0,
            "nav_usd": 10000.0,
            "tier_name": "CORE",
            "price_hint": 50000.0,
            "reduce_only": False,
        }
        with patch("execution.sizing_snapshot._load_conviction_mode", return_value=mode):
            return emit_sizing_snapshot(**kwargs)

    # --- mode = off → no conviction fields ---

    def test_mode_off_no_conviction_fields(self):
        snap = self._emit_with_mode("off")
        assert "conviction_mult_shadow" not in snap
        assert "conviction_band_shadow" not in snap
        assert "conviction_score_shadow" not in snap

    # --- mode = shadow → conviction fields present ---

    def test_mode_shadow_has_conviction_mult(self):
        snap = self._emit_with_mode("shadow")
        assert "conviction_mult_shadow" in snap
        assert isinstance(snap["conviction_mult_shadow"], float)

    def test_mode_shadow_has_conviction_band(self):
        snap = self._emit_with_mode("shadow")
        assert "conviction_band_shadow" in snap
        assert isinstance(snap["conviction_band_shadow"], str)

    def test_mode_shadow_has_conviction_score(self):
        snap = self._emit_with_mode("shadow")
        assert "conviction_score_shadow" in snap
        assert isinstance(snap["conviction_score_shadow"], float)

    def test_mode_shadow_score_in_range(self):
        snap = self._emit_with_mode("shadow")
        assert 0.0 <= snap["conviction_score_shadow"] <= 1.0

    # --- Shadow NEVER mutates final_qty ---

    def test_shadow_does_not_change_final_qty(self):
        snap_off = self._emit_with_mode("off")
        snap_shadow = self._emit_with_mode("shadow")
        assert snap_off["final_qty"] == snap_shadow["final_qty"]

    def test_shadow_does_not_change_target_notional(self):
        snap_off = self._emit_with_mode("off")
        snap_shadow = self._emit_with_mode("shadow")
        assert snap_off["target_notional_usd"] == snap_shadow["target_notional_usd"]

    def test_shadow_does_not_change_final_nav_pct(self):
        snap_off = self._emit_with_mode("off")
        snap_shadow = self._emit_with_mode("shadow")
        assert snap_off["final_nav_pct"] == snap_shadow["final_nav_pct"]

    # --- Required keys still present with shadow fields ---

    def test_shadow_required_keys_still_present(self):
        snap = self._emit_with_mode("shadow")
        missing = REQUIRED_KEYS - set(snap.keys())
        assert not missing, f"Missing required keys: {missing}"

    # --- Shadow with hybrid_score propagation ---

    def test_shadow_uses_intent_hybrid_score(self):
        """Different hybrid_scores should produce different conviction scores."""
        snap_low = self._emit_with_mode("shadow", {"hybrid_score": 0.1})
        snap_high = self._emit_with_mode("shadow", {"hybrid_score": 0.9})
        # Scores should differ (they use the intent's hybrid_score)
        assert snap_low["conviction_score_shadow"] != snap_high["conviction_score_shadow"]

    # --- Shadow fail-open: bad conviction engine → omit fields, no crash ---

    def test_shadow_failopen_on_engine_error(self):
        """If _compute_conviction_shadow raises or returns None, fields are simply absent."""
        with patch(
            "execution.sizing_snapshot._compute_conviction_shadow", return_value=None
        ):
            snap = self._emit_with_mode("shadow")
        # No crash, fields absent
        assert "conviction_mult_shadow" not in snap
        # All other required keys still there
        missing = REQUIRED_KEYS - set(snap.keys())
        assert not missing

    # --- JSON serializable with shadow fields ---

    def test_shadow_snapshot_json_serializable(self):
        snap = self._emit_with_mode("shadow")
        json.dumps(snap)  # Must not raise
