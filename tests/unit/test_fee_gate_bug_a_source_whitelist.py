"""
v7.9-FG-A — Fee gate edge-source whitelist (Bug A).

Regression test for the field-source mismatch where the direct-edge path
in `_check_fee_edge_gate` only fired for `intent_expected_edge`, silently
dropping `hybrid_components.expectancy` and `metadata.expectancy` into
the ATR fallback path.

Tests:
  1. hybrid_expectancy present  → direct edge path used
  2. metadata_expectancy present → direct edge path used
  3. no expectancy              → ATR fallback still used
  4. fee threshold unchanged    → numeric parity with existing config
"""
from __future__ import annotations

import pytest


# ---- Test environment ------------------------------------------------------

@pytest.fixture(autouse=True)
def _safe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setenv("EXECUTOR_ONCE", "1")


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch):
    """Capture the TrueEdgeResult passed into the fee gate.

    The executor calls `check_fee_edge_v2(_te_result)` with the
    TrueEdgeResult it constructed.  We patch that symbol to (a) record
    the TrueEdgeResult and (b) return an `(allowed, details)` tuple so
    the caller does not try to log/veto.
    """
    import execution.fee_gate as fg
    import execution.true_edge as te

    box: dict = {"te": None, "calls": 0}

    def _fake_check_fee_edge_v2(te_result):
        box["te"] = te_result
        box["calls"] += 1
        return True, {"gate_status": "pass", "shortfall_usd": 0.0}

    monkeypatch.setattr(fg, "check_fee_edge_v2", _fake_check_fee_edge_v2)

    # Also stub structured event writer to keep test side-effect free.
    import execution.executor_live as el

    monkeypatch.setattr(
        el, "_record_structured_event", lambda *a, **k: None, raising=True
    )

    # Force compute_true_edge to a deterministic ATR-path result.
    def _fake_compute_true_edge(
        confidence: float,
        price: float,
        atr,
        notional_usd: float,
        timeframe=None,
    ):
        return te.TrueEdgeResult(
            expected_edge_pct=0.0001,
            expected_edge_usd=notional_usd * 0.0001,
            atr_pct=0.001,
            k_atr=0.6,
            adv=0.0,
            confidence=confidence,
            notional_usd=notional_usd,
            source="atr_conf_v1",
        )

    monkeypatch.setattr(te, "compute_true_edge", _fake_compute_true_edge)
    # The function does `from execution.true_edge import compute_true_edge`
    # at call time, so patching the source module is sufficient.

    return box


def _call_gate(intent: dict, *, gross_target: float = 200.0):
    from execution.executor_live import _check_fee_edge_gate

    return _check_fee_edge_gate(
        intent=intent,
        symbol_upper="ETHUSDT",
        side="LONG",
        price=2000.0,
        price_hint=2000.0,
        gross_target=gross_target,
        intent_id="test_intent",
        attempt_id="test_attempt",
    )


# ---- Test 1: hybrid_expectancy → direct path ------------------------------

def test_hybrid_expectancy_uses_direct_edge_path(captured):
    """Bug A regression: hybrid_components.expectancy must hit direct path."""
    intent = {
        "symbol": "ETHUSDT",
        "expected_edge": 0.0,                       # absent at top level
        "hybrid_components": {"expectancy": 0.05},  # 5% expectancy
        "confidence": 0.7,
    }
    allowed = _call_gate(intent)
    assert captured["calls"] == 1
    te = captured["te"]
    assert te is not None
    # Direct path tags TrueEdgeResult source with the resolved field name.
    assert te.source == "hybrid_expectancy", (
        f"Expected direct-edge path (source='hybrid_expectancy'), "
        f"got source={te.source!r} — Bug A regression"
    )
    # Direct path uses the resolved edge, not ATR.
    assert te.expected_edge_pct == pytest.approx(0.05, rel=1e-6)
    assert te.expected_edge_usd == pytest.approx(200.0 * 0.05, rel=1e-6)
    assert te.atr_pct == 0.0
    assert allowed is True  # stubbed v2 returns True


# ---- Test 2: metadata_expectancy → direct path ----------------------------

def test_metadata_expectancy_uses_direct_edge_path(captured):
    """Bug A regression: metadata.expectancy must hit direct path."""
    intent = {
        "symbol": "ETHUSDT",
        "expected_edge": 0.0,
        "hybrid_components": {},
        "metadata": {"expectancy": 0.04},
        "confidence": 0.7,
    }
    allowed = _call_gate(intent)
    assert captured["calls"] == 1
    te = captured["te"]
    assert te is not None
    assert te.source == "metadata_expectancy", (
        f"Expected direct-edge path (source='metadata_expectancy'), "
        f"got source={te.source!r} — Bug A regression"
    )
    assert te.expected_edge_pct == pytest.approx(0.04, rel=1e-6)
    assert te.atr_pct == 0.0
    assert allowed is True


# ---- Test 3: no expectancy → ATR fallback ---------------------------------

def test_no_expectancy_uses_atr_fallback(captured):
    """Backwards compatibility: no edge anywhere → ATR fallback unchanged."""
    intent = {
        "symbol": "ETHUSDT",
        "expected_edge": 0.0,
        "hybrid_components": {"expectancy": 0.0},
        "metadata": {"expectancy": 0.0},
        "confidence": 0.0,
    }
    _call_gate(intent)
    assert captured["calls"] == 1
    te = captured["te"]
    assert te is not None
    # Fallback path is tagged "atr_conf_v1" by compute_true_edge.
    assert te.source == "atr_conf_v1", (
        f"Expected ATR fallback (source='atr_conf_v1'), got {te.source!r}"
    )
    # ATR fields populated by the (stubbed) ATR model.
    assert te.atr_pct > 0.0


# ---- Test 4: top-level expected_edge still wins (priority) ----------------

def test_top_level_expected_edge_takes_priority(captured):
    """Priority: intent['expected_edge'] beats hybrid and metadata."""
    intent = {
        "symbol": "ETHUSDT",
        "expected_edge": 0.10,
        "hybrid_components": {"expectancy": 0.05},
        "metadata": {"expectancy": 0.04},
        "confidence": 0.7,
    }
    _call_gate(intent)
    te = captured["te"]
    assert te is not None
    assert te.source == "intent_expected_edge"
    assert te.expected_edge_pct == pytest.approx(0.10, rel=1e-6)


# ---- Test 5: fee threshold unchanged --------------------------------------

def test_fee_threshold_config_unchanged():
    """The Bug A fix does not modify any fee gate threshold or buffer."""
    from execution.fee_gate import FeeGateConfig, load_fee_gate_config

    cfg = load_fee_gate_config()
    assert isinstance(cfg, FeeGateConfig)
    # Anchor the live numbers — these must not move with this patch.
    assert cfg.taker_fee_rate == pytest.approx(0.0004, rel=1e-9)
    assert cfg.fee_buffer_mult == pytest.approx(1.5, rel=1e-9)


# ---- Test 6: edge cap still enforced --------------------------------------

def test_intent_edge_cap_still_applied(captured):
    """A runaway expectancy is capped at 25% on the direct path."""
    intent = {
        "symbol": "ETHUSDT",
        "hybrid_components": {"expectancy": 0.99},  # would-be 99%
        "confidence": 0.7,
    }
    _call_gate(intent, gross_target=1000.0)
    te = captured["te"]
    assert te is not None
    assert te.source == "hybrid_expectancy"
    assert te.expected_edge_pct == pytest.approx(0.25, rel=1e-6)
    assert te.expected_edge_usd == pytest.approx(1000.0 * 0.25, rel=1e-6)
