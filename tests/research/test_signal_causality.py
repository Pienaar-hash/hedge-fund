"""Sanity test for research/signal_causality_audit.py.

Confirms the audit pipeline runs on a synthetic episode dataset and that
the verdict logic responds correctly to a perfectly-predictive vs.
random signal. Does NOT validate the live audit verdict — that lives in
docs/SIGNAL_OUTCOME_CAUSALITY_AUDIT_2026-03-18.md Appendix A.

Marked `runtime` because it imports from `execution.*` and touches the
heavier monotonicity module path.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest

from research.signal_causality_audit import (
    PVAL_GATE,
    RHO_GATE,
    SPREAD_GATE,
    _audit_block,
    _extract_pairs,
    _spearman_p_value,
    render_appendix,
    run_audit,
)

pytestmark = pytest.mark.runtime


def _mk_episode(
    epid: str,
    symbol: str,
    side: str,
    entry: float,
    exit_: float,
    hybrid_score: float,
    duration_hours: float = 1.0,
    entry_ts: str = "2026-01-01T00:00:00+00:00",
) -> Dict[str, Any]:
    return {
        "episode_id": epid,
        "symbol": symbol,
        "side": side,
        "entry_ts": entry_ts,
        "exit_ts": entry_ts,
        "avg_entry_price": entry,
        "avg_exit_price": exit_,
        "entry_notional": entry * 1.0,
        "net_pnl": (exit_ - entry) if side == "LONG" else (entry - exit_),
        "duration_hours": duration_hours,
        "hybrid_score": hybrid_score,
    }


def test_perfectly_predictive_signal_passes_gate() -> None:
    """If higher hybrid_score → higher return monotonically, verdict = PASS."""
    eps: List[Dict[str, Any]] = []
    for i in range(100):
        score = 0.3 + i * 0.005  # 0.30 .. 0.795
        # Return increases with score
        ret_pct = 0.005 + i * 0.0001
        exit_price = 100.0 * (1.0 + ret_pct)
        eps.append(_mk_episode(f"EP_{i:04d}", "BTCUSDT", "LONG", 100.0, exit_price, score))
    pairs = _extract_pairs(eps)
    assert len(pairs) == 100
    block = _audit_block(pairs, "TEST")
    assert block["rho"] is not None and block["rho"] > 0.95
    assert block["rho_p_value"] is not None and block["rho_p_value"] < 1e-6
    assert block["q5_minus_q1"] > 0


def test_random_signal_fails_gate() -> None:
    """Random scores → ρ ≈ 0 → verdict = FAIL."""
    rng = random.Random(12345)
    eps: List[Dict[str, Any]] = []
    for i in range(200):
        score = rng.uniform(0.30, 0.60)
        # Returns independent of score
        ret_pct = rng.gauss(0.0, 0.01)
        exit_price = 100.0 * (1.0 + ret_pct)
        eps.append(_mk_episode(f"EP_{i:04d}", "BTCUSDT", "LONG", 100.0, exit_price, score))
    pairs = _extract_pairs(eps)
    block = _audit_block(pairs, "TEST")
    assert block["rho"] is not None
    assert abs(block["rho"]) < 0.2  # Random signal: |ρ| should be small


def test_audit_handles_empty_dataset(tmp_path) -> None:
    p = tmp_path / "empty_ledger.json"
    p.write_text('{"episodes": []}')
    report = run_audit(str(p))
    assert report["verdict"] == "NO_EPISODES"


def test_audit_skips_zero_score_episodes(tmp_path) -> None:
    p = tmp_path / "ledger.json"
    eps = [
        _mk_episode("EP_001", "BTCUSDT", "LONG", 100.0, 101.0, 0.0),  # excluded
        _mk_episode("EP_002", "BTCUSDT", "LONG", 100.0, 101.0, 0.5),
    ]
    import json as _json
    p.write_text(_json.dumps({"episodes": eps}))
    report = run_audit(str(p))
    assert report["n_total_episodes"] == 2
    assert report["n_scored_episodes"] == 1


def test_spearman_p_value_known_values() -> None:
    # rho=0 → p=1; large rho with reasonable n → p tiny
    assert _spearman_p_value(0.0, 100) == pytest.approx(1.0, abs=1e-9)
    assert _spearman_p_value(0.5, 100) is not None
    assert _spearman_p_value(0.5, 100) < 1e-5
    assert _spearman_p_value(0.05, 5) is None  # too few samples


def test_verdict_thresholds_are_locked() -> None:
    """Plan.md decision: thresholds are FIXED. Guard against drift."""
    assert RHO_GATE == 0.15
    assert PVAL_GATE == 0.05
    assert SPREAD_GATE == 0.0


def test_render_appendix_produces_markdown_with_verdict(tmp_path) -> None:
    eps = [_mk_episode(f"EP_{i:04d}", "BTCUSDT", "LONG", 100.0, 100.0 + i * 0.01, 0.3 + i * 0.003)
           for i in range(60)]
    p = tmp_path / "ledger.json"
    import json as _json
    p.write_text(_json.dumps({"episodes": eps}))
    report = run_audit(str(p))
    md = render_appendix(report)
    assert "Appendix A" in md
    assert "Verdict" in md
    assert report["verdict"] in md
