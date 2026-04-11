"""
Shadow Selector v2 — Phase 5 Research (Observation Only).

Evaluates three experimental routing rules against live ECS decisions
without touching execution. Logs counterfactual records to a dedicated
JSONL file for offline analysis.

**Never gates execution. Never modifies intents.**

Three candidate selectors evaluated simultaneously:

  A — Hydra preference band:
      If hydra_score falls in empirically positive Sharpe peak → prefer Hydra.
      Else fallback to ECS.

  B — Positive-region routing:
      If hydra_score < regime_boundary(symbol) → prefer Hydra.
      Else abstain.

  C — Calibrated EV routing (placeholder):
      Future: argmax(EV_hydra, EV_legacy) using calibrated EV functions.
      Currently inactive — logs null verdicts.

Env:
    SELECTOR_V2_SHADOW=1   Enable shadow v2 telemetry.
    SELECTOR_V2_SHADOW=0   (default) No-op.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

_V2_LOG_PATH = Path("logs/execution/selector_v2_shadow.jsonl")

# ── Regime boundaries from phase map analysis (866 soak events, 2026-03-15) ─
REGIME_BOUNDARIES: Dict[str, List[float]] = {
    "BTCUSDT": [0.5236],
    "ETHUSDT": [0.4291, 0.4883],
    "SOLUSDT": [],
}

REGIME_LABELS: Dict[str, List[str]] = {
    "BTCUSDT": ["HYDRA_REGIME", "LEGACY_REGIME"],
    "ETHUSDT": ["LEGACY_LOW", "HYDRA_REGIME", "LEGACY_HIGH"],
    "SOLUSDT": ["LEGACY_ONLY"],
}

# ── V2 boundaries: monotonicity-constrained (DEC_ECS_PROFIT_MASK_V2) ────────
REGIME_BOUNDARIES_V2: Dict[str, List[float]] = {
    "BTCUSDT": [0.4197, 0.5369],   # 3-regime: COLD / PROFIT / LEGACY
    "ETHUSDT": [0.4291, 0.4883],   # unchanged from V1 (pending monotonicity)
    "SOLUSDT": [],                   # unchanged from V1 (pending monotonicity)
}

REGIME_LABELS_V2: Dict[str, List[str]] = {
    "BTCUSDT": ["COLD_ZONE", "PROFIT_ZONE", "LEGACY_ZONE"],
    "ETHUSDT": ["LEGACY_LOW", "HYDRA_REGIME", "LEGACY_HIGH"],
    "SOLUSDT": ["LEGACY_ONLY"],
}

# ── Candidate A: Sharpe peak bands per symbol ──
# Derived from Sharpe surface analysis: score ranges where Hydra Sharpe > 0
HYDRA_PREFERENCE_BANDS: Dict[str, List[tuple]] = {
    "BTCUSDT": [(0.42, 0.52)],
    "ETHUSDT": [(0.43, 0.49)],
    "SOLUSDT": [],  # no Hydra regime found
}

# ── Candidate D: PnL-positive regions per symbol ──
# Derived from ecs_profit_mask.py (780 episodes, 2026-03-17).
# These are hydra_score ranges where cumulative PnL is empirically positive.
# Unlike REGIME_BOUNDARIES (arbitration surface — who wins the model comparison),
# these represent the profit surface (where money is actually made).
PNL_POSITIVE_REGIONS: Dict[str, List[tuple]] = {
    "BTCUSDT": [(0.4197, 0.4953)],
    "ETHUSDT": [(0.4393, 0.511)],
    "SOLUSDT": [],  # no profitable region found
}

# ── V2 profit masks: only stable regions retained (DEC_ECS_PROFIT_MASK_V2) ──
# ETH mask discarded (subsample overlap < 0.70 = UNSTABLE).
# SOL mask unchanged (no profitable region found).
PNL_POSITIVE_REGIONS_V2: Dict[str, List[tuple]] = {
    "BTCUSDT": [(0.4197, 0.4953)],  # STABLE (overlap 0.889)
    "ETHUSDT": [],                    # DISCARDED (unstable)
    "SOLUSDT": [],                    # no profitable region
}

# Version tag logged in shadow events for mask lineage tracking.
# Bump this whenever PNL_POSITIVE_REGIONS is refreshed.
PROFIT_MASK_VERSION = "2026-03-17_v1"
PROFIT_MASK_VERSION_V2 = "2026-04-07_v2"

# ── INV-5: mask upper bound must not exceed regime boundary ─────────────────
for _sym, _bands in PNL_POSITIVE_REGIONS_V2.items():
    _regime_bounds = REGIME_BOUNDARIES_V2.get(_sym, [])
    if _bands and _regime_bounds:
        _mask_hi = max(hi for _, hi in _bands)
        _regime_hi = _regime_bounds[-1]
        assert _mask_hi <= _regime_hi, (
            f"INV-5 violation: {_sym} mask_hi={_mask_hi} > regime_hi={_regime_hi}"
        )

# ── ZERO_SCORE deterministic routing policy ─────────────────────────────────
# Episodes with hybrid_score == 0 or None cannot be placed on the profit mask
# axis. They are routed to Legacy (ECS fallback) and tracked separately.
ZERO_SCORE_POLICY: Dict[str, Any] = {
    "action": "ROUTE_TO_LEGACY",
    "candidate_d_verdict": "ABSTAIN",
    "log_tag": "zero_score_fallback",
    "rescore": False,
    "reason": "Score=0 cannot be placed on profit mask axis; default to Legacy",
}


def in_profit_region(symbol: str, hydra_score: float) -> bool:
    """Check if (symbol, hydra_score) falls inside a PnL-positive region (V2)."""
    bands = PNL_POSITIVE_REGIONS_V2.get(symbol, [])
    return any(lo <= hydra_score <= hi for lo, hi in bands)


def _v2_enabled() -> bool:
    return (os.getenv("SELECTOR_V2_SHADOW", "0") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def classify_regime(symbol: str, hydra_score: float) -> str:
    """Map hydra_score to regime label via boundary thresholds."""
    bounds = REGIME_BOUNDARIES.get(symbol, [])
    labels = REGIME_LABELS.get(symbol, ["UNKNOWN"])
    if not bounds:
        return labels[0] if labels else "UNKNOWN"
    for i, threshold in enumerate(bounds):
        if hydra_score < threshold:
            return labels[i] if i < len(labels) else "UNKNOWN"
    return labels[-1] if labels else "UNKNOWN"


# ── Candidate A: Hydra preference band ──────────────────────────────────────

def _selector_a(
    symbol: str, hydra_score: float, ecs_choice: str,
) -> Dict[str, Any]:
    """Hydra preference band: prefer Hydra inside empirical Sharpe peak."""
    bands = HYDRA_PREFERENCE_BANDS.get(symbol, [])
    in_band = any(lo <= hydra_score <= hi for lo, hi in bands)
    if in_band:
        return {"v2_choice": "hydra", "v2_abstain": False, "rule": "A_band_hit"}
    # Outside band: fall back to ECS decision
    return {"v2_choice": ecs_choice, "v2_abstain": False, "rule": "A_ecs_fallback"}


# ── Candidate B: Positive-region routing ─────────────────────────────────────

def _selector_b(
    symbol: str, hydra_score: float,
) -> Dict[str, Any]:
    """Positive-region routing: prefer Hydra in Hydra regime, else abstain."""
    regime = classify_regime(symbol, hydra_score)
    if "HYDRA" in regime:
        return {"v2_choice": "hydra", "v2_abstain": False, "rule": "B_hydra_regime"}
    return {"v2_choice": "none", "v2_abstain": True, "rule": "B_abstain"}


# ── Candidate C: Calibrated EV routing (placeholder) ────────────────────────

def _selector_c(
    symbol: str, hydra_score: float, legacy_score: float,
) -> Dict[str, Any]:
    """Calibrated EV routing — placeholder for future implementation.

    Once enough data exists, this uses empirical EV functions:
        EV_hydra  = f(hydra_score)
        EV_legacy = g(legacy_score)
        winner    = argmax(EV_hydra, EV_legacy)

    Currently logs null verdicts (inactive).
    """
    return {"v2_choice": None, "v2_abstain": None, "rule": "C_inactive"}

# ── Candidate D: Profit-conditioned routing ──────────────────────────────────

def _selector_d(
    symbol: str, hydra_score: Optional[float],
) -> Dict[str, Any]:
    """Profit-conditioned routing: prefer Hydra in PnL-positive regions only.

    Unlike Candidate B (arbitration surface — where Hydra wins the model
    comparison), D operates on the profit surface (where Hydra actually
    makes money).  This is a stricter filter: BTC mid-band only (V2).

    Zero-score policy:
        None  → D_no_score   (missing signal path — pipeline gap)
        <= 0  → D_zero_score (computed but neutral/blind — model blindness)
    """
    # ── ZERO_SCORE deterministic policy ──
    if hydra_score is None:
        return {
            "v2_choice": "none",
            "v2_abstain": True,
            "rule": "D_no_score",
            "d_zero_score": True,
        }
    if hydra_score <= 0:
        return {
            "v2_choice": "none",
            "v2_abstain": True,
            "rule": "D_zero_score",
            "d_zero_score": True,
        }
    # ── Profit mask routing ──
    if in_profit_region(symbol, hydra_score):
        return {"v2_choice": "hydra", "v2_abstain": False, "rule": "D_profit_region", "d_zero_score": False}
    return {"v2_choice": "none", "v2_abstain": True, "rule": "D_abstain", "d_zero_score": False}

# ── Main shadow evaluation ───────────────────────────────────────────────────

def evaluate_v2_shadow(
    *,
    symbol: str,
    hydra_score: Optional[float],
    legacy_score: Optional[float],
    ecs_choice: str,
    score_delta: Optional[float] = None,
    merge_conflict: bool = False,
    cycle: int = 0,
) -> Optional[Dict[str, Any]]:
    """Evaluate all three v2 candidate selectors and log shadow record.

    Args:
        symbol: Trading symbol (e.g. "BTCUSDT").
        hydra_score: Hydra absolute score (merge_hydra_score from soak).
        legacy_score: Legacy absolute score (merge_legacy_score from soak).
        ecs_choice: Engine the live ECS actually chose ("hydra" or "legacy").
        score_delta: hydra_score - legacy_score (pre-computed or derived).
        merge_conflict: Whether this was a conflict case (both engines had signals).
        cycle: Executor cycle number.

    Returns:
        Shadow event dict (also appended to JSONL), or None if disabled/error.
    """
    if not _v2_enabled():
        return None

    if hydra_score is None:
        return None  # cannot evaluate without Hydra score

    try:
        h_score = float(hydra_score)
        l_score = float(legacy_score) if legacy_score is not None else 0.0

        if score_delta is None and legacy_score is not None:
            score_delta = round(h_score - l_score, 6)

        regime = classify_regime(symbol, h_score)

        # Evaluate all four candidates
        result_a = _selector_a(symbol, h_score, ecs_choice)
        result_b = _selector_b(symbol, h_score)
        result_c = _selector_c(symbol, h_score, l_score)
        result_d = _selector_d(symbol, h_score)

        event: Dict[str, Any] = {
            "ts": time.time(),
            "schema": "selector_v2_shadow_v3",
            "symbol": symbol,
            "cycle": cycle,
            # ── Input scores ──
            "hydra_score": h_score,
            "legacy_score": l_score,
            "score_delta": score_delta,
            "hydra_regime_band": regime,
            "profit_region": in_profit_region(symbol, h_score),
            "profit_mask_version": PROFIT_MASK_VERSION_V2,
            "ecs_conflict": merge_conflict,
            # ── Live ECS decision ──
            "ecs_choice": ecs_choice,
            # ── Candidate A ──
            "a_choice": result_a["v2_choice"],
            "a_abstain": result_a["v2_abstain"],
            "a_rule": result_a["rule"],
            # ── Candidate B ──
            "b_choice": result_b["v2_choice"],
            "b_abstain": result_b["v2_abstain"],
            "b_rule": result_b["rule"],
            # ── Candidate C (placeholder) ──
            "c_choice": result_c["v2_choice"],
            "c_abstain": result_c["v2_abstain"],
            "c_rule": result_c["rule"],
            # ── Candidate D (profit-conditioned) ──
            "d_choice": result_d["v2_choice"],
            "d_abstain": result_d["v2_abstain"],
            "d_rule": result_d["rule"],
            # ── Candidate D lineage (v3) ──
            "d_zero_score": result_d.get("d_zero_score", False),
            "d_mask_boundaries": list(PNL_POSITIVE_REGIONS_V2.get(symbol, [])),
            "d_regime_boundary": (REGIME_BOUNDARIES_V2.get(symbol) or [None])[-1],
        }

        _append_v2_event(event)

        # Log divergences for visibility
        for label, res in [("A", result_a), ("B", result_b), ("D", result_d)]:
            if res["v2_choice"] and res["v2_choice"] != ecs_choice:
                LOG.info(
                    "[selector_v2] %s divergence sym=%s ecs=%s v2=%s rule=%s",
                    label, symbol, ecs_choice, res["v2_choice"], res["rule"],
                )

        return event

    except Exception as exc:
        LOG.debug("[selector_v2] shadow_eval_failed: %s", exc)
        return None


def _append_v2_event(event: Dict[str, Any]) -> None:
    """Append a v2 shadow JSONL event (fail-open)."""
    try:
        _V2_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_V2_LOG_PATH, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as exc:
        LOG.debug("[selector_v2] log_write_failed: %s", exc)
