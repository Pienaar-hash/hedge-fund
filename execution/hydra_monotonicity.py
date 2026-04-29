"""
Hydra Score Monotonicity — Does higher Hydra score → higher return?

Buckets trades by hybrid_score quintiles and computes mean realized
return per bucket.  Also computes Spearman rank correlation as the
single summary statistic.

Healthy model:   spearman > 0.2  (upward slope)
Weak signal:     0 <= spearman < 0.2
Broken model:    spearman < 0    (inverted or noise)

State output: logs/state/hydra_monotonicity.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

_STATE_PATH = os.path.join("logs", "state", "hydra_monotonicity.json")


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _realized_return(ep: Dict[str, Any]) -> Optional[float]:
    entry = _safe_float(ep.get("avg_entry_price"))
    exit_ = _safe_float(ep.get("avg_exit_price"))
    if entry <= 0 or exit_ <= 0:
        return None
    side = str(ep.get("side", "")).upper()
    if side == "LONG":
        return (exit_ - entry) / entry
    elif side == "SHORT":
        return (entry - exit_) / entry
    return None


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Compute Spearman rank correlation without scipy."""
    n = len(xs)
    if n < 5:
        return None

    def _rank(vals: List[float]) -> List[float]:
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)

    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1.0))


def compute_monotonicity(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Bucket trades by Hydra score and compute mean return per bucket.

    Returns dict with buckets list, spearman correlation, and metadata.
    """
    # Filter to episodes with valid score and realized return
    pairs: List[tuple] = []  # (score, return)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        ret = _realized_return(ep)
        if ret is None:
            continue
        pairs.append((score, ret))

    if len(pairs) < 5:
        return {
            "buckets": [],
            "spearman": None,
            "slope": "insufficient_data",
            "n": len(pairs),
            "score_field": score_field,
            "ts": time.time(),
        }

    # Sort by score
    pairs.sort(key=lambda p: p[0])
    scores = [p[0] for p in pairs]
    returns = [p[1] for p in pairs]

    # Spearman rank correlation
    rho = _spearman(scores, returns)

    # Build equal-count buckets (quintiles by default)
    bucket_size = max(1, len(pairs) // n_buckets)
    buckets: List[Dict[str, Any]] = []
    for i in range(0, len(pairs), bucket_size):
        chunk = pairs[i: i + bucket_size]
        chunk_scores = [c[0] for c in chunk]
        chunk_rets = [c[1] for c in chunk]
        lo, hi = min(chunk_scores), max(chunk_scores)
        buckets.append({
            "range": f"{lo:.3f}–{hi:.3f}",
            "lo": round(lo, 5),
            "hi": round(hi, 5),
            "mean_score": round(sum(chunk_scores) / len(chunk_scores), 5),
            "mean_return": round(sum(chunk_rets) / len(chunk_rets), 6),
            "n": len(chunk),
        })

    # Classify slope
    if rho is None:
        slope = "unknown"
    elif rho > 0.15:
        slope = "upward"
    elif rho >= -0.05:
        slope = "flat"
    else:
        slope = "inverted"

    return {
        "buckets": buckets,
        "spearman": round(rho, 4) if rho is not None else None,
        "slope": slope,
        "n": len(pairs),
        "score_field": score_field,
        "ts": time.time(),
    }


def compute_quintile_spread(
    episodes: List[Dict[str, Any]],
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Compute Q5-Q1 return spread (top vs bottom quintile).

    Returns dict with q5_q1_spread, quintiles list, and n.
    Requires at least 10 scored episodes to produce meaningful quintiles.
    """
    pairs: List[tuple] = []
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        ret = _realized_return(ep)
        if ret is None:
            continue
        pairs.append((score, ret))

    if len(pairs) < 10:
        return {
            "q5_q1_spread": None,
            "quintiles": [],
            "n": len(pairs),
        }

    pairs.sort(key=lambda p: p[0])
    n = len(pairs)
    bucket_size = max(1, n // 5)
    quintiles: List[Dict[str, Any]] = []
    for qi in range(5):
        lo_idx = qi * bucket_size
        hi_idx = (qi + 1) * bucket_size if qi < 4 else n
        chunk = pairs[lo_idx:hi_idx]
        chunk_rets = [c[1] for c in chunk]
        chunk_scores = [c[0] for c in chunk]
        mean_ret = sum(chunk_rets) / len(chunk_rets)
        quintiles.append({
            "label": f"Q{qi + 1}",
            "mean_return": round(mean_ret, 6),
            "mean_score": round(sum(chunk_scores) / len(chunk_scores), 5),
            "n": len(chunk),
        })

    q1_ret = quintiles[0]["mean_return"]
    q5_ret = quintiles[4]["mean_return"]
    spread = round(q5_ret - q1_ret, 6)

    return {
        "q5_q1_spread": spread,
        "quintiles": quintiles,
        "n": len(pairs),
    }


# Regime → head mapping.  Sentinel-X regimes map to the Hydra head
# that would have been active for that trade.
_REGIME_TO_HEAD: Dict[str, str] = {
    "TREND_UP": "TREND",
    "TREND_DOWN": "TREND",
    "MEAN_REVERT": "MEAN_REVERT",
    "BREAKOUT": "BREAKOUT",
    "CHOPPY": "CHOPPY",
    "CRISIS": "CRISIS",
}


def _head_from_regime(regime: str) -> str:
    return _REGIME_TO_HEAD.get(regime.upper(), "OTHER")


def compute_monotonicity_by_head(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 3,
    score_field: str = "hybrid_score",
    regime_field: str = "regime_at_entry",
) -> List[Dict[str, Any]]:
    """Run monotonicity per Hydra head (regime proxy).

    Returns list of per-head summaries sorted by n descending.
    """
    # Group scored episodes by head
    head_eps: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        regime = str(ep.get(regime_field, "") or "").upper()
        if not regime or regime == "UNKNOWN":
            continue
        head = _head_from_regime(regime)
        head_eps.setdefault(head, []).append(ep)

    results: List[Dict[str, Any]] = []
    for head, eps in head_eps.items():
        mono = compute_monotonicity(eps, n_buckets=n_buckets, score_field=score_field)
        mono["head"] = head
        results.append(mono)

    results.sort(key=lambda r: r["n"], reverse=True)
    return results


def persist_snapshot(
    episodes: List[Dict[str, Any]],
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute global + per-head monotonicity and write to state file."""
    dest = path or _STATE_PATH
    snap = compute_monotonicity(episodes)

    # Per-head breakdown
    per_head = compute_monotonicity_by_head(episodes)
    snap["per_head"] = [
        {
            "head": h["head"],
            "spearman": h["spearman"],
            "slope": h["slope"],
            "n": h["n"],
            "buckets": h["buckets"],
        }
        for h in per_head
    ]

    # Head contamination flag: any head upward while global is flat/inverted
    global_weak = snap["slope"] in ("flat", "inverted", "insufficient_data")
    any_head_strong = any(h["slope"] == "upward" for h in per_head)
    snap["head_contamination"] = global_weak and any_head_strong

    # Quintile return spread (Q5-Q1)
    qs = compute_quintile_spread(episodes)
    snap["q5_q1_spread"] = qs["q5_q1_spread"]
    snap["quintiles"] = qs["quintiles"]

    try:
        tmp = dest + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(snap, fh, indent=2)
        os.replace(tmp, dest)
    except OSError as exc:
        LOG.debug("[hydra_monotonicity] persist failed: %s", exc)

    return snap
