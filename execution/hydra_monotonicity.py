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
import random
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


def _spearman_p_value(rho: float, n: int) -> Optional[float]:
    """Approximate two-tailed p-value for Spearman rho.

    Uses t-distribution approximation:
        t = rho * sqrt((n-2) / (1 - rho^2)),  df = n - 2
    then computes p via regularized incomplete beta function.
    """
    if n < 5 or abs(rho) >= 1.0:
        return None
    df = n - 2
    t_stat = rho * math.sqrt(df / (1.0 - rho * rho))
    # Two-tailed p-value from t-distribution using incomplete beta
    x = df / (df + t_stat * t_stat)
    p = _regularized_beta(x, df / 2.0, 0.5)
    return min(p, 1.0)


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Accurate to ~1e-10 for typical statistical use (a,b < 1000).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use symmetry relation when x > (a+1)/(a+b+2) for convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_beta(1.0 - x, b, a)
    # Log-beta for normalization
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta) / a
    # Lentz continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, 200):
        # even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c
        # odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * f


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

    # p-value for Spearman rho
    p_value = _spearman_p_value(rho, len(pairs)) if rho is not None else None

    # Build equal-count buckets (quintiles by default)
    bucket_size = max(1, len(pairs) // n_buckets)
    buckets: List[Dict[str, Any]] = []
    for i in range(0, len(pairs), bucket_size):
        chunk = pairs[i: i + bucket_size]
        chunk_scores = [c[0] for c in chunk]
        chunk_rets = [c[1] for c in chunk]
        lo, hi = min(chunk_scores), max(chunk_scores)
        hits = sum(1 for r in chunk_rets if r > 0)
        buckets.append({
            "range": f"{lo:.3f}–{hi:.3f}",
            "lo": round(lo, 5),
            "hi": round(hi, 5),
            "mean_score": round(sum(chunk_scores) / len(chunk_scores), 5),
            "mean_return": round(sum(chunk_rets) / len(chunk_rets), 6),
            "hit_rate": round(hits / len(chunk_rets), 4),
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
        "p_value": round(p_value, 6) if p_value is not None else None,
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


def compute_time_stability(
    episodes: List[Dict[str, Any]],
    n_slices: int = 3,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Split episodes into temporal slices and compute per-slice monotonicity.

    Returns dict with slices list, overall stability class, and metadata.
    Stability classes: stable, unstable, degrading, insufficient.
    """
    pairs: List[tuple] = []  # (entry_ts, score, return)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        ret = _realized_return(ep)
        if ret is None:
            continue
        ts_raw = ep.get("entry_ts", "")
        pairs.append((str(ts_raw), score, ret))

    if len(pairs) < n_slices * 10:
        return {
            "slices": [],
            "stability": "insufficient",
            "n": len(pairs),
        }

    # Sort by entry timestamp
    pairs.sort(key=lambda p: p[0])
    slice_size = max(1, len(pairs) // n_slices)

    slices: List[Dict[str, Any]] = []
    rhos: List[Optional[float]] = []
    for si in range(n_slices):
        lo_idx = si * slice_size
        hi_idx = (si + 1) * slice_size if si < n_slices - 1 else len(pairs)
        chunk = pairs[lo_idx:hi_idx]
        chunk_scores = [c[1] for c in chunk]
        chunk_rets = [c[2] for c in chunk]
        rho = _spearman(chunk_scores, chunk_rets)
        p_val = _spearman_p_value(rho, len(chunk)) if rho is not None else None
        rhos.append(rho)
        slices.append({
            "label": f"T{si + 1}",
            "ts_start": chunk[0][0],
            "ts_end": chunk[-1][0],
            "n": len(chunk),
            "spearman": round(rho, 4) if rho is not None else None,
            "p_value": round(p_val, 6) if p_val is not None else None,
            "slope": (
                "upward" if rho is not None and rho > 0.15
                else "flat" if rho is not None and rho >= -0.05
                else "inverted" if rho is not None
                else "unknown"
            ),
        })

    # Classify stability
    valid_rhos = [r for r in rhos if r is not None]
    if len(valid_rhos) < n_slices:
        stability = "insufficient"
    elif any(r < 0 for r in valid_rhos) and any(r > 0 for r in valid_rhos):
        stability = "unstable"
    elif len(valid_rhos) >= 3 and valid_rhos[0] > valid_rhos[1] > valid_rhos[2] and valid_rhos[-1] < 0.05:
        stability = "degrading"
    elif all(r > 0.10 for r in valid_rhos):
        stability = "stable"
    else:
        stability = "weak"

    return {
        "slices": slices,
        "stability": stability,
        "n": len(pairs),
    }


def compute_duration_by_bucket(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> List[Dict[str, Any]]:
    """Compute mean/std hold duration per score bucket.

    Detects whether high-score trades have systematically different
    hold durations, which would confound return-vs-score analysis.
    """
    triples: List[tuple] = []  # (score, duration_hours)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        dur = _safe_float(ep.get("duration_hours"))
        if dur <= 0:
            continue
        triples.append((score, dur))

    if len(triples) < 10:
        return []

    triples.sort(key=lambda p: p[0])
    bucket_size = max(1, len(triples) // n_buckets)
    result: List[Dict[str, Any]] = []
    for i in range(0, len(triples), bucket_size):
        chunk = triples[i: i + bucket_size]
        durs = [c[1] for c in chunk]
        scores = [c[0] for c in chunk]
        mean_dur = sum(durs) / len(durs)
        var_dur = sum((d - mean_dur) ** 2 for d in durs) / max(1, len(durs) - 1)
        result.append({
            "label": f"Q{len(result) + 1}",
            "mean_score": round(sum(scores) / len(scores), 5),
            "mean_duration_hrs": round(mean_dur, 2),
            "std_duration_hrs": round(math.sqrt(var_dur), 2),
            "n": len(chunk),
        })
    return result


def compute_bootstrap_q5_q1(
    episodes: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    score_field: str = "hybrid_score",
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap test for Q5-Q1 spread significance.

    Resamples episodes with replacement and computes Q5-Q1 spread
    for each resample. Reports the fraction of resamples with
    spread <= 0 as the bootstrap p-value.
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
        return {"p_value": None, "observed_spread": None, "n": len(pairs)}

    def _q5_q1(data: List[tuple]) -> float:
        data_sorted = sorted(data, key=lambda p: p[0])
        n = len(data_sorted)
        q_size = max(1, n // 5)
        q1_rets = [d[1] for d in data_sorted[:q_size]]
        q5_rets = [d[1] for d in data_sorted[-q_size:]]
        return (sum(q5_rets) / len(q5_rets)) - (sum(q1_rets) / len(q1_rets))

    observed = _q5_q1(pairs)

    rng = random.Random(seed)
    n_leq_zero = 0
    for _ in range(n_bootstrap):
        resample = [pairs[rng.randint(0, len(pairs) - 1)] for _ in range(len(pairs))]
        spread = _q5_q1(resample)
        if spread <= 0:
            n_leq_zero += 1

    return {
        "p_value": round(n_leq_zero / n_bootstrap, 4),
        "observed_spread": round(observed, 6),
        "n_bootstrap": n_bootstrap,
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


def compute_threshold_sweep(
    episodes: List[Dict[str, Any]],
    thresholds: Optional[List[float]] = None,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Sweep entry threshold T and compute performance at each level.

    For each T, considers only episodes with score >= T.
    Reports trade frequency, average return, hit rate, and expectancy.

    Returns dict with sweep table, optimal T*, and comparison to current.
    """
    pairs: List[tuple] = []
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        ret = _realized_return(ep)
        if ret is None or score <= 0:
            continue
        pairs.append((score, ret))

    if len(pairs) < 10:
        return {"sweep": [], "optimal_threshold": None, "n": len(pairs)}

    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95

    total = len(pairs)
    sweep: List[Dict[str, Any]] = []
    best_t = 0.0
    best_expectancy = -999.0

    for t in thresholds:
        above = [p for p in pairs if p[0] >= t]
        if len(above) < 3:
            continue
        rets = [p[1] for p in above]
        mean_ret = sum(rets) / len(rets)
        hits = sum(1 for r in rets if r > 0)
        hit_rate = hits / len(rets)
        losses = [r for r in rets if r <= 0]
        avg_win = sum(r for r in rets if r > 0) / max(1, hits)
        avg_loss = sum(losses) / max(1, len(losses)) if losses else 0.0
        expectancy = hit_rate * avg_win + (1.0 - hit_rate) * avg_loss

        row = {
            "threshold": t,
            "n_trades": len(above),
            "trade_pct": round(len(above) / total, 4),
            "mean_return": round(mean_ret, 6),
            "hit_rate": round(hit_rate, 4),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "expectancy": round(expectancy, 6),
        }
        sweep.append(row)

        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_t = t

    # Compare to current default threshold (0.50)
    current_row = next((r for r in sweep if r["threshold"] == 0.50), None)

    return {
        "sweep": sweep,
        "optimal_threshold": best_t,
        "optimal_expectancy": round(best_expectancy, 6),
        "current_threshold": 0.50,
        "current_expectancy": current_row["expectancy"] if current_row else None,
        "delta": round(best_expectancy - (current_row["expectancy"] if current_row else 0), 6),
        "n": len(pairs),
    }


def compute_edge_curve(
    episodes: List[Dict[str, Any]],
    n_points: int = 20,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Fit a smooth score → expected return curve via local averaging.

    Uses a sliding window (kernel smoothing with rectangular kernel)
    to estimate E[return | score] as a continuous function.

    Classifies the curve shape: linear, convex, concave, step, noise.
    """
    pairs: List[tuple] = []
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        ret = _realized_return(ep)
        if ret is None or score <= 0:
            continue
        pairs.append((score, ret))

    if len(pairs) < 20:
        return {"points": [], "shape": "insufficient_data", "n": len(pairs)}

    pairs.sort(key=lambda p: p[0])
    scores = [p[0] for p in pairs]
    returns = [p[1] for p in pairs]
    s_min, s_max = scores[0], scores[-1]

    if s_max - s_min < 0.01:
        return {"points": [], "shape": "degenerate_range", "n": len(pairs)}

    # Evaluate at n_points equally spaced positions
    bandwidth = max(0.05, (s_max - s_min) / (n_points * 0.8))
    points: List[Dict[str, Any]] = []
    y_values: List[float] = []

    for i in range(n_points):
        center = s_min + (s_max - s_min) * (i + 0.5) / n_points
        # Rectangular kernel: all points within bandwidth
        window_rets = [
            returns[j] for j in range(len(scores))
            if abs(scores[j] - center) <= bandwidth
        ]
        if len(window_rets) < 3:
            continue
        y = sum(window_rets) / len(window_rets)
        y_values.append(y)
        points.append({
            "score": round(center, 4),
            "expected_return": round(y, 6),
            "n_window": len(window_rets),
        })

    # Classify shape via second differences
    shape = "noise"
    if len(y_values) >= 5:
        # First differences (slope)
        diffs = [y_values[i + 1] - y_values[i] for i in range(len(y_values) - 1)]
        # Second differences (curvature)
        d2 = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]

        positive_slope = sum(1 for d in diffs if d > 0) / len(diffs)
        avg_d2 = sum(d2) / len(d2) if d2 else 0

        if positive_slope >= 0.75:
            if avg_d2 > 0.0001:
                shape = "convex"
            elif avg_d2 < -0.0001:
                shape = "concave"
            else:
                shape = "linear"
        elif positive_slope <= 0.25:
            shape = "inverted"
        else:
            # Check for step function: large jump in a small region
            max_diff = max(abs(d) for d in diffs) if diffs else 0
            total_range = abs(y_values[-1] - y_values[0]) if y_values else 0
            if total_range > 0 and max_diff > 0.5 * total_range:
                shape = "step"
            else:
                shape = "noise"

    # Monotonicity on the curve points
    curve_rho = _spearman(
        [p["score"] for p in points],
        [p["expected_return"] for p in points],
    )

    return {
        "points": points,
        "shape": shape,
        "curve_spearman": round(curve_rho, 4) if curve_rho is not None else None,
        "n": len(pairs),
        "bandwidth": round(bandwidth, 4),
    }


def compute_direction_accuracy(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Compute P(correct direction | score bucket).

    Direction is correct if:
    - LONG and exit > entry
    - SHORT and entry > exit

    Compares each bucket's accuracy to the 50% random baseline.
    Reports lift = accuracy - 0.50 per bucket.
    """
    pairs: List[tuple] = []
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        entry = _safe_float(ep.get("avg_entry_price"))
        exit_ = _safe_float(ep.get("avg_exit_price"))
        if entry <= 0 or exit_ <= 0:
            continue
        side = str(ep.get("side", "")).upper()
        if side == "LONG":
            correct = 1.0 if exit_ > entry else 0.0
        elif side == "SHORT":
            correct = 1.0 if entry > exit_ else 0.0
        else:
            continue
        pairs.append((score, correct))

    if len(pairs) < 10:
        return {"buckets": [], "overall_accuracy": None, "n": len(pairs)}

    pairs.sort(key=lambda p: p[0])
    bucket_size = max(1, len(pairs) // n_buckets)
    buckets: List[Dict[str, Any]] = []

    for i in range(0, len(pairs), bucket_size):
        chunk = pairs[i: i + bucket_size]
        chunk_scores = [c[0] for c in chunk]
        chunk_correct = [c[1] for c in chunk]
        acc = sum(chunk_correct) / len(chunk_correct)
        lift = acc - 0.50
        buckets.append({
            "label": f"Q{len(buckets) + 1}",
            "mean_score": round(sum(chunk_scores) / len(chunk_scores), 5),
            "accuracy": round(acc, 4),
            "lift_vs_random": round(lift, 4),
            "n": len(chunk),
        })

    overall_correct = sum(c[1] for c in pairs) / len(pairs)
    # Spearman on (bucket_mean_score, bucket_accuracy)
    rho = _spearman(
        [b["mean_score"] for b in buckets],
        [b["accuracy"] for b in buckets],
    )

    return {
        "buckets": buckets,
        "overall_accuracy": round(overall_correct, 4),
        "overall_lift": round(overall_correct - 0.50, 4),
        "direction_spearman": round(rho, 4) if rho is not None else None,
        "n": len(pairs),
    }


def compute_friction_overlay(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Compute raw vs friction-adjusted return per score bucket.

    Uses episode gross_pnl (pre-fee) and net_pnl (post-fee) to show
    which score buckets survive after friction and which are erased.
    """
    quads: List[tuple] = []  # (score, raw_return, net_return, notional)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        entry_px = _safe_float(ep.get("avg_entry_price"))
        exit_px = _safe_float(ep.get("avg_exit_price"))
        notional = _safe_float(ep.get("entry_notional"))
        fees = _safe_float(ep.get("fees"))
        if entry_px <= 0 or exit_px <= 0 or notional <= 0:
            continue

        side = str(ep.get("side", "")).upper()
        if side == "LONG":
            raw_ret = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            raw_ret = (entry_px - exit_px) / entry_px
        else:
            continue

        net_ret = raw_ret - (fees / notional if notional > 0 else 0)
        quads.append((score, raw_ret, net_ret, fees))

    if len(quads) < 10:
        return {"buckets": [], "n": len(quads)}

    quads.sort(key=lambda p: p[0])
    bucket_size = max(1, len(quads) // n_buckets)
    buckets: List[Dict[str, Any]] = []

    for i in range(0, len(quads), bucket_size):
        chunk = quads[i: i + bucket_size]
        chunk_scores = [c[0] for c in chunk]
        chunk_raw = [c[1] for c in chunk]
        chunk_net = [c[2] for c in chunk]
        chunk_fees = [c[3] for c in chunk]
        mean_raw = sum(chunk_raw) / len(chunk_raw)
        mean_net = sum(chunk_net) / len(chunk_net)
        mean_fees = sum(chunk_fees) / len(chunk_fees)
        edge_erased = mean_raw > 0 and mean_net <= 0
        buckets.append({
            "label": f"Q{len(buckets) + 1}",
            "mean_score": round(sum(chunk_scores) / len(chunk_scores), 5),
            "mean_raw_return": round(mean_raw, 6),
            "mean_net_return": round(mean_net, 6),
            "mean_fees_usd": round(mean_fees, 4),
            "friction_drag": round(mean_raw - mean_net, 6),
            "edge_erased": edge_erased,
            "n": len(chunk),
        })

    # Friction-adjusted monotonicity
    rho_raw = _spearman(
        [b["mean_score"] for b in buckets],
        [b["mean_raw_return"] for b in buckets],
    )
    rho_net = _spearman(
        [b["mean_score"] for b in buckets],
        [b["mean_net_return"] for b in buckets],
    )
    erased_count = sum(1 for b in buckets if b["edge_erased"])

    return {
        "buckets": buckets,
        "spearman_raw": round(rho_raw, 4) if rho_raw is not None else None,
        "spearman_net": round(rho_net, 4) if rho_net is not None else None,
        "buckets_erased": erased_count,
        "n": len(quads),
    }


def compute_selection_bias(
    all_observations: List[Dict[str, Any]],
    traded_episodes: List[Dict[str, Any]],
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Compare monotonicity between full scored universe and traded subset.

    all_observations: passive observation records with hybrid_score
        (requires realized_return to be backfilled for counterfactual).
    traded_episodes: standard episode records with score and return.

    The delta between rho_full and rho_traded quantifies selection bias.
    """
    # Traded subset monotonicity
    traded = compute_monotonicity(traded_episodes, score_field=score_field)

    # Full universe monotonicity (only if observations have returns)
    full_pairs: List[tuple] = []
    for obs in all_observations:
        score = _safe_float(obs.get(score_field))
        ret_val = obs.get("realized_return")
        if ret_val is None:
            ret_val = _realized_return(obs)
        if ret_val is None or score <= 0:
            continue
        full_pairs.append((score, float(ret_val)))

    if len(full_pairs) < 10:
        return {
            "rho_traded": traded.get("spearman"),
            "rho_full": None,
            "selection_bias_delta": None,
            "n_traded": traded.get("n", 0),
            "n_full": len(full_pairs),
            "verdict": "insufficient_full_data",
        }

    full_scores = [p[0] for p in full_pairs]
    full_returns = [p[1] for p in full_pairs]
    rho_full = _spearman(full_scores, full_returns)

    rho_traded = traded.get("spearman")
    delta = None
    verdict = "insufficient"
    if rho_full is not None and rho_traded is not None:
        delta = round(rho_traded - rho_full, 4)
        if abs(delta) < 0.10:
            verdict = "signal_real"
        elif delta > 0.10:
            verdict = "selection_bias_suspected"
        else:
            verdict = "signal_stronger_in_full"

    return {
        "rho_traded": rho_traded,
        "rho_full": round(rho_full, 4) if rho_full is not None else None,
        "selection_bias_delta": delta,
        "n_traded": traded.get("n", 0),
        "n_full": len(full_pairs),
        "score_range_traded": {
            "min": round(min(p[0] for p in full_pairs if True), 4) if full_pairs else None,
            "max": round(max(p[0] for p in full_pairs), 4) if full_pairs else None,
        },
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Friction-Aware Edge Audit
# ---------------------------------------------------------------------------


def _median(vals: List[float]) -> float:
    """Pure-Python median."""
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _percentile(vals: List[float], pct: float) -> float:
    """Pure-Python percentile via nearest-rank."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


def _std(vals: List[float]) -> float:
    """Population standard deviation (pure Python)."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def compute_friction_decomposition(
    episodes: List[Dict[str, Any]],
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Decompose every trade into raw edge, fee drag, and net edge in BPS.

    Returns per-trade statistics and distribution summaries.
    A trade is 'friction-killed' when raw_edge_bps > 0 but net_edge_bps <= 0.
    """
    trades: List[Dict[str, Any]] = []
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        entry_px = _safe_float(ep.get("avg_entry_price"))
        exit_px = _safe_float(ep.get("avg_exit_price"))
        notional = _safe_float(ep.get("entry_notional"))
        fees = _safe_float(ep.get("fees"))
        gross = _safe_float(ep.get("gross_pnl"))
        net = _safe_float(ep.get("net_pnl"))
        if entry_px <= 0 or notional <= 0:
            continue

        side = str(ep.get("side", "")).upper()
        if side == "LONG":
            raw_ret = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            raw_ret = (entry_px - exit_px) / entry_px
        else:
            continue

        raw_bps = raw_ret * 10_000.0
        fee_bps = (fees / notional) * 10_000.0 if notional > 0 else 0.0
        net_bps = raw_bps - fee_bps

        # Cross-validate with gross/net PnL when available
        if notional > 0 and gross != 0.0:
            gross_ret_bps = (gross / notional) * 10_000.0
            net_ret_bps = (net / notional) * 10_000.0
        else:
            gross_ret_bps = raw_bps
            net_ret_bps = net_bps

        killed = raw_bps > 0 and net_bps <= 0

        trades.append({
            "symbol": ep.get("symbol", ""),
            "score": round(score, 5),
            "side": side,
            "raw_edge_bps": round(raw_bps, 2),
            "fee_drag_bps": round(fee_bps, 2),
            "net_edge_bps": round(net_bps, 2),
            "gross_ret_bps": round(gross_ret_bps, 2),
            "net_ret_bps": round(net_ret_bps, 2),
            "notional_usd": round(notional, 2),
            "fees_usd": round(fees, 4),
            "friction_killed": killed,
            "duration_hours": _safe_float(ep.get("duration_hours")),
        })

    if not trades:
        return {"n": 0, "distribution": {}, "trades_summary": {}}

    raw_all = [t["raw_edge_bps"] for t in trades]
    fee_all = [t["fee_drag_bps"] for t in trades]
    net_all = [t["net_edge_bps"] for t in trades]

    def _dist(vals: List[float]) -> Dict[str, float]:
        return {
            "mean": round(sum(vals) / len(vals), 2),
            "median": round(_median(vals), 2),
            "std": round(_std(vals), 2),
            "p10": round(_percentile(vals, 10), 2),
            "p25": round(_percentile(vals, 25), 2),
            "p75": round(_percentile(vals, 75), 2),
            "p90": round(_percentile(vals, 90), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    killed_count = sum(1 for t in trades if t["friction_killed"])
    positive_raw = [t for t in trades if t["raw_edge_bps"] > 0]
    kill_rate = (killed_count / len(positive_raw)) if positive_raw else 0.0

    # Fee-to-edge ratio: how much of gross edge is consumed by fees
    total_gross_edge = sum(t["raw_edge_bps"] for t in trades)
    total_fee_drag = sum(t["fee_drag_bps"] for t in trades)
    fee_to_edge_ratio = (
        total_fee_drag / total_gross_edge
        if total_gross_edge > 0
        else None
    )

    return {
        "n": len(trades),
        "distribution": {
            "raw_edge_bps": _dist(raw_all),
            "fee_drag_bps": _dist(fee_all),
            "net_edge_bps": _dist(net_all),
        },
        "trades_summary": {
            "positive_raw_count": len(positive_raw),
            "friction_killed_count": killed_count,
            "friction_kill_rate": round(kill_rate, 4),
            "fee_to_edge_ratio": (
                round(fee_to_edge_ratio, 4) if fee_to_edge_ratio is not None else None
            ),
            "total_fee_drag_bps": round(total_fee_drag, 2),
            "total_gross_edge_bps": round(total_gross_edge, 2),
            "total_net_edge_bps": round(total_gross_edge - total_fee_drag, 2),
        },
    }


def compute_friction_kill_analysis(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Friction-kill rate by score bucket and by symbol.

    Identifies which score regions and symbols are most vulnerable
    to having positive raw edge consumed by costs.
    """
    records: List[tuple] = []  # (score, raw_bps, net_bps, symbol)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        entry_px = _safe_float(ep.get("avg_entry_price"))
        notional = _safe_float(ep.get("entry_notional"))
        fees = _safe_float(ep.get("fees"))
        if entry_px <= 0 or notional <= 0:
            continue

        side = str(ep.get("side", "")).upper()
        exit_px = _safe_float(ep.get("avg_exit_price"))
        if side == "LONG":
            raw_ret = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            raw_ret = (entry_px - exit_px) / entry_px
        else:
            continue

        raw_bps = raw_ret * 10_000.0
        fee_bps = (fees / notional) * 10_000.0 if notional > 0 else 0.0
        net_bps = raw_bps - fee_bps
        records.append((score, raw_bps, net_bps, ep.get("symbol", "")))

    if len(records) < 5:
        return {"by_bucket": [], "by_symbol": [], "n": len(records)}

    # --- By score bucket ---
    records.sort(key=lambda r: r[0])
    bucket_size = max(1, len(records) // n_buckets)
    by_bucket: List[Dict[str, Any]] = []
    for i in range(0, len(records), bucket_size):
        chunk = records[i: i + bucket_size]
        positive_raw = [r for r in chunk if r[1] > 0]
        killed = sum(1 for r in positive_raw if r[2] <= 0)
        kill_rate = killed / len(positive_raw) if positive_raw else 0.0
        scores = [r[0] for r in chunk]
        raw_vals = [r[1] for r in chunk]
        net_vals = [r[2] for r in chunk]
        by_bucket.append({
            "label": f"Q{len(by_bucket) + 1}",
            "score_range": f"{min(scores):.3f}–{max(scores):.3f}",
            "mean_raw_bps": round(sum(raw_vals) / len(raw_vals), 2),
            "mean_net_bps": round(sum(net_vals) / len(net_vals), 2),
            "positive_raw": len(positive_raw),
            "friction_killed": killed,
            "kill_rate": round(kill_rate, 4),
            "n": len(chunk),
        })

    # --- By symbol ---
    sym_map: Dict[str, List[tuple]] = {}
    for r in records:
        sym_map.setdefault(r[3], []).append(r)

    by_symbol: List[Dict[str, Any]] = []
    for sym, sym_recs in sorted(sym_map.items()):
        if len(sym_recs) < 3:
            continue
        pos_raw = [r for r in sym_recs if r[1] > 0]
        killed = sum(1 for r in pos_raw if r[2] <= 0)
        kill_rate = killed / len(pos_raw) if pos_raw else 0.0
        raw_vals = [r[1] for r in sym_recs]
        net_vals = [r[2] for r in sym_recs]
        fee_vals = [r[1] - r[2] for r in sym_recs]
        by_symbol.append({
            "symbol": sym,
            "mean_raw_bps": round(sum(raw_vals) / len(raw_vals), 2),
            "mean_net_bps": round(sum(net_vals) / len(net_vals), 2),
            "mean_fee_bps": round(sum(fee_vals) / len(fee_vals), 2),
            "kill_rate": round(kill_rate, 4),
            "n": len(sym_recs),
        })

    # Sort by kill rate descending so worst symbols are first
    by_symbol.sort(key=lambda s: s["kill_rate"], reverse=True)

    return {
        "by_bucket": by_bucket,
        "by_symbol": by_symbol,
        "n": len(records),
    }


def compute_break_even_edge(
    episodes: List[Dict[str, Any]],
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Compute the minimum raw edge (in BPS) required to survive friction.

    Returns:
        break_even_bps: mean fee drag across all trades (the hurdle)
        pct_above_hurdle: fraction of trades with raw_edge > hurdle
        implied_min_score: approximate score at the break-even point
        duration_vs_friction: whether longer holds dilute fee impact
    """
    data: List[tuple] = []  # (score, raw_bps, fee_bps, duration_h)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        entry_px = _safe_float(ep.get("avg_entry_price"))
        notional = _safe_float(ep.get("entry_notional"))
        fees = _safe_float(ep.get("fees"))
        if entry_px <= 0 or notional <= 0:
            continue

        side = str(ep.get("side", "")).upper()
        exit_px = _safe_float(ep.get("avg_exit_price"))
        if side == "LONG":
            raw_ret = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            raw_ret = (entry_px - exit_px) / entry_px
        else:
            continue

        raw_bps = raw_ret * 10_000.0
        fee_bps = (fees / notional) * 10_000.0 if notional > 0 else 0.0
        dur = _safe_float(ep.get("duration_hours"))
        data.append((score, raw_bps, fee_bps, dur))

    if len(data) < 5:
        return {
            "break_even_bps": None,
            "pct_above_hurdle": None,
            "n": len(data),
        }

    fee_vals = [d[2] for d in data]
    mean_fee = sum(fee_vals) / len(fee_vals)
    median_fee = _median(fee_vals)
    hurdle = mean_fee  # mean fee drag as the break-even hurdle

    above = sum(1 for d in data if d[1] > hurdle)
    pct_above = above / len(data)

    # Approximate the implied minimum score at the break-even point
    # Sort by score and find where mean raw edge crosses hurdle
    data.sort(key=lambda d: d[0])
    implied_min_score = None
    window = max(5, len(data) // 10)
    for i in range(len(data) - window + 1):
        chunk_raw = [d[1] for d in data[i: i + window]]
        if sum(chunk_raw) / len(chunk_raw) >= hurdle:
            implied_min_score = data[i][0]
            break

    # Duration vs friction: compare fee drag per hour for short vs long holds
    dur_data = [(d[2], d[3]) for d in data if d[3] > 0]
    duration_insight = None
    if len(dur_data) >= 10:
        dur_data.sort(key=lambda d: d[1])
        mid = len(dur_data) // 2
        short_holds = dur_data[:mid]
        long_holds = dur_data[mid:]
        short_fee_mean = sum(d[0] for d in short_holds) / len(short_holds)
        long_fee_mean = sum(d[0] for d in long_holds) / len(long_holds)
        duration_insight = {
            "short_hold_mean_fee_bps": round(short_fee_mean, 2),
            "long_hold_mean_fee_bps": round(long_fee_mean, 2),
            "short_hold_mean_hours": round(
                sum(d[1] for d in short_holds) / len(short_holds), 2
            ),
            "long_hold_mean_hours": round(
                sum(d[1] for d in long_holds) / len(long_holds), 2
            ),
        }

    return {
        "break_even_bps": round(hurdle, 2),
        "median_fee_bps": round(median_fee, 2),
        "pct_above_hurdle": round(pct_above, 4),
        "implied_min_score": (
            round(implied_min_score, 4) if implied_min_score is not None else None
        ),
        "duration_vs_friction": duration_insight,
        "n": len(data),
    }


def compute_friction_audit(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Master friction-aware edge audit.

    Orchestrates decomposition, kill analysis, and break-even computation
    into a single verdict: TRADABLE, MARGINAL, or NOT_TRADABLE.

    Verdict criteria:
        TRADABLE       — net edge positive AND kill rate < 25%
        MARGINAL       — net edge positive BUT kill rate >= 25%
                         OR net edge negative but > -2 bps (within noise)
        NOT_TRADABLE   — net edge <= -2 bps OR kill rate >= 50%
    """
    decomp = compute_friction_decomposition(episodes, score_field=score_field)
    kills = compute_friction_kill_analysis(
        episodes, n_buckets=n_buckets, score_field=score_field,
    )
    breakev = compute_break_even_edge(episodes, score_field=score_field)

    # Also pull existing friction overlay for bucket-level view
    overlay = compute_friction_overlay(
        episodes, n_buckets=n_buckets, score_field=score_field,
    )

    n = decomp.get("n", 0)
    if n < 10:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n": n,
            "decomposition": decomp,
            "kill_analysis": kills,
            "break_even": breakev,
            "friction_overlay": overlay,
        }

    net_dist = decomp.get("distribution", {}).get("net_edge_bps", {})
    mean_net = net_dist.get("mean", 0.0)
    summary = decomp.get("trades_summary", {})
    kill_rate = summary.get("friction_kill_rate", 0.0)
    fee_to_edge = summary.get("fee_to_edge_ratio")

    # Verdict logic
    if mean_net > 0 and kill_rate < 0.25:
        verdict = "TRADABLE"
    elif kill_rate >= 0.50 or mean_net <= -2.0:
        verdict = "NOT_TRADABLE"
    else:
        verdict = "MARGINAL"

    # Severity narrative
    if fee_to_edge is not None:
        if fee_to_edge > 1.0:
            severity = "fees_exceed_edge"
        elif fee_to_edge > 0.5:
            severity = "fees_consume_majority"
        elif fee_to_edge > 0.25:
            severity = "moderate_drag"
        else:
            severity = "low_drag"
    else:
        severity = "unknown"

    return {
        "verdict": verdict,
        "severity": severity,
        "mean_net_edge_bps": round(mean_net, 2),
        "friction_kill_rate": round(kill_rate, 4),
        "fee_to_edge_ratio": (
            round(fee_to_edge, 4) if fee_to_edge is not None else None
        ),
        "n": n,
        "decomposition": decomp,
        "kill_analysis": kills,
        "break_even": breakev,
        "friction_overlay": overlay,
    }


# ---------------------------------------------------------------------------
# Calibration Audit (Probability Discipline)
# ---------------------------------------------------------------------------


def _binary_outcome(ep: Dict[str, Any]) -> Optional[int]:
    """Return 1 if trade was profitable (direction-adjusted), 0 if not, None if unusable."""
    entry = _safe_float(ep.get("avg_entry_price"))
    exit_ = _safe_float(ep.get("avg_exit_price"))
    if entry <= 0 or exit_ <= 0:
        return None
    side = str(ep.get("side", "")).upper()
    if side == "LONG":
        return 1 if exit_ > entry else 0
    elif side == "SHORT":
        return 1 if exit_ < entry else 0
    return None


def compute_calibration_curve(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 10,
    score_field: str = "conviction_score",
) -> Dict[str, Any]:
    """Build reliability diagram data: predicted probability vs realized frequency.

    Groups episodes into equal-count buckets by score_field and computes
    the realized hit rate (P(profitable)) in each bucket.

    A perfectly calibrated model has predicted ≈ realized in every bucket.
    """
    pairs: List[tuple] = []  # (predicted, outcome)
    for ep in episodes:
        pred = _safe_float(ep.get(score_field))
        if pred <= 0:
            continue
        outcome = _binary_outcome(ep)
        if outcome is None:
            continue
        pairs.append((pred, outcome))

    if len(pairs) < 10:
        return {"buckets": [], "n": len(pairs), "score_field": score_field}

    pairs.sort(key=lambda p: p[0])
    bucket_size = max(1, len(pairs) // n_buckets)
    buckets: List[Dict[str, Any]] = []

    for i in range(0, len(pairs), bucket_size):
        chunk = pairs[i: i + bucket_size]
        preds = [c[0] for c in chunk]
        outcomes = [c[1] for c in chunk]
        mean_pred = sum(preds) / len(preds)
        mean_outcome = sum(outcomes) / len(outcomes)
        gap = mean_pred - mean_outcome  # positive = overconfident

        buckets.append({
            "label": f"B{len(buckets) + 1}",
            "pred_lo": round(min(preds), 4),
            "pred_hi": round(max(preds), 4),
            "mean_predicted": round(mean_pred, 4),
            "realized_frequency": round(mean_outcome, 4),
            "gap": round(gap, 4),
            "n": len(chunk),
        })

    return {
        "buckets": buckets,
        "n": len(pairs),
        "score_field": score_field,
    }


def compute_brier_score(
    episodes: List[Dict[str, Any]],
    score_field: str = "conviction_score",
) -> Dict[str, Any]:
    """Compute Brier score and decomposition.

    Brier = (1/N) Σ (predicted_i − outcome_i)²

    Lower is better. Range [0, 1].
    Decomposition: reliability (calibration error), resolution (discrimination),
    and uncertainty (base rate entropy).

    Also computes naive baseline Brier (always predict base rate).
    """
    pairs: List[tuple] = []
    for ep in episodes:
        pred = _safe_float(ep.get(score_field))
        if pred <= 0:
            continue
        outcome = _binary_outcome(ep)
        if outcome is None:
            continue
        pairs.append((pred, outcome))

    n = len(pairs)
    if n < 5:
        return {
            "brier": None,
            "brier_baseline": None,
            "brier_skill_score": None,
            "n": n,
            "score_field": score_field,
        }

    preds = [p[0] for p in pairs]
    outcomes = [p[1] for p in pairs]
    base_rate = sum(outcomes) / n

    # Brier score
    brier = sum((p - o) ** 2 for p, o in pairs) / n

    # Naive baseline: always predict base rate
    brier_baseline = sum((base_rate - o) ** 2 for o in outcomes) / n

    # Brier Skill Score: BSS = 1 - Brier/Brier_baseline
    # BSS > 0 means model beats baseline, BSS < 0 means worse
    bss = 1.0 - (brier / brier_baseline) if brier_baseline > 0 else None

    # Decomposition (Murphy 1973): Brier = Reliability - Resolution + Uncertainty
    # Reliability = (1/N) Σ_k n_k (predicted_k - observed_k)²
    # Resolution  = (1/N) Σ_k n_k (observed_k - base_rate)²
    # Uncertainty = base_rate × (1 - base_rate)
    uncertainty = base_rate * (1.0 - base_rate)

    # Bucket for decomposition (10 buckets)
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    bkt_size = max(1, n // 10)
    reliability = 0.0
    resolution = 0.0
    for i in range(0, n, bkt_size):
        chunk = pairs_sorted[i: i + bkt_size]
        nk = len(chunk)
        pk = sum(c[0] for c in chunk) / nk
        ok = sum(c[1] for c in chunk) / nk
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - base_rate) ** 2
    reliability /= n
    resolution /= n

    return {
        "brier": round(brier, 6),
        "brier_baseline": round(brier_baseline, 6),
        "brier_skill_score": round(bss, 4) if bss is not None else None,
        "base_rate": round(base_rate, 4),
        "decomposition": {
            "reliability": round(reliability, 6),
            "resolution": round(resolution, 6),
            "uncertainty": round(uncertainty, 6),
        },
        "mean_predicted": round(sum(preds) / n, 4),
        "mean_outcome": round(sum(outcomes) / n, 4),
        "n": n,
        "score_field": score_field,
    }


def compute_calibration_diagnosis(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 10,
    score_field: str = "conviction_score",
) -> Dict[str, Any]:
    """Diagnose overconfidence, underconfidence, and collapse-to-mean.

    Overconfidence:   model predicts high but realized is lower (gap > 0 in top buckets)
    Underconfidence:  model predicts low but realized is higher (gap < 0 in low buckets)
    Collapse to mean: model clusters predictions near base rate — low spread between
                      min and max predicted values (< 0.15 range)

    Returns diagnosis dict with per-bucket gaps, ECE, MCE, and pattern verdict.
    """
    curve = compute_calibration_curve(
        episodes, n_buckets=n_buckets, score_field=score_field,
    )
    buckets = curve.get("buckets", [])
    n = curve.get("n", 0)

    if len(buckets) < 3:
        return {
            "ece": None,
            "mce": None,
            "pattern": "insufficient_data",
            "buckets": buckets,
            "n": n,
            "score_field": score_field,
        }

    # Expected Calibration Error (ECE) = weighted mean |gap|
    total_n = sum(b["n"] for b in buckets)
    ece = sum(b["n"] * abs(b["gap"]) for b in buckets) / total_n if total_n > 0 else 0.0

    # Maximum Calibration Error (MCE) = worst bucket
    mce = max(abs(b["gap"]) for b in buckets)

    # Detect patterns
    # Top third and bottom third of buckets
    n_third = max(1, len(buckets) // 3)
    bottom = buckets[:n_third]
    top = buckets[-n_third:]

    avg_gap_top = sum(b["gap"] for b in top) / len(top)
    avg_gap_bottom = sum(b["gap"] for b in bottom) / len(bottom)

    # Prediction spread (is the model collapsing to a narrow band?)
    pred_min = min(b["mean_predicted"] for b in buckets)
    pred_max = max(b["mean_predicted"] for b in buckets)
    pred_spread = pred_max - pred_min

    # Classify pattern
    patterns: List[str] = []
    if avg_gap_top > 0.05:
        patterns.append("overconfident_top")
    if avg_gap_bottom < -0.05:
        patterns.append("underconfident_bottom")
    if pred_spread < 0.15:
        patterns.append("collapse_to_mean")
    if ece < 0.03:
        patterns.append("well_calibrated")
    elif ece < 0.08:
        patterns.append("mildly_miscalibrated")
    else:
        patterns.append("severely_miscalibrated")

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "pred_spread": round(pred_spread, 4),
        "avg_gap_top_third": round(avg_gap_top, 4),
        "avg_gap_bottom_third": round(avg_gap_bottom, 4),
        "patterns": patterns,
        "buckets": buckets,
        "n": n,
        "score_field": score_field,
    }


def compute_calibration_audit(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 10,
) -> Dict[str, Any]:
    """Master calibration audit: calibrate both conviction_score and hybrid_score.

    Produces:
        - Calibration curve (reliability diagram) for each score
        - Brier score + decomposition + baseline comparison
        - Diagnosis: overconfidence / underconfidence / collapse
        - Lift: model Brier vs naive baseline
        - Verdict: CALIBRATED / MISCALIBRATED / OVERCONFIDENT / UNDERCONFIDENT

    Verdict criteria:
        CALIBRATED      — ECE < 0.05 AND BSS > 0
        OVERCONFIDENT    — top-third gap > 0.08 (predicts too high)
        UNDERCONFIDENT   — bottom-third gap < -0.08 (predicts too low)
        COLLAPSED        — pred_spread < 0.10 (model outputs cluster near mean)
        MISCALIBRATED    — ECE ≥ 0.08 without clear directional pattern
        INSUFFICIENT_DATA — fewer than 20 usable episodes
    """
    results: Dict[str, Any] = {}

    for field in ("conviction_score", "hybrid_score"):
        brier = compute_brier_score(episodes, score_field=field)
        diag = compute_calibration_diagnosis(
            episodes, n_buckets=n_buckets, score_field=field,
        )

        n = brier.get("n", 0)
        if n < 20:
            verdict = "INSUFFICIENT_DATA"
        else:
            ece = diag.get("ece", 1.0)
            bss = brier.get("brier_skill_score")
            gap_top = diag.get("avg_gap_top_third", 0.0)
            gap_bottom = diag.get("avg_gap_bottom_third", 0.0)
            spread = diag.get("pred_spread", 0.0)

            if spread < 0.10:
                verdict = "COLLAPSED"
            elif gap_top > 0.08:
                verdict = "OVERCONFIDENT"
            elif gap_bottom < -0.08:
                verdict = "UNDERCONFIDENT"
            elif ece is not None and ece < 0.05 and bss is not None and bss > 0:
                verdict = "CALIBRATED"
            elif ece is not None and ece >= 0.08:
                verdict = "MISCALIBRATED"
            else:
                verdict = "WEAKLY_CALIBRATED"

        results[field] = {
            "verdict": verdict,
            "brier": brier,
            "diagnosis": diag,
        }

    # Cross-score comparison: which score is better calibrated?
    conv_brier = results.get("conviction_score", {}).get("brier", {}).get("brier")
    hyb_brier = results.get("hybrid_score", {}).get("brier", {}).get("brier")
    if conv_brier is not None and hyb_brier is not None:
        if conv_brier < hyb_brier:
            better = "conviction_score"
        elif hyb_brier < conv_brier:
            better = "hybrid_score"
        else:
            better = "tie"
        delta = round(abs(conv_brier - hyb_brier), 6)
    else:
        better = "insufficient_data"
        delta = None

    return {
        "conviction_score": results.get("conviction_score", {}),
        "hybrid_score": results.get("hybrid_score", {}),
        "comparison": {
            "better_calibrated": better,
            "brier_delta": delta,
        },
    }


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

    # Temporal stability (3 time slices)
    ts_stab = compute_time_stability(episodes)
    snap["time_stability"] = ts_stab["stability"]
    snap["time_slices"] = ts_stab["slices"]

    # Duration confound check
    snap["duration_by_bucket"] = compute_duration_by_bucket(episodes)

    # Bootstrap significance for Q5-Q1
    boot = compute_bootstrap_q5_q1(episodes)
    snap["q5_q1_bootstrap_p"] = boot["p_value"]

    # Threshold optimality sweep
    snap["threshold_sweep"] = compute_threshold_sweep(episodes)

    # Continuous edge curve
    snap["edge_curve"] = compute_edge_curve(episodes)

    # Direction correctness
    snap["direction_accuracy"] = compute_direction_accuracy(episodes)

    # Friction overlay (raw vs net)
    snap["friction_overlay"] = compute_friction_overlay(episodes)

    # Friction-aware edge audit (comprehensive)
    snap["friction_audit"] = compute_friction_audit(episodes)

    # Calibration audit (probability discipline)
    snap["calibration_audit"] = compute_calibration_audit(episodes)

    try:
        tmp = dest + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(snap, fh, indent=2)
        os.replace(tmp, dest)
    except OSError as exc:
        LOG.debug("[hydra_monotonicity] persist failed: %s", exc)

    return snap
