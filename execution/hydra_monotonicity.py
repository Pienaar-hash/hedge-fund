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


# ===========================================================================
# Data-room evidence compute functions
# ===========================================================================
# Pure read-only analytics over the episode ledger. No side effects, no
# state writes, no execution influence. Consumed by
# scripts/generate_data_room_evidence.py.

def _bucket_label(qi: int) -> str:
    return f"Q{qi + 1}"


def _quintile_groups(
    pairs: List[tuple],
    n_buckets: int = 5,
) -> List[List[tuple]]:
    """Split sorted-by-first-element pairs into n equal-count buckets."""
    if not pairs:
        return []
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    n = len(pairs_sorted)
    bucket_size = max(1, n // n_buckets)
    groups: List[List[tuple]] = []
    for qi in range(n_buckets):
        lo_idx = qi * bucket_size
        hi_idx = (qi + 1) * bucket_size if qi < n_buckets - 1 else n
        groups.append(pairs_sorted[lo_idx:hi_idx])
    return groups


def _net_realized_return(ep: Dict[str, Any]) -> Optional[float]:
    """Return net (post-fee) return as a fraction of entry notional."""
    raw = _realized_return(ep)
    if raw is None:
        return None
    notional = _safe_float(ep.get("entry_notional"))
    fees = _safe_float(ep.get("fees"))
    if notional <= 0:
        return raw
    fee_drag = fees / notional
    return raw - fee_drag


# ---------------------------------------------------------------------------
# Direction accuracy (signal causality, layer 02)
# ---------------------------------------------------------------------------
def compute_direction_accuracy(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Bucket trades by score and compute hit rate (return > 0).

    Returns per-bucket accuracy and lift vs random (0.5 baseline).
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

    n = len(pairs)
    if n < 5:
        return {
            "buckets": [], "overall_accuracy": None, "overall_lift": None,
            "direction_spearman": None, "n": n,
        }

    overall_acc = sum(1 for _, r in pairs if r > 0) / n

    groups = _quintile_groups(pairs, n_buckets=n_buckets)
    buckets: List[Dict[str, Any]] = []
    bucket_means: List[float] = []
    bucket_accs: List[float] = []
    for qi, group in enumerate(groups):
        if not group:
            continue
        accs = [1.0 if r > 0 else 0.0 for _, r in group]
        mean_score = sum(s for s, _ in group) / len(group)
        acc = sum(accs) / len(accs)
        buckets.append({
            "label": _bucket_label(qi),
            "mean_score": round(mean_score, 5),
            "accuracy": round(acc, 4),
            "lift_vs_random": round(acc - 0.5, 4),
            "n": len(group),
        })
        bucket_means.append(mean_score)
        bucket_accs.append(acc)

    rho = _spearman(bucket_means, bucket_accs) if len(bucket_means) >= 5 else None

    return {
        "buckets": buckets,
        "overall_accuracy": round(overall_acc, 4),
        "overall_lift": round(overall_acc - 0.5, 4),
        "direction_spearman": round(rho, 4) if rho is not None else None,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Friction decomposition (layer 03)
# ---------------------------------------------------------------------------
def compute_friction_decomposition(
    episodes: List[Dict[str, Any]],
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Aggregate per-trade friction.

    Returns trades_summary with fee_to_edge_ratio, friction_kill_rate, and
    total gross/net edge sums in basis points.
    """
    n = 0
    sum_gross_bps = 0.0
    sum_fee_bps = 0.0
    sum_net_bps = 0.0
    killed = 0
    sum_gross_abs_bps = 0.0  # for fee/edge ratio (denominator uses |gross|)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        raw = _realized_return(ep)
        if raw is None:
            continue
        notional = _safe_float(ep.get("entry_notional"))
        if notional <= 0:
            continue
        fees = _safe_float(ep.get("fees"))
        raw_bps = raw * 10_000
        fee_bps = (fees / notional) * 10_000
        net_bps = raw_bps - fee_bps
        n += 1
        sum_gross_bps += raw_bps
        sum_fee_bps += fee_bps
        sum_net_bps += net_bps
        sum_gross_abs_bps += abs(raw_bps)
        if raw_bps > 0 and net_bps <= 0:
            killed += 1

    if n == 0:
        return {
            "n": 0,
            "trades_summary": {
                "fee_to_edge_ratio": None,
                "friction_kill_rate": None,
                "total_gross_edge_bps": 0.0,
                "total_fee_drag_bps": 0.0,
                "total_net_edge_bps": 0.0,
            },
        }

    fee_to_edge = (sum_fee_bps / sum_gross_abs_bps) if sum_gross_abs_bps > 0 else None

    return {
        "n": n,
        "trades_summary": {
            "fee_to_edge_ratio": round(fee_to_edge, 4) if fee_to_edge is not None else None,
            "friction_kill_rate": round(killed / n, 4),
            "total_gross_edge_bps": round(sum_gross_bps, 2),
            "total_fee_drag_bps": round(sum_fee_bps, 2),
            "total_net_edge_bps": round(sum_net_bps, 2),
        },
    }


# ---------------------------------------------------------------------------
# Friction overlay (layer 03 — bucket-level raw vs net)
# ---------------------------------------------------------------------------
def compute_friction_overlay(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 5,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Bucket trades by score; report mean raw and mean net return per bucket.

    edge_erased = True when bucket's raw mean > 0 but net mean <= 0.
    """
    triples: List[tuple] = []  # (score, raw_return, net_return)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        raw = _realized_return(ep)
        net = _net_realized_return(ep)
        if raw is None or net is None:
            continue
        triples.append((score, raw, net))

    n = len(triples)
    if n < 5:
        return {
            "buckets": [], "spearman_raw": None, "spearman_net": None,
            "buckets_erased": 0, "n": n,
        }

    pairs_score_sorted = [(t[0], t) for t in triples]
    groups = _quintile_groups(pairs_score_sorted, n_buckets=n_buckets)
    buckets: List[Dict[str, Any]] = []
    bucket_scores: List[float] = []
    bucket_raws: List[float] = []
    bucket_nets: List[float] = []
    erased = 0
    for qi, group in enumerate(groups):
        if not group:
            continue
        items = [g[1] for g in group]
        mean_score = sum(s for s, _, _ in items) / len(items)
        mean_raw = sum(r for _, r, _ in items) / len(items)
        mean_net = sum(net for _, _, net in items) / len(items)
        is_erased = (mean_raw > 0) and (mean_net <= 0)
        if is_erased:
            erased += 1
        buckets.append({
            "label": _bucket_label(qi),
            "mean_score": round(mean_score, 5),
            "mean_raw_return": round(mean_raw, 6),
            "mean_net_return": round(mean_net, 6),
            "friction_drag": round(mean_raw - mean_net, 6),
            "edge_erased": is_erased,
            "n": len(items),
        })
        bucket_scores.append(mean_score)
        bucket_raws.append(mean_raw)
        bucket_nets.append(mean_net)

    rho_raw = _spearman(bucket_scores, bucket_raws) if len(bucket_scores) >= 5 else None
    rho_net = _spearman(bucket_scores, bucket_nets) if len(bucket_scores) >= 5 else None

    return {
        "buckets": buckets,
        "spearman_raw": round(rho_raw, 4) if rho_raw is not None else None,
        "spearman_net": round(rho_net, 4) if rho_net is not None else None,
        "buckets_erased": erased,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Break-even edge (layer 03 summary)
# ---------------------------------------------------------------------------
def compute_break_even_edge(
    episodes: List[Dict[str, Any]],
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Compute break-even raw edge (bps) and what fraction of trades clear it.

    Break-even per-trade is fees/notional (round-trip already in `fees`).
    pct_above_hurdle = fraction of trades where raw_bps > fee_bps.
    implied_min_score = lowest score whose mean raw return clears the
    median fee_bps across all scored trades.
    """
    samples: List[tuple] = []  # (score, raw_bps, fee_bps)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        raw = _realized_return(ep)
        if raw is None:
            continue
        notional = _safe_float(ep.get("entry_notional"))
        if notional <= 0:
            continue
        fees = _safe_float(ep.get("fees"))
        raw_bps = raw * 10_000
        fee_bps = (fees / notional) * 10_000
        samples.append((score, raw_bps, fee_bps))

    n = len(samples)
    if n == 0:
        return {
            "break_even_bps": None,
            "pct_above_hurdle": None,
            "implied_min_score": None,
            "n": 0,
        }

    fee_bps_sorted = sorted(s[2] for s in samples)
    median_fee_bps = fee_bps_sorted[n // 2]
    above = sum(1 for _, raw_bps, fee_bps in samples if raw_bps > fee_bps)

    # Find the lowest score whose forward (>=score) sample mean raw_bps
    # clears the median fee hurdle. None when no such bin exists.
    samples_sorted = sorted(samples, key=lambda s: s[0])
    implied_min = None
    for i in range(n):
        tail = samples_sorted[i:]
        if len(tail) < max(5, n // 10):
            break
        mean_raw = sum(t[1] for t in tail) / len(tail)
        if mean_raw > median_fee_bps:
            implied_min = round(samples_sorted[i][0], 5)
            break

    return {
        "break_even_bps": round(median_fee_bps, 2),
        "pct_above_hurdle": round(above / n, 4),
        "implied_min_score": implied_min,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Calibration curve (layer 04)
# ---------------------------------------------------------------------------
def _normalize_score_to_prob(score: float) -> float:
    """Clip score to [0,1] (hybrid_score is already in [0,1])."""
    return max(0.0, min(1.0, float(score)))


def compute_calibration_curve(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 10,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Reliability curve: predicted probability vs realized hit frequency.

    Hit = realized_return > 0. Buckets by predicted probability deciles.
    """
    pairs: List[tuple] = []  # (predicted_prob, hit)
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        ret = _realized_return(ep)
        if ret is None:
            continue
        pairs.append((_normalize_score_to_prob(score), 1 if ret > 0 else 0))

    n = len(pairs)
    if n < 5:
        return {"buckets": [], "n": n}

    pairs.sort(key=lambda p: p[0])
    bucket_size = max(1, n // n_buckets)
    buckets: List[Dict[str, Any]] = []
    for qi in range(n_buckets):
        lo_idx = qi * bucket_size
        hi_idx = (qi + 1) * bucket_size if qi < n_buckets - 1 else n
        chunk = pairs[lo_idx:hi_idx]
        if not chunk:
            continue
        preds = [c[0] for c in chunk]
        hits = [c[1] for c in chunk]
        mean_pred = sum(preds) / len(preds)
        realized = sum(hits) / len(hits)
        buckets.append({
            "label": _bucket_label(qi) if n_buckets <= 5 else f"D{qi + 1}",
            "pred_lo": round(min(preds), 5),
            "pred_hi": round(max(preds), 5),
            "mean_predicted": round(mean_pred, 4),
            "realized_frequency": round(realized, 4),
            "gap": round(mean_pred - realized, 4),
            "n": len(chunk),
        })

    return {"buckets": buckets, "n": n}


# ---------------------------------------------------------------------------
# Brier score + Murphy decomposition (layer 04)
# ---------------------------------------------------------------------------
def compute_brier_score(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 10,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Brier score with Murphy decomposition (reliability, resolution, uncertainty).

    Brier = mean((p_i - o_i)^2)
    Brier = Reliability - Resolution + Uncertainty
    """
    pairs: List[tuple] = []
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        ret = _realized_return(ep)
        if ret is None:
            continue
        pairs.append((_normalize_score_to_prob(score), 1.0 if ret > 0 else 0.0))

    n = len(pairs)
    if n < 5:
        return {
            "brier": None, "brier_baseline": None, "brier_skill_score": None,
            "base_rate": None, "decomposition": {}, "n": n,
        }

    base_rate = sum(o for _, o in pairs) / n
    brier = sum((p - o) ** 2 for p, o in pairs) / n
    brier_baseline = base_rate * (1.0 - base_rate)
    bss = (
        1.0 - (brier / brier_baseline) if brier_baseline > 0 else None
    )

    # Murphy decomposition via probability bins
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    bucket_size = max(1, n // n_buckets)
    rel = 0.0
    res = 0.0
    for qi in range(n_buckets):
        lo_idx = qi * bucket_size
        hi_idx = (qi + 1) * bucket_size if qi < n_buckets - 1 else n
        chunk = pairs_sorted[lo_idx:hi_idx]
        if not chunk:
            continue
        nk = len(chunk)
        pk = sum(p for p, _ in chunk) / nk
        ok = sum(o for _, o in chunk) / nk
        rel += (nk / n) * (pk - ok) ** 2
        res += (nk / n) * (ok - base_rate) ** 2
    unc = base_rate * (1.0 - base_rate)

    return {
        "brier": round(brier, 5),
        "brier_baseline": round(brier_baseline, 5),
        "brier_skill_score": round(bss, 4) if bss is not None else None,
        "base_rate": round(base_rate, 4),
        "decomposition": {
            "reliability": round(rel, 5),
            "resolution": round(res, 5),
            "uncertainty": round(unc, 5),
        },
        "n": n,
    }


# ---------------------------------------------------------------------------
# Calibration diagnosis (ECE, MCE, pattern verdicts) (layer 04)
# ---------------------------------------------------------------------------
def compute_calibration_diagnosis(
    episodes: List[Dict[str, Any]],
    n_buckets: int = 10,
    score_field: str = "hybrid_score",
) -> Dict[str, Any]:
    """Expected/Maximum Calibration Error + qualitative pattern verdicts."""
    curve = compute_calibration_curve(
        episodes, n_buckets=n_buckets, score_field=score_field,
    )
    buckets = curve.get("buckets", [])
    n_total = curve.get("n", 0)
    if not buckets or n_total == 0:
        return {
            "ece": None, "mce": None, "pred_spread": None,
            "patterns": [], "n": n_total,
        }

    ece = 0.0
    mce = 0.0
    preds = []
    for b in buckets:
        nk = b.get("n", 0)
        gap = abs(b.get("gap", 0.0))
        if nk and n_total:
            ece += (nk / n_total) * gap
        if gap > mce:
            mce = gap
        preds.append(b.get("mean_predicted", 0.0))

    pred_spread = (max(preds) - min(preds)) if preds else 0.0

    patterns: List[str] = []
    # Overconfident: mean gap > 0 and ECE > 0.1
    mean_gap = sum(b.get("gap", 0.0) for b in buckets) / len(buckets)
    if ece > 0.10 and mean_gap > 0.05:
        patterns.append("overconfident")
    elif ece > 0.10 and mean_gap < -0.05:
        patterns.append("underconfident")
    elif ece <= 0.05:
        patterns.append("well_calibrated")
    else:
        patterns.append("modestly_miscalibrated")

    if pred_spread < 0.10:
        patterns.append("flat_predictions")
    if mce > 0.25:
        patterns.append("severe_local_bias")

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "pred_spread": round(pred_spread, 4),
        "patterns": patterns,
        "n": n_total,
    }
