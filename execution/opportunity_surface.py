"""
Opportunity Surface (v7.9-P5)

Decomposes the episode pool to identify dominant loss mechanisms and test
whether regime conditioning hides local positive structure.

Two analysis phases share this module:

  P5A — Band Composition Audit:
    Per-conviction-band breakdown by symbol, side, regime, exit_reason,
    duration bucket.  Includes friction_dominated classification, top
    dragger identification, and conservation assertions.

  P5C — Episode-Quality Audit:
    Cross-cutting loss-driver analysis over ALL episodes (not just scored).
    Friction burden, exit-class PnL, duration quality, symbol drag, regime
    mismatch cost.  Produces an ordered loss-driver summary.

Hard invariants (conservation):
  - Band subcohort PnL sums exactly to band PnL
  - Symbol drag totals sum exactly to pool total
  - Exit class totals sum exactly to pool total
  - friction_dominated_pnl + non_friction_pnl == total pool net_pnl

Frozen definitions:
  - friction_dominated = (gross_pnl > 0) and (net_pnl <= 0)
  - regime_mismatch = (exit_reason == "REGIME_CHANGE")
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOG = logging.getLogger("opportunity_surface")

# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_BAND_WIDTH: float = 0.05
DEFAULT_MIN_CONVICTION: float = 0.20

DURATION_BUCKETS = [
    ("lt_1h", 0.0, 1.0),
    ("1h_6h", 1.0, 6.0),
    ("6h_24h", 6.0, 24.0),
    ("gt_24h", 24.0, float("inf")),
]

STATE_PATH = "logs/state/opportunity_surface.json"


# ── Helpers ───────────────────────────────────────────────────────────────

def _band_key(conviction: float, band_width: float) -> str:
    """Same band key logic as expectancy_bridge._band_key."""
    band_idx = math.floor(conviction / band_width + 1e-9)
    band_lo = round(band_idx * band_width, 4)
    band_hi = round(band_lo + band_width, 4)
    return f"{band_lo:.2f}-{band_hi:.2f}"


def _duration_bucket(hours: float) -> str:
    """Classify duration into a named bucket."""
    for name, lo, hi in DURATION_BUCKETS:
        if lo <= hours < hi:
            return name
    return "gt_24h"


def _is_friction_dominated(ep: Dict[str, Any]) -> bool:
    """Canonical frozen definition: gross_pnl > 0 and net_pnl <= 0."""
    gross = float(ep.get("gross_pnl") or 0)
    net = float(ep.get("net_pnl") or 0)
    return gross > 0 and net <= 0


def _safe_share(part: float, total: float) -> float:
    """Share of total loss. Returns 0 if total is zero or positive."""
    if total >= 0 or part >= 0:
        return 0.0
    return part / total


def _safe_div(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return num / denom


# ── P5A: Band Composition Audit ──────────────────────────────────────────

def _empty_cohort() -> Dict[str, Any]:
    return {"n": 0, "gross_pnl": 0.0, "fees": 0.0, "net_pnl": 0.0}


def _add_to_cohort(cohort: Dict[str, Any], ep: Dict[str, Any]) -> None:
    cohort["n"] += 1
    cohort["gross_pnl"] += float(ep.get("gross_pnl") or 0)
    cohort["fees"] += float(ep.get("fees") or 0)
    cohort["net_pnl"] += float(ep.get("net_pnl") or 0)


def _finalize_cohort(cohort: Dict[str, Any], band_net_pnl: float) -> Dict[str, Any]:
    """Add share_of_band_loss to a cohort."""
    cohort["share_of_band_loss"] = _safe_share(cohort["net_pnl"], band_net_pnl)
    # Round for JSON
    cohort["gross_pnl"] = round(cohort["gross_pnl"], 6)
    cohort["fees"] = round(cohort["fees"], 6)
    cohort["net_pnl"] = round(cohort["net_pnl"], 6)
    cohort["share_of_band_loss"] = round(cohort["share_of_band_loss"], 6)
    return cohort


def build_composition_audit(
    episodes: Sequence[Dict[str, Any]],
    band_width: float = DEFAULT_BAND_WIDTH,
    min_conviction: float = DEFAULT_MIN_CONVICTION,
) -> Dict[str, Any]:
    """P5A: Per-band decomposition of scored episodes.

    Returns a dict keyed by band (e.g. "0.40-0.45") with full breakdowns.
    Only includes episodes with conviction_score >= min_conviction and valid
    net_pnl / entry_notional.

    Conservation invariant: sum of each dimension's cohort net_pnl == band net_pnl.
    """
    # Accumulate raw data per band
    band_episodes: Dict[str, List[Dict[str, Any]]] = {}

    for ep in episodes:
        conviction = float(ep.get("conviction_score") or 0)
        if conviction < min_conviction:
            continue
        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue
        notional = float(ep.get("entry_notional") or 0)
        if notional <= 0:
            continue

        key = _band_key(conviction, band_width)
        band_episodes.setdefault(key, []).append(ep)

    result: Dict[str, Any] = {}

    for band_key, eps in sorted(band_episodes.items()):
        band_data = _build_band_detail(eps, band_key, band_width)
        result[band_key] = band_data

    return result


def _build_band_detail(
    eps: List[Dict[str, Any]],
    band_key: str,
    band_width: float,
) -> Dict[str, Any]:
    """Build detailed breakdown for a single band."""

    n = len(eps)
    gross_pnl = sum(float(e.get("gross_pnl") or 0) for e in eps)
    fees = sum(float(e.get("fees") or 0) for e in eps)
    net_pnl = sum(float(e.get("net_pnl") or 0) for e in eps)
    notional = sum(float(e.get("entry_notional") or 0) for e in eps)
    avg_net_edge_pct = _safe_div(net_pnl, notional) if notional > 0 else 0.0

    # ── Dimension breakdowns ──
    symbol_bk: Dict[str, Dict[str, Any]] = {}
    side_bk: Dict[str, Dict[str, Any]] = {}
    regime_bk: Dict[str, Dict[str, Any]] = {}
    exit_bk: Dict[str, Dict[str, Any]] = {}
    duration_bk: Dict[str, Dict[str, Any]] = {}

    friction_count = 0
    friction_pnl = 0.0

    for ep in eps:
        sym = ep.get("symbol") or "UNKNOWN"
        side = ep.get("side") or "UNKNOWN"
        regime = ep.get("regime_at_entry") or "unknown"
        exit_r = ep.get("exit_reason") or "UNKNOWN"
        dur = float(ep.get("duration_hours") or 0)
        dur_bucket = _duration_bucket(dur)

        for dim_map, dim_key in [
            (symbol_bk, sym),
            (side_bk, side),
            (regime_bk, regime),
            (exit_bk, exit_r),
            (duration_bk, dur_bucket),
        ]:
            if dim_key not in dim_map:
                dim_map[dim_key] = _empty_cohort()
            _add_to_cohort(dim_map[dim_key], ep)

        if _is_friction_dominated(ep):
            friction_count += 1
            friction_pnl += float(ep.get("net_pnl") or 0)

    # Finalize cohorts with share_of_band_loss
    for dim_map in [symbol_bk, side_bk, regime_bk, exit_bk, duration_bk]:
        for cohort in dim_map.values():
            _finalize_cohort(cohort, net_pnl)

    # Top dragger: symbol with largest absolute net loss
    top_dragger = ""
    top_dragger_loss = 0.0
    for sym, cohort in symbol_bk.items():
        if cohort["net_pnl"] < top_dragger_loss:
            top_dragger = sym
            top_dragger_loss = cohort["net_pnl"]

    top_dragger_share = _safe_share(top_dragger_loss, net_pnl) if top_dragger else 0.0

    # ── Conservation assertions ──
    for label, dim_map in [
        ("symbol", symbol_bk),
        ("side", side_bk),
        ("regime", regime_bk),
        ("exit_reason", exit_bk),
        ("duration", duration_bk),
    ]:
        dim_sum = sum(c["net_pnl"] for c in dim_map.values())
        assert abs(dim_sum - net_pnl) < 1e-6, (
            f"Conservation violation in band {band_key}, dimension {label}: "
            f"sum={dim_sum}, total={net_pnl}"
        )

    non_friction_pnl = net_pnl - friction_pnl
    # friction + non_friction == total (conservation)
    assert abs((friction_pnl + non_friction_pnl) - net_pnl) < 1e-6, (
        f"Friction conservation violation in band {band_key}: "
        f"friction={friction_pnl}, non_friction={non_friction_pnl}, total={net_pnl}"
    )

    return {
        "episode_count": n,
        "gross_pnl": round(gross_pnl, 6),
        "fees": round(fees, 6),
        "net_pnl": round(net_pnl, 6),
        "notional": round(notional, 2),
        "avg_net_edge_pct": round(avg_net_edge_pct, 8),
        "friction_dominated_count": friction_count,
        "friction_dominated_pnl": round(friction_pnl, 6),
        "non_friction_pnl": round(non_friction_pnl, 6),
        "top_dragger": top_dragger,
        "top_dragger_share": round(top_dragger_share, 6),
        "symbol_breakdown": symbol_bk,
        "side_breakdown": side_bk,
        "regime_breakdown": regime_bk,
        "exit_reason_breakdown": exit_bk,
        "duration_breakdown": duration_bk,
    }


def diagnose_band_drag(audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank drag sources across all bands by net_pnl magnitude.

    Returns a list of {band_key, dimension, key, net_pnl, share_of_band_loss}
    ordered by loss severity (most negative first).
    """
    drags: List[Dict[str, Any]] = []

    for band_key, band_data in audit.items():
        band_net = band_data["net_pnl"]
        for dim_name in ["symbol_breakdown", "side_breakdown", "regime_breakdown",
                         "exit_reason_breakdown", "duration_breakdown"]:
            breakdown = band_data.get(dim_name, {})
            for key, cohort in breakdown.items():
                if cohort["net_pnl"] < 0:
                    drags.append({
                        "band_key": band_key,
                        "dimension": dim_name.replace("_breakdown", ""),
                        "key": key,
                        "net_pnl": cohort["net_pnl"],
                        "share_of_band_loss": cohort["share_of_band_loss"],
                        "episode_count": cohort["n"],
                    })

    drags.sort(key=lambda d: d["net_pnl"])
    return drags


# ── P5C: Episode-Quality Audit ───────────────────────────────────────────

def compute_friction_burden(
    episodes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fraction of episodes where friction_dominated=True, per symbol, regime, band.

    Uses ALL episodes (not just scored).
    """
    total = 0
    friction_total = 0

    per_symbol: Dict[str, Dict[str, int]] = {}
    per_regime: Dict[str, Dict[str, int]] = {}

    for ep in episodes:
        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue
        total += 1
        is_f = _is_friction_dominated(ep)
        if is_f:
            friction_total += 1

        sym = ep.get("symbol") or "UNKNOWN"
        regime = ep.get("regime_at_entry") or "unknown"

        for dim_map, dim_key in [(per_symbol, sym), (per_regime, regime)]:
            if dim_key not in dim_map:
                dim_map[dim_key] = {"n": 0, "friction": 0}
            dim_map[dim_key]["n"] += 1
            if is_f:
                dim_map[dim_key]["friction"] += 1

    def _rate(d: Dict[str, int]) -> float:
        return _safe_div(d["friction"], d["n"])

    return {
        "total_episodes": total,
        "friction_dominated_count": friction_total,
        "friction_rate": round(_safe_div(friction_total, total), 4),
        "per_symbol": {k: {"n": v["n"], "friction": v["friction"],
                           "rate": round(_rate(v), 4)}
                       for k, v in sorted(per_symbol.items())},
        "per_regime": {k: {"n": v["n"], "friction": v["friction"],
                           "rate": round(_rate(v), 4)}
                       for k, v in sorted(per_regime.items())},
    }


def compute_exit_class_pnl(
    episodes: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """PnL breakdown by exit_reason.

    Returns {exit_reason → {n, gross_pnl, fees, net_pnl, share_of_total_loss, median_loss}}.
    Conservation: sum of all exit class net_pnl == total net_pnl.
    """
    by_reason: Dict[str, List[Dict[str, Any]]] = {}
    total_net = 0.0

    for ep in episodes:
        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue
        reason = ep.get("exit_reason") or "UNKNOWN"
        by_reason.setdefault(reason, []).append(ep)
        total_net += float(net_pnl)

    result: Dict[str, Dict[str, Any]] = {}
    check_sum = 0.0

    for reason, eps in sorted(by_reason.items()):
        gross = sum(float(e.get("gross_pnl") or 0) for e in eps)
        fees = sum(float(e.get("fees") or 0) for e in eps)
        net = sum(float(e.get("net_pnl") or 0) for e in eps)
        losses = [float(e["net_pnl"]) for e in eps if float(e.get("net_pnl") or 0) < 0]
        median_loss = round(statistics.median(losses), 6) if losses else 0.0

        result[reason] = {
            "n": len(eps),
            "gross_pnl": round(gross, 6),
            "fees": round(fees, 6),
            "net_pnl": round(net, 6),
            "share_of_total_loss": round(_safe_share(net, total_net), 6),
            "median_loss": median_loss,
        }
        check_sum += net

    # Conservation
    assert abs(check_sum - total_net) < 1e-6, (
        f"Exit class conservation violation: sum={check_sum}, total={total_net}"
    )

    return result


def compute_duration_quality(
    episodes: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Edge quality by duration bucket.

    Returns {bucket → {n, gross_pnl, fees, net_pnl, avg_net_edge_pct}}.
    """
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}

    for ep in episodes:
        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue
        dur = float(ep.get("duration_hours") or 0)
        bucket = _duration_bucket(dur)
        by_bucket.setdefault(bucket, []).append(ep)

    result: Dict[str, Dict[str, Any]] = {}
    for bucket, eps in sorted(by_bucket.items()):
        gross = sum(float(e.get("gross_pnl") or 0) for e in eps)
        fees = sum(float(e.get("fees") or 0) for e in eps)
        net = sum(float(e.get("net_pnl") or 0) for e in eps)
        notional = sum(float(e.get("entry_notional") or 0) for e in eps)
        result[bucket] = {
            "n": len(eps),
            "gross_pnl": round(gross, 6),
            "fees": round(fees, 6),
            "net_pnl": round(net, 6),
            "avg_net_edge_pct": round(_safe_div(net, notional), 8) if notional > 0 else 0.0,
        }

    return result


def compute_symbol_drag(
    episodes: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """PnL breakdown by symbol.

    Conservation: sum of all symbol net_pnl == total net_pnl.
    Returns {symbol → {n, gross_pnl, fees, net_pnl, share_of_total_loss}}.
    """
    by_sym: Dict[str, List[Dict[str, Any]]] = {}
    total_net = 0.0

    for ep in episodes:
        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue
        sym = ep.get("symbol") or "UNKNOWN"
        by_sym.setdefault(sym, []).append(ep)
        total_net += float(net_pnl)

    result: Dict[str, Dict[str, Any]] = {}
    check_sum = 0.0

    for sym, eps in sorted(by_sym.items()):
        gross = sum(float(e.get("gross_pnl") or 0) for e in eps)
        fees = sum(float(e.get("fees") or 0) for e in eps)
        net = sum(float(e.get("net_pnl") or 0) for e in eps)
        result[sym] = {
            "n": len(eps),
            "gross_pnl": round(gross, 6),
            "fees": round(fees, 6),
            "net_pnl": round(net, 6),
            "share_of_total_loss": round(_safe_share(net, total_net), 6),
        }
        check_sum += net

    # Conservation
    assert abs(check_sum - total_net) < 1e-6, (
        f"Symbol drag conservation violation: sum={check_sum}, total={total_net}"
    )

    return result


def compute_regime_mismatch_cost(
    episodes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """PnL of regime_mismatch (exit_reason=REGIME_CHANGE) vs thesis-driven.

    Frozen definition: regime_mismatch = exit_reason == "REGIME_CHANGE"
    Observable only — no narrative inference.
    """
    mismatch_net = 0.0
    mismatch_gross = 0.0
    mismatch_fees = 0.0
    mismatch_n = 0

    thesis_net = 0.0
    thesis_gross = 0.0
    thesis_fees = 0.0
    thesis_n = 0

    other_net = 0.0
    other_gross = 0.0
    other_fees = 0.0
    other_n = 0

    total_net = 0.0

    for ep in episodes:
        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue
        net = float(net_pnl)
        gross = float(ep.get("gross_pnl") or 0)
        fees = float(ep.get("fees") or 0)
        total_net += net
        reason = ep.get("exit_reason") or "UNKNOWN"

        if reason == "REGIME_CHANGE":
            mismatch_n += 1
            mismatch_net += net
            mismatch_gross += gross
            mismatch_fees += fees
        elif reason == "THESIS_INVALIDATED":
            thesis_n += 1
            thesis_net += net
            thesis_gross += gross
            thesis_fees += fees
        else:
            other_n += 1
            other_net += net
            other_gross += gross
            other_fees += fees

    # Conservation: mismatch + thesis + other == total
    check_sum = mismatch_net + thesis_net + other_net
    assert abs(check_sum - total_net) < 1e-6, (
        f"Regime mismatch conservation violation: sum={check_sum}, total={total_net}"
    )

    return {
        "regime_mismatch": {
            "n": mismatch_n,
            "gross_pnl": round(mismatch_gross, 6),
            "fees": round(mismatch_fees, 6),
            "net_pnl": round(mismatch_net, 6),
            "share_of_total_loss": round(_safe_share(mismatch_net, total_net), 6),
        },
        "thesis_driven": {
            "n": thesis_n,
            "gross_pnl": round(thesis_gross, 6),
            "fees": round(thesis_fees, 6),
            "net_pnl": round(thesis_net, 6),
            "share_of_total_loss": round(_safe_share(thesis_net, total_net), 6),
        },
        "other": {
            "n": other_n,
            "gross_pnl": round(other_gross, 6),
            "fees": round(other_fees, 6),
            "net_pnl": round(other_net, 6),
            "share_of_total_loss": round(_safe_share(other_net, total_net), 6),
        },
        "total_net_pnl": round(total_net, 6),
    }


def summarize_loss_drivers(
    episodes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ordered loss-driver summary across all episodes.

    Produces a list ordered by loss magnitude AND coverage:
      - Each driver: {driver, net_pnl, gross_pnl, fees, episode_count,
                       share_of_total_loss, median_loss, coverage_pct}

    Drivers are drawn from: symbol, exit_reason, regime, duration_bucket.
    """
    total_n = 0
    total_net = 0.0
    for ep in episodes:
        if ep.get("net_pnl") is not None:
            total_n += 1
            total_net += float(ep["net_pnl"])

    if total_n == 0:
        return []

    # Accumulate by driver
    drivers: Dict[str, Dict[str, Any]] = {}

    for ep in episodes:
        net_pnl = ep.get("net_pnl")
        if net_pnl is None:
            continue
        net = float(net_pnl)
        gross = float(ep.get("gross_pnl") or 0)
        fees = float(ep.get("fees") or 0)

        tags = [
            f"symbol:{ep.get('symbol') or 'UNKNOWN'}",
            f"exit_reason:{ep.get('exit_reason') or 'UNKNOWN'}",
            f"regime:{ep.get('regime_at_entry') or 'unknown'}",
            f"duration:{_duration_bucket(float(ep.get('duration_hours') or 0))}",
        ]

        if _is_friction_dominated(ep):
            tags.append("class:friction_dominated")

        for tag in tags:
            if tag not in drivers:
                drivers[tag] = {
                    "driver": tag,
                    "net_pnl": 0.0,
                    "gross_pnl": 0.0,
                    "fees": 0.0,
                    "episode_count": 0,
                    "losses": [],
                }
            d = drivers[tag]
            d["net_pnl"] += net
            d["gross_pnl"] += gross
            d["fees"] += fees
            d["episode_count"] += 1
            if net < 0:
                d["losses"].append(net)

    result: List[Dict[str, Any]] = []
    for tag, d in drivers.items():
        losses = d.pop("losses")
        d["share_of_total_loss"] = round(_safe_share(d["net_pnl"], total_net), 6)
        d["median_loss"] = round(statistics.median(losses), 6) if losses else 0.0
        d["coverage_pct"] = round(d["episode_count"] / total_n, 4)
        d["net_pnl"] = round(d["net_pnl"], 6)
        d["gross_pnl"] = round(d["gross_pnl"], 6)
        d["fees"] = round(d["fees"], 6)
        result.append(d)

    # Sort: most negative net_pnl first, then by coverage descending
    result.sort(key=lambda d: (d["net_pnl"], -d["coverage_pct"]))

    return result


# ── Full Surface Build ────────────────────────────────────────────────────

def build_full_surface(
    episodes: Sequence[Dict[str, Any]],
    band_width: float = DEFAULT_BAND_WIDTH,
    min_conviction: float = DEFAULT_MIN_CONVICTION,
) -> Dict[str, Any]:
    """Build the complete opportunity surface (P5A + P5C).

    Returns a dict with:
      - band_audit: per-band composition (P5A)
      - friction_burden: friction analysis (P5C)
      - exit_class_pnl: by exit reason (P5C)
      - duration_quality: by duration bucket (P5C)
      - symbol_drag: by symbol (P5C)
      - regime_mismatch: mismatch vs thesis (P5C)
      - loss_drivers: ordered summary (P5C)
      - meta: build metadata

    Conservation assertions are enforced during build.
    """
    ep_list = list(episodes)

    # Filter to episodes with valid net_pnl for pool-level conservation
    valid_eps = [e for e in ep_list if e.get("net_pnl") is not None]
    total_net = sum(float(e["net_pnl"]) for e in valid_eps)
    total_gross = sum(float(e.get("gross_pnl") or 0) for e in valid_eps)
    total_fees = sum(float(e.get("fees") or 0) for e in valid_eps)

    # Scored subset
    scored = [e for e in valid_eps
              if float(e.get("conviction_score") or 0) >= min_conviction
              and float(e.get("entry_notional") or 0) > 0]

    # P5A
    band_audit = build_composition_audit(ep_list, band_width, min_conviction)

    # Band-level conservation: sum of band net_pnl == scored pool net_pnl
    scored_net = sum(float(e["net_pnl"]) for e in scored)
    band_sum = sum(b["net_pnl"] for b in band_audit.values())
    assert abs(band_sum - scored_net) < 1e-4, (
        f"Band-pool conservation violation: band_sum={band_sum}, scored_net={scored_net}"
    )

    # Friction conservation across full pool
    friction_pnl = sum(float(e["net_pnl"]) for e in valid_eps if _is_friction_dominated(e))
    non_friction_pnl = sum(float(e["net_pnl"]) for e in valid_eps if not _is_friction_dominated(e))
    assert abs((friction_pnl + non_friction_pnl) - total_net) < 1e-6, (
        f"Pool friction conservation: friction={friction_pnl}, "
        f"non_friction={non_friction_pnl}, total={total_net}"
    )

    # P5C
    friction_burden = compute_friction_burden(ep_list)
    exit_class_pnl = compute_exit_class_pnl(ep_list)
    duration_quality = compute_duration_quality(ep_list)
    symbol_drag = compute_symbol_drag(ep_list)
    regime_mismatch = compute_regime_mismatch_cost(ep_list)
    loss_drivers = summarize_loss_drivers(ep_list)

    return {
        "meta": {
            "build_ts": time.time(),
            "build_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_episodes_total": len(ep_list),
            "n_episodes_valid": len(valid_eps),
            "n_episodes_scored": len(scored),
            "band_width": band_width,
            "min_conviction": min_conviction,
            "total_gross_pnl": round(total_gross, 6),
            "total_fees": round(total_fees, 6),
            "total_net_pnl": round(total_net, 6),
            "friction_dominated_pnl": round(friction_pnl, 6),
            "non_friction_pnl": round(non_friction_pnl, 6),
        },
        "band_audit": band_audit,
        "friction_burden": friction_burden,
        "exit_class_pnl": exit_class_pnl,
        "duration_quality": duration_quality,
        "symbol_drag": symbol_drag,
        "regime_mismatch": regime_mismatch,
        "loss_drivers": loss_drivers,
    }


# ── Persistence ───────────────────────────────────────────────────────────

def save_surface(surface: Dict[str, Any], path: Optional[str] = None) -> str:
    """Persist opportunity surface to JSON."""
    if path is None:
        path = STATE_PATH
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(surface, f, indent=2)
    LOG.info("[opportunity_surface] saved to %s", path)
    return str(out)


def load_surface(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load opportunity surface from JSON."""
    if path is None:
        path = STATE_PATH
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)
