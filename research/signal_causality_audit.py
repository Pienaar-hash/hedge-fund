"""Signal-outcome causality audit (Phase 1a of resume remediation).

Reads logs/state/episode_ledger.json, computes for each scored episode:
- Spearman rho (raw) + asymptotic p-value
- Quintile breakdown: mean score, mean return, hit rate, N, avg hold hours
- Q5-Q1 spread + bootstrap p-value
- Temporal stability across 3 chronological time slices
- Duration confound check

Reuses execution.hydra_monotonicity._spearman.

Outputs:
- JSON to logs/state/signal_causality_audit.json
- Markdown verdict block appended to
  docs/SIGNAL_OUTCOME_CAUSALITY_AUDIT_2026-03-18.md (only the new
  "Appendix A" section; existing template tables in section 5 are
  left unchanged).

Verdict gate (from /memories/session/plan.md):
- PASS:  pooled rho > 0.15 AND p < 0.05 AND Q5-Q1 > 0
- FAIL:  any criterion missed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Path so we can import execution.* when run as `python -m research.signal_causality_audit`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.hydra_monotonicity import _spearman  # type: ignore  # noqa: E402

LEDGER_PATH = os.path.join("logs", "state", "episode_ledger.json")
OUT_PATH = os.path.join("logs", "state", "signal_causality_audit.json")
DOC_PATH = os.path.join("docs", "SIGNAL_OUTCOME_CAUSALITY_AUDIT_2026-03-18.md")

# Verdict gate thresholds (locked, no auto-loosening — see plan.md)
RHO_GATE = 0.15
PVAL_GATE = 0.05
SPREAD_GATE = 0.0


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        v = float(val)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _realized_return(ep: Dict[str, Any]) -> Optional[float]:
    """Side-aware realized return (gross of fees). Excludes fee impact so the
    test is on signal quality, not execution overhead."""
    entry = _safe_float(ep.get("avg_entry_price"))
    exit_ = _safe_float(ep.get("avg_exit_price"))
    if entry <= 0 or exit_ <= 0:
        return None
    side = str(ep.get("side", "")).upper()
    if side == "LONG":
        return (exit_ - entry) / entry
    if side == "SHORT":
        return (entry - exit_) / entry
    return None


def _net_return_pct(ep: Dict[str, Any]) -> Optional[float]:
    """Net (after-fee) return as fraction of entry notional. Reported alongside
    raw return so we can see fee drag separately."""
    notional = _safe_float(ep.get("entry_notional"))
    if notional <= 0:
        return None
    net = _safe_float(ep.get("net_pnl"))
    return net / notional


def _parse_ts(ts: Any) -> Optional[float]:
    if isinstance(ts, (int, float)):
        return float(ts)
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _spearman_p_value(rho: float, n: int) -> Optional[float]:
    """Two-sided asymptotic p-value for Spearman rho under H0: rho == 0.

    For n >= 10, sqrt(n-1) * rho is approximately N(0,1) under H0.
    For n=377 this is extremely accurate.
    """
    if n < 10 or rho is None:
        return None
    z = abs(rho) * math.sqrt(n - 1)
    return 2.0 * (1.0 - _normal_cdf(z))


def _quintile_table(
    pairs: List[Tuple[float, float, float]],  # (score, return, hold_hrs)
    n_buckets: int = 5,
) -> List[Dict[str, Any]]:
    """Equal-count quintile breakdown. Last bucket absorbs any remainder."""
    if len(pairs) < n_buckets:
        return []
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    n = len(sorted_pairs)
    bucket_size = n // n_buckets
    buckets: List[Dict[str, Any]] = []
    for q in range(n_buckets):
        lo = q * bucket_size
        hi = (q + 1) * bucket_size if q < n_buckets - 1 else n
        chunk = sorted_pairs[lo:hi]
        if not chunk:
            continue
        scores = [c[0] for c in chunk]
        rets = [c[1] for c in chunk]
        holds = [c[2] for c in chunk if c[2] > 0]
        mean_ret = sum(rets) / len(rets)
        std_hold = (
            math.sqrt(sum((h - sum(holds) / len(holds)) ** 2 for h in holds) / len(holds))
            if len(holds) > 1
            else 0.0
        )
        buckets.append(
            {
                "bucket": f"Q{q+1}",
                "score_range": [round(min(scores), 4), round(max(scores), 4)],
                "mean_score": round(sum(scores) / len(scores), 4),
                "mean_return": round(mean_ret, 6),
                "hit_rate_pct": round(100.0 * sum(1 for r in rets if r > 0) / len(rets), 2),
                "n": len(rets),
                "avg_hold_hrs": round(sum(holds) / len(holds), 3) if holds else 0.0,
                "std_hold_hrs": round(std_hold, 3),
            }
        )
    return buckets


def _bootstrap_q5_q1_pvalue(
    pairs: List[Tuple[float, float, float]],
    n_iter: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap Q5-Q1 spread distribution; return (observed_spread, two_sided_p).

    p is the fraction of bootstrap samples in which spread <= 0 (one-sided),
    doubled to two-sided. Tests H0: signal does NOT predict ordering.
    """
    rng = random.Random(seed)
    n = len(pairs)
    if n < 25:
        return 0.0, 1.0
    observed = _quintile_table(pairs)
    if len(observed) < 5:
        return 0.0, 1.0
    obs_spread = observed[-1]["mean_return"] - observed[0]["mean_return"]
    le_zero = 0
    for _ in range(n_iter):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        bt = _quintile_table(sample)
        if len(bt) < 5:
            continue
        spread = bt[-1]["mean_return"] - bt[0]["mean_return"]
        if spread <= 0:
            le_zero += 1
    one_sided = le_zero / n_iter
    return obs_spread, min(1.0, 2.0 * one_sided)


# --------------------------------------------------------------------------- #
# Core audit
# --------------------------------------------------------------------------- #

def _extract_pairs(
    episodes: List[Dict[str, Any]],
    score_field: str = "hybrid_score",
    sign_flip: bool = False,
) -> List[Tuple[float, float, float, float, str]]:
    """Returns (score, raw_return, hold_hrs, entry_ts, symbol).

    If sign_flip=True, returns the return that WOULD have been realized had
    every signal traded in the opposite direction (i.e. the signal-inverted
    counterfactual). Algebraically equivalent to negating the return field.
    """
    out: List[Tuple[float, float, float, float, str]] = []
    flip_mul = -1.0 if sign_flip else 1.0
    for ep in episodes:
        score = _safe_float(ep.get(score_field))
        if score <= 0:
            continue
        ret = _realized_return(ep)
        if ret is None:
            continue
        hold = _safe_float(ep.get("duration_hours"))
        entry_ts = _parse_ts(ep.get("entry_ts")) or 0.0
        symbol = str(ep.get("symbol") or "UNKNOWN")
        out.append((score, ret * flip_mul, hold, entry_ts, symbol))
    return out


def _audit_block(
    pairs_full: List[Tuple[float, float, float, float, str]],
    label: str,
) -> Dict[str, Any]:
    """Compute the full audit on one cohort of pairs."""
    if not pairs_full:
        return {"label": label, "n": 0, "verdict": "INSUFFICIENT_DATA"}
    scores = [p[0] for p in pairs_full]
    rets = [p[1] for p in pairs_full]
    holds = [p[2] for p in pairs_full]
    n = len(scores)

    rho = _spearman(scores, rets)
    p_val = _spearman_p_value(rho, n) if rho is not None else None

    quintile_pairs = [(s, r, h) for s, r, h, _, _ in pairs_full]
    quintiles = _quintile_table(quintile_pairs)
    spread, boot_p = _bootstrap_q5_q1_pvalue(quintile_pairs)

    return {
        "label": label,
        "n": n,
        "rho": round(rho, 6) if rho is not None else None,
        "rho_p_value": round(p_val, 6) if p_val is not None else None,
        "q5_minus_q1": round(spread, 6),
        "q5_minus_q1_bootstrap_p": round(boot_p, 4),
        "quintiles": quintiles,
        "mean_return": round(sum(rets) / n, 6),
        "hit_rate_pct": round(100.0 * sum(1 for r in rets if r > 0) / n, 2),
        "avg_hold_hrs": round(sum(h for h in holds if h > 0) / max(1, sum(1 for h in holds if h > 0)), 3),
    }


def _temporal_slices(
    pairs_full: List[Tuple[float, float, float, float, str]],
    n_slices: int = 3,
) -> List[Dict[str, Any]]:
    sorted_pairs = sorted(pairs_full, key=lambda p: p[3])
    n = len(sorted_pairs)
    if n < n_slices * 10:
        return []
    slice_size = n // n_slices
    out: List[Dict[str, Any]] = []
    for i in range(n_slices):
        lo = i * slice_size
        hi = (i + 1) * slice_size if i < n_slices - 1 else n
        chunk = sorted_pairs[lo:hi]
        ts_lo = chunk[0][3]
        ts_hi = chunk[-1][3]
        block = _audit_block(chunk, f"T{i+1}")
        block["date_range"] = [
            datetime.fromtimestamp(ts_lo, tz=timezone.utc).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(ts_hi, tz=timezone.utc).strftime("%Y-%m-%d"),
        ]
        # classify slope using same thresholds as hydra_monotonicity
        rho = block.get("rho")
        if rho is None:
            slope = "insufficient_data"
        elif rho > 0.15:
            slope = "upward"
        elif rho < -0.05:
            slope = "inverted"
        else:
            slope = "flat"
        block["slope"] = slope
        out.append(block)
    return out


def run_audit(
    ledger_path: str = LEDGER_PATH,
    score_field: str = "hybrid_score",
    sign_flip: bool = False,
) -> Dict[str, Any]:
    with open(ledger_path) as f:
        data = json.load(f)
    episodes = data.get("episodes") if isinstance(data, dict) else data
    if not episodes:
        return {"ts": time.time(), "verdict": "NO_EPISODES", "n_total": 0}

    pairs = _extract_pairs(episodes, score_field=score_field, sign_flip=sign_flip)
    pooled = _audit_block(pairs, "POOLED")
    slices = _temporal_slices(pairs)

    # Per-symbol breakdown (only symbols with enough samples)
    by_symbol: Dict[str, List[Tuple[float, float, float, float, str]]] = {}
    for p in pairs:
        by_symbol.setdefault(p[4], []).append(p)
    per_symbol = [
        _audit_block(v, sym)
        for sym, v in sorted(by_symbol.items(), key=lambda kv: -len(kv[1]))
        if len(v) >= 25
    ]

    # Verdict
    rho = pooled.get("rho")
    p_val = pooled.get("rho_p_value")
    spread = pooled.get("q5_minus_q1")
    criteria = {
        "rho_gt_0_15": bool(rho is not None and rho > RHO_GATE),
        "p_lt_0_05": bool(p_val is not None and p_val < PVAL_GATE),
        "spread_gt_0": bool(spread is not None and spread > SPREAD_GATE),
    }
    passed = all(criteria.values())
    verdict = "PASS" if passed else "FAIL"

    return {
        "ts": time.time(),
        "ts_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ledger_path": ledger_path,
        "score_field": score_field,
        "sign_flip": sign_flip,
        "n_total_episodes": len(episodes),
        "n_scored_episodes": len(pairs),
        "pooled": pooled,
        "temporal_slices": slices,
        "per_symbol": per_symbol,
        "gate_thresholds": {
            "rho_min": RHO_GATE,
            "p_max": PVAL_GATE,
            "spread_min": SPREAD_GATE,
        },
        "criteria": criteria,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Markdown verdict appendix
# --------------------------------------------------------------------------- #

APPENDIX_MARKER = "\n<!-- BEGIN: Phase 1a Verdict Appendix (auto-generated) -->\n"
APPENDIX_END = "\n<!-- END: Phase 1a Verdict Appendix -->\n"
APPENDIX_B_MARKER = "\n<!-- BEGIN: Appendix B Sign-Flip (auto-generated) -->\n"
APPENDIX_B_END = "\n<!-- END: Appendix B Sign-Flip -->\n"


def _fmt_pval(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def render_appendix(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("## Appendix A — Phase 1a Verdict (auto-generated)\n")
    lines.append(f"_Generated {report['ts_iso']} — `research/signal_causality_audit.py`_\n")
    lines.append(
        f"**Source:** `{report['ledger_path']}` "
        f"({report['n_scored_episodes']} scored episodes "
        f"of {report['n_total_episodes']} total)\n"
    )
    lines.append(
        f"**Score field:** `{report['score_field']}`. "
        f"Return = raw side-aware return on `avg_entry_price`/`avg_exit_price` (gross of fees).\n"
    )
    lines.append("")
    lines.append("### A.1 Verdict\n")
    crit = report["criteria"]
    pooled = report["pooled"]
    lines.append("| Criterion | Threshold | Observed | Pass? |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Spearman $\\rho$ | > {RHO_GATE} | {pooled.get('rho')} | "
        f"{'✅' if crit['rho_gt_0_15'] else '❌'} |"
    )
    lines.append(
        f"| $p$-value | < {PVAL_GATE} | {_fmt_pval(pooled.get('rho_p_value'))} | "
        f"{'✅' if crit['p_lt_0_05'] else '❌'} |"
    )
    lines.append(
        f"| Q5 − Q1 spread | > {SPREAD_GATE} | {pooled.get('q5_minus_q1')} | "
        f"{'✅' if crit['spread_gt_0'] else '❌'} |"
    )
    lines.append("")
    lines.append(f"**Overall verdict: `{report['verdict']}`**\n")
    if report["verdict"] == "FAIL":
        lines.append(
            "> Signal does not satisfy the causality gate on the current "
            "executed-trade cohort. Resume blocked on Phase 1a alone. "
            "Per `/memories/session/plan.md`, thresholds are locked — no "
            "auto-loosening. Escalate to user.\n"
        )
    lines.append("")
    lines.append("### A.2 Quintile table (pooled)\n")
    lines.append(
        "| Bucket | Score range | $\\bar{S}$ | $\\bar{O}$ (mean return) | "
        "Hit rate (%) | $N$ | Avg hold (hrs) | Std hold (hrs) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for b in pooled.get("quintiles", []):
        sr = b["score_range"]
        lines.append(
            f"| {b['bucket']} | [{sr[0]}, {sr[1]}] | {b['mean_score']} | "
            f"{b['mean_return']} | {b['hit_rate_pct']} | {b['n']} | "
            f"{b['avg_hold_hrs']} | {b['std_hold_hrs']} |"
        )
    lines.append("")
    lines.append("### A.3 Summary statistics (pooled)\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| $N$ | {pooled.get('n')} |")
    lines.append(f"| Spearman $\\rho$ (raw) | {pooled.get('rho')} |")
    lines.append(f"| Spearman $\\rho$ $p$-value (asymptotic, two-sided) | {_fmt_pval(pooled.get('rho_p_value'))} |")
    lines.append(f"| Q5 − Q1 spread | {pooled.get('q5_minus_q1')} |")
    lines.append(f"| Q5 − Q1 bootstrap $p$ (2000 iter) | {_fmt_pval(pooled.get('q5_minus_q1_bootstrap_p'))} |")
    lines.append(f"| Mean return (cohort) | {pooled.get('mean_return')} |")
    lines.append(f"| Hit rate (%) | {pooled.get('hit_rate_pct')} |")
    lines.append(f"| Avg hold (hrs) | {pooled.get('avg_hold_hrs')} |")
    lines.append("")
    lines.append("### A.4 Temporal stability\n")
    lines.append("| Slice | Date range | $N$ | $\\rho$ | $p$ | Q5−Q1 | Slope |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in report.get("temporal_slices", []):
        dr = s.get("date_range", ["—", "—"])
        lines.append(
            f"| {s['label']} | {dr[0]} → {dr[1]} | {s['n']} | {s.get('rho')} | "
            f"{_fmt_pval(s.get('rho_p_value'))} | {s.get('q5_minus_q1')} | {s.get('slope')} |"
        )
    lines.append("")
    lines.append("### A.5 Duration confound check\n")
    lines.append("| Bucket | Avg hold (hrs) | Std hold (hrs) |")
    lines.append("|---|---|---|")
    qs = pooled.get("quintiles", [])
    if qs:
        for b in (qs[0], qs[-1]):
            lines.append(f"| {b['bucket']} | {b['avg_hold_hrs']} | {b['std_hold_hrs']} |")
    lines.append("")
    lines.append("### A.6 Per-symbol breakdown (≥25 episodes)\n")
    lines.append("| Symbol | $N$ | $\\rho$ | $p$ | Q5−Q1 | Hit rate (%) | Mean return |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in report.get("per_symbol", []):
        lines.append(
            f"| {s['label']} | {s['n']} | {s.get('rho')} | {_fmt_pval(s.get('rho_p_value'))} | "
            f"{s.get('q5_minus_q1')} | {s.get('hit_rate_pct')} | {s.get('mean_return')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_appendix(doc_path: str, appendix_md: str) -> None:
    with open(doc_path) as f:
        body = f.read()
    if APPENDIX_MARKER in body:
        # Replace existing block
        before = body.split(APPENDIX_MARKER)[0]
        after_split = body.split(APPENDIX_END)
        after = after_split[1] if len(after_split) > 1 else ""
        new_body = before + APPENDIX_MARKER + appendix_md + APPENDIX_END + after
    else:
        new_body = body.rstrip() + "\n" + APPENDIX_MARKER + appendix_md + APPENDIX_END
    with open(doc_path, "w") as f:
        f.write(new_body)


def render_appendix_b(
    baseline: Dict[str, Any],
    flipped: Dict[str, Any],
    fee_per_trade_pct: float = 0.0008,
) -> str:
    """Sign-flip counterfactual report.

    Computes what would have happened if every signal traded the opposite
    direction. Algebraically: rho_flipped = -rho_baseline,
    mean_return_flipped = -mean_return_baseline. p-value is symmetric.
    Hit rate must be recomputed (not simple complement because of zero-PnL
    trades).
    """
    lines: List[str] = []
    lines.append("## Appendix B — Sign-Flip Counterfactual (auto-generated)\n")
    lines.append(f"_Generated {flipped['ts_iso']} — `research/signal_causality_audit.py --sign-flip`_\n")
    lines.append(
        "**Hypothesis:** If the live signal is statistically anti-predictive "
        "(Appendix A: BTC ρ=-0.213 p=0.027, ETH ρ=-0.172 p=0.049), then "
        "trading the *opposite* side of every intent would have produced "
        "positive ρ on the same episode set. This appendix tests that "
        "counterfactual on the *same closed episodes* by negating the "
        "realized return per episode (algebraically equivalent to "
        "side-inversion).\n"
    )
    lines.append(
        f"**Fee assumption:** {fee_per_trade_pct*100:.3f}% round-trip cost per trade "
        "(approximated from sample fee/notional ratios in the ledger).\n"
    )
    lines.append("")
    lines.append("### B.1 Verdict — flipped vs baseline\n")
    bp, fp = baseline["pooled"], flipped["pooled"]
    lines.append("| Metric | Baseline (live signal) | Sign-flipped | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Pooled ρ | {bp.get('rho')} | {fp.get('rho')} | "
                 f"{round((fp.get('rho') or 0) - (bp.get('rho') or 0), 4)} |")
    lines.append(f"| Pooled $p$ | {_fmt_pval(bp.get('rho_p_value'))} | "
                 f"{_fmt_pval(fp.get('rho_p_value'))} | (symmetric) |")
    lines.append(f"| Mean return | {bp.get('mean_return')} | {fp.get('mean_return')} | "
                 f"{round((fp.get('mean_return') or 0) - (bp.get('mean_return') or 0), 6)} |")
    lines.append(f"| Hit rate (%) | {bp.get('hit_rate_pct')} | {fp.get('hit_rate_pct')} | "
                 f"{round((fp.get('hit_rate_pct') or 0) - (bp.get('hit_rate_pct') or 0), 2)} |")
    lines.append(f"| Q5 − Q1 | {bp.get('q5_minus_q1')} | {fp.get('q5_minus_q1')} | "
                 f"{round((fp.get('q5_minus_q1') or 0) - (bp.get('q5_minus_q1') or 0), 6)} |")
    lines.append(f"| Verdict | {baseline['verdict']} | {flipped['verdict']} | — |")
    lines.append("")
    lines.append("### B.2 Per-symbol gate-pass under sign-flip\n")
    lines.append("| Symbol | N | Baseline ρ (p) | Flipped ρ (p) | Flipped passes ρ>0.15 & p<0.05? |")
    lines.append("|---|---:|---:|---:|:-:|")
    baseline_per_sym = {s["label"]: s for s in baseline.get("per_symbol", [])}
    for fs in flipped.get("per_symbol", []):
        bs = baseline_per_sym.get(fs["label"], {})
        f_rho = fs.get("rho")
        f_p = fs.get("rho_p_value")
        passes = (
            f_rho is not None and f_rho > RHO_GATE
            and f_p is not None and f_p < PVAL_GATE
        )
        lines.append(
            f"| {fs['label']} | {fs['n']} | "
            f"{bs.get('rho')} ({_fmt_pval(bs.get('rho_p_value'))}) | "
            f"{f_rho} ({_fmt_pval(f_p)}) | "
            f"{'✅' if passes else '❌'} |"
        )
    lines.append("")
    lines.append("### B.3 Economic implication\n")
    mean_ret_flipped = fp.get("mean_return") or 0.0
    net_per_trade = mean_ret_flipped - fee_per_trade_pct
    n_eps = flipped.get("n_scored_episodes", 0)
    lines.append(
        f"Per-trade gross return (flipped): **{mean_ret_flipped*100:+.4f}%**. "
        f"Subtracting assumed round-trip fee ({fee_per_trade_pct*100:.3f}%) → "
        f"per-trade net: **{net_per_trade*100:+.4f}%**. "
        f"Across the {n_eps} scored episodes this corresponds to a "
        f"cumulative arithmetic edge of **{net_per_trade*100*n_eps:+.2f} bps "
        f"× avg-notional**.\n"
    )
    lines.append(
        "⚠️ Caveat: this is a counterfactual on *executed* trades only "
        "(left-truncated above the score threshold). It does not prove that "
        "naive sign-flipping would have been profitable in live execution — "
        "slippage, fee tiering, position sizing and risk gates would all "
        "behave differently. The result is *evidence of anti-prediction*, "
        "not a recommendation to invert the production signal.\n"
    )
    lines.append("")
    lines.append("### B.4 Conclusion\n")
    pooled_passes = (
        fp.get("rho") is not None and fp.get("rho") > RHO_GATE
        and fp.get("rho_p_value") is not None and fp.get("rho_p_value") < PVAL_GATE
    )
    if pooled_passes:
        lines.append(
            "The flipped pooled signal **passes** the resume-gate ρ and p "
            "criteria. The live signal contains a recoverable inverted edge. "
            "This is a concrete remediation candidate: investigate the side-"
            "decision logic in `execution/signal_screener.py` and the "
            "Hydra TREND head sign convention.\n"
        )
    else:
        lines.append(
            "Even under sign-flip the **pooled** ρ does not clear the 0.15 "
            "gate (observed: "
            f"{fp.get('rho')}, p={_fmt_pval(fp.get('rho_p_value'))}). "
            "Per-symbol BTC and ETH would clear the gate under inversion, "
            "but the universe-level signal does not. Interpretation: the "
            "anti-prediction is concentrated in the two highest-volume "
            "symbols; on the broader universe the signal is mostly noise. "
            "Naive inversion would not fix the model — the scoring system "
            "needs a redesign, not a sign change.\n"
        )
    return "\n".join(lines)


def write_appendix_b(doc_path: str, appendix_md: str) -> None:
    with open(doc_path) as f:
        body = f.read()
    if APPENDIX_B_MARKER in body:
        before = body.split(APPENDIX_B_MARKER)[0]
        after_split = body.split(APPENDIX_B_END)
        after = after_split[1] if len(after_split) > 1 else ""
        new_body = before + APPENDIX_B_MARKER + appendix_md + APPENDIX_B_END + after
    else:
        new_body = body.rstrip() + "\n" + APPENDIX_B_MARKER + appendix_md + APPENDIX_B_END
    with open(doc_path, "w") as f:
        f.write(new_body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Signal-outcome causality audit")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--doc", default=DOC_PATH)
    parser.add_argument("--score-field", default="hybrid_score")
    parser.add_argument("--no-doc", action="store_true", help="Skip writing markdown appendices")
    parser.add_argument(
        "--sign-flip",
        action="store_true",
        help="Also run the sign-flip counterfactual and append Appendix B",
    )
    args = parser.parse_args()

    report = run_audit(args.ledger, score_field=args.score_field)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[ok] wrote {args.out}")

    appendix = render_appendix(report)
    if not args.no_doc:
        write_appendix(args.doc, appendix)
        print(f"[ok] appended Appendix A to {args.doc}")

    print(f"\nVERDICT: {report['verdict']}")
    print(f"  rho={report['pooled'].get('rho')} "
          f"p={report['pooled'].get('rho_p_value')} "
          f"Q5-Q1={report['pooled'].get('q5_minus_q1')} "
          f"(n={report['pooled'].get('n')})")

    if args.sign_flip:
        flipped = run_audit(args.ledger, score_field=args.score_field, sign_flip=True)
        flipped_out = args.out.replace(".json", "_signflip.json")
        with open(flipped_out, "w") as f:
            json.dump(flipped, f, indent=2, default=str)
        print(f"\n[ok] wrote {flipped_out}")
        if not args.no_doc:
            md_b = render_appendix_b(report, flipped)
            write_appendix_b(args.doc, md_b)
            print(f"[ok] appended Appendix B to {args.doc}")
        print(f"\nSIGN-FLIP VERDICT: {flipped['verdict']}")
        print(f"  rho={flipped['pooled'].get('rho')} "
              f"p={flipped['pooled'].get('rho_p_value')} "
              f"mean_ret={flipped['pooled'].get('mean_return')} "
              f"hit_rate={flipped['pooled'].get('hit_rate_pct')}%")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
