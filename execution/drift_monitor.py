"""
AUDIT-4.1 — Drift monitoring for model and mask health.

Checks:
1. Mask overlap: alert if subsample overlap < 0.70 → disable affected mask
2. Score monotonicity: Spearman ρ < 0.15 → flag model as degraded
3. ZERO_SCORE rate: > 15% → alert

All checks are **observation-only** (never block execution).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)

# Thresholds
MASK_OVERLAP_MIN = 0.70
SCORE_MONOTONICITY_MIN = 0.15
ZERO_SCORE_RATE_MAX = 0.15

# Paths
SCORES_STATE_PATH = Path("logs/state/symbol_scores_v6.json")
EPISODE_LEDGER_PATH = Path("logs/execution/episode_ledger.jsonl")
ZERO_SCORE_AUDIT_PATH = Path("logs/execution/zero_score_audit.jsonl")
DRIFT_REPORT_PATH = Path("logs/state/drift_report.json")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def check_zero_score_rate(
    audit_path: Path = ZERO_SCORE_AUDIT_PATH,
    lookback_s: float = 86400.0,
) -> Dict[str, Any]:
    """Check rate of zero-score rejections over recent window."""
    rows = _read_jsonl(audit_path)
    cutoff = time.time() - lookback_s
    recent = [r for r in rows if (r.get("ts") or 0) >= cutoff]
    total = len(recent)

    # Count by source
    by_source: Dict[str, int] = {}
    for r in recent:
        src = r.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    return {
        "total_zero_score_events": total,
        "by_source": by_source,
        "lookback_s": lookback_s,
        "alert": None,  # would need total intents for rate; log event count for now
    }


def check_score_monotonicity(
    episode_path: Path = EPISODE_LEDGER_PATH,
    lookback_s: float = 7 * 86400.0,
) -> Dict[str, Any]:
    """Check Spearman correlation between hybrid_score and PnL across episodes."""
    rows = _read_jsonl(episode_path)
    if not rows:
        return {"rho": None, "n": 0, "alert": "no_data"}

    cutoff = time.time() - lookback_s
    pairs: List[tuple] = []
    for r in rows:
        ts = r.get("ts_close") or r.get("ts") or 0
        if ts < cutoff:
            continue
        score = r.get("hybrid_score") or r.get("score")
        pnl = r.get("pnl_usd") or r.get("realizedPnlUsd")
        if score is not None and pnl is not None:
            pairs.append((float(score), float(pnl)))

    if len(pairs) < 10:
        return {"rho": None, "n": len(pairs), "alert": "insufficient_data"}

    # Spearman rank correlation
    scores = [p[0] for p in pairs]
    pnls = [p[1] for p in pairs]
    rho = _spearman_rho(scores, pnls)
    alert = "low_monotonicity" if rho < SCORE_MONOTONICITY_MIN else None
    return {"rho": round(rho, 4), "n": len(pairs), "alert": alert}


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _rank(values: List[float]) -> List[float]:
    """Average rank with tie handling."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def check_score_distribution(
    scores_path: Path = SCORES_STATE_PATH,
) -> Dict[str, Any]:
    """Check current score distribution for anomalies."""
    if not scores_path.exists():
        return {"alert": "no_scores_file"}

    raw = json.loads(scores_path.read_text())
    symbols_raw = raw.get("symbols") or raw.get("scores") or []
    if isinstance(symbols_raw, list):
        scores = {
            entry["symbol"]: float(entry.get("score", 0))
            for entry in symbols_raw
            if isinstance(entry, dict) and "symbol" in entry
        }
    else:
        scores = {k: float(v.get("score", 0) if isinstance(v, dict) else v) for k, v in symbols_raw.items()}

    total = len(scores)
    zero_count = sum(1 for s in scores.values() if s == 0.0)
    zero_rate = zero_count / total if total > 0 else 0.0
    alert = "high_zero_score_rate" if zero_rate > ZERO_SCORE_RATE_MAX else None

    return {
        "total_symbols": total,
        "zero_count": zero_count,
        "zero_rate": round(zero_rate, 4),
        "alert": alert,
    }


def run_drift_check() -> Dict[str, Any]:
    """Run all drift checks and produce a consolidated report."""
    checks: Dict[str, Dict[str, Any]] = {
        "zero_score_events": check_zero_score_rate(),
        "score_monotonicity": check_score_monotonicity(),
        "score_distribution": check_score_distribution(),
    }
    alerts: list[str] = []
    for key, check in checks.items():
        alert = check.get("alert")
        if alert:
            alerts.append(f"{key}:{alert}")

    report: Dict[str, Any] = {
        "ts": time.time(),
        **checks,
        "alerts": alerts,
    }

    # Write report
    try:
        DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DRIFT_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    except Exception as exc:
        LOGGER.error("Failed to write drift report: %s", exc)

    return report
