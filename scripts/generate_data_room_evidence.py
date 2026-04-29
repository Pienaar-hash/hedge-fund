#!/usr/bin/env python3
"""
Generate Data Room Evidence — 06_SUPPORTING

Produces 7 evidence layers as CSVs from existing system logs.
All compute functions come from execution/hydra_monotonicity.py — zero
re-implementation.

Usage:
    PYTHONPATH=. python scripts/generate_data_room_evidence.py

Output:
    data_room/06_SUPPORTING/
      01_EPISODE_EVIDENCE/    — authority-traced trade lifecycle
      02_SIGNAL_CAUSALITY/    — score → return monotonicity
      03_FRICTION/            — edge survival after costs
      04_CALIBRATION/         — probability honesty
      05_RAW_DATA/            — simple exports
      06_DENIALS/             — refused trades (falsification surface)
      07_PASSIVE_OBSERVATIONS — scored but not traded (selection bias kill)
      README.md               — investor-facing explanation
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EPISODE_LEDGER = Path("logs/state/episode_ledger.json")
DOCTRINE_EVENTS = Path("logs/doctrine_events.jsonl")
RISK_VETOES = Path("logs/execution/risk_vetoes.jsonl")
PASSIVE_OBS = Path("logs/execution/passive_observations.jsonl")

OUT_ROOT = Path("data_room/06_SUPPORTING")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> Dict[str, Path]:
    dirs = {
        "episode":     OUT_ROOT / "01_EPISODE_EVIDENCE",
        "causality":   OUT_ROOT / "02_SIGNAL_CAUSALITY",
        "friction":    OUT_ROOT / "03_FRICTION",
        "calibration": OUT_ROOT / "04_CALIBRATION",
        "raw":         OUT_ROOT / "05_RAW_DATA",
        "denials":     OUT_ROOT / "06_DENIALS",
        "passive":     OUT_ROOT / "07_PASSIVE_OBSERVATIONS",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> int:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    LOG.info(f"  wrote {len(rows)} rows → {path}")
    return len(rows)


def _load_jsonl(path: Path, max_lines: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        LOG.warning(f"  {path} not found — skipping")
        return rows
    with open(path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Phase A — Episode Evidence
# ---------------------------------------------------------------------------

EPISODE_FIELDS = [
    "episode_id", "symbol", "side",
    "entry_ts", "exit_ts", "duration_hours",
    "entry_fills", "exit_fills",
    "entry_notional", "exit_notional", "total_qty",
    "avg_entry_price", "avg_exit_price",
    "gross_pnl", "fees", "net_pnl",
    "regime_at_entry", "regime_at_exit",
    "exit_reason", "exit_reason_raw",
    "strategy",
    "intent_id", "attempt_id",
    "confidence", "hybrid_score",
    "conviction_score", "conviction_band",
    "entry_regime_confidence", "expected_edge",
    "engine_source",
]


def load_episodes() -> List[Dict[str, Any]]:
    if not EPISODE_LEDGER.exists():
        LOG.error(f"Episode ledger not found: {EPISODE_LEDGER}")
        sys.exit(1)
    data = json.loads(EPISODE_LEDGER.read_text())
    episodes = data.get("episodes", [])
    LOG.info(f"Loaded {len(episodes)} episodes from {EPISODE_LEDGER}")
    return episodes


def phase_a(episodes: List[Dict[str, Any]], dirs: Dict[str, Path]) -> None:
    LOG.info("Phase A — Episode Evidence")
    _write_csv(
        dirs["episode"] / "futures_episode_sample.csv",
        episodes,
        EPISODE_FIELDS,
    )


# ---------------------------------------------------------------------------
# Phase B — Signal Causality
# ---------------------------------------------------------------------------

def phase_b(episodes: List[Dict[str, Any]], dirs: Dict[str, Path]) -> None:
    LOG.info("Phase B — Signal Causality")
    from execution.hydra_monotonicity import (
        compute_monotonicity,
        compute_quintile_spread,
        compute_direction_accuracy,
    )

    # B1: Monotonicity + quintile spread
    mono = compute_monotonicity(episodes)
    qspread = compute_quintile_spread(episodes)

    mono_rows = []
    for b in mono.get("buckets", []):
        mono_rows.append({
            "bucket": b.get("range", ""),
            "score_lo": b.get("lo", ""),
            "score_hi": b.get("hi", ""),
            "mean_score": b.get("mean_score", ""),
            "mean_return": b.get("mean_return", ""),
            "hit_rate": b.get("hit_rate", ""),
            "n": b.get("n", ""),
        })
    # Append summary row
    mono_rows.append({
        "bucket": "SUMMARY",
        "score_lo": "",
        "score_hi": "",
        "mean_score": "",
        "mean_return": "",
        "hit_rate": "",
        "n": mono.get("n", ""),
    })
    # Append Spearman + Q5-Q1 as metadata rows
    mono_rows.append({
        "bucket": f"spearman={mono.get('spearman', '')}",
        "score_lo": f"p_value={mono.get('p_value', '')}",
        "score_hi": f"slope={mono.get('slope', '')}",
        "mean_score": f"q5_q1_spread={qspread.get('q5_q1_spread', '')}",
        "mean_return": "",
        "hit_rate": "",
        "n": "",
    })

    _write_csv(
        dirs["causality"] / "futures_monotonicity.csv",
        mono_rows,
        ["bucket", "score_lo", "score_hi", "mean_score", "mean_return", "hit_rate", "n"],
    )

    # B2: Direction accuracy
    da = compute_direction_accuracy(episodes)
    da_rows = []
    for b in da.get("buckets", []):
        da_rows.append({
            "bucket": b.get("label", ""),
            "mean_score": b.get("mean_score", ""),
            "accuracy": b.get("accuracy", ""),
            "lift_vs_random": b.get("lift_vs_random", ""),
            "n": b.get("n", ""),
        })
    da_rows.append({
        "bucket": "OVERALL",
        "mean_score": "",
        "accuracy": da.get("overall_accuracy", ""),
        "lift_vs_random": da.get("overall_lift", ""),
        "n": da.get("n", ""),
    })
    da_rows.append({
        "bucket": f"direction_spearman={da.get('direction_spearman', '')}",
        "mean_score": "",
        "accuracy": "",
        "lift_vs_random": "",
        "n": "",
    })

    _write_csv(
        dirs["causality"] / "futures_direction_accuracy.csv",
        da_rows,
        ["bucket", "mean_score", "accuracy", "lift_vs_random", "n"],
    )


# ---------------------------------------------------------------------------
# Phase C — Friction
# ---------------------------------------------------------------------------

def phase_c(episodes: List[Dict[str, Any]], dirs: Dict[str, Path]) -> None:
    LOG.info("Phase C — Friction")
    from execution.hydra_monotonicity import (
        compute_friction_decomposition,
        compute_friction_overlay,
        compute_break_even_edge,
    )

    # C1: Per-trade friction decomposition
    fd = compute_friction_decomposition(episodes)
    # Build per-trade rows from the raw return we can recompute
    fd_rows: List[Dict[str, Any]] = []
    for ep in episodes:
        score = float(ep.get("hybrid_score") or 0)
        if score <= 0:
            continue
        entry_px = float(ep.get("avg_entry_price") or 0)
        exit_px = float(ep.get("avg_exit_price") or 0)
        notional = float(ep.get("entry_notional") or 0)
        fees = float(ep.get("fees") or 0)
        if entry_px <= 0 or notional <= 0:
            continue
        side = str(ep.get("side", "")).upper()
        if side == "LONG":
            raw_ret = (exit_px - entry_px) / entry_px
        elif side == "SHORT":
            raw_ret = (entry_px - exit_px) / entry_px
        else:
            continue
        raw_bps = raw_ret * 10_000
        fee_bps = (fees / notional) * 10_000 if notional > 0 else 0
        net_bps = raw_bps - fee_bps
        killed = raw_bps > 0 and net_bps <= 0
        fd_rows.append({
            "episode_id": ep.get("episode_id", ""),
            "symbol": ep.get("symbol", ""),
            "side": side,
            "hybrid_score": round(score, 5),
            "raw_edge_bps": round(raw_bps, 2),
            "fee_drag_bps": round(fee_bps, 2),
            "net_edge_bps": round(net_bps, 2),
            "notional_usd": round(notional, 2),
            "fees_usd": round(fees, 4),
            "friction_killed": killed,
            "duration_hours": ep.get("duration_hours", ""),
        })

    _write_csv(
        dirs["friction"] / "futures_friction_decomposition.csv",
        fd_rows,
        ["episode_id", "symbol", "side", "hybrid_score",
         "raw_edge_bps", "fee_drag_bps", "net_edge_bps",
         "notional_usd", "fees_usd", "friction_killed", "duration_hours"],
    )

    # C2: Friction overlay (bucket-level raw vs net)
    fo = compute_friction_overlay(episodes)
    fo_rows = []
    for b in fo.get("buckets", []):
        fo_rows.append({
            "bucket": b.get("label", ""),
            "mean_score": b.get("mean_score", ""),
            "mean_raw_return": b.get("mean_raw_return", ""),
            "mean_net_return": b.get("mean_net_return", ""),
            "friction_drag": b.get("friction_drag", ""),
            "edge_erased": b.get("edge_erased", ""),
            "n": b.get("n", ""),
        })
    fo_rows.append({
        "bucket": "SUMMARY",
        "mean_score": "",
        "mean_raw_return": f"spearman_raw={fo.get('spearman_raw', '')}",
        "mean_net_return": f"spearman_net={fo.get('spearman_net', '')}",
        "friction_drag": f"buckets_erased={fo.get('buckets_erased', '')}",
        "edge_erased": "",
        "n": fo.get("n", ""),
    })

    _write_csv(
        dirs["friction"] / "futures_friction_overlay.csv",
        fo_rows,
        ["bucket", "mean_score", "mean_raw_return", "mean_net_return",
         "friction_drag", "edge_erased", "n"],
    )

    # C3: Break-even edge summary
    be = compute_break_even_edge(episodes)
    summary = fd.get("trades_summary", {})
    be_rows = [{
        "metric": "break_even_bps",
        "value": be.get("break_even_bps", ""),
    }, {
        "metric": "pct_above_hurdle",
        "value": be.get("pct_above_hurdle", ""),
    }, {
        "metric": "implied_min_score",
        "value": be.get("implied_min_score", ""),
    }, {
        "metric": "fee_to_edge_ratio",
        "value": summary.get("fee_to_edge_ratio", ""),
    }, {
        "metric": "friction_kill_rate",
        "value": summary.get("friction_kill_rate", ""),
    }, {
        "metric": "total_gross_edge_bps",
        "value": summary.get("total_gross_edge_bps", ""),
    }, {
        "metric": "total_fee_drag_bps",
        "value": summary.get("total_fee_drag_bps", ""),
    }, {
        "metric": "total_net_edge_bps",
        "value": summary.get("total_net_edge_bps", ""),
    }, {
        "metric": "n_trades",
        "value": fd.get("n", ""),
    }]

    _write_csv(
        dirs["friction"] / "futures_friction_summary.csv",
        be_rows,
        ["metric", "value"],
    )


# ---------------------------------------------------------------------------
# Phase D — Calibration
# ---------------------------------------------------------------------------

def phase_d(episodes: List[Dict[str, Any]], dirs: Dict[str, Path]) -> None:
    LOG.info("Phase D — Calibration")
    from execution.hydra_monotonicity import (
        compute_calibration_curve,
        compute_brier_score,
        compute_calibration_diagnosis,
    )

    # D1: Calibration curve
    curve = compute_calibration_curve(episodes)
    curve_rows = []
    for b in curve.get("buckets", []):
        curve_rows.append({
            "bucket": b.get("label", ""),
            "pred_lo": b.get("pred_lo", ""),
            "pred_hi": b.get("pred_hi", ""),
            "mean_predicted": b.get("mean_predicted", ""),
            "realized_frequency": b.get("realized_frequency", ""),
            "gap": b.get("gap", ""),
            "n": b.get("n", ""),
        })

    _write_csv(
        dirs["calibration"] / "futures_calibration.csv",
        curve_rows,
        ["bucket", "pred_lo", "pred_hi", "mean_predicted",
         "realized_frequency", "gap", "n"],
    )

    # D2: Brier + ECE summary
    brier = compute_brier_score(episodes)
    diag = compute_calibration_diagnosis(episodes)
    decomp = brier.get("decomposition", {})

    summary_rows = [{
        "metric": "brier_score",
        "value": brier.get("brier", ""),
    }, {
        "metric": "brier_baseline",
        "value": brier.get("brier_baseline", ""),
    }, {
        "metric": "brier_skill_score",
        "value": brier.get("brier_skill_score", ""),
    }, {
        "metric": "base_rate",
        "value": brier.get("base_rate", ""),
    }, {
        "metric": "reliability",
        "value": decomp.get("reliability", ""),
    }, {
        "metric": "resolution",
        "value": decomp.get("resolution", ""),
    }, {
        "metric": "uncertainty",
        "value": decomp.get("uncertainty", ""),
    }, {
        "metric": "ECE",
        "value": diag.get("ece", ""),
    }, {
        "metric": "MCE",
        "value": diag.get("mce", ""),
    }, {
        "metric": "pred_spread",
        "value": diag.get("pred_spread", ""),
    }, {
        "metric": "pattern_verdict",
        "value": "|".join(diag.get("patterns", [])),
    }, {
        "metric": "n",
        "value": brier.get("n", ""),
    }]

    _write_csv(
        dirs["calibration"] / "futures_calibration_summary.csv",
        summary_rows,
        ["metric", "value"],
    )


# ---------------------------------------------------------------------------
# Phase E — Denials
# ---------------------------------------------------------------------------

DENIAL_FIELDS = [
    "ts", "symbol", "verdict", "reason",
    "regime", "confidence", "direction",
    "source_head", "multiplier",
]


def phase_e(dirs: Dict[str, Path]) -> None:
    LOG.info("Phase E — Denials")

    # E1: Doctrine vetoes (entry vetoes only)
    doctrine_rows: List[Dict[str, Any]] = []
    for ev in _load_jsonl(DOCTRINE_EVENTS):
        if ev.get("allowed", True):
            continue  # only denials
        doctrine_rows.append({
            "ts": ev.get("ts", ""),
            "symbol": ev.get("symbol", ""),
            "verdict": ev.get("verdict", ""),
            "reason": ev.get("reason", ""),
            "regime": ev.get("regime", ""),
            "confidence": ev.get("confidence", ""),
            "direction": ev.get("direction", ""),
            "source_head": ev.get("source_head", ""),
            "multiplier": ev.get("multiplier", ""),
        })

    LOG.info(f"  doctrine denials: {len(doctrine_rows)}")

    # E2: Risk vetoes (slim extract)
    risk_rows: List[Dict[str, Any]] = []
    for ev in _load_jsonl(RISK_VETOES):
        risk_rows.append({
            "ts": ev.get("ts", ""),
            "symbol": ev.get("symbol", ""),
            "verdict": "RISK_VETO",
            "reason": ev.get("veto_reason", ""),
            "regime": "",
            "confidence": "",
            "direction": ev.get("side", ""),
            "source_head": ev.get("source_head", ""),
            "multiplier": "",
        })

    LOG.info(f"  risk vetoes: {len(risk_rows)}")

    # Merge and sort by timestamp
    all_denials = doctrine_rows + risk_rows
    all_denials.sort(key=lambda r: r.get("ts", ""))

    _write_csv(
        dirs["denials"] / "futures_denials_sample.csv",
        all_denials,
        DENIAL_FIELDS,
    )


# ---------------------------------------------------------------------------
# Phase F — Passive Observations
# ---------------------------------------------------------------------------

PASSIVE_FIELDS = [
    "ts", "symbol", "direction", "price",
    "hybrid_score", "passes_threshold",
    "regime", "rq_score", "rv_score",
    "trend", "carry", "expectancy", "router",
]


def phase_f(dirs: Dict[str, Path]) -> None:
    LOG.info("Phase F — Passive Observations")
    rows: List[Dict[str, Any]] = []
    for ev in _load_jsonl(PASSIVE_OBS):
        components = ev.get("components", {})
        rows.append({
            "ts": ev.get("ts", ""),
            "symbol": ev.get("symbol", ""),
            "direction": ev.get("direction", ""),
            "price": ev.get("price", ""),
            "hybrid_score": ev.get("hybrid_score", ""),
            "passes_threshold": ev.get("passes_threshold", ""),
            "regime": ev.get("regime", ""),
            "rq_score": ev.get("rq_score", ""),
            "rv_score": ev.get("rv_score", ""),
            "trend": components.get("trend", ""),
            "carry": components.get("carry", ""),
            "expectancy": components.get("expectancy", ""),
            "router": components.get("router", ""),
        })

    _write_csv(
        dirs["passive"] / "futures_passive_sample.csv",
        rows,
        PASSIVE_FIELDS,
    )


# ---------------------------------------------------------------------------
# Phase G — Raw Data + README
# ---------------------------------------------------------------------------

RAW_FIELDS = [
    "episode_id", "symbol", "side",
    "entry_ts", "exit_ts", "duration_hours",
    "avg_entry_price", "avg_exit_price",
    "gross_pnl", "fees", "net_pnl",
    "exit_reason",
]


def phase_g(episodes: List[Dict[str, Any]], dirs: Dict[str, Path]) -> None:
    LOG.info("Phase G — Raw Data + README")

    _write_csv(
        dirs["raw"] / "futures_trades_sample.csv",
        episodes,
        RAW_FIELDS,
    )

    readme = OUT_ROOT / "README.md"
    readme.write_text(_README_TEXT)
    LOG.info(f"  wrote {readme}")


_README_TEXT = """\
# 06_SUPPORTING — Data Room Evidence

## What this proves

| Layer | Question answered | Artifact |
|-------|-------------------|----------|
| **01 Episode Evidence** | What happened? (authority-traced) | `futures_episode_sample.csv` |
| **02 Signal Causality** | Why did it work? (score → return) | `futures_monotonicity.csv`, `futures_direction_accuracy.csv` |
| **03 Friction** | Does edge survive costs? | `futures_friction_decomposition.csv`, `futures_friction_overlay.csv`, `futures_friction_summary.csv` |
| **04 Calibration** | Are probabilities honest? | `futures_calibration.csv`, `futures_calibration_summary.csv` |
| **05 Raw Data** | Simple trade export | `futures_trades_sample.csv` |
| **06 Denials** | What was refused? (falsification) | `futures_denials_sample.csv` |
| **07 Passive Observations** | What was scored but not traded? (selection bias kill) | `futures_passive_sample.csv` |

## How to reproduce

```bash
cd /root/hedge-fund
PYTHONPATH=. python scripts/generate_data_room_evidence.py
```

## Key fields

### Episode CSV (01)
Every trade traces back to a decision via `intent_id`. Scoring fields
(`hybrid_score`, `conviction_score`, `conviction_band`) make every episode
attributable to the signal that triggered it. `regime_at_entry` and
`regime_at_exit` show the market regime context.

### Monotonicity CSV (02)
Trades bucketed by `hybrid_score` quintile. A healthy model shows
Spearman ρ > 0.2 (higher score → higher return). Q5−Q1 spread is the
top-vs-bottom quintile return gap.

### Friction CSVs (03)
Per-trade: `raw_edge_bps` (before fees), `fee_drag_bps`, `net_edge_bps`
(after fees). `friction_killed = True` means a trade had positive raw
edge but negative net edge — fees consumed the profit.

Summary: `fee_to_edge_ratio` shows what fraction of gross edge goes to
fees. `break_even_bps` is the minimum raw edge required to survive costs.

### Calibration CSVs (04)
Reliability diagram: `mean_predicted` vs `realized_frequency` per bucket.
`gap > 0` = overconfident. ECE (Expected Calibration Error) is the weighted
mean |gap|. Brier Skill Score > 0 means the model beats the naïve baseline.

### Denials CSV (06)
Every trade the system **refused** — doctrine vetoes (regime mismatch,
direction conflict) and risk vetoes (symbol cap, drawdown). This proves the
system knows when NOT to trade.

### Passive Observations CSV (07)
Every symbol scored by the signal pipeline, including those that did NOT
result in a trade. Shows the full scoring universe, not just cherry-picked
executions. Enables independent threshold analysis.

## Data provenance

- Episodes: `logs/state/episode_ledger.json`
- Doctrine events: `logs/doctrine_events.jsonl`
- Risk vetoes: `logs/execution/risk_vetoes.jsonl`
- Passive observations: `logs/execution/passive_observations.jsonl`
- Compute functions: `execution/hydra_monotonicity.py`

Generated by `scripts/generate_data_room_evidence.py`
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    LOG.info("=" * 60)
    LOG.info("Generating Data Room Evidence — 06_SUPPORTING")
    LOG.info("=" * 60)

    dirs = _ensure_dirs()
    episodes = load_episodes()

    phase_a(episodes, dirs)
    phase_b(episodes, dirs)
    phase_c(episodes, dirs)
    phase_d(episodes, dirs)
    phase_e(dirs)
    phase_f(dirs)
    phase_g(episodes, dirs)

    # Summary
    csv_count = sum(1 for p in OUT_ROOT.rglob("*.csv"))
    LOG.info("=" * 60)
    LOG.info(f"Done — {csv_count} CSVs + README written to {OUT_ROOT}")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()
