#!/usr/bin/env python3
"""
Post-audit validation — 4 steps.

Step 1: Score→PnL relationship (monotonicity, quintile spread)
Step 2: ZERO_SCORE elimination effectiveness
Step 3: Determinism certification (replay hash comparison)
Step 4: Drift baseline snapshot (clean-state anchor)

Data sources:
  - logs/execution/p6_replay_signals.jsonl   (conviction + PnL, 1700 episodes)
  - logs/execution/selector_v2_shadow.jsonl  (hydra_score per cycle)
  - logs/execution/orders_executed.jsonl     (order_close → realizedPnlUsd)
  - logs/execution/zero_score_audit.jsonl    (zero-score rejections)
  - logs/state/symbol_scores_v6.json         (current score distribution)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPLAY_PATH = Path("logs/execution/p6_replay_signals.jsonl")
SHADOW_PATH = Path("logs/execution/selector_v2_shadow.jsonl")
ORDERS_PATH = Path("logs/execution/orders_executed.jsonl")
ZERO_SCORE_PATH = Path("logs/execution/zero_score_audit.jsonl")
SCORES_PATH = Path("logs/state/symbol_scores_v6.json")
DOCTRINE_PATH = Path("logs/doctrine_events.jsonl")
OUTPUT_DIR = Path("data/post_audit_validation")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ─────────────────────────────────────────────────────────────────────
# Step 1: Score → PnL validation
# ─────────────────────────────────────────────────────────────────────

def _rank(values: list[float]) -> list[float]:
    """Average rank with tie handling."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and values[indexed[j]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def quintile_spread(scores: list[float], pnls: list[float]) -> dict:
    """Compute PnL by score quintile."""
    n = len(scores)
    if n < 5:
        return {"error": "insufficient data"}
    pairs = sorted(zip(scores, pnls), key=lambda p: p[0])
    q_size = n // 5
    result = {}
    for q_idx in range(5):
        start = q_idx * q_size
        end = start + q_size if q_idx < 4 else n
        q_pnls = [p[1] for p in pairs[start:end]]
        q_scores = [p[0] for p in pairs[start:end]]
        result[f"Q{q_idx + 1}"] = {
            "n": len(q_pnls),
            "score_range": [round(min(q_scores), 4), round(max(q_scores), 4)],
            "mean_pnl": round(statistics.mean(q_pnls), 4),
            "total_pnl": round(sum(q_pnls), 2),
        }
    q5_mean = result["Q5"]["mean_pnl"]
    q1_mean = result["Q1"]["mean_pnl"]
    result["Q5_minus_Q1"] = round(q5_mean - q1_mean, 4)
    return result


def monotonicity_ratio(scores: list[float], pnls: list[float], n_bins: int = 10) -> float:
    """Fraction of adjacent bins where higher score → higher mean PnL."""
    n = len(scores)
    if n < n_bins:
        return float("nan")
    pairs = sorted(zip(scores, pnls), key=lambda p: p[0])
    bin_size = n // n_bins
    bin_means = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        bin_means.append(statistics.mean(p[1] for p in pairs[start:end]))
    increasing = sum(1 for i in range(1, len(bin_means)) if bin_means[i] >= bin_means[i - 1])
    return round(increasing / (len(bin_means) - 1), 4)


def step1_score_pnl_validation() -> dict:
    """Validate score→PnL relationship using available data."""
    print("\n" + "=" * 70)
    print("STEP 1: Score → PnL Relationship Validation")
    print("=" * 70)

    results: dict[str, Any] = {"data_sources": {}}

    # ── Source A: P6 Replay (conviction × PnL) ──
    replay = _read_jsonl(REPLAY_PATH)
    signaled = [r for r in replay if r.get("signal_generated") and r.get("conviction") is not None]
    results["data_sources"]["p6_replay"] = {
        "total_episodes": len(replay),
        "signaled_with_conviction": len(signaled),
    }

    results["by_symbol"] = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        sym_episodes = [r for r in signaled if r.get("symbol") == sym or r.get("episode_symbol") == sym]
        if len(sym_episodes) < 10:
            results["by_symbol"][sym] = {"error": f"insufficient data ({len(sym_episodes)} episodes)"}
            continue

        convictions = [float(r["conviction"]) for r in sym_episodes]
        pnls = [float(r["realized_net_pnl"]) for r in sym_episodes]

        rho = spearman_rho(convictions, pnls)
        qspread = quintile_spread(convictions, pnls)
        mono = monotonicity_ratio(convictions, pnls)

        results["by_symbol"][sym] = {
            "n": len(sym_episodes),
            "score_field": "conviction",
            "spearman_rho": round(rho, 4) if not math.isnan(rho) else None,
            "monotonicity_ratio": mono if not math.isnan(mono) else None,
            "quintile_spread": qspread,
            "total_pnl": round(sum(pnls), 2),
            "mean_pnl": round(statistics.mean(pnls), 4),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4),
        }

        # Verdicts
        pass_rho = rho > 0.15 if not math.isnan(rho) else False
        pass_q5q1 = qspread.get("Q5_minus_Q1", 0) > 0
        pass_mono = mono >= 0.75 if not math.isnan(mono) else False
        results["by_symbol"][sym]["verdicts"] = {
            "rho_gt_0.15": "PASS" if pass_rho else "FAIL",
            "Q5_Q1_gt_0": "PASS" if pass_q5q1 else "FAIL",
            "monotonicity_gte_0.75": "PASS" if pass_mono else "FAIL",
        }

        print(f"\n  {sym} ({len(sym_episodes)} episodes):")
        print(f"    Spearman ρ:        {rho:+.4f}  {'✓' if pass_rho else '✗'} (need > 0.15)")
        print(f"    Q5-Q1 spread:      {qspread.get('Q5_minus_Q1', 0):+.4f}  {'✓' if pass_q5q1 else '✗'} (need > 0)")
        print(f"    Monotonicity:      {mono:.4f}  {'✓' if pass_mono else '✗'} (need ≥ 0.75)")
        print(f"    Win rate:          {results['by_symbol'][sym]['win_rate']:.2%}")
        print(f"    Total PnL:         {sum(pnls):+.2f} USDT")
        for q_name, q_data in qspread.items():
            if q_name.startswith("Q"):
                q_info = q_data
                if isinstance(q_info, dict):
                    print(f"    {q_name}: n={q_info['n']}, score=[{q_info['score_range'][0]:.3f}-{q_info['score_range'][1]:.3f}], mean_pnl={q_info['mean_pnl']:+.4f}")

    # ── Source B: Shadow selector (hydra_score — no direct PnL, distribution only) ──
    shadow = _read_jsonl(SHADOW_PATH)
    if shadow:
        by_sym: dict[str, list[float]] = defaultdict(list)
        for r in shadow:
            hs = r.get("hydra_score")
            if hs is not None:
                by_sym[str(r.get("symbol", "?"))].append(float(hs))
        results["hydra_score_distribution"] = {}
        print("\n  Hydra score distribution (from selector shadow):")
        for sym in sorted(by_sym):
            scores = by_sym[sym]
            dist = {
                "n": len(scores),
                "mean": round(statistics.mean(scores), 4),
                "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
            results["hydra_score_distribution"][sym] = dist
            print(f"    {sym}: n={dist['n']}, μ={dist['mean']:.4f}, σ={dist['std']:.4f}, [{dist['min']:.4f}, {dist['max']:.4f}]")

    return results


# ─────────────────────────────────────────────────────────────────────
# Step 2: ZERO_SCORE elimination effectiveness
# ─────────────────────────────────────────────────────────────────────

def step2_zero_score_effectiveness() -> dict:
    """Validate zero-score guard is working."""
    print("\n" + "=" * 70)
    print("STEP 2: ZERO_SCORE Elimination Effectiveness")
    print("=" * 70)

    results: dict[str, Any] = {}

    # Current zero-score audit log
    zs_events = _read_jsonl(ZERO_SCORE_PATH)
    results["zero_score_audit_events"] = len(zs_events)

    by_source: dict[str, int] = defaultdict(int)
    by_layer: dict[str, int] = defaultdict(int)
    by_symbol: dict[str, int] = defaultdict(int)
    for r in zs_events:
        by_source[str(r.get("source", "unknown"))] += 1
        by_layer[str(r.get("layer", r.get("source", "unknown")))] += 1
        by_symbol[str(r.get("symbol", "unknown"))] += 1

    results["by_source"] = dict(by_source)
    results["by_layer"] = dict(by_layer)
    results["by_symbol"] = dict(by_symbol)

    # Count from p6 replay: how many episodes had conviction=0?
    replay = _read_jsonl(REPLAY_PATH)
    signaled = [r for r in replay if r.get("signal_generated")]
    zero_conv = [r for r in signaled if float(r.get("conviction", 0)) == 0.0]
    no_signal = [r for r in replay if not r.get("signal_generated")]

    results["p6_replay_analysis"] = {
        "total_episodes": len(replay),
        "signaled": len(signaled),
        "no_signal": len(no_signal),
        "zero_conviction_in_signaled": len(zero_conv),
        "zero_conviction_rate": round(len(zero_conv) / len(signaled), 4) if signaled else None,
        "no_signal_rate": round(len(no_signal) / len(replay), 4) if replay else None,
    }

    # PnL of zero-conviction episodes vs nonzero
    nonzero_conv = [r for r in signaled if float(r.get("conviction", 0)) > 0.0]
    if zero_conv:
        zc_pnl = [float(r["realized_net_pnl"]) for r in zero_conv]
        results["zero_conviction_pnl"] = {
            "n": len(zc_pnl),
            "mean": round(statistics.mean(zc_pnl), 4),
            "total": round(sum(zc_pnl), 2),
        }
    if nonzero_conv:
        nz_pnl = [float(r["realized_net_pnl"]) for r in nonzero_conv]
        results["nonzero_conviction_pnl"] = {
            "n": len(nz_pnl),
            "mean": round(statistics.mean(nz_pnl), 4),
            "total": round(sum(nz_pnl), 2),
        }

    # Current score distribution zero rate
    if SCORES_PATH.exists():
        raw = json.loads(SCORES_PATH.read_text())
        syms = raw.get("symbols") or []
        if isinstance(syms, list):
            all_scores = [float(e.get("score", 0)) for e in syms if isinstance(e, dict)]
        else:
            all_scores = [float(v.get("score", 0) if isinstance(v, dict) else v) for v in syms.values()]
        total = len(all_scores)
        zeros = sum(1 for s in all_scores if s == 0.0)
        results["current_score_state"] = {
            "total_symbols": total,
            "zero_score_count": zeros,
            "zero_rate": round(zeros / total, 4) if total else None,
        }

    print(f"\n  Zero-score audit events:  {len(zs_events)}")
    print(f"  By source: {dict(by_source)}")
    print(f"  By symbol: {dict(by_symbol)}")
    if signaled:
        print(f"\n  P6 replay analysis:")
        print(f"    Total episodes:        {len(replay)}")
        print(f"    No-signal episodes:    {len(no_signal)} ({len(no_signal)/len(replay):.1%})")
        print(f"    Zero-conviction:       {len(zero_conv)} ({len(zero_conv)/len(signaled):.1%} of signaled)")
        if zero_conv:
            zc_pnl = [float(r["realized_net_pnl"]) for r in zero_conv]
            nz_pnl = [float(r["realized_net_pnl"]) for r in nonzero_conv] if nonzero_conv else []
            print(f"    Zero-conv mean PnL:    {statistics.mean(zc_pnl):+.4f}")
            if nz_pnl:
                print(f"    Nonzero-conv mean PnL: {statistics.mean(nz_pnl):+.4f}")
                bias = statistics.mean(zc_pnl) - statistics.mean(nz_pnl)
                print(f"    PnL bias (zero-nonzero): {bias:+.4f} {'⚠ negative bias' if bias < 0 else '≈ neutral'}")

    if "current_score_state" in results:
        cs = results["current_score_state"]
        print(f"\n  Current live score state:")
        print(f"    Symbols: {cs['total_symbols']}, Zero-score: {cs['zero_score_count']} ({cs['zero_rate']:.0%})")
        target = cs["zero_rate"] < 0.10 if cs["zero_rate"] is not None else False
        print(f"    Zero rate < 10%: {'PASS ✓' if target else 'FAIL ✗'}")

    return results


# ─────────────────────────────────────────────────────────────────────
# Step 3: Determinism certification
# ─────────────────────────────────────────────────────────────────────

def step3_determinism_certification() -> dict:
    """Test determinism by running scoring pipeline twice with identical inputs."""
    print("\n" + "=" * 70)
    print("STEP 3: Determinism Certification")
    print("=" * 70)

    results: dict[str, Any] = {}

    # Test 1: Score computation determinism
    # Run hybrid_score_universe with identical inputs twice
    try:
        from execution.intel.symbol_score_v6 import (
            hybrid_score_universe,
            load_hybrid_config,
        )

        cfg = load_hybrid_config()

        # Build fixed synthetic inputs
        mock_intents = [
            {"symbol": "BTCUSDT", "direction": "LONG", "conviction": 0.5},
            {"symbol": "ETHUSDT", "direction": "LONG", "conviction": 0.4},
        ]
        mock_expectancy = {
            "symbols": {
                "BTCUSDT": {"expectancy": 0.01, "count": 5, "is_mature": False, "is_prior": True,
                            "prior_components": {"regime_base": -0.01, "vol_adj": 0, "router_adj": 0, "trend_adj": 0, "carry_adj": 0}},
                "ETHUSDT": {"expectancy": -0.005, "count": 3, "is_mature": False, "is_prior": True,
                            "prior_components": {"regime_base": -0.005, "vol_adj": 0, "router_adj": 0, "trend_adj": 0, "carry_adj": 0}},
            },
            "updated_ts": 1700000000.0,
        }

        mock_router = {
            "BTCUSDT": {"quality": "ok", "maker_fill_rate": 0.8, "offset_bps": 1.0, "slippage_p50": 0.5},
            "ETHUSDT": {"quality": "ok", "maker_fill_rate": 0.7, "offset_bps": 1.2, "slippage_p50": 0.8},
        }

        mock_funding = {"BTCUSDT": {"rate": 0.0001}, "ETHUSDT": {"rate": -0.0002}}
        mock_basis = {"BTCUSDT": {"basis_pct": 0.001}, "ETHUSDT": {"basis_pct": -0.001}}

        # Run A
        result_a = hybrid_score_universe(
            intents=mock_intents,
            expectancy_snapshot=mock_expectancy,
            router_health_snapshot=mock_router,
            funding_snapshot=mock_funding,
            basis_snapshot=mock_basis,
            regime="TREND_UP",
            config=cfg,
        )

        # Run B (identical inputs)
        result_b = hybrid_score_universe(
            intents=mock_intents,
            expectancy_snapshot=mock_expectancy,
            router_health_snapshot=mock_router,
            funding_snapshot=mock_funding,
            basis_snapshot=mock_basis,
            regime="TREND_UP",
            config=cfg,
        )

        # Compare
        hash_a = hashlib.sha256(json.dumps(result_a, sort_keys=True, default=str).encode()).hexdigest()
        hash_b = hashlib.sha256(json.dumps(result_b, sort_keys=True, default=str).encode()).hexdigest()

        score_match = hash_a == hash_b
        results["scoring_determinism"] = {
            "hash_a": hash_a[:16],
            "hash_b": hash_b[:16],
            "match": score_match,
            "verdict": "PASS" if score_match else "FAIL",
        }

        print(f"\n  Scoring pipeline determinism:")
        print(f"    Run A hash: {hash_a[:16]}...")
        print(f"    Run B hash: {hash_b[:16]}...")
        print(f"    Match: {'PASS ✓' if score_match else 'FAIL ✗'}")

        # Check individual symbol scores
        if not score_match:
            for sym in ("BTCUSDT", "ETHUSDT"):
                sa = result_a.get(sym, {})
                sb = result_b.get(sym, {})
                for key in ("hybrid_score", "score"):
                    va = sa.get(key)
                    vb = sb.get(key)
                    if va != vb:
                        print(f"    MISMATCH {sym}.{key}: {va} vs {vb}")

    except Exception as exc:
        results["scoring_determinism"] = {"error": str(exc)}
        print(f"\n  Scoring determinism: ERROR - {exc}")

    # Test 2: Regime classification determinism
    try:
        from execution.sentinel_x import (
            run_sentinel_x_step,
            SentinelXConfig,
        )
        import tempfile

        prices = [100.0 + i * 0.1 for i in range(60)]
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_sx = SentinelXConfig(enabled=True)
            state_a = run_sentinel_x_step(prices=prices, cfg=cfg_sx, state_path=Path(tmpdir) / "a.json", dry_run=True)
            state_b = run_sentinel_x_step(prices=prices, cfg=cfg_sx, state_path=Path(tmpdir) / "b.json", dry_run=True)

            regime_match = (
                state_a is not None
                and state_b is not None
                and state_a.primary_regime == state_b.primary_regime
                and state_a.smoothed_probs == state_b.smoothed_probs
            )

            results["regime_determinism"] = {
                "regime_a": state_a.primary_regime if state_a else None,
                "regime_b": state_b.primary_regime if state_b else None,
                "probs_match": regime_match,
                "verdict": "PASS" if regime_match else "FAIL",
            }
            print(f"\n  Regime classification determinism:")
            print(f"    Run A: {state_a.primary_regime if state_a else 'None'}")
            print(f"    Run B: {state_b.primary_regime if state_b else 'None'}")
            print(f"    Match: {'PASS ✓' if regime_match else 'FAIL ✗'}")

    except Exception as exc:
        results["regime_determinism"] = {"error": str(exc)}
        print(f"\n  Regime determinism: ERROR - {exc}")

    # Test 3: Dedup cache determinism (monotonic clock)
    try:
        from execution.signal_screener import reset_dedup_cache
        reset_dedup_cache()
        results["dedup_reset"] = {"available": True, "verdict": "PASS"}
        print(f"\n  Dedup cache reset: PASS ✓ (reset_dedup_cache available)")
    except Exception as exc:
        results["dedup_reset"] = {"error": str(exc)}
        print(f"\n  Dedup cache reset: ERROR - {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────
# Step 4: Drift baseline snapshot
# ─────────────────────────────────────────────────────────────────────

def step4_drift_baseline() -> dict:
    """Capture clean-state baseline for future drift detection."""
    print("\n" + "=" * 70)
    print("STEP 4: Drift Baseline Snapshot (post-audit anchor)")
    print("=" * 70)

    results: dict[str, Any] = {"ts": time.time()}

    # Score distribution
    if SCORES_PATH.exists():
        raw = json.loads(SCORES_PATH.read_text())
        syms = raw.get("symbols") or []
        if isinstance(syms, list):
            scores = {e["symbol"]: float(e.get("score", 0)) for e in syms if isinstance(e, dict) and "symbol" in e}
        else:
            scores = {k: float(v.get("score", 0) if isinstance(v, dict) else v) for k, v in syms.items()}

        all_s = sorted(scores.values())
        results["score_distribution"] = {
            "n": len(all_s),
            "mean": round(statistics.mean(all_s), 4) if all_s else None,
            "std": round(statistics.stdev(all_s), 4) if len(all_s) > 1 else 0,
            "min": round(min(all_s), 4) if all_s else None,
            "max": round(max(all_s), 4) if all_s else None,
            "zero_count": sum(1 for s in all_s if s == 0.0),
            "per_symbol": {k: round(v, 4) for k, v in scores.items()},
        }
        print(f"\n  Score distribution: {len(all_s)} symbols")
        for sym, sc in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"    {sym}: {sc:.4f}")

    # Regime distribution from sentinel state
    sx_path = Path("logs/state/sentinel_x.json")
    if sx_path.exists():
        sx = json.loads(sx_path.read_text())
        results["regime_state"] = {
            "primary": sx.get("primary_regime"),
            "probs": sx.get("smoothed_probs", {}),
            "cycle_count": sx.get("cycle_count"),
        }
        print(f"\n  Regime state: {sx.get('primary_regime')} (cycle {sx.get('cycle_count')})")
        for regime, prob in sorted((sx.get("smoothed_probs") or {}).items(), key=lambda x: -x[1]):
            print(f"    {regime}: {prob:.4f}")

    # Hydra score distribution from shadow log
    shadow = _read_jsonl(SHADOW_PATH)
    if shadow:
        by_sym: dict[str, list[float]] = defaultdict(list)
        for r in shadow:
            hs = r.get("hydra_score")
            if hs is not None:
                by_sym[str(r.get("symbol", "?"))].append(float(hs))
        results["hydra_shadow_distribution"] = {
            sym: {
                "n": len(scores),
                "mean": round(statistics.mean(scores), 4),
                "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
            }
            for sym, scores in sorted(by_sym.items())
        }
        print(f"\n  Hydra shadow scores: {len(shadow)} events across {len(by_sym)} symbols")

    # PnL by conviction band (from replay)
    replay = _read_jsonl(REPLAY_PATH)
    signaled = [r for r in replay if r.get("signal_generated")]
    if signaled:
        bands: dict[str, list[float]] = defaultdict(list)
        for r in signaled:
            conv = float(r.get("conviction", 0))
            pnl = float(r.get("realized_net_pnl", 0))
            if conv == 0:
                band = "0.00"
            elif conv < 0.2:
                band = "0.00-0.20"
            elif conv < 0.4:
                band = "0.20-0.40"
            elif conv < 0.6:
                band = "0.40-0.60"
            else:
                band = "0.60+"
            bands[band].append(pnl)

        results["pnl_by_conviction_band"] = {
            band: {
                "n": len(pnls),
                "mean_pnl": round(statistics.mean(pnls), 4),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4) if pnls else 0,
            }
            for band, pnls in sorted(bands.items())
        }
        print(f"\n  PnL by conviction band:")
        for band in sorted(bands):
            pnls = bands[band]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            print(f"    [{band:>9}]: n={len(pnls):4d}, mean_pnl={statistics.mean(pnls):+.4f}, wr={wr:.1%}, total={sum(pnls):+.1f}")

    # Config hashes
    config_files = sorted(Path("config").glob("*.json")) + sorted(Path("config").glob("*.yaml"))
    config_hashes = {}
    for f in config_files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        config_hashes[str(f)] = h
    results["config_hashes"] = config_hashes
    print(f"\n  Config files: {len(config_hashes)} hashed")

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          POST-AUDIT VALIDATION — 4-STEP CERTIFICATION          ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    all_results: dict[str, Any] = {"ts": time.time()}

    all_results["step1_score_pnl"] = step1_score_pnl_validation()
    all_results["step2_zero_score"] = step2_zero_score_effectiveness()
    all_results["step3_determinism"] = step3_determinism_certification()
    all_results["step4_baseline"] = step4_drift_baseline()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY — VALIDATION VERDICTS")
    print("=" * 70)

    # Step 1 verdicts
    s1 = all_results["step1_score_pnl"].get("by_symbol", {})
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        sym_data = s1.get(sym, {})
        verdicts = sym_data.get("verdicts", {})
        if verdicts:
            all_pass = all(v == "PASS" for v in verdicts.values())
            status = "PASS ✓" if all_pass else "FAIL ✗"
            print(f"  Step 1 ({sym}): {status}  {verdicts}")
        elif "error" in sym_data:
            print(f"  Step 1 ({sym}): SKIP — {sym_data['error']}")

    # Step 2
    s2 = all_results["step2_zero_score"]
    zr = s2.get("current_score_state", {}).get("zero_rate")
    if zr is not None:
        print(f"  Step 2 (zero rate): {'PASS ✓' if zr < 0.10 else 'FAIL ✗'}  (rate={zr:.0%})")
    else:
        print(f"  Step 2 (zero rate): N/A")

    # Step 3
    s3 = all_results["step3_determinism"]
    sd = s3.get("scoring_determinism", {}).get("verdict", "ERROR")
    rd = s3.get("regime_determinism", {}).get("verdict", "ERROR")
    print(f"  Step 3 (scoring):   {sd}")
    print(f"  Step 3 (regime):    {rd}")

    # Step 4
    s4 = all_results["step4_baseline"]
    print(f"  Step 4 (baseline):  CAPTURED ({len(s4.get('config_hashes', {}))} config files)")

    # Write results
    output_path = OUTPUT_DIR / "validation_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Full results written to: {output_path}")

    # Overall
    print("\n" + "═" * 70)


if __name__ == "__main__":
    main()
