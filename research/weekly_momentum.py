"""
Hypothesis D — Weekly Time-Series Momentum: causality test.

Statement: Past-N-week return predicts next-week return with the same sign.
Assets that rose over the past N weeks tend to continue rising; assets
that fell tend to continue falling. (Moskowitz, Ooi, Pedersen 2012.)

Signal definition:
    raw:             signal = (close_t − close_{t − N×7d}) / close_{t − N×7d}
    vol-normalized:  signal = raw / rolling_std(weekly_returns, 26 weeks)

Observations are sampled every horizon_weeks×7 days so forward returns
do not overlap, preserving statistical independence.

Falsification criteria (pre-registered — do not change after first data run):
    ρ(signal, fwd_return_1w)  > +0.15   (Spearman)
    p-value                   <  0.05
    Q5 − Q1                   >  0.0
    Monotonicity ratio        ≥  0.75
    Fee-adjusted ρ            >  0.0

Usage:
    python -m research.weekly_momentum
    python -m research.weekly_momentum --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --lookback-weeks 4 --horizon-weeks 1 --days 730
    python -m research.weekly_momentum --vol-normalize
    python -m research.weekly_momentum --lookback-weeks 1 --horizon-weeks 2
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_BASE_URL = "https://fapi.binance.com"
_ROUND_TRIP_FEE = 0.001  # 0.10% conservative round-trip

# ---------------------------------------------------------------------------
# Falsification thresholds — written before seeing any data
# ---------------------------------------------------------------------------
_MIN_RHO = 0.15
_MAX_PVALUE = 0.05
_MIN_Q5_MINUS_Q1 = 0.0
_MIN_MONOTONICITY = 0.75

# Rolling window for vol normalization: 26 weekly returns (~6 months)
_VOL_WINDOW_WEEKS = 26

# Minimum usable pairs before issuing a verdict
_MIN_PAIRS = 30


@dataclass
class Bar:
    ts_ms: int
    close: float


@dataclass
class AnalysisResult:
    symbol: str
    n: int = 0
    lookback_weeks: int = 4
    horizon_weeks: int = 1
    vol_normalize: bool = False
    rho: float = 0.0
    pvalue: float = 1.0
    q5_minus_q1: float = 0.0
    monotonicity: float = 0.0
    rho_fee_adj: float = 0.0
    quintile_means: list[float] = field(default_factory=list)
    criteria: dict[str, bool] = field(default_factory=dict)
    verdict: str = "PENDING"
    error: str = ""


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _get(path: str, params: dict) -> list | dict:
    try:
        import requests as _requests
    except ImportError as exc:
        raise RuntimeError("requests is required: pip install requests") from exc
    r = _requests.get(_BASE_URL + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_daily_closes(symbol: str, days: int) -> list[Bar]:
    """Return daily close prices for the past `days` calendar days."""
    now_ms = int(time.time() * 1000)
    # Extra buffer for lookback + vol-window
    start_ms = now_ms - (days + (_VOL_WINDOW_WEEKS + 4) * 7) * 24 * 3_600_000
    bars: list[Bar] = []
    cursor = start_ms

    while cursor < now_ms:
        data = _get("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": "1d",
            "startTime": cursor,
            "endTime": now_ms,
            "limit": 1500,
        })
        if not data:
            break
        for row in data:
            bars.append(Bar(ts_ms=int(row[0]), close=float(row[4])))
        if len(data) < 1500:
            break
        cursor = bars[-1].ts_ms + 1
        time.sleep(0.05)

    seen: set[int] = set()
    out: list[Bar] = []
    for b in sorted(bars, key=lambda b: b.ts_ms):
        if b.ts_ms not in seen:
            seen.add(b.ts_ms)
            out.append(b)
    return out


# ---------------------------------------------------------------------------
# Statistics (duplicated from funding_rate_extremes.py for self-containment)
# ---------------------------------------------------------------------------

def _rank(vals: list[float]) -> list[float]:
    n = len(vals)
    indexed = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (ρ, two-tailed p-value) via t-approximation."""
    n = len(x)
    if n < 4:
        return 0.0, 1.0

    rx, ry = _rank(x), _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    rho = max(-1.0, min(1.0, rho))

    if abs(rho) == 1.0:
        return rho, 0.0

    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    df = n - 2
    x_val = df / (df + t * t)

    def _ibeta_cf(x_v: float, a: float, b: float, max_iter: int = 200) -> float:
        if x_v <= 0:
            return 0.0
        if x_v >= 1:
            return 1.0
        lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        front = math.exp(lbeta + a * math.log(x_v) + b * math.log(1 - x_v)) / a
        f = C = 1.0
        D = 0.0
        for m in range(max_iter):
            for step in (0, 1):
                if m == 0 and step == 0:
                    d = 1.0
                elif step == 0:
                    d = m * (b - m) * x_v / ((a + 2 * m - 1) * (a + 2 * m))
                else:
                    d = -(a + m) * (a + b + m) * x_v / ((a + 2 * m) * (a + 2 * m + 1))
                D = 1.0 / max(abs(1.0 + d * D), 1e-30) * (1 if (1.0 + d * D) >= 0 else -1)
                C = max(abs(1.0 + d / C), 1e-30) * (1 if (1.0 + d / C) >= 0 else -1)
                D = 1.0 / (1.0 + d * (1.0 / f - 1.0) / f) if abs(f) > 1e-30 else 1.0
                # Simplified: use standard Lentz
                D_raw = 1.0 + d * D
                if abs(D_raw) < 1e-30:
                    D_raw = 1e-30
                C_raw = 1.0 + d / C if abs(C) > 1e-30 else 1e-30
                if abs(C_raw) < 1e-30:
                    C_raw = 1e-30
                D = 1.0 / D_raw
                C = C_raw
                delta = C * D
                f *= delta
                if abs(delta - 1.0) < 1e-10:
                    break
        return front * f

    p_one = 0.5 * _ibeta_cf(x_val, df / 2.0, 0.5)
    return rho, min(1.0, max(0.0, 2.0 * p_one))


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def build_signal_return_pairs(
    bars: list[Bar],
    lookback_weeks: int,
    horizon_weeks: int,
    vol_normalize: bool,
) -> tuple[list[float], list[float]]:
    """
    Return (signals, forward_returns) sampled at horizon_weeks intervals.

    Sampling is non-overlapping so forward returns are independent.
    A bar at index i is a sample point if it has:
      - lookback_weeks*7 calendar days of history available
      - horizon_weeks*7 calendar days of future available
    """
    if not bars:
        return [], []

    lookback_ms = lookback_weeks * 7 * 24 * 3_600_000
    horizon_ms = horizon_weeks * 7 * 24 * 3_600_000
    step_ms = horizon_weeks * 7 * 24 * 3_600_000  # non-overlapping step

    # Build a timestamp → close lookup
    close_by_ts: dict[int, float] = {b.ts_ms: b.close for b in bars}

    def closest_close(target_ms: int, tolerance_ms: int = 2 * 24 * 3_600_000) -> Optional[float]:
        """Find the closest bar within tolerance."""
        best_dist = tolerance_ms + 1
        best_close = None
        for b in bars:
            dist = abs(b.ts_ms - target_ms)
            if dist < best_dist:
                best_dist = dist
                best_close = b.close
        return best_close

    # Pre-build sorted timestamps for efficient lookup
    sorted_ts = sorted(b.ts_ms for b in bars)

    def snap_close(target_ms: int) -> Optional[float]:
        """Binary search to find closest bar."""
        lo, hi = 0, len(sorted_ts) - 1
        best_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_ts[mid] <= target_ms:
                best_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        # Check both sides
        candidates = []
        for idx in (best_idx, best_idx + 1):
            if 0 <= idx < len(sorted_ts):
                candidates.append((abs(sorted_ts[idx] - target_ms), close_by_ts.get(sorted_ts[idx])))
        candidates = [(d, c) for d, c in candidates if c is not None]
        if not candidates:
            return None
        _, c = min(candidates)
        return c

    # Precompute weekly returns for vol normalization
    # Weekly return at bar i: (close_i - close_{i-7d}) / close_{i-7d}
    weekly_ret_by_ts: dict[int, float] = {}
    if vol_normalize:
        week_ms = 7 * 24 * 3_600_000
        for b in bars:
            px_prev = snap_close(b.ts_ms - week_ms)
            if px_prev and px_prev > 0:
                weekly_ret_by_ts[b.ts_ms] = (b.close - px_prev) / px_prev

    # Sample non-overlapping observation points
    signals: list[float] = []
    returns: list[float] = []

    # Start from the first bar that has enough lookback history
    first_eligible = bars[0].ts_ms + lookback_ms + _VOL_WINDOW_WEEKS * 7 * 24 * 3_600_000
    last_eligible = bars[-1].ts_ms - horizon_ms

    t = first_eligible
    while t <= last_eligible:
        px_now = snap_close(t)
        px_back = snap_close(t - lookback_ms)
        px_fwd = snap_close(t + horizon_ms)

        if px_now is None or px_back is None or px_fwd is None:
            t += step_ms
            continue
        if px_back <= 0 or px_now <= 0:
            t += step_ms
            continue

        raw_signal = (px_now - px_back) / px_back

        if vol_normalize:
            # Collect weekly returns over the past _VOL_WINDOW_WEEKS weeks
            vol_window_ms = _VOL_WINDOW_WEEKS * 7 * 24 * 3_600_000
            window_rets = [
                v for ts_k, v in weekly_ret_by_ts.items()
                if t - vol_window_ms <= ts_k <= t
            ]
            if len(window_rets) < 4:
                t += step_ms
                continue
            std = statistics.stdev(window_rets)
            if std < 1e-9:
                t += step_ms
                continue
            signal = raw_signal / std
        else:
            signal = raw_signal

        fwd_return = (px_fwd - px_now) / px_now
        signals.append(signal)
        returns.append(fwd_return)
        t += step_ms

    return signals, returns


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_symbol(
    symbol: str,
    days: int,
    lookback_weeks: int,
    horizon_weeks: int,
    vol_normalize: bool,
) -> AnalysisResult:
    result = AnalysisResult(
        symbol=symbol,
        lookback_weeks=lookback_weeks,
        horizon_weeks=horizon_weeks,
        vol_normalize=vol_normalize,
    )
    vol_label = " vol-norm" if vol_normalize else ""
    print(f"\n{'='*60}")
    print(f"  {symbol}  |  {days}d  |  lookback={lookback_weeks}w  |  fwd={horizon_weeks}w{vol_label}")
    print(f"{'='*60}")

    print("  Fetching daily closes ...")
    bars = fetch_daily_closes(symbol, days)
    print(f"  {len(bars)} daily bars")

    if len(bars) < (lookback_weeks + horizon_weeks + _VOL_WINDOW_WEEKS) * 7:
        result.verdict = "INSUFFICIENT_DATA"
        result.error = f"only {len(bars)} bars"
        return result

    signals, returns = build_signal_return_pairs(bars, lookback_weeks, horizon_weeks, vol_normalize)
    result.n = len(signals)
    print(f"  Non-overlapping sample pairs: {result.n}")

    if result.n < _MIN_PAIRS:
        result.verdict = "INSUFFICIENT_PAIRS"
        result.error = f"only {result.n} pairs (need {_MIN_PAIRS})"
        return result

    # Spearman ρ
    result.rho, result.pvalue = spearman_rho(signals, returns)
    print(f"  ρ = {result.rho:+.4f}   p = {result.pvalue:.4f}")

    # Quintile analysis
    n = result.n
    paired = sorted(zip(signals, returns), key=lambda t: t[0])
    q_size = n // 5
    result.quintile_means = []
    for q in range(5):
        s = q * q_size
        e = (q + 1) * q_size if q < 4 else n
        result.quintile_means.append(statistics.mean(r for _, r in paired[s:e]))

    result.q5_minus_q1 = result.quintile_means[4] - result.quintile_means[0]
    in_order = sum(1 for i in range(4) if result.quintile_means[i] <= result.quintile_means[i + 1])
    result.monotonicity = in_order / 4.0

    qstr = "  ".join(f"{v:+.5f}" for v in result.quintile_means)
    print(f"  Quintiles Q1→Q5: {qstr}")
    print(f"  Q5−Q1 = {result.q5_minus_q1:+.5f}   monotonicity = {result.monotonicity:.2f}")

    # Fee-adjusted ρ
    fee_adj = [r - _ROUND_TRIP_FEE for r in returns]
    result.rho_fee_adj, _ = spearman_rho(signals, fee_adj)
    print(f"  Fee-adj ρ = {result.rho_fee_adj:+.4f}")

    # Verdict
    result.criteria = {
        "rho_above_0.15": result.rho > _MIN_RHO,
        "pvalue_below_0.05": result.pvalue < _MAX_PVALUE,
        "q5_q1_positive": result.q5_minus_q1 > _MIN_Q5_MINUS_Q1,
        "monotonicity_above_0.75": result.monotonicity >= _MIN_MONOTONICITY,
        "fee_adj_rho_positive": result.rho_fee_adj > 0.0,
    }
    passed = sum(result.criteria.values())
    result.verdict = "PASS" if passed == 5 else ("CONDITIONAL" if passed >= 3 else "FAIL")

    print(f"\n  Criteria:")
    for k, v in result.criteria.items():
        print(f"    {'✓' if v else '✗'}  {k}")
    print(f"\n  VERDICT: {result.verdict}  ({passed}/5)")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hypothesis D: weekly time-series momentum causality test"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--days", type=int, default=730,
                        help="Calendar days of history (default 730 = 2 years)")
    parser.add_argument("--lookback-weeks", type=int, default=4,
                        help="Momentum lookback in weeks (default 4)")
    parser.add_argument("--horizon-weeks", type=int, default=1,
                        help="Forward return horizon in weeks (default 1)")
    parser.add_argument("--vol-normalize", action="store_true",
                        help="Divide signal by rolling 26-week return std")
    parser.add_argument("--out", default="data/hypothesis_d_results.json",
                        help="Path to write JSON results")
    args = parser.parse_args()

    vol_label = " (vol-normalized)" if args.vol_normalize else ""
    print("\nFALSIFICATION CRITERIA (pre-registered):")
    print(f"  ρ(signal, fwd_return_{args.horizon_weeks}w)  > {_MIN_RHO}")
    print(f"  p-value                                    < {_MAX_PVALUE}")
    print(f"  Q5 − Q1                                    > {_MIN_Q5_MINUS_Q1}")
    print(f"  Monotonicity ratio                         ≥ {_MIN_MONOTONICITY}")
    print(f"  Fee-adjusted ρ                             > 0.0")
    print(f"\n  Signal: past_{args.lookback_weeks}w_return{vol_label}")
    print(f"  Sampling: non-overlapping, every {args.horizon_weeks} week(s)")
    print(f"  Rationale: trend persistence — winners keep winning, losers keep losing")

    results = []
    for sym in args.symbols:
        try:
            r = analyse_symbol(sym, args.days, args.lookback_weeks, args.horizon_weeks, args.vol_normalize)
        except Exception as exc:
            r = AnalysisResult(symbol=sym, verdict="ERROR", error=str(exc))
            print(f"\nERROR for {sym}: {exc}")
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.symbol:12s}  {r.verdict:12s}  ρ={r.rho:+.4f}  p={r.pvalue:.4f}  n={r.n}")

    all_pass = all(r.verdict == "PASS" for r in results)
    any_error = any(r.verdict == "ERROR" for r in results)
    any_fail = any(r.verdict == "FAIL" for r in results)

    print()
    if all_pass:
        print("HYPOTHESIS SUPPORTED — proceed to paper trade design")
    elif any_error:
        print("ERRORS ENCOUNTERED — fix data fetch issues before concluding")
    elif any_fail:
        print("HYPOTHESIS NOT SUPPORTED — do not deploy live")
    else:
        print("CONDITIONAL SUPPORT — review individual criteria before proceeding")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([
            {
                "symbol": r.symbol,
                "n": r.n,
                "lookback_weeks": r.lookback_weeks,
                "horizon_weeks": r.horizon_weeks,
                "vol_normalize": r.vol_normalize,
                "rho": r.rho,
                "pvalue": r.pvalue,
                "q5_minus_q1": r.q5_minus_q1,
                "monotonicity": r.monotonicity,
                "rho_fee_adj": r.rho_fee_adj,
                "quintile_means": r.quintile_means,
                "criteria": r.criteria,
                "verdict": r.verdict,
                "error": r.error,
            }
            for r in results
        ], f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
