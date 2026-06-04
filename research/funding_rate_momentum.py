"""
Hypothesis F — Funding Rate Momentum: causality test.

Statement: When the 8-hour perpetual funding rate is trending upward over the
past N funding periods, it signals growing bullish positioning; price continues
upward over the next M days.

This is distinct from Hypothesis A (funding rate *level* extremes → mean
reversion). Here the signal is the *slope* (recent window vs prior window),
testing trend continuation rather than reversion.

Signal definition:
    signal = mean(fr[-N:]) - mean(fr[-2N:-N])

    Positive signal → funding accelerating up → expect positive forward return
    Negative signal → funding accelerating down → expect negative forward return

Observations are sampled at every forward-horizon boundary (non-overlapping)
so forward returns are independent.

Falsification criteria (pre-registered — do not change after first data run):
    ρ(signal, fwd_return)  > +0.15   (Spearman)
    p-value                <  0.05
    Q5 − Q1                >  0.0
    Monotonicity ratio     ≥  0.75
    Fee-adjusted ρ         >  0.0

Usage:
    python -m research.funding_rate_momentum
    python -m research.funding_rate_momentum --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --days 730 --window 7 --horizon-days 1
    python -m research.funding_rate_momentum --window 14 --horizon-days 4
    python -m research.funding_rate_momentum --window 3 --horizon-days 1
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

# Funding periods per day: Binance pays every 8h → 3 per day
_FR_PER_DAY = 3

# ---------------------------------------------------------------------------
# Falsification thresholds — written before seeing any data
# ---------------------------------------------------------------------------
_MIN_RHO = 0.15
_MAX_PVALUE = 0.05
_MIN_Q5_MINUS_Q1 = 0.0
_MIN_MONOTONICITY = 0.75

_MIN_PAIRS = 30


@dataclass
class FundingRecord:
    ts_ms: int
    rate: float


@dataclass
class Bar:
    ts_ms: int
    close: float


@dataclass
class AnalysisResult:
    symbol: str
    n: int = 0
    window: int = 7        # funding window in days (each window = window * 3 periods)
    horizon_days: int = 1
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


def fetch_funding_history(symbol: str, days: int) -> list[FundingRecord]:
    """Paginate /fapi/v1/fundingRate to get the full history window."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3_600_000
    records: list[FundingRecord] = []
    cursor = start_ms

    while True:
        data = _get("/fapi/v1/fundingRate", {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": now_ms,
            "limit": 1000,
        })
        if not data:
            break
        for row in data:
            records.append(FundingRecord(
                ts_ms=int(row["fundingTime"]),
                rate=float(row["fundingRate"]),
            ))
        if len(data) < 1000:
            break
        cursor = records[-1].ts_ms + 1
        time.sleep(0.1)

    records.sort(key=lambda r: r.ts_ms)
    return records


def fetch_daily_closes(symbol: str, days: int) -> list[Bar]:
    """Return daily close prices for the past `days` + buffer calendar days."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days + 30) * 24 * 3_600_000
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
# Statistics
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
    """Return (ρ, two-tailed p-value). Uses t-approximation; no scipy required."""
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
        """Regularised incomplete beta via Lentz continued fraction."""
        if x_v <= 0:
            return 0.0
        if x_v >= 1:
            return 1.0
        lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        front = math.exp(lbeta + a * math.log(x_v) + b * math.log(1 - x_v)) / a
        f, C, D = 1.0, 1.0, 0.0
        for m in range(max_iter):
            for step in (0, 1):
                if m == 0 and step == 0:
                    d = 1.0
                elif step == 0:
                    d = m * (b - m) * x_v / ((a + 2 * m - 1) * (a + 2 * m))
                else:
                    d = -(a + m) * (a + b + m) * x_v / ((a + 2 * m) * (a + 2 * m + 1))
                D = 1.0 + d * D
                if abs(D) < 1e-30:
                    D = 1e-30
                C = 1.0 + d / C
                if abs(C) < 1e-30:
                    C = 1e-30
                D = 1.0 / D
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
    funding: list[FundingRecord],
    bars: list[Bar],
    window: int,
    horizon_days: int,
) -> tuple[list[float], list[float]]:
    """
    Return (signals, forward_returns) sampled at horizon_days intervals.

    signal = mean(fr[-window_periods:]) - mean(fr[-2*window_periods:-window_periods])

    At each sample point t:
      - The signal is computed from the N=window*3 funding periods ending at t
      - The forward return is the daily close return over the next horizon_days

    Sampling is non-overlapping (step = horizon_days) for independence.
    """
    if not funding or not bars:
        return [], []

    window_periods = window * _FR_PER_DAY   # 3 funding events per day
    required_periods = 2 * window_periods    # need 2 windows of history
    horizon_ms = horizon_days * 24 * 3_600_000
    step_ms = horizon_ms  # non-overlapping

    # Build sorted bar lookup for snap_close
    sorted_bar_ts = sorted(b.ts_ms for b in bars)
    close_by_ts: dict[int, float] = {b.ts_ms: b.close for b in bars}

    def snap_close(target_ms: int, tol_ms: int = 2 * 24 * 3_600_000) -> Optional[float]:
        lo, hi = 0, len(sorted_bar_ts) - 1
        best_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_bar_ts[mid] <= target_ms:
                best_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        best_dist, best_close = tol_ms + 1, None
        for idx in (best_idx, best_idx + 1):
            if 0 <= idx < len(sorted_bar_ts):
                dist = abs(sorted_bar_ts[idx] - target_ms)
                if dist < best_dist:
                    c = close_by_ts.get(sorted_bar_ts[idx])
                    if c is not None:
                        best_dist, best_close = dist, c
        return best_close

    signals: list[float] = []
    returns: list[float] = []

    # First eligible index in funding array: need 2*window_periods of history
    first_idx = required_periods
    if first_idx >= len(funding):
        return [], []

    # Sample at every horizon_days step starting from the first eligible funding event
    first_ts = funding[first_idx].ts_ms
    last_ts = funding[-1].ts_ms - horizon_ms

    t = first_ts
    # Find starting funding index for t
    fr_idx = first_idx

    while t <= last_ts:
        # Advance fr_idx to the last funding event <= t
        while fr_idx + 1 < len(funding) and funding[fr_idx + 1].ts_ms <= t:
            fr_idx += 1

        if fr_idx < required_periods:
            t += step_ms
            continue

        # Two windows ending at fr_idx
        recent = [funding[i].rate for i in range(fr_idx - window_periods + 1, fr_idx + 1)]
        prior = [funding[i].rate for i in range(fr_idx - required_periods + 1, fr_idx - window_periods + 1)]

        if len(recent) < window_periods or len(prior) < window_periods:
            t += step_ms
            continue

        signal = statistics.mean(recent) - statistics.mean(prior)

        # Forward return from close at t to close at t + horizon_days
        px_now = snap_close(t)
        px_fwd = snap_close(t + horizon_ms)

        if px_now is None or px_fwd is None or px_now <= 0:
            t += step_ms
            continue

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
    window: int,
    horizon_days: int,
) -> AnalysisResult:
    result = AnalysisResult(symbol=symbol, window=window, horizon_days=horizon_days)

    print(f"\n{'='*60}")
    print(f"  {symbol}  |  {days}d  |  window={window}d  |  fwd={horizon_days}d")
    print(f"{'='*60}")

    print("  Fetching funding rate history ...")
    funding = fetch_funding_history(symbol, days)
    print(f"  {len(funding)} funding records")

    print("  Fetching daily closes ...")
    bars = fetch_daily_closes(symbol, days)
    print(f"  {len(bars)} daily bars")

    signals, returns = build_signal_return_pairs(funding, bars, window, horizon_days)
    result.n = len(signals)
    print(f"  Non-overlapping sample pairs: {result.n}")

    if result.n < _MIN_PAIRS:
        result.verdict = "INSUFFICIENT_PAIRS"
        result.error = f"only {result.n} pairs (need {_MIN_PAIRS})"
        return result

    result.rho, result.pvalue = spearman_rho(signals, returns)
    print(f"  ρ = {result.rho:+.4f}   p = {result.pvalue:.4f}")

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

    fee_adj = [r - _ROUND_TRIP_FEE for r in returns]
    result.rho_fee_adj, _ = spearman_rho(signals, fee_adj)
    print(f"  Fee-adj ρ = {result.rho_fee_adj:+.4f}")

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
        description="Hypothesis F: funding rate momentum causality test"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--days", type=int, default=730,
                        help="Calendar days of history (default 730)")
    parser.add_argument("--window", type=int, default=7,
                        help="Window size in days for each half of the slope signal (default 7)")
    parser.add_argument("--horizon-days", type=int, default=1,
                        help="Forward return horizon in days (default 1)")
    parser.add_argument("--out", default="data/hypothesis_f_results.json",
                        help="Path to write JSON results")
    args = parser.parse_args()

    print("\nFALSIFICATION CRITERIA (pre-registered):")
    print(f"  ρ(signal, fwd_return_{args.horizon_days}d)  > {_MIN_RHO}")
    print(f"  p-value                             < {_MAX_PVALUE}")
    print(f"  Q5 − Q1                             > {_MIN_Q5_MINUS_Q1}")
    print(f"  Monotonicity ratio                  ≥ {_MIN_MONOTONICITY}")
    print(f"  Fee-adjusted ρ                      > 0.0")
    print(f"\n  Signal: mean(fr[-{args.window}d:]) - mean(fr[-{2*args.window}d:-{args.window}d])")
    print(f"  Rationale: rising funding → growing bullish positioning → trend continuation")

    results = []
    for sym in args.symbols:
        try:
            r = analyse_symbol(sym, args.days, args.window, args.horizon_days)
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
                "window": r.window,
                "horizon_days": r.horizon_days,
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
