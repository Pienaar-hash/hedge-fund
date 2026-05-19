"""
Hypothesis A — Funding Rate Extremes: causality test.

Statement: When the 8-hour perpetual funding rate deviates beyond 2× its
30-day rolling mean-absolute value, price mean-reverts over the following
4 hours as carry traders unwind.

Anti-signal definition:
    anti_signal = -(fr_t / rolling_30d_mean_abs)
    Positive anti_signal → expect positive forward return (short squeeze / long unwind)
    Negative anti_signal → expect negative forward return (short unwind)

Falsification criteria (pre-registered — do not change after first data run):
    ρ(anti_signal, return_4h)  > +0.15   (Spearman)
    p-value                    <  0.05
    Q5 − Q1 quintile spread    >  0.0
    Monotonicity ratio         ≥  0.75   (quintiles in order)
    Fee-adjusted ρ             >  0.0

Usage:
    python -m research.funding_rate_extremes
    python -m research.funding_rate_extremes --symbols BTCUSDT ETHUSDT --days 180 --horizon 4
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_BASE_URL = "https://fapi.binance.com"

# Conservative round-trip fee assumption: mostly taker fills (0.05% each way)
_ROUND_TRIP_FEE = 0.001

# ---------------------------------------------------------------------------
# Falsification thresholds — written before seeing any data
# ---------------------------------------------------------------------------
_MIN_RHO = 0.15
_MAX_PVALUE = 0.05
_MIN_Q5_MINUS_Q1 = 0.0
_MIN_MONOTONICITY = 0.75

# 30-day rolling window in funding events (3 per day × 30 days)
_ROLLING_WINDOW = 90


@dataclass
class FundingRecord:
    ts_ms: int
    rate: float


@dataclass
class Bar:
    ts_ms: int
    open: float
    close: float


@dataclass
class AnalysisResult:
    symbol: str
    n: int = 0
    horizon_h: int = 4
    rho: float = 0.0
    pvalue: float = 1.0
    q5_minus_q1: float = 0.0
    monotonicity: float = 0.0
    rho_fee_adj: float = 0.0
    quintile_means: list[float] = field(default_factory=list)
    extreme_pct: float = 0.0
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
        raise RuntimeError("requests is required for data fetching: pip install requests") from exc
    url = _BASE_URL + path
    r = _requests.get(url, params=params, timeout=30)
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


def fetch_klines_range(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[Bar]:
    """Paginate /fapi/v1/klines across a time range."""
    bars: list[Bar] = []
    cursor = start_ms

    while cursor < end_ms:
        data = _get("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        })
        if not data:
            break
        for row in data:
            bars.append(Bar(ts_ms=int(row[0]), open=float(row[1]), close=float(row[4])))
        if len(data) < 1500:
            break
        cursor = bars[-1].ts_ms + 1
        time.sleep(0.05)

    # Deduplicate and sort
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

    # Two-tailed p-value via regularised incomplete beta (Abramowitz & Stegun 26.7.8)
    df = n - 2
    x_val = df / (df + t * t)
    # I_x(a, b) where a=df/2, b=1/2 — use continued fraction expansion
    def _ibeta_cf(x_v: float, a: float, b: float, max_iter: int = 200) -> float:
        """Regularised incomplete beta via Lentz continued fraction."""
        if x_v <= 0:
            return 0.0
        if x_v >= 1:
            return 1.0
        lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        front = math.exp(lbeta + a * math.log(x_v) + b * math.log(1 - x_v)) / a
        # Continued fraction (even part)
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
    p = 2.0 * p_one
    return rho, min(1.0, max(0.0, p))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_symbol(symbol: str, days: int, horizon_h: int) -> AnalysisResult:
    result = AnalysisResult(symbol=symbol, horizon_h=horizon_h)

    print(f"\n{'='*60}")
    print(f"  {symbol}  |  {days}d  |  {horizon_h}h forward return")
    print(f"{'='*60}")

    print("  Fetching funding rate history ...")
    funding = fetch_funding_history(symbol, days)
    print(f"  {len(funding)} funding records")

    if len(funding) < _ROLLING_WINDOW + 50:
        result.verdict = "INSUFFICIENT_DATA"
        result.error = f"only {len(funding)} funding records"
        return result

    # Fetch 1h klines: need price at funding time and at funding time + horizon_h hours
    first_ts = funding[_ROLLING_WINDOW].ts_ms
    last_ts = funding[-1].ts_ms + (horizon_h + 2) * 3_600_000
    print("  Fetching 1h klines ...")
    bars = fetch_klines_range(symbol, "1h", first_ts - 3_600_000, last_ts)
    print(f"  {len(bars)} hourly bars")

    # Price lookup: snap to nearest 1h bar open
    bar_open: dict[int, float] = {b.ts_ms: b.open for b in bars}

    def price_at(ts_ms: int) -> Optional[float]:
        snapped = (ts_ms // 3_600_000) * 3_600_000
        return bar_open.get(snapped)

    # Build signal / return pairs
    signals: list[float] = []
    returns: list[float] = []

    for i in range(_ROLLING_WINDOW, len(funding)):
        fr = funding[i].rate
        ts = funding[i].ts_ms

        window_abs = [abs(funding[j].rate) for j in range(i - _ROLLING_WINDOW, i)]
        mean_abs = statistics.mean(window_abs)
        if mean_abs < 1e-9:
            continue  # degenerate window

        anti_signal = -(fr / mean_abs)

        px_now = price_at(ts)
        px_fwd = price_at(ts + horizon_h * 3_600_000)
        if px_now is None or px_fwd is None or px_now <= 0:
            continue

        signals.append(anti_signal)
        returns.append((px_fwd - px_now) / px_now)

    result.n = len(signals)
    print(f"  Usable pairs: {result.n}")

    if result.n < 50:
        result.verdict = "INSUFFICIENT_PAIRS"
        result.error = f"only {result.n} usable pairs"
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

    # Extreme signal fraction
    result.extreme_pct = 100.0 * sum(1 for s in signals if abs(s) > 2.0) / n
    print(f"  |signal| > 2σ: {result.extreme_pct:.1f}% of observations")

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
        description="Hypothesis A: funding rate extremes causality test"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--days", type=int, default=180,
                        help="Calendar days of history to fetch")
    parser.add_argument("--horizon", type=int, default=4,
                        help="Forward return horizon in hours")
    parser.add_argument("--out", default="data/hypothesis_a_results.json",
                        help="Path to write JSON results")
    args = parser.parse_args()

    print("\nFALSIFICATION CRITERIA (pre-registered):")
    print(f"  ρ(anti_signal, return_{args.horizon}h)  > {_MIN_RHO}")
    print(f"  p-value                               < {_MAX_PVALUE}")
    print(f"  Q5 − Q1                               > {_MIN_Q5_MINUS_Q1}")
    print(f"  Monotonicity ratio                    ≥ {_MIN_MONOTONICITY}")
    print(f"  Fee-adjusted ρ                        > 0.0")
    print(f"\n  Signal: anti_signal = -(fr_t / rolling_30d_mean_abs)")
    print(f"  Rationale: high +funding → longs unwind → price falls → short earns")

    results = []
    for sym in args.symbols:
        try:
            r = analyse_symbol(sym, args.days, args.horizon)
        except Exception as exc:
            r = AnalysisResult(symbol=sym, verdict="ERROR", error=str(exc))
            print(f"\nERROR for {sym}: {exc}")
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.symbol:12s}  {r.verdict:12s}  ρ={r.rho:+.4f}  p={r.pvalue:.4f}  n={r.n}")

    any_fail = any(r.verdict == "FAIL" for r in results)
    any_error = any(r.verdict == "ERROR" for r in results)
    all_pass = all(r.verdict == "PASS" for r in results)

    print()
    if all_pass:
        print("HYPOTHESIS SUPPORTED — proceed to paper trade design")
    elif any_error:
        print("ERRORS ENCOUNTERED — fix data fetch issues before concluding")
    elif any_fail:
        print("HYPOTHESIS NOT SUPPORTED — do not deploy live")
        print("Redesign signal before re-testing on new data.")
    else:
        print("CONDITIONAL SUPPORT — review individual criteria before proceeding")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "symbol": r.symbol,
            "n": r.n,
            "horizon_h": r.horizon_h,
            "rho": r.rho,
            "pvalue": r.pvalue,
            "q5_minus_q1": r.q5_minus_q1,
            "monotonicity": r.monotonicity,
            "rho_fee_adj": r.rho_fee_adj,
            "extreme_pct": r.extreme_pct,
            "quintile_means": r.quintile_means,
            "criteria": r.criteria,
            "verdict": r.verdict,
            "error": r.error,
        }
        for r in results
    ]
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
