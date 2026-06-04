"""
Hypothesis G — Open Interest Change Rate: causality test.

Statement: When open interest is growing (new positions entering), price
continues in the direction it has been moving. When OI is shrinking (positions
closing), trend exhaustion is likely.

Signal definition:
    oi_return = (oi_t - oi_{t - window_days}) / oi_{t - window_days}
    price_direction = sign(close_t - close_{t - window_days})
    signal = oi_return * price_direction

    signal > 0:  OI growing in the direction of the recent trend → continuation
    signal < 0:  OI growing against the trend (contrarian) or shrinking with trend

Forward return: (close_{t + horizon_days} - close_t) / close_t

Alternatively: raw OI change rate (unsigned) tested against unsigned return
magnitude is NOT the hypothesis — we are testing directional continuation only.

Falsification criteria (pre-registered — do not change after first data run):
    ρ(signal, fwd_return)  > +0.15   (Spearman)
    p-value                <  0.05
    Q5 − Q1                >  0.0
    Monotonicity ratio     ≥  0.75
    Fee-adjusted ρ         >  0.0

Data source: Binance Futures /fapi/v1/openInterest (snapshot at a point in
time) — requires multiple calls over a history window using the Binance OI
history endpoint.

Usage:
    python -m research.oi_momentum
    python -m research.oi_momentum --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --days 730 --window 7 --horizon-days 1
    python -m research.oi_momentum --window 14 --horizon-days 4
    python -m research.oi_momentum --window 3 --horizon-days 1

NOTE: Binance provides historical OI via /futures/data/openInterestHist with
period intervals (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d). We use 1d.
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

_MIN_PAIRS = 30


@dataclass
class OIRecord:
    ts_ms: int
    oi: float      # sumOpenInterest (in contracts / BTC equivalent)


@dataclass
class Bar:
    ts_ms: int
    close: float


@dataclass
class AnalysisResult:
    symbol: str
    n: int = 0
    window: int = 7
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


def fetch_oi_history(symbol: str, days: int) -> list[OIRecord]:
    """
    Fetch daily OI history via /futures/data/openInterestHist.

    Binance returns up to 500 data points per request. We paginate backwards
    from now to cover the full `days` window.
    """
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days + 30) * 24 * 3_600_000
    records: list[OIRecord] = []
    cursor = start_ms

    while cursor < now_ms:
        data = _get("/futures/data/openInterestHist", {
            "symbol": symbol,
            "period": "1d",
            "startTime": cursor,
            "endTime": now_ms,
            "limit": 500,
        })
        if not data:
            break
        for row in data:
            records.append(OIRecord(
                ts_ms=int(row["timestamp"]),
                oi=float(row["sumOpenInterest"]),
            ))
        if len(data) < 500:
            break
        cursor = records[-1].ts_ms + 1
        time.sleep(0.05)

    seen: set[int] = set()
    out: list[OIRecord] = []
    for r in sorted(records, key=lambda r: r.ts_ms):
        if r.ts_ms not in seen:
            seen.add(r.ts_ms)
            out.append(r)
    return out


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

def _snap(
    target_ms: int,
    sorted_ts: list[int],
    val_by_ts: dict[int, float],
    tol_ms: int = 2 * 24 * 3_600_000,
) -> Optional[float]:
    """Binary search: closest bar within tolerance."""
    lo, hi = 0, len(sorted_ts) - 1
    best_idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_ts[mid] <= target_ms:
            best_idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    best_dist, best_val = tol_ms + 1, None
    for idx in (best_idx, best_idx + 1):
        if 0 <= idx < len(sorted_ts):
            dist = abs(sorted_ts[idx] - target_ms)
            if dist < best_dist:
                v = val_by_ts.get(sorted_ts[idx])
                if v is not None:
                    best_dist, best_val = dist, v
    return best_val


def build_signal_return_pairs(
    oi_records: list[OIRecord],
    bars: list[Bar],
    window: int,
    horizon_days: int,
) -> tuple[list[float], list[float]]:
    """
    Return (signals, forward_returns) sampled at horizon_days intervals.

    signal = oi_return * price_direction
    oi_return = (oi_t - oi_{t - window_days}) / oi_{t - window_days}
    price_direction = sign(close_t - close_{t - window_days})
    """
    if not oi_records or not bars:
        return [], []

    window_ms = window * 24 * 3_600_000
    horizon_ms = horizon_days * 24 * 3_600_000
    step_ms = horizon_ms

    oi_sorted_ts = sorted(r.ts_ms for r in oi_records)
    oi_by_ts: dict[int, float] = {r.ts_ms: r.oi for r in oi_records}

    bar_sorted_ts = sorted(b.ts_ms for b in bars)
    close_by_ts: dict[int, float] = {b.ts_ms: b.close for b in bars}

    first_eligible = oi_sorted_ts[0] + window_ms
    last_eligible = min(oi_sorted_ts[-1], bar_sorted_ts[-1]) - horizon_ms

    signals: list[float] = []
    returns: list[float] = []

    t = first_eligible
    while t <= last_eligible:
        oi_now = _snap(t, oi_sorted_ts, oi_by_ts)
        oi_back = _snap(t - window_ms, oi_sorted_ts, oi_by_ts)
        px_now = _snap(t, bar_sorted_ts, close_by_ts)
        px_back = _snap(t - window_ms, bar_sorted_ts, close_by_ts)
        px_fwd = _snap(t + horizon_ms, bar_sorted_ts, close_by_ts)

        if any(v is None for v in (oi_now, oi_back, px_now, px_back, px_fwd)):
            t += step_ms
            continue
        if oi_back <= 0 or px_back <= 0 or px_now <= 0:
            t += step_ms
            continue

        oi_return = (oi_now - oi_back) / oi_back
        price_direction = 1.0 if px_now > px_back else (-1.0 if px_now < px_back else 0.0)
        signal = oi_return * price_direction

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

    print("  Fetching OI history ...")
    oi_records = fetch_oi_history(symbol, days)
    print(f"  {len(oi_records)} OI records")

    print("  Fetching daily closes ...")
    bars = fetch_daily_closes(symbol, days)
    print(f"  {len(bars)} daily bars")

    signals, returns = build_signal_return_pairs(oi_records, bars, window, horizon_days)
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
        description="Hypothesis G: open interest change rate causality test"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--days", type=int, default=730,
                        help="Calendar days of history (default 730)")
    parser.add_argument("--window", type=int, default=7,
                        help="OI lookback window in days (default 7)")
    parser.add_argument("--horizon-days", type=int, default=1,
                        help="Forward return horizon in days (default 1)")
    parser.add_argument("--out", default="data/hypothesis_g_results.json",
                        help="Path to write JSON results")
    args = parser.parse_args()

    print("\nFALSIFICATION CRITERIA (pre-registered):")
    print(f"  ρ(signal, fwd_return_{args.horizon_days}d)  > {_MIN_RHO}")
    print(f"  p-value                             < {_MAX_PVALUE}")
    print(f"  Q5 − Q1                             > {_MIN_Q5_MINUS_Q1}")
    print(f"  Monotonicity ratio                  ≥ {_MIN_MONOTONICITY}")
    print(f"  Fee-adjusted ρ                      > 0.0")
    print(f"\n  Signal: oi_return({args.window}d) × sign(price_return({args.window}d))")
    print(f"  Rationale: OI growing with trend → new committed positions → continuation")
    print(f"\n  NOTE: If Hypothesis G fails, research program halts.")
    print(f"  Five hypotheses with zero passes = no causal signal found in this universe.")

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
