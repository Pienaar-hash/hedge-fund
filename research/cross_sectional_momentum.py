"""
Hypothesis E — Cross-Sectional Momentum: causality test.

Statement: Among BTC/ETH/SOL, the asset with the highest past-N-week return
continues to outperform the asset with the lowest past-N-week return over the
next M weeks.  The mechanism is relative mispricing / underreaction to news.
Unlike Hypothesis D (time-series), common market beta cancels here.

Signal definition:
    At each sample time t, rank assets by past_return(t, lookback_weeks).
    signal_i = (rank_i − 1) / (n_assets − 1)   [0 = worst, 1 = best]
    forward_return_i = (close_{t+horizon} − close_t) / close_t

Observations are pooled across assets and sampled every horizon_weeks×7 days
(non-overlapping) so forward returns are independent within each asset.

Falsification criteria (pre-registered — do not change after first data run):
    ρ(signal, fwd_return)   > +0.15   (Spearman, pooled across assets)
    p-value                 <  0.05
    Top − Bottom spread     >  0.0    (top-tercile mean fwd return > bottom)
    Monotonicity            ≥  0.75   (Q1 < Q2 < Q3 across signal terciles)
    Fee-adjusted ρ          >  0.0

Usage:
    python -m research.cross_sectional_momentum
    python -m research.cross_sectional_momentum --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --lookback-weeks 4 --horizon-weeks 1 --days 1500
    python -m research.cross_sectional_momentum --lookback-weeks 1 --horizon-weeks 1
    python -m research.cross_sectional_momentum --lookback-weeks 12 --horizon-weeks 4
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
_MIN_SPREAD = 0.0        # top − bottom tercile mean forward return
_MIN_MONOTONICITY = 0.75

# Rolling vol-window (not used in raw signal; kept for potential extension)
_VOL_WINDOW_WEEKS = 26
_MIN_PAIRS = 30          # minimum per-asset pairs before accepting a run


@dataclass
class Bar:
    ts_ms: int
    close: float


@dataclass
class AnalysisResult:
    symbols: list[str] = field(default_factory=list)
    n_total: int = 0          # pooled (signal, return) pairs
    n_per_asset: int = 0      # pairs per asset (approx)
    lookback_weeks: int = 4
    horizon_weeks: int = 1
    rho: float = 0.0
    pvalue: float = 1.0
    spread: float = 0.0       # top_tercile_mean − bottom_tercile_mean
    monotonicity: float = 0.0
    rho_fee_adj: float = 0.0
    tercile_means: list[float] = field(default_factory=list)
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
    """Return daily close prices for the past `days` calendar days (plus buffer)."""
    now_ms = int(time.time() * 1000)
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

def _snap_close(
    target_ms: int,
    sorted_ts: list[int],
    close_by_ts: dict[int, float],
    tolerance_ms: int = 2 * 24 * 3_600_000,
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
    best_dist, best_close = tolerance_ms + 1, None
    for idx in (best_idx, best_idx + 1):
        if 0 <= idx < len(sorted_ts):
            dist = abs(sorted_ts[idx] - target_ms)
            if dist < best_dist:
                c = close_by_ts.get(sorted_ts[idx])
                if c is not None:
                    best_dist, best_close = dist, c
    return best_close


def build_cross_sectional_pairs(
    bars_by_symbol: dict[str, list[Bar]],
    lookback_weeks: int,
    horizon_weeks: int,
) -> tuple[list[float], list[float], list[str]]:
    """
    Return (signals, forward_returns, symbol_labels) pooled across all assets.

    At each sample time t:
      - Compute past return for every asset.
      - Rank assets: signal = (rank − 1) / (n_assets − 1)  [0=worst, 1=best]
      - Record forward return for each asset.

    Sample times advance by horizon_weeks * 7 days (non-overlapping).
    Times where any asset lacks lookback/forward data are skipped entirely
    to keep the cross-section balanced.
    """
    n_assets = len(bars_by_symbol)
    if n_assets < 2:
        return [], [], []

    lookback_ms = lookback_weeks * 7 * 24 * 3_600_000
    horizon_ms = horizon_weeks * 7 * 24 * 3_600_000
    step_ms = horizon_ms

    # Pre-build per-symbol sorted-ts + close-lookup
    sym_data: dict[str, tuple[list[int], dict[int, float]]] = {}
    for sym, bars in bars_by_symbol.items():
        sts = sorted(b.ts_ms for b in bars)
        cbt = {b.ts_ms: b.close for b in bars}
        sym_data[sym] = (sts, cbt)

    # Determine the common eligible range across all symbols
    earliest = max(bars[0].ts_ms for bars in bars_by_symbol.values())
    latest = min(bars[-1].ts_ms for bars in bars_by_symbol.values())

    first_eligible = earliest + lookback_ms + _VOL_WINDOW_WEEKS * 7 * 24 * 3_600_000
    last_eligible = latest - horizon_ms

    signals: list[float] = []
    returns: list[float] = []
    labels: list[str] = []

    t = first_eligible
    symbols = list(bars_by_symbol.keys())

    while t <= last_eligible:
        past_returns: list[float] = []
        fwd_returns: list[float] = []
        valid = True

        for sym in symbols:
            sts, cbt = sym_data[sym]
            px_now = _snap_close(t, sts, cbt)
            px_back = _snap_close(t - lookback_ms, sts, cbt)
            px_fwd = _snap_close(t + horizon_ms, sts, cbt)

            if px_now is None or px_back is None or px_fwd is None:
                valid = False
                break
            if px_back <= 0 or px_now <= 0:
                valid = False
                break

            past_returns.append((px_now - px_back) / px_back)
            fwd_returns.append((px_fwd - px_now) / px_now)

        if valid and len(past_returns) == n_assets:
            # Rank assets by past return: 0 = worst, 1 = best
            raw_ranks = _rank(past_returns)  # 1-based
            for i, sym in enumerate(symbols):
                signal = (raw_ranks[i] - 1.0) / (n_assets - 1.0)
                signals.append(signal)
                returns.append(fwd_returns[i])
                labels.append(sym)

        t += step_ms

    return signals, returns, labels


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse(
    symbols: list[str],
    days: int,
    lookback_weeks: int,
    horizon_weeks: int,
) -> AnalysisResult:
    result = AnalysisResult(
        symbols=symbols,
        lookback_weeks=lookback_weeks,
        horizon_weeks=horizon_weeks,
    )

    print(f"\n{'='*60}")
    print(f"  Cross-sectional momentum")
    print(f"  symbols={symbols}")
    print(f"  {days}d  |  lookback={lookback_weeks}w  |  fwd={horizon_weeks}w")
    print(f"{'='*60}")

    bars_by_symbol: dict[str, list[Bar]] = {}
    for sym in symbols:
        print(f"  Fetching {sym} ...")
        bars = fetch_daily_closes(sym, days)
        print(f"    {len(bars)} daily bars")
        bars_by_symbol[sym] = bars

    signals, returns, labels = build_cross_sectional_pairs(
        bars_by_symbol, lookback_weeks, horizon_weeks
    )

    n_total = len(signals)
    n_per_asset = n_total // len(symbols) if symbols else 0
    result.n_total = n_total
    result.n_per_asset = n_per_asset
    print(f"  Total pooled pairs: {n_total}  ({n_per_asset} per asset)")

    if n_per_asset < _MIN_PAIRS:
        result.verdict = "INSUFFICIENT_PAIRS"
        result.error = f"only {n_per_asset} pairs per asset (need {_MIN_PAIRS})"
        return result

    # Spearman ρ (pooled)
    result.rho, result.pvalue = spearman_rho(signals, returns)
    print(f"  ρ = {result.rho:+.4f}   p = {result.pvalue:.4f}")

    # Tercile analysis (3 groups: bottom, middle, top signal)
    n = n_total
    paired = sorted(zip(signals, returns), key=lambda t: t[0])
    t_size = n // 3
    tercile_means = []
    for q in range(3):
        s = q * t_size
        e = (q + 1) * t_size if q < 2 else n
        tercile_means.append(statistics.mean(r for _, r in paired[s:e]))
    result.tercile_means = tercile_means
    result.spread = tercile_means[2] - tercile_means[0]
    result.monotonicity = sum(
        1 for i in range(2) if tercile_means[i] <= tercile_means[i + 1]
    ) / 2.0

    tstr = "  ".join(f"{v:+.5f}" for v in tercile_means)
    print(f"  Terciles T1→T3: {tstr}")
    print(f"  Spread (T3−T1) = {result.spread:+.5f}   mono = {result.monotonicity:.2f}")

    # Fee-adjusted ρ
    fee_adj = [r - _ROUND_TRIP_FEE for r in returns]
    result.rho_fee_adj, _ = spearman_rho(signals, fee_adj)
    print(f"  Fee-adj ρ = {result.rho_fee_adj:+.4f}")

    # Verdict
    result.criteria = {
        "rho_above_0.15": result.rho > _MIN_RHO,
        "pvalue_below_0.05": result.pvalue < _MAX_PVALUE,
        "spread_positive": result.spread > _MIN_SPREAD,
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
        description="Hypothesis E: cross-sectional momentum causality test"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--days", type=int, default=1500,
                        help="Calendar days of history (default 1500 ≈ 4 years)")
    parser.add_argument("--lookback-weeks", type=int, default=4,
                        help="Momentum lookback in weeks (default 4)")
    parser.add_argument("--horizon-weeks", type=int, default=1,
                        help="Forward return horizon in weeks (default 1)")
    parser.add_argument("--out", default="data/hypothesis_e_results.json",
                        help="Path to write JSON results")
    args = parser.parse_args()

    print("\nFALSIFICATION CRITERIA (pre-registered):")
    print(f"  ρ(signal, fwd_return_{args.horizon_weeks}w)  > {_MIN_RHO}   (pooled Spearman)")
    print(f"  p-value                             < {_MAX_PVALUE}")
    print(f"  Spread (top − bottom tercile)       > {_MIN_SPREAD}")
    print(f"  Monotonicity                        ≥ {_MIN_MONOTONICITY}")
    print(f"  Fee-adjusted ρ                      > 0.0")
    print(f"\n  Signal: cross-sectional rank(past_{args.lookback_weeks}w_return) in [{args.symbols}]")
    print(f"  Sampling: non-overlapping, every {args.horizon_weeks} week(s)")
    print(f"  Rationale: relative winners continue to outperform relative losers")

    try:
        r = analyse(args.symbols, args.days, args.lookback_weeks, args.horizon_weeks)
    except Exception as exc:
        r = AnalysisResult(symbols=args.symbols, verdict="ERROR", error=str(exc))
        print(f"\nERROR: {exc}")

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"  ρ={r.rho:+.4f}  p={r.pvalue:.4f}  n_total={r.n_total}  VERDICT: {r.verdict}")

    if r.verdict == "PASS":
        print("\nHYPOTHESIS SUPPORTED — proceed to paper trade design")
    elif r.verdict == "ERROR":
        print("\nERROR — fix data fetch issues before concluding")
    elif r.verdict == "FAIL":
        print("\nHYPOTHESIS NOT SUPPORTED — do not deploy live")
    else:
        print("\nCONDITIONAL SUPPORT — review individual criteria before proceeding")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "symbols": r.symbols,
            "n_total": r.n_total,
            "n_per_asset": r.n_per_asset,
            "lookback_weeks": r.lookback_weeks,
            "horizon_weeks": r.horizon_weeks,
            "rho": r.rho,
            "pvalue": r.pvalue,
            "spread": r.spread,
            "monotonicity": r.monotonicity,
            "rho_fee_adj": r.rho_fee_adj,
            "tercile_means": r.tercile_means,
            "criteria": r.criteria,
            "verdict": r.verdict,
            "error": r.error,
        }, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
