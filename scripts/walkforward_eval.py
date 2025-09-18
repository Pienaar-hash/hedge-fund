#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.perf_metrics import compute_performance_stats


def _load_returns(equity_path: Path | None, returns_path: Path | None) -> Tuple[pd.Series, pd.Series]:
    if returns_path and returns_path.exists():
        df = pd.read_csv(returns_path, parse_dates=[0], index_col=0)
        if df.shape[1] > 1:
            series = df.iloc[:, 0]
        else:
            series = df.iloc[:, 0]
        equity = (1.0 + series).cumprod()
        return equity, series

    if not equity_path or not equity_path.exists():
        raise FileNotFoundError("Provide either --equity or --returns with a valid path")

    df = pd.read_csv(equity_path, parse_dates=[0], index_col=0)
    col = df.columns[0]
    series = df[col]
    returns = series.pct_change().dropna()
    return series.loc[returns.index], returns


def _walk_forward(
    equity: pd.Series,
    returns: pd.Series,
    train: int,
    test: int,
    periods_per_year: float,
    sr_ref: float,
) -> pd.DataFrame:
    records: List[Dict[str, float | str]] = []
    idx = 0
    total = len(returns)
    while idx + train + test <= total:
        train_slice = returns.iloc[idx : idx + train]
        test_slice = returns.iloc[idx + train : idx + train + test]
        eq_slice = equity.loc[test_slice.index]
        stats = compute_performance_stats(
            returns=test_slice.values,
            equity=eq_slice.values,
            periods_per_year=periods_per_year,
            sr_ref=sr_ref,
        )
        records.append(
            {
                "train_start": train_slice.index[0].isoformat(),
                "train_end": train_slice.index[-1].isoformat(),
                "test_start": test_slice.index[0].isoformat(),
                "test_end": test_slice.index[-1].isoformat(),
                "sharpe": stats.sharpe,
                "calmar": stats.calmar,
                "cagr": stats.cagr,
                "max_drawdown": stats.max_drawdown,
                "psr": stats.psr,
            }
        )
        idx += test
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation harness")
    parser.add_argument("--equity", type=Path, help="CSV with timestamp + equity columns", default=None)
    parser.add_argument("--returns", type=Path, help="CSV with timestamp + returns columns", default=None)
    parser.add_argument("--train", type=int, default=500, help="In-sample window (rows)")
    parser.add_argument("--test", type=int, default=100, help="Out-of-sample window (rows)")
    parser.add_argument("--freq", type=float, default=252.0, help="Periods per year for Sharpe")
    parser.add_argument("--sr-ref", type=float, default=0.0, help="Sharpe ratio hurdle for PSR")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/walkforward_metrics.csv"),
        help="Where to write walk-forward metrics",
    )
    args = parser.parse_args()

    equity, returns = _load_returns(args.equity, args.returns)
    wf = _walk_forward(equity, returns, args.train, args.test, args.freq, args.sr_ref)
    if wf.empty:
        print("No walk-forward windows produced. Check window sizes.")
        return 1

    wf.to_csv(args.output, index=False)
    summary = {
        "windows": len(wf),
        "avg_sharpe": float(wf["sharpe"].mean()),
        "median_sharpe": float(wf["sharpe"].median()),
        "avg_calmar": float(wf["calmar"].mean()),
        "avg_psr": float(wf["psr"].mean()),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Walk-forward metrics saved to", args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
