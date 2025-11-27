#!/usr/bin/env python3
"""
Strategy Parameter Optimizer (v1.0)

Grid search and optimization for trading strategy parameters.
Supports parallel execution and exports results for analysis.

Usage:
    # Optimize RSI strategy
    python scripts/optimize.py --strategy rsi --symbol BTCUSDT --interval 15m --days 30

    # Optimize with custom parameter grid
    python scripts/optimize.py --strategy momentum --fast 5,8,12 --slow 20,26,50

    # Multi-symbol optimization
    python scripts/optimize.py --strategy rsi --symbols BTCUSDT,ETHUSDT --interval 1h

    # Export results to CSV
    python scripts/optimize.py --strategy rsi --output results.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest import (
    Backtest,
    BacktestConfig,
    BacktestResult,
    Strategy,
    RSIStrategy,
    MomentumStrategy,
    MACDStrategy,
    BollingerBandStrategy,
    BreakoutStrategy,
)
from utils.ohlcv_loader import OHLCVLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("optimizer")


# --- Parameter Grids ---
DEFAULT_GRIDS = {
    "rsi": {
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80],
        "period": [7, 14, 21],
    },
    "momentum": {
        "fast_period": [5, 8, 12, 15],
        "slow_period": [20, 26, 30, 50],
    },
    "macd": {
        "fast": [8, 12, 15],
        "slow": [21, 26, 30],
        "signal": [7, 9, 12],
    },
    "bollinger": {
        "period": [15, 20, 25, 30],
        "std_dev": [1.5, 2.0, 2.5, 3.0],
    },
    "breakout": {
        "period": [10, 15, 20, 25, 30, 40],
    },
}


# --- Enhanced Strategies with Configurable Parameters ---
class OptimizableRSI(Strategy):
    """RSI strategy with configurable parameters."""
    
    name = "RSI_Optimized"
    
    def __init__(self, oversold: int = 30, overbought: int = 70, period: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period
        self.name = f"RSI({period},{oversold},{overbought})"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate RSI with custom period
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss
        df["RSI_custom"] = 100 - (100 / (1 + rs))
        
        df["signal"] = 0
        
        # Buy when RSI crosses above oversold
        df.loc[
            (df["RSI_custom"] > self.oversold) & 
            (df["RSI_custom"].shift(1) <= self.oversold), 
            "signal"
        ] = 1
        
        # Sell when RSI crosses below overbought
        df.loc[
            (df["RSI_custom"] < self.overbought) & 
            (df["RSI_custom"].shift(1) >= self.overbought), 
            "signal"
        ] = -1
        
        return df


class OptimizableMomentum(Strategy):
    """Momentum strategy with configurable EMA periods."""
    
    name = "Momentum_Optimized"
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"Momentum({fast_period},{slow_period})"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        fast_ema = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        
        df["signal"] = 0
        
        # Long when fast > slow
        df.loc[(fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1)), "signal"] = 1
        
        # Short when fast < slow
        df.loc[(fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1)), "signal"] = -1
        
        return df


class OptimizableMACD(Strategy):
    """MACD strategy with configurable periods."""
    
    name = "MACD_Optimized"
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.name = f"MACD({fast},{slow},{signal})"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        fast_ema = df["close"].ewm(span=self.fast, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        df["signal"] = 0
        
        # Long when MACD crosses above signal
        df.loc[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), "signal"] = 1
        
        # Short when MACD crosses below signal
        df.loc[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), "signal"] = -1
        
        return df


class OptimizableBollinger(Strategy):
    """Bollinger Band strategy with configurable parameters."""
    
    name = "Bollinger_Optimized"
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = f"BB({period},{std_dev})"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        middle = df["close"].rolling(self.period).mean()
        std = df["close"].rolling(self.period).std()
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)
        
        df["signal"] = 0
        
        # Long when price touches lower band and bounces
        df.loc[(df["close"] > lower) & (df["close"].shift(1) <= lower.shift(1)), "signal"] = 1
        
        # Short when price touches upper band and reverses
        df.loc[(df["close"] < upper) & (df["close"].shift(1) >= upper.shift(1)), "signal"] = -1
        
        return df


class OptimizableBreakout(Strategy):
    """Breakout strategy with configurable period."""
    
    name = "Breakout_Optimized"
    
    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"Breakout({period})"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        upper = df["high"].rolling(self.period).max()
        lower = df["low"].rolling(self.period).min()
        
        df["signal"] = 0
        
        df.loc[df["close"] > upper.shift(1), "signal"] = 1
        df.loc[df["close"] < lower.shift(1), "signal"] = -1
        
        return df


# Strategy factory
OPTIMIZABLE_STRATEGIES = {
    "rsi": OptimizableRSI,
    "momentum": OptimizableMomentum,
    "macd": OptimizableMACD,
    "bollinger": OptimizableBollinger,
    "breakout": OptimizableBreakout,
}


@dataclass
class OptimizationResult:
    """Single optimization run result."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    strategy_name: str
    symbol: str
    interval: str


def run_single_backtest(
    strategy_name: str,
    params: Dict[str, Any],
    symbol: str,
    interval: str,
    days: int,
    config: BacktestConfig,
    df: Optional[pd.DataFrame] = None,
) -> OptimizationResult:
    """Run a single backtest with given parameters."""
    try:
        strategy_class = OPTIMIZABLE_STRATEGIES[strategy_name]
        strategy = strategy_class(**params)
        
        bt = Backtest(
            strategy=strategy,
            symbol=symbol,
            interval=interval,
            days=days,
            config=config,
            df=df,
        )
        result = bt.run()
        
        return OptimizationResult(
            params=params,
            metrics={
                "total_return_pct": result.total_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "expectancy": result.expectancy,
                "net_pnl": result.net_pnl,
            },
            strategy_name=strategy.name,
            symbol=symbol,
            interval=interval,
        )
    except Exception as e:
        LOG.debug(f"Backtest failed for {params}: {e}")
        return OptimizationResult(
            params=params,
            metrics={
                "total_return_pct": -999,
                "sharpe_ratio": -999,
                "sortino_ratio": -999,
                "max_drawdown_pct": 100,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
                "expectancy": -999,
                "net_pnl": -999,
            },
            strategy_name=f"{strategy_name}_ERROR",
            symbol=symbol,
            interval=interval,
        )


def generate_param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from a grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def filter_valid_combinations(
    strategy_name: str,
    combinations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter out invalid parameter combinations."""
    valid = []
    
    for params in combinations:
        # Strategy-specific validation
        if strategy_name == "momentum":
            if params.get("fast_period", 12) >= params.get("slow_period", 26):
                continue  # Fast must be < slow
        
        if strategy_name == "macd":
            if params.get("fast", 12) >= params.get("slow", 26):
                continue
        
        if strategy_name == "rsi":
            if params.get("oversold", 30) >= params.get("overbought", 70):
                continue
        
        valid.append(params)
    
    return valid


class ParameterOptimizer:
    """Main optimizer class."""
    
    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        interval: str,
        days: int,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        config: Optional[BacktestConfig] = None,
        n_jobs: int = 1,
    ):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.interval = interval
        self.days = days
        self.param_grid = param_grid or DEFAULT_GRIDS.get(strategy_name, {})
        self.config = config or BacktestConfig()
        self.n_jobs = n_jobs
        
        self.results: List[OptimizationResult] = []
        self.loader = OHLCVLoader()
    
    def run(self) -> pd.DataFrame:
        """Run optimization across all parameter combinations."""
        combinations = generate_param_combinations(self.param_grid)
        combinations = filter_valid_combinations(self.strategy_name, combinations)
        
        total_runs = len(combinations) * len(self.symbols)
        LOG.info(f"Starting optimization: {self.strategy_name}")
        LOG.info(f"Parameter grid: {self.param_grid}")
        LOG.info(f"Valid combinations: {len(combinations)}")
        LOG.info(f"Symbols: {self.symbols}")
        LOG.info(f"Total backtests: {total_runs}")
        
        self.results = []
        completed = 0
        
        # Pre-load data for each symbol
        data_cache = {}
        for symbol in self.symbols:
            df = self.loader.load(symbol, self.interval, days=self.days)
            if not df.empty:
                data_cache[symbol] = df
        
        # Run backtests
        for symbol in self.symbols:
            df = data_cache.get(symbol)
            if df is None or df.empty:
                LOG.warning(f"No data for {symbol}, skipping")
                continue
            
            for params in combinations:
                result = run_single_backtest(
                    strategy_name=self.strategy_name,
                    params=params,
                    symbol=symbol,
                    interval=self.interval,
                    days=self.days,
                    config=self.config,
                    df=df,
                )
                self.results.append(result)
                completed += 1
                
                if completed % 50 == 0:
                    LOG.info(f"Progress: {completed}/{total_runs} ({100*completed/total_runs:.1f}%)")
        
        LOG.info(f"Optimization complete: {completed} backtests")
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for r in self.results:
            row = {
                "symbol": r.symbol,
                "interval": r.interval,
                "strategy": r.strategy_name,
                **r.params,
                **r.metrics,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    
    def best_params(
        self,
        metric: str = "sharpe_ratio",
        min_trades: int = 10,
    ) -> Dict[str, Any]:
        """Get best parameters based on a metric."""
        df = self.to_dataframe()
        
        # Filter by minimum trades
        df = df[df["total_trades"] >= min_trades]
        
        if df.empty:
            return {}
        
        best_row = df.loc[df[metric].idxmax()]
        return {
            "params": {k: best_row[k] for k in self.param_grid.keys()},
            "metrics": {
                "sharpe_ratio": best_row["sharpe_ratio"],
                "total_return_pct": best_row["total_return_pct"],
                "max_drawdown_pct": best_row["max_drawdown_pct"],
                "win_rate": best_row["win_rate"],
                "profit_factor": best_row["profit_factor"],
                "total_trades": best_row["total_trades"],
            },
            "symbol": best_row["symbol"],
            "strategy": best_row["strategy"],
        }
    
    def summary(self) -> str:
        """Generate optimization summary."""
        df = self.to_dataframe()
        
        lines = [
            f"\n{'='*80}",
            f"OPTIMIZATION RESULTS: {self.strategy_name.upper()}",
            f"{'='*80}",
            f"Symbols: {', '.join(self.symbols)}",
            f"Interval: {self.interval} | Days: {self.days}",
            f"Total combinations tested: {len(self.results)}",
            f"{'='*80}",
            "",
            "TOP 10 PARAMETER SETS (by Sharpe Ratio):",
            "-" * 80,
        ]
        
        # Filter for valid results
        valid = df[df["total_trades"] >= 5].head(10)
        
        if valid.empty:
            lines.append("No valid results (min 5 trades required)")
        else:
            # Format columns based on strategy
            param_cols = list(self.param_grid.keys())
            display_cols = ["symbol", "strategy"] + param_cols + [
                "sharpe_ratio", "total_return_pct", "max_drawdown_pct", 
                "win_rate", "profit_factor", "total_trades"
            ]
            
            for idx, row in valid[display_cols].iterrows():
                param_str = ", ".join([f"{k}={row[k]}" for k in param_cols])
                lines.append(
                    f"{row['symbol']:8} | {param_str:30} | "
                    f"Sharpe: {row['sharpe_ratio']:>6.2f} | "
                    f"Return: {row['total_return_pct']:>6.1f}% | "
                    f"DD: {row['max_drawdown_pct']:>5.1f}% | "
                    f"WR: {row['win_rate']:>5.1f}% | "
                    f"PF: {row['profit_factor']:>5.2f} | "
                    f"Trades: {int(row['total_trades']):>3}"
                )
        
        lines.extend([
            "",
            "WORST 5 PARAMETER SETS:",
            "-" * 80,
        ])
        
        worst = df[df["total_trades"] >= 5].tail(5)
        for idx, row in worst[display_cols].iterrows():
            param_str = ", ".join([f"{k}={row[k]}" for k in param_cols])
            lines.append(
                f"{row['symbol']:8} | {param_str:30} | "
                f"Sharpe: {row['sharpe_ratio']:>6.2f} | "
                f"Return: {row['total_return_pct']:>6.1f}%"
            )
        
        # Best overall
        best = self.best_params()
        if best:
            lines.extend([
                "",
                "=" * 80,
                "RECOMMENDED PARAMETERS:",
                "=" * 80,
                f"Strategy: {best['strategy']}",
                f"Symbol: {best['symbol']}",
                f"Parameters: {best['params']}",
                f"Sharpe Ratio: {best['metrics']['sharpe_ratio']:.3f}",
                f"Total Return: {best['metrics']['total_return_pct']:.2f}%",
                f"Max Drawdown: {best['metrics']['max_drawdown_pct']:.2f}%",
                f"Win Rate: {best['metrics']['win_rate']:.1f}%",
                f"Profit Factor: {best['metrics']['profit_factor']:.3f}",
                f"Total Trades: {int(best['metrics']['total_trades'])}",
                "=" * 80,
            ])
        
        return "\n".join(lines)


def run_comprehensive_optimization(
    symbols: List[str],
    interval: str,
    days: int,
) -> pd.DataFrame:
    """Run optimization for all strategies and combine results."""
    all_results = []
    
    for strategy_name in OPTIMIZABLE_STRATEGIES.keys():
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Optimizing: {strategy_name.upper()}")
        LOG.info(f"{'='*60}")
        
        optimizer = ParameterOptimizer(
            strategy_name=strategy_name,
            symbols=symbols,
            interval=interval,
            days=days,
        )
        df = optimizer.run()
        df["base_strategy"] = strategy_name
        all_results.append(df)
    
    combined = pd.concat(all_results, ignore_index=True)
    return combined.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Strategy Parameter Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="rsi",
        help=f"Strategy to optimize: {', '.join(OPTIMIZABLE_STRATEGIES.keys())}, or 'all'",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        help="Candlestick interval (default: 15m)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output best params as JSON",
    )
    
    # Custom parameter overrides
    parser.add_argument("--oversold", type=str, default="", help="RSI oversold levels (e.g., 20,25,30)")
    parser.add_argument("--overbought", type=str, default="", help="RSI overbought levels (e.g., 70,75,80)")
    parser.add_argument("--period", type=str, default="", help="Period values (e.g., 14,21,28)")
    parser.add_argument("--fast", type=str, default="", help="Fast EMA periods")
    parser.add_argument("--slow", type=str, default="", help="Slow EMA periods")
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = [args.symbol.upper()]
    
    # Run all strategies
    if args.strategy == "all":
        df = run_comprehensive_optimization(symbols, args.interval, args.days)
        print("\n" + "=" * 80)
        print("COMPREHENSIVE OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"\nTop 20 parameter sets across all strategies:\n")
        display_cols = ["base_strategy", "symbol", "strategy", "sharpe_ratio", 
                       "total_return_pct", "max_drawdown_pct", "win_rate", 
                       "profit_factor", "total_trades"]
        print(df[display_cols].head(20).to_string(index=False))
        
        if args.output:
            df.to_csv(args.output, index=False)
            LOG.info(f"Results saved to {args.output}")
        return
    
    # Validate strategy
    if args.strategy not in OPTIMIZABLE_STRATEGIES:
        LOG.error(f"Unknown strategy: {args.strategy}")
        sys.exit(1)
    
    # Build custom parameter grid if provided
    param_grid = DEFAULT_GRIDS.get(args.strategy, {}).copy()
    
    if args.oversold:
        param_grid["oversold"] = [int(x) for x in args.oversold.split(",")]
    if args.overbought:
        param_grid["overbought"] = [int(x) for x in args.overbought.split(",")]
    if args.period:
        param_grid["period"] = [int(x) for x in args.period.split(",")]
    if args.fast:
        param_grid["fast_period"] = [int(x) for x in args.fast.split(",")]
    if args.slow:
        param_grid["slow_period"] = [int(x) for x in args.slow.split(",")]
    
    # Run optimization
    optimizer = ParameterOptimizer(
        strategy_name=args.strategy,
        symbols=symbols,
        interval=args.interval,
        days=args.days,
        param_grid=param_grid,
    )
    
    df = optimizer.run()
    
    if args.json:
        best = optimizer.best_params()
        print(json.dumps(best, indent=2))
    else:
        print(optimizer.summary())
    
    if args.output:
        df.to_csv(args.output, index=False)
        LOG.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
