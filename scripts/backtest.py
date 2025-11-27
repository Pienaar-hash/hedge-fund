#!/usr/bin/env python3
"""
Simple Backtest Framework for Hedge Fund Strategies (v1.0)

A lightweight, event-driven backtesting engine using collected OHLCV data.
Supports vectorized and bar-by-bar strategy execution with realistic fee modeling.

Usage:
    # Run built-in RSI strategy
    python scripts/backtest.py --strategy rsi --symbol BTCUSDT --interval 15m --days 30

    # Run momentum strategy with custom params
    python scripts/backtest.py --strategy momentum --symbols BTCUSDT,ETHUSDT --days 14

    # Compare multiple strategies
    python scripts/backtest.py --compare rsi,momentum,macd --symbol BTCUSDT --days 30

Example Custom Strategy:
    from scripts.backtest import Backtest, Strategy

    class MyStrategy(Strategy):
        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['close'] > df['close'].shift(1), 'signal'] = 1  # Buy on up move
            df.loc[df['close'] < df['close'].shift(1), 'signal'] = -1  # Sell on down move
            return df

    bt = Backtest(strategy=MyStrategy(), symbol="BTCUSDT", interval="15m", days=30)
    results = bt.run()
    bt.plot()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ohlcv_loader import OHLCVLoader, add_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("backtest")


# --- Configuration ---
@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""
    initial_capital: float = 10_000.0
    leverage: float = 1.0
    maker_fee: float = 0.0002  # 2 bps
    taker_fee: float = 0.0004  # 4 bps
    slippage: float = 0.0001   # 1 bp average slippage
    use_maker: bool = False    # Assume taker by default (conservative)
    position_size: float = 1.0 # Fraction of capital per trade
    max_positions: int = 1     # Max concurrent positions
    stop_loss: Optional[float] = None  # e.g., 0.02 for 2% stop
    take_profit: Optional[float] = None  # e.g., 0.03 for 3% TP


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: Optional[float]
    size: float  # In base currency
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Backtest results container."""
    # Core metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Position metrics
    avg_trade_duration: float = 0.0  # in hours
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Fee impact
    total_fees: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    
    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    
    # Metadata
    strategy_name: str = ""
    symbol: str = ""
    interval: str = ""
    start_date: str = ""
    end_date: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding time series)."""
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "period": f"{self.start_date} to {self.end_date}",
            "total_return": round(self.total_return, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy": round(self.expectancy, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "total_fees": round(self.total_fees, 2),
            "net_pnl": round(self.net_pnl, 2),
        }

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"\n{'='*60}",
            f"BACKTEST RESULTS: {self.strategy_name}",
            f"{'='*60}",
            f"Symbol: {self.symbol} | Interval: {self.interval}",
            f"Period: {self.start_date} to {self.end_date}",
            f"{'-'*60}",
            f"Total Return:     ${self.total_return:>10,.2f} ({self.total_return_pct:>6.2f}%)",
            f"Sharpe Ratio:     {self.sharpe_ratio:>10.3f}",
            f"Sortino Ratio:    {self.sortino_ratio:>10.3f}",
            f"Max Drawdown:     {self.max_drawdown_pct:>10.2f}%",
            f"{'-'*60}",
            f"Total Trades:     {self.total_trades:>10}",
            f"Win Rate:         {self.win_rate:>10.1f}%",
            f"Profit Factor:    {self.profit_factor:>10.3f}",
            f"Expectancy:       ${self.expectancy:>10.4f}",
            f"Avg Win:          ${self.avg_win:>10.2f}",
            f"Avg Loss:         ${self.avg_loss:>10.2f}",
            f"{'-'*60}",
            f"Gross PnL:        ${self.gross_pnl:>10,.2f}",
            f"Total Fees:       ${self.total_fees:>10,.2f}",
            f"Net PnL:          ${self.net_pnl:>10,.2f}",
            f"{'='*60}\n",
        ]
        return "\n".join(lines)


# --- Strategy Base Class ---
class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    name: str = "BaseStrategy"
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Must add a 'signal' column to df:
            1 = Long entry
           -1 = Short entry
            0 = No position / exit
        
        Args:
            df: DataFrame with OHLCV + indicators
        
        Returns:
            DataFrame with 'signal' column added
        """
        pass
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data before signal generation (add indicators, etc.)."""
        return add_indicators(df)


# --- Built-in Strategies ---
class RSIStrategy(Strategy):
    """RSI mean reversion strategy."""
    
    name = "RSI_MeanReversion"
    
    def __init__(self, oversold: int = 30, overbought: int = 70, period: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.period = period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        
        # Buy when RSI crosses above oversold
        df.loc[
            (df["RSI_14"] > self.oversold) & 
            (df["RSI_14"].shift(1) <= self.oversold), 
            "signal"
        ] = 1
        
        # Sell when RSI crosses below overbought
        df.loc[
            (df["RSI_14"] < self.overbought) & 
            (df["RSI_14"].shift(1) >= self.overbought), 
            "signal"
        ] = -1
        
        return df


class MomentumStrategy(Strategy):
    """Simple momentum/trend following strategy."""
    
    name = "Momentum"
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Use EMA crossover
        fast_ema = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        
        df["signal"] = 0
        
        # Long when fast > slow
        df.loc[(fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1)), "signal"] = 1
        
        # Short when fast < slow
        df.loc[(fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1)), "signal"] = -1
        
        return df


class MACDStrategy(Strategy):
    """MACD signal line crossover strategy."""
    
    name = "MACD_Crossover"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        
        # Long when MACD crosses above signal
        df.loc[
            (df["MACD"] > df["MACD_signal"]) & 
            (df["MACD"].shift(1) <= df["MACD_signal"].shift(1)), 
            "signal"
        ] = 1
        
        # Short when MACD crosses below signal
        df.loc[
            (df["MACD"] < df["MACD_signal"]) & 
            (df["MACD"].shift(1) >= df["MACD_signal"].shift(1)), 
            "signal"
        ] = -1
        
        return df


class BollingerBandStrategy(Strategy):
    """Bollinger Band mean reversion strategy."""
    
    name = "BollingerBand_MeanReversion"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        
        # Long when price touches lower band and bounces
        df.loc[
            (df["close"] > df["BB_lower"]) & 
            (df["close"].shift(1) <= df["BB_lower"].shift(1)), 
            "signal"
        ] = 1
        
        # Short when price touches upper band and reverses
        df.loc[
            (df["close"] < df["BB_upper"]) & 
            (df["close"].shift(1) >= df["BB_upper"].shift(1)), 
            "signal"
        ] = -1
        
        return df


class BreakoutStrategy(Strategy):
    """Donchian channel breakout strategy."""
    
    name = "Breakout"
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Donchian channels
        upper = df["high"].rolling(self.period).max()
        lower = df["low"].rolling(self.period).min()
        
        df["signal"] = 0
        
        # Long on breakout above upper channel
        df.loc[df["close"] > upper.shift(1), "signal"] = 1
        
        # Short on breakdown below lower channel
        df.loc[df["close"] < lower.shift(1), "signal"] = -1
        
        return df


# Strategy registry
STRATEGIES: Dict[str, Type[Strategy]] = {
    "rsi": RSIStrategy,
    "momentum": MomentumStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerBandStrategy,
    "breakout": BreakoutStrategy,
}


# --- Backtest Engine ---
class Backtest:
    """Main backtest engine."""
    
    def __init__(
        self,
        strategy: Strategy,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        days: int = 30,
        config: Optional[BacktestConfig] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.days = days
        self.config = config or BacktestConfig()
        self.loader = OHLCVLoader()
        
        # Allow passing pre-loaded data
        self._df = df
        
        # State
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.position: Optional[Trade] = None
        self.capital = self.config.initial_capital
    
    def _load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        if self._df is not None:
            df = self._df.copy()
        else:
            df = self.loader.load(self.symbol, self.interval, days=self.days)
        
        if df.empty:
            raise ValueError(f"No data available for {self.symbol} {self.interval}")
        
        # Add indicators
        df = self.strategy.prepare_data(df)
        
        # Generate signals
        df = self.strategy.generate_signals(df)
        
        if df is None:
            raise ValueError(f"Strategy {self.strategy.name} returned None - check generate_signals()")
        
        # Drop NaN from indicator warmup
        df = df.dropna(subset=["close", "signal"]).reset_index(drop=True)
        
        if len(df) < 50:
            raise ValueError(f"Insufficient data after indicator warmup: {len(df)} bars")
        
        return df
    
    def _calculate_fee(self, notional: float) -> float:
        """Calculate trading fee."""
        fee_rate = self.config.maker_fee if self.config.use_maker else self.config.taker_fee
        return notional * (fee_rate + self.config.slippage)
    
    def _open_position(self, row: pd.Series, side: str):
        """Open a new position."""
        price = row["close"]
        notional = self.capital * self.config.position_size * self.config.leverage
        size = notional / price
        fee = self._calculate_fee(notional)
        
        self.position = Trade(
            entry_time=row["datetime"],
            exit_time=None,
            symbol=self.symbol,
            side=side,
            entry_price=price,
            exit_price=None,
            size=size,
            fees=fee,
        )
        self.capital -= fee
    
    def _close_position(self, row: pd.Series, reason: str = "signal"):
        """Close current position."""
        if self.position is None:
            return
        
        exit_price = row["close"]
        notional = self.position.size * exit_price
        fee = self._calculate_fee(notional)
        
        # Calculate PnL
        if self.position.side == "LONG":
            pnl = (exit_price - self.position.entry_price) * self.position.size
        else:  # SHORT
            pnl = (self.position.entry_price - exit_price) * self.position.size
        
        pnl_pct = (pnl / (self.position.entry_price * self.position.size)) * 100
        
        # Update trade
        self.position.exit_time = row["datetime"]
        self.position.exit_price = exit_price
        self.position.pnl = pnl - fee - self.position.fees  # Net of all fees
        self.position.pnl_pct = pnl_pct
        self.position.fees += fee
        self.position.exit_reason = reason
        
        # Update capital
        self.capital += pnl - fee
        
        # Record trade
        self.trades.append(self.position)
        self.position = None
    
    def _check_stops(self, row: pd.Series) -> bool:
        """Check stop loss and take profit. Returns True if position was closed."""
        if self.position is None:
            return False
        
        current_price = row["close"]
        entry_price = self.position.entry_price
        
        if self.position.side == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if self.config.stop_loss and pnl_pct <= -self.config.stop_loss:
            self._close_position(row, reason="stop_loss")
            return True
        
        # Check take profit
        if self.config.take_profit and pnl_pct >= self.config.take_profit:
            self._close_position(row, reason="take_profit")
            return True
        
        return False
    
    def run(self) -> BacktestResult:
        """Run the backtest."""
        df = self._load_data()
        
        self.trades = []
        self.equity_curve = []
        self.position = None
        self.capital = self.config.initial_capital
        
        LOG.info(f"Running backtest: {self.strategy.name} on {self.symbol} {self.interval}")
        LOG.info(f"Data: {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        
        # Bar-by-bar simulation
        for idx, row in df.iterrows():
            # Check stops first
            if self._check_stops(row):
                pass  # Position was closed by stop
            
            signal = row.get("signal", 0)
            
            # Process signals
            if self.position is None:
                # No position - check for entry
                if signal == 1:
                    self._open_position(row, "LONG")
                elif signal == -1:
                    self._open_position(row, "SHORT")
            else:
                # Have position - check for exit or reversal
                if self.position.side == "LONG" and signal == -1:
                    self._close_position(row, reason="signal")
                    self._open_position(row, "SHORT")
                elif self.position.side == "SHORT" and signal == 1:
                    self._close_position(row, reason="signal")
                    self._open_position(row, "LONG")
            
            # Record equity (including unrealized PnL)
            equity = self.capital
            if self.position:
                current_price = row["close"]
                if self.position.side == "LONG":
                    unrealized = (current_price - self.position.entry_price) * self.position.size
                else:
                    unrealized = (self.position.entry_price - current_price) * self.position.size
                equity += unrealized
            self.equity_curve.append(equity)
        
        # Close any open position at end
        if self.position:
            self._close_position(df.iloc[-1], reason="end_of_backtest")
        
        # Calculate results
        return self._calculate_results(df)
    
    def _calculate_results(self, df: pd.DataFrame) -> BacktestResult:
        """Calculate backtest statistics."""
        result = BacktestResult()
        result.strategy_name = self.strategy.name
        result.symbol = self.symbol
        result.interval = self.interval
        result.start_date = str(df["datetime"].iloc[0].date())
        result.end_date = str(df["datetime"].iloc[-1].date())
        result.config = {
            "initial_capital": self.config.initial_capital,
            "leverage": self.config.leverage,
            "fees": f"maker={self.config.maker_fee}, taker={self.config.taker_fee}",
        }
        result.trades = self.trades
        
        # Equity curve
        result.equity_curve = pd.Series(
            self.equity_curve, 
            index=df["datetime"].iloc[:len(self.equity_curve)]
        )
        
        # Returns
        initial = self.config.initial_capital
        final = self.equity_curve[-1] if self.equity_curve else initial
        result.total_return = final - initial
        result.total_return_pct = (result.total_return / initial) * 100
        
        # Drawdown
        equity = result.equity_curve
        rolling_max = equity.cummax()
        drawdown = equity - rolling_max
        result.drawdown_curve = drawdown
        result.max_drawdown = abs(drawdown.min())
        result.max_drawdown_pct = (result.max_drawdown / rolling_max[drawdown.idxmin()]) * 100 if len(drawdown) > 0 else 0
        
        # Returns for Sharpe/Sortino
        returns = equity.pct_change().dropna()
        if len(returns) > 1:
            # Annualize based on interval
            intervals_per_year = {
                "1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4,
                "30m": 365 * 24 * 2, "1h": 365 * 24, "4h": 365 * 6, "1d": 365,
            }
            factor = np.sqrt(intervals_per_year.get(self.interval, 365 * 24 * 4))
            
            result.sharpe_ratio = (returns.mean() / returns.std()) * factor if returns.std() > 0 else 0
            
            downside = returns[returns < 0]
            downside_std = downside.std() if len(downside) > 0 else returns.std()
            result.sortino_ratio = (returns.mean() / downside_std) * factor if downside_std > 0 else 0
        
        # Trade statistics
        if self.trades:
            result.total_trades = len(self.trades)
            
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]
            
            result.winning_trades = len(wins)
            result.losing_trades = len(losses)
            result.win_rate = (len(wins) / len(self.trades)) * 100
            
            result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
            result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
            
            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            result.total_fees = sum(t.fees for t in self.trades)
            result.gross_pnl = sum(t.pnl + t.fees for t in self.trades)  # Add back fees for gross
            result.net_pnl = sum(t.pnl for t in self.trades)
            
            result.expectancy = result.net_pnl / result.total_trades
            
            # Consecutive wins/losses
            streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            for t in self.trades:
                if t.pnl > 0:
                    if streak > 0:
                        streak += 1
                    else:
                        streak = 1
                    max_win_streak = max(max_win_streak, streak)
                else:
                    if streak < 0:
                        streak -= 1
                    else:
                        streak = -1
                    max_loss_streak = max(max_loss_streak, abs(streak))
            
            result.max_consecutive_wins = max_win_streak
            result.max_consecutive_losses = max_loss_streak
            
            # Average trade duration
            durations = []
            for t in self.trades:
                if t.exit_time and t.entry_time:
                    duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                    durations.append(duration)
            result.avg_trade_duration = np.mean(durations) if durations else 0
        
        return result
    
    def plot(self, save_path: Optional[str] = None):
        """Plot equity curve and drawdown."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            LOG.warning("matplotlib not installed, skipping plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Equity curve
        ax1 = axes[0]
        equity = pd.Series(self.equity_curve)
        ax1.plot(equity, label="Equity", color="blue", linewidth=1.5)
        ax1.axhline(y=self.config.initial_capital, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(f"{self.strategy.name} - {self.symbol} {self.interval}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2 = axes[1]
        rolling_max = equity.cummax()
        drawdown_pct = ((equity - rolling_max) / rolling_max) * 100
        ax2.fill_between(range(len(drawdown_pct)), drawdown_pct, 0, alpha=0.3, color="red")
        ax2.plot(drawdown_pct, color="red", linewidth=1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Bar")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            LOG.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def run_comparison(
    strategies: List[str],
    symbol: str,
    interval: str,
    days: int,
    config: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """Run multiple strategies and compare results."""
    results = []
    
    for strat_name in strategies:
        if strat_name not in STRATEGIES:
            LOG.warning(f"Unknown strategy: {strat_name}")
            continue
        
        strategy = STRATEGIES[strat_name]()
        bt = Backtest(strategy, symbol, interval, days, config)
        result = bt.run()
        results.append(result.to_dict())
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest trading strategies on OHLCV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="rsi",
        help=f"Strategy to run: {', '.join(STRATEGIES.keys())}",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="Comma-separated strategies to compare",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to test",
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
        "--capital",
        type=float,
        default=10_000,
        help="Initial capital (default: 10000)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Leverage (default: 1.0)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss percentage (e.g., 0.02 for 2%%)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit percentage (e.g., 0.03 for 3%%)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show equity curve plot",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available strategies",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Strategies:")
        for name, cls in STRATEGIES.items():
            print(f"  {name:15} - {cls().name}")
        return
    
    # Build config
    config = BacktestConfig(
        initial_capital=args.capital,
        leverage=args.leverage,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )
    
    # Comparison mode
    if args.compare:
        strategies = [s.strip() for s in args.compare.split(",")]
        df = run_comparison(strategies, args.symbol, args.interval, args.days, config)
        
        if args.json:
            print(df.to_json(orient="records", indent=2))
        else:
            print("\n" + "="*80)
            print("STRATEGY COMPARISON")
            print("="*80)
            print(df.to_string(index=False))
        return
    
    # Multi-symbol mode
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        all_results = []
        
        for symbol in symbols:
            if args.strategy not in STRATEGIES:
                LOG.error(f"Unknown strategy: {args.strategy}")
                sys.exit(1)
            
            strategy = STRATEGIES[args.strategy]()
            bt = Backtest(strategy, symbol, args.interval, args.days, config)
            result = bt.run()
            all_results.append(result.to_dict())
            
            if not args.json:
                print(result.summary())
        
        if args.json:
            print(json.dumps(all_results, indent=2))
        return
    
    # Single backtest
    if args.strategy not in STRATEGIES:
        LOG.error(f"Unknown strategy: {args.strategy}. Use --list to see available strategies.")
        sys.exit(1)
    
    strategy = STRATEGIES[args.strategy]()
    bt = Backtest(strategy, args.symbol, args.interval, args.days, config)
    result = bt.run()
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
        
        # Print trade log
        if result.trades:
            print("\nRecent Trades (last 10):")
            print("-" * 100)
            for t in result.trades[-10:]:
                print(f"{t.entry_time.strftime('%Y-%m-%d %H:%M')} | {t.side:5} | "
                      f"Entry: {t.entry_price:>10.2f} | Exit: {t.exit_price:>10.2f} | "
                      f"PnL: ${t.pnl:>8.2f} ({t.pnl_pct:>5.2f}%) | {t.exit_reason}")
    
    if args.plot:
        bt.plot()


if __name__ == "__main__":
    main()
