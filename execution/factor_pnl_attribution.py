"""
v7.5_C2 — Factor PnL Attribution.

Provides approximate PnL attribution by factor based on:
- Factor component weights at trade entry
- Realized PnL per trade
- Aggregation over lookback window

This module is analysis-only and does NOT affect trading decisions.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

from execution.factor_diagnostics import load_factor_diagnostics_config

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

DEFAULT_FACTOR_NAMES = [
    "trend",
    "carry",
    "rv_momentum",
    "router_quality",
    "expectancy",
    "vol_regime",
]


@dataclass
class TradeRecord:
    """Trade record for PnL attribution."""

    symbol: str
    direction: str  # LONG or SHORT
    realized_pnl_usd: float
    factor_components: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0  # Unix timestamp
    trade_id: str = ""


@dataclass
class FactorPnlSlice:
    """PnL slice for a single factor."""

    factor: str
    pnl_usd: float


@dataclass
class FactorPnlSnapshot:
    """Snapshot of factor PnL attribution."""

    by_factor: Dict[str, float] = field(default_factory=dict)  # factor -> pnl_usd
    total_pnl_usd: float = 0.0
    window_days: int = 14
    trade_count: int = 0
    updated_ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        # Compute percentages
        pct_by_factor: Dict[str, float] = {}
        if abs(self.total_pnl_usd) > 1e-9:
            for factor, pnl in self.by_factor.items():
                pct_by_factor[factor] = round(pnl / self.total_pnl_usd * 100, 2)
        else:
            for factor in self.by_factor:
                pct_by_factor[factor] = 0.0

        return {
            "by_factor": {k: round(v, 4) for k, v in self.by_factor.items()},
            "pct_by_factor": pct_by_factor,
            "total_pnl_usd": round(self.total_pnl_usd, 4),
            "window_days": self.window_days,
            "trade_count": self.trade_count,
            "updated_ts": self.updated_ts,
        }


# ---------------------------------------------------------------------------
# Attribution Logic
# ---------------------------------------------------------------------------


def compute_factor_pnl_snapshot(
    trades: List[TradeRecord],
    factor_names: List[str] | None = None,
    window_days: int = 14,
) -> FactorPnlSnapshot:
    """
    Compute factor PnL attribution from a list of trades.

    Attribution method:
    - For each trade, allocate PnL to factors proportionally based on
      the absolute value of each factor component at entry time.
    - Sum across all trades in the window.

    Args:
        trades: List of TradeRecord objects with factor_components
        factor_names: Factor names to attribute (defaults to all known factors)
        window_days: Lookback window (for metadata only)

    Returns:
        FactorPnlSnapshot with per-factor PnL attribution
    """
    if factor_names is None:
        factor_names = DEFAULT_FACTOR_NAMES

    # Initialize factor PnL accumulators
    factor_pnl: Dict[str, float] = {f: 0.0 for f in factor_names}
    total_pnl = 0.0

    for trade in trades:
        pnl = trade.realized_pnl_usd
        total_pnl += pnl

        # Get factor components
        components = trade.factor_components or {}
        if not components:
            # No factor breakdown available - distribute equally
            equal_share = pnl / len(factor_names) if factor_names else 0.0
            for factor in factor_names:
                factor_pnl[factor] += equal_share
            continue

        # Compute weights based on absolute factor values
        total_abs = sum(abs(components.get(f, 0.0)) for f in factor_names)
        if total_abs < 1e-9:
            # All factors near zero - distribute equally
            equal_share = pnl / len(factor_names) if factor_names else 0.0
            for factor in factor_names:
                factor_pnl[factor] += equal_share
            continue

        # Allocate PnL proportionally
        for factor in factor_names:
            weight = abs(components.get(factor, 0.0)) / total_abs
            factor_pnl[factor] += pnl * weight

    return FactorPnlSnapshot(
        by_factor=factor_pnl,
        total_pnl_usd=total_pnl,
        window_days=window_days,
        trade_count=len(trades),
        updated_ts=time.time(),
    )


def load_trades_from_orders_executed(
    path: Path | str | None = None,
    lookback_days: int = 14,
    factor_names: List[str] | None = None,
) -> List[TradeRecord]:
    """
    Load trade records from orders_executed.jsonl for factor attribution.

    Args:
        path: Path to orders_executed.jsonl (defaults to logs/execution/orders_executed.jsonl)
        lookback_days: Only include trades from last N days
        factor_names: Factor names for component extraction

    Returns:
        List of TradeRecord objects
    """
    if path is None:
        path = Path("logs/execution/orders_executed.jsonl")
    else:
        path = Path(path)

    if factor_names is None:
        factor_names = DEFAULT_FACTOR_NAMES

    if not path.exists():
        return []

    cutoff_ts = time.time() - (lookback_days * 86400)
    trades: List[TradeRecord] = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip if no PnL
                pnl = record.get("realized_pnl") or record.get("realized_pnl_usd")
                if pnl is None:
                    continue

                # Parse timestamp
                ts = record.get("ts") or record.get("timestamp") or record.get("time")
                if ts is None:
                    continue

                # Convert timestamp
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        ts_float = dt.timestamp()
                    else:
                        ts_float = float(ts)
                        if ts_float > 1e12:  # Milliseconds
                            ts_float /= 1000.0
                except Exception:
                    continue

                # Skip if outside window
                if ts_float < cutoff_ts:
                    continue

                # Extract factor components if available
                factor_components: Dict[str, float] = {}
                
                # Try to get from hybrid_score metadata
                metadata = record.get("metadata") or {}
                hybrid_meta = metadata.get("hybrid_score") or {}
                components = hybrid_meta.get("components") or record.get("components") or {}

                for factor in factor_names:
                    if factor in components:
                        try:
                            factor_components[factor] = float(components[factor])
                        except (TypeError, ValueError):
                            factor_components[factor] = 0.0

                # Also check factor_vector directly
                factor_vector = record.get("factor_vector") or metadata.get("factor_vector") or {}
                for factor in factor_names:
                    if factor in factor_vector and factor not in factor_components:
                        try:
                            factor_components[factor] = float(factor_vector[factor])
                        except (TypeError, ValueError):
                            pass

                trades.append(
                    TradeRecord(
                        symbol=str(record.get("symbol", "")).upper(),
                        direction=str(record.get("side", record.get("direction", "LONG"))).upper(),
                        realized_pnl_usd=float(pnl),
                        factor_components=factor_components,
                        timestamp=ts_float,
                        trade_id=str(record.get("id", record.get("order_id", ""))),
                    )
                )
    except Exception:
        pass

    return trades


def build_factor_pnl_from_logs(
    lookback_days: int | None = None,
    factor_names: List[str] | None = None,
    strategy_config: Mapping[str, Any] | None = None,
) -> FactorPnlSnapshot:
    """
    Build factor PnL snapshot from order execution logs.

    Convenience function that loads config and trades, then computes attribution.

    Args:
        lookback_days: Override lookback days (defaults to config)
        factor_names: Override factor names (defaults to config)
        strategy_config: Strategy config dict

    Returns:
        FactorPnlSnapshot with attribution
    """
    cfg = load_factor_diagnostics_config(strategy_config)

    if lookback_days is None:
        lookback_days = cfg.pnl_attribution_lookback_days
    if factor_names is None:
        factor_names = cfg.factors

    trades = load_trades_from_orders_executed(
        lookback_days=lookback_days,
        factor_names=factor_names,
    )

    return compute_factor_pnl_snapshot(
        trades=trades,
        factor_names=factor_names,
        window_days=lookback_days,
    )


# ---------------------------------------------------------------------------
# State File I/O
# ---------------------------------------------------------------------------

DEFAULT_FACTOR_PNL_PATH = Path("logs/state/factor_pnl.json")


def load_factor_pnl_state(
    path: Path | str = DEFAULT_FACTOR_PNL_PATH,
) -> Dict[str, Any]:
    """Load factor PnL state from file."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def write_factor_pnl_state(
    snapshot: FactorPnlSnapshot,
    path: Path | str = DEFAULT_FACTOR_PNL_PATH,
) -> None:
    """Write factor PnL state to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        tmp.replace(path)
    except Exception:
        pass


__all__ = [
    "TradeRecord",
    "FactorPnlSlice",
    "FactorPnlSnapshot",
    "compute_factor_pnl_snapshot",
    "load_trades_from_orders_executed",
    "build_factor_pnl_from_logs",
    "load_factor_pnl_state",
    "write_factor_pnl_state",
]
