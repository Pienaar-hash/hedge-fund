"""
Treasury Aggregator — v7.6

Aggregates holdings from multiple collectors and publishes to the
canonical state surfaces:
    - logs/state/offchain_assets.json
    - logs/state/offchain_yield.json

In v7.6, this supports only the CoinTracker CSV collector.
In v7.7, this will support multiple CEX APIs and on-chain collectors.

Usage:
    from treasury.aggregator import TreasuryAggregator
    from treasury.collectors import CoinTrackerCollector
    
    aggregator = TreasuryAggregator()
    
    # Add holdings from CSV
    collector = CoinTrackerCollector("treasury/cointracker/holdings.csv")
    aggregator.add_holdings(collector.collect(), source="cointracker")
    
    # Add manual holdings from config
    aggregator.add_holdings_from_config("config/offexchange_holdings.json")
    
    # Publish to state surfaces
    aggregator.publish()
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("treasury.aggregator")

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "logs" / "state"
CONFIG_DIR = PROJECT_ROOT / "config"

OFFCHAIN_ASSETS_PATH = STATE_DIR / "offchain_assets.json"
OFFCHAIN_YIELD_PATH = STATE_DIR / "offchain_yield.json"
OFFEXCHANGE_CONFIG_PATH = CONFIG_DIR / "offexchange_holdings.json"


@dataclass
class AggregatedHolding:
    """Aggregated holding from multiple sources."""
    symbol: str
    qty: float = 0.0
    avg_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    current_price_usd: Optional[float] = None
    usd_value: Optional[float] = None
    sources: List[str] = field(default_factory=list)


class TreasuryAggregator:
    """
    Aggregates treasury holdings from multiple sources and publishes
    to canonical state surfaces.
    """
    
    def __init__(
        self,
        assets_path: Optional[Path] = None,
        yield_path: Optional[Path] = None,
    ):
        self.assets_path = assets_path or OFFCHAIN_ASSETS_PATH
        self.yield_path = yield_path or OFFCHAIN_YIELD_PATH
        
        self.holdings: Dict[str, AggregatedHolding] = {}
        self.yields: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_holdings(
        self,
        holdings: Dict[str, Dict[str, Any]],
        source: str = "unknown",
    ) -> None:
        """
        Add holdings from a collector.
        
        Args:
            holdings: Dict mapping symbol to holding data
            source: Source identifier (e.g., "cointracker", "binance")
        """
        for symbol, data in holdings.items():
            if symbol not in self.holdings:
                self.holdings[symbol] = AggregatedHolding(symbol=symbol)
            
            holding = self.holdings[symbol]
            
            # Aggregate quantities
            new_qty = float(data.get("qty", 0))
            new_cost = float(data.get("total_cost_usd", 0) or data.get("avg_cost_usd", 0) * new_qty)
            
            # Weighted average for cost basis
            old_total = holding.total_cost_usd
            holding.qty += new_qty
            holding.total_cost_usd = old_total + new_cost
            
            if holding.qty > 0:
                holding.avg_cost_usd = holding.total_cost_usd / holding.qty
            
            # Track sources
            if source not in holding.sources:
                holding.sources.append(source)
            
            # Copy price if available
            if "current_price_usd" in data:
                holding.current_price_usd = float(data["current_price_usd"])
            if "usd_value" in data:
                holding.usd_value = float(data["usd_value"])
        
        LOG.info("Added %d holdings from %s", len(holdings), source)
    
    def add_holdings_from_config(
        self,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Add holdings from the legacy config file.
        
        This reads config/offexchange_holdings.json and normalizes it.
        """
        path = Path(config_path) if config_path else OFFEXCHANGE_CONFIG_PATH
        
        if not path.exists():
            LOG.warning("Config file not found: %s", path)
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Normalize the config format
            holdings = {}
            for symbol, data in config.items():
                if isinstance(data, dict):
                    qty = float(data.get("qty", 0))
                    avg_cost = float(data.get("avg_cost", 0))
                    holdings[symbol] = {
                        "qty": qty,
                        "avg_cost_usd": avg_cost,
                        "total_cost_usd": qty * avg_cost,
                    }
            
            self.add_holdings(holdings, source="config")
        except Exception as e:
            LOG.error("Failed to load config: %s", e)
    
    def add_yield(
        self,
        symbol: str,
        apr_pct: float,
        strategy: str = "unknown",
    ) -> None:
        """
        Add yield rate for an asset.
        
        Args:
            symbol: Asset symbol (e.g., "BTC")
            apr_pct: Annual percentage rate
            strategy: Yield strategy (e.g., "lending", "staking")
        """
        self.yields[symbol] = {
            "apr_pct": apr_pct,
            "strategy": strategy,
        }
    
    def set_prices(
        self,
        prices: Dict[str, float],
    ) -> None:
        """
        Set current prices and compute USD values.
        
        Args:
            prices: Dict mapping symbol to USD price
        """
        for symbol, price in prices.items():
            if symbol in self.holdings:
                holding = self.holdings[symbol]
                holding.current_price_usd = price
                holding.usd_value = holding.qty * price
    
    def publish(self) -> None:
        """
        Publish aggregated holdings to state surfaces.
        
        Writes:
            - logs/state/offchain_assets.json
            - logs/state/offchain_yield.json (if yields exist)
        """
        self._publish_assets()
        
        if self.yields:
            self._publish_yields()
    
    def _publish_assets(self) -> None:
        """Write offchain_assets.json."""
        assets = {}
        for symbol, holding in self.holdings.items():
            assets[symbol] = {
                "qty": holding.qty,
                "avg_cost_usd": holding.avg_cost_usd,
                "total_cost_usd": holding.total_cost_usd,
                "current_price_usd": holding.current_price_usd,
                "usd_value": holding.usd_value,
                "source": ",".join(holding.sources),
            }
        
        output = {
            "assets": assets,
            "metadata": {
                "sources": list(set(
                    src
                    for h in self.holdings.values()
                    for src in h.sources
                )),
                "asset_count": len(assets),
                **self.metadata,
            },
            "updated_ts": time.time(),
        }
        
        self.assets_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.assets_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        LOG.info("Published %d assets to %s", len(assets), self.assets_path)
    
    def _publish_yields(self) -> None:
        """Write offchain_yield.json."""
        output = {
            "yields": self.yields,
            "updated_ts": time.time(),
        }
        
        self.yield_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.yield_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        LOG.info("Published %d yields to %s", len(self.yields), self.yield_path)
    
    def get_total_value(self) -> float:
        """Get total USD value of all holdings."""
        total = 0.0
        for holding in self.holdings.values():
            if holding.usd_value:
                total += holding.usd_value
            elif holding.current_price_usd:
                total += holding.qty * holding.current_price_usd
        return total


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def sync_treasury_from_sources(
    csv_path: Optional[str] = None,
    config_path: Optional[str] = None,
    prices: Optional[Dict[str, float]] = None,
    yields: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TreasuryAggregator:
    """
    Convenience function to sync treasury from all available sources.
    
    Args:
        csv_path: Path to CoinTracker CSV (optional)
        config_path: Path to offexchange_holdings.json (optional)
        prices: Current prices for assets (optional)
        yields: Yield rates per asset (optional)
    
    Returns:
        TreasuryAggregator with all holdings loaded
    """
    from treasury.collectors import CoinTrackerCollector
    
    aggregator = TreasuryAggregator()
    
    # Load from CSV if provided
    if csv_path and Path(csv_path).exists():
        collector = CoinTrackerCollector(csv_path)
        aggregator.add_holdings(collector.collect(), source="cointracker")
    
    # Load from config
    if config_path:
        aggregator.add_holdings_from_config(config_path)
    else:
        # Try default config
        aggregator.add_holdings_from_config()
    
    # Set prices
    if prices:
        aggregator.set_prices(prices)
    
    # Set yields
    if yields:
        for symbol, yield_data in yields.items():
            aggregator.add_yield(
                symbol,
                apr_pct=yield_data.get("apr_pct", 0),
                strategy=yield_data.get("strategy", "unknown"),
            )
    
    # Publish
    aggregator.publish()
    
    return aggregator


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Default: sync from config + CSV
    csv_file = "treasury/cointracker/cointracker_holdings.csv"
    
    # Example prices (in production, fetch from exchange)
    prices = {
        "BTC": 97500.0,
        "XAUT": 2680.0,
        "USDC": 1.0,
        "ETH": 3800.0,
    }
    
    # Example yields
    yields = {
        "BTC": {"apr_pct": 1.2, "strategy": "lending"},
        "XAUT": {"apr_pct": 2.2, "strategy": "staking"},
    }
    
    aggregator = sync_treasury_from_sources(
        csv_path=csv_file if Path(csv_file).exists() else None,
        prices=prices,
        yields=yields,
    )
    
    print(f"Total treasury value: ${aggregator.get_total_value():,.2f}")
