"""
Treasury Package — v7.6

Provides off-exchange asset tracking and yield management.

Components:
    collectors/     - Source-specific collectors (CSV, CEX APIs)
    aggregator.py   - Aggregates holdings and publishes state
    cointracker/    - CoinTracker CSV exports (input data)

Canonical State Surfaces:
    logs/state/offchain_assets.json  - Aggregated holdings
    logs/state/offchain_yield.json   - Yield rates per asset

v7.6 Scope:
    - CoinTracker CSV collector (functional)
    - Config file collector (functional)
    - Aggregator (functional)
    
v7.7 Roadmap:
    - CEX API collectors (Binance, OKX, Bitfinex)
    - On-chain collectors (EVM L1/L2, Solana)
    - Automated price fetching
    - Yield strategy configuration via runtime.yaml
"""
from treasury.aggregator import TreasuryAggregator, sync_treasury_from_sources
from treasury.collectors import CoinTrackerCollector

__all__ = [
    "TreasuryAggregator",
    "CoinTrackerCollector",
    "sync_treasury_from_sources",
]
