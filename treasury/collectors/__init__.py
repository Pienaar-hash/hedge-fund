"""
Treasury Collectors Package — v7.6 Stubs

This package provides a framework for collecting treasury/off-exchange 
holdings from various sources. In v7.6, only the CSV collector is 
functional. API-based collectors are stubbed for v7.7.

Collectors:
    csv_cointracker.py  - Parse CoinTracker CSV exports (v7.6)
    cex_binance.py      - Binance spot/savings API (v7.7 stub)
    cex_okx.py          - OKX spot/earn API (v7.7 stub)
    onchain_evm.py      - EVM chain balance collector (v7.7 stub)

Usage:
    from treasury.collectors import CoinTrackerCollector
    from treasury.aggregator import TreasuryAggregator
    
    collector = CoinTrackerCollector("treasury/cointracker/holdings.csv")
    holdings = collector.collect()
    
    aggregator = TreasuryAggregator()
    aggregator.add_holdings(holdings, source="cointracker")
    aggregator.publish()  # writes to logs/state/offchain_assets.json
"""
from treasury.collectors.csv_cointracker import CoinTrackerCollector

__all__ = ["CoinTrackerCollector"]
