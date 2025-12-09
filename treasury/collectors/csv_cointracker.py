"""
CoinTracker CSV Collector — v7.6

Parses CoinTracker CSV transaction exports and computes current holdings
with average cost basis.

CSV Expected Format:
    Date,Type,Received Quantity,Received Currency,Sent Quantity,
    Sent Currency,Fee Amount,Fee Currency,Fiat Value,Fiat Currency

This collector:
    1. Reads the CSV export
    2. Computes running balances per asset
    3. Computes weighted average cost basis
    4. Returns normalized holdings dict
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("treasury.csv_cointracker")


@dataclass
class AssetHolding:
    """Represents a single asset holding with cost basis."""
    symbol: str
    qty: float = 0.0
    total_cost_usd: float = 0.0
    
    @property
    def avg_cost_usd(self) -> float:
        """Compute average cost per unit."""
        if self.qty <= 0:
            return 0.0
        return self.total_cost_usd / self.qty


@dataclass
class CoinTrackerTransaction:
    """Parsed CoinTracker transaction."""
    date: datetime
    tx_type: str
    received_qty: float
    received_currency: str
    sent_qty: float
    sent_currency: str
    fee_qty: float
    fee_currency: str
    fiat_value: float
    fiat_currency: str
    
    @classmethod
    def from_row(cls, row: Dict[str, str]) -> Optional["CoinTrackerTransaction"]:
        """Parse a CSV row into a transaction."""
        try:
            date_str = row.get("Date", "")
            # Handle multiple date formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
                try:
                    date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                LOG.warning("Could not parse date: %s", date_str)
                return None
            
            return cls(
                date=date,
                tx_type=row.get("Type", "").lower(),
                received_qty=_safe_float(row.get("Received Quantity", "")),
                received_currency=row.get("Received Currency", "").upper(),
                sent_qty=_safe_float(row.get("Sent Quantity", "")),
                sent_currency=row.get("Sent Currency", "").upper(),
                fee_qty=_safe_float(row.get("Fee Amount", "")),
                fee_currency=row.get("Fee Currency", "").upper(),
                fiat_value=_safe_float(row.get("Fiat Value", "")),
                fiat_currency=row.get("Fiat Currency", "USD").upper(),
            )
        except Exception as e:
            LOG.warning("Failed to parse row: %s", e)
            return None


def _safe_float(val: str) -> float:
    """Safely convert string to float."""
    if not val or val.strip() == "":
        return 0.0
    try:
        # Remove commas and currency symbols
        cleaned = val.replace(",", "").replace("$", "").strip()
        return float(cleaned)
    except ValueError:
        return 0.0


class CoinTrackerCollector:
    """
    Collector that parses CoinTracker CSV exports.
    
    Usage:
        collector = CoinTrackerCollector("treasury/cointracker/holdings.csv")
        holdings = collector.collect()
        # holdings = {"BTC": {"qty": 0.5, "avg_cost_usd": 45000}, ...}
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.transactions: List[CoinTrackerTransaction] = []
        self.holdings: Dict[str, AssetHolding] = {}
    
    def collect(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse CSV and compute holdings.
        
        Returns:
            Dict mapping symbol to holding data:
            {
                "BTC": {
                    "qty": 0.035,
                    "avg_cost_usd": 45000.0,
                    "total_cost_usd": 1575.0,
                    "source": "cointracker",
                },
                ...
            }
        """
        if not self.csv_path.exists():
            LOG.warning("CSV file not found: %s", self.csv_path)
            return {}
        
        self._parse_csv()
        self._compute_holdings()
        
        return self._to_dict()
    
    def _parse_csv(self) -> None:
        """Parse CSV file into transactions."""
        self.transactions = []
        
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tx = CoinTrackerTransaction.from_row(row)
                    if tx:
                        self.transactions.append(tx)
            
            # Sort by date
            self.transactions.sort(key=lambda t: t.date)
            LOG.info("Parsed %d transactions from %s", len(self.transactions), self.csv_path)
        except Exception as e:
            LOG.error("Failed to parse CSV: %s", e)
    
    def _compute_holdings(self) -> None:
        """Compute current holdings from transactions."""
        self.holdings = {}
        
        for tx in self.transactions:
            # Handle received assets (buys, deposits, income)
            if tx.received_qty > 0 and tx.received_currency:
                symbol = tx.received_currency
                if symbol not in self.holdings:
                    self.holdings[symbol] = AssetHolding(symbol=symbol)
                
                holding = self.holdings[symbol]
                holding.qty += tx.received_qty
                
                # Add cost basis (fiat value of the transaction)
                if tx.fiat_value > 0:
                    holding.total_cost_usd += tx.fiat_value
            
            # Handle sent assets (sells, withdrawals, payments)
            if tx.sent_qty > 0 and tx.sent_currency:
                symbol = tx.sent_currency
                if symbol not in self.holdings:
                    self.holdings[symbol] = AssetHolding(symbol=symbol)
                
                holding = self.holdings[symbol]
                
                # Reduce cost basis proportionally on sells
                if holding.qty > 0:
                    reduction_ratio = min(1.0, tx.sent_qty / holding.qty)
                    holding.total_cost_usd *= (1 - reduction_ratio)
                
                holding.qty -= tx.sent_qty
            
            # Handle fees
            if tx.fee_qty > 0 and tx.fee_currency:
                symbol = tx.fee_currency
                if symbol in self.holdings:
                    self.holdings[symbol].qty -= tx.fee_qty
        
        # Remove zero or negative balances
        self.holdings = {
            k: v for k, v in self.holdings.items()
            if v.qty > 0.0001  # Small threshold for dust
        }
    
    def _to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert holdings to dictionary format."""
        result = {}
        for symbol, holding in self.holdings.items():
            result[symbol] = {
                "qty": holding.qty,
                "avg_cost_usd": holding.avg_cost_usd,
                "total_cost_usd": holding.total_cost_usd,
                "source": "cointracker",
            }
        return result


# =============================================================================
# CLI ENTRY POINT (for testing)
# =============================================================================

if __name__ == "__main__":
    import json
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "treasury/cointracker/cointracker_holdings.csv"
    
    collector = CoinTrackerCollector(csv_file)
    holdings = collector.collect()
    
    print(json.dumps(holdings, indent=2))
