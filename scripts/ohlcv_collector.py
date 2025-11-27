#!/usr/bin/env python3
"""
OHLCV Data Collector for Backtesting (v1.0)

Collects and stores historical candlestick data from Binance Futures for backtesting.
Supports multiple timeframes and automatic gap filling.

Usage:
    # Collect last 30 days of 15m candles for all symbols
    python scripts/ohlcv_collector.py --days 30 --interval 15m

    # Collect specific symbols with multiple timeframes
    python scripts/ohlcv_collector.py --symbols BTCUSDT,ETHUSDT --intervals 5m,15m,1h --days 7

    # Backfill missing data
    python scripts/ohlcv_collector.py --backfill --interval 15m

    # Run as continuous service (collects new candles every interval)
    python scripts/ohlcv_collector.py --daemon --interval 15m

Storage: Parquet files in data/ohlcv/{symbol}/{interval}/
Format: Each file contains ~1 day of data, named YYYY-MM-DD.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("ohlcv_collector")

# --- Configuration ---
DATA_DIR = Path(os.getenv("OHLCV_DATA_DIR") or PROJECT_ROOT / "data" / "ohlcv")
CONFIG_PATH = PROJECT_ROOT / "config" / "pairs_universe.json"
STATE_PATH = DATA_DIR / ".collector_state.json"

# Binance limits
MAX_KLINES_PER_REQUEST = 1500
RATE_LIMIT_DELAY_S = 0.1  # 100ms between requests

# Interval to milliseconds mapping
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

# Default intervals to collect
DEFAULT_INTERVALS = ["5m", "15m", "1h", "4h"]


def load_universe() -> List[str]:
    """Load trading symbols from pairs_universe.json."""
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        symbols = config.get("symbols", [])
        if not symbols:
            symbols = [u["symbol"] for u in config.get("universe", []) if u.get("enabled")]
        return symbols
    except Exception as e:
        LOG.warning(f"Failed to load universe config: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def fetch_klines(
    symbol: str,
    interval: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = MAX_KLINES_PER_REQUEST,
) -> List[Dict[str, Any]]:
    """
    Fetch klines from Binance Futures API.
    
    Returns list of dicts with keys:
        open_time, open, high, low, close, volume, close_time, quote_volume, trades
    """
    from execution.exchange_utils import _req
    
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, MAX_KLINES_PER_REQUEST),
    }
    if start_time:
        params["startTime"] = int(start_time)
    if end_time:
        params["endTime"] = int(end_time)
    
    try:
        resp = _req("GET", "/fapi/v1/klines", params=params)
        data = resp.json()
    except Exception as e:
        LOG.error(f"Failed to fetch klines for {symbol} {interval}: {e}")
        return []
    
    result = []
    for row in data:
        try:
            result.append({
                "open_time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": int(row[6]),
                "quote_volume": float(row[7]),
                "trades": int(row[8]),
            })
        except (IndexError, TypeError, ValueError) as e:
            LOG.debug(f"Skipping malformed row: {row} - {e}")
            continue
    
    return result


def fetch_all_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> List[Dict[str, Any]]:
    """
    Fetch all klines between start_time and end_time, handling pagination.
    """
    all_klines = []
    current_start = start_time
    interval_ms = INTERVAL_MS.get(interval, 900_000)
    
    while current_start < end_time:
        klines = fetch_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_time,
            limit=MAX_KLINES_PER_REQUEST,
        )
        
        if not klines:
            break
        
        all_klines.extend(klines)
        
        # Move start to after the last candle
        last_open_time = klines[-1]["open_time"]
        current_start = last_open_time + interval_ms
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY_S)
        
        # Progress logging for long fetches
        if len(all_klines) % 5000 == 0:
            LOG.info(f"  {symbol} {interval}: fetched {len(all_klines)} candles...")
    
    return all_klines


def save_klines_parquet(
    symbol: str,
    interval: str,
    klines: List[Dict[str, Any]],
) -> int:
    """
    Save klines to Parquet files, organized by date.
    Returns number of new rows saved.
    """
    if not klines:
        return 0
    
    # Create output directory
    out_dir = DATA_DIR / symbol / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(klines)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date
    
    total_saved = 0
    
    # Group by date and save
    for date, group in df.groupby("date"):
        file_path = out_dir / f"{date}.parquet"
        
        # If file exists, merge and deduplicate
        if file_path.exists():
            try:
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, group], ignore_index=True)
                combined = combined.drop_duplicates(subset=["open_time"], keep="last")
                combined = combined.sort_values("open_time").reset_index(drop=True)
                new_rows = len(combined) - len(existing)
            except Exception as e:
                LOG.warning(f"Failed to read existing {file_path}: {e}")
                combined = group
                new_rows = len(group)
        else:
            combined = group.sort_values("open_time").reset_index(drop=True)
            new_rows = len(combined)
        
        # Save (without datetime/date columns to save space)
        save_cols = ["open_time", "open", "high", "low", "close", "volume", 
                     "close_time", "quote_volume", "trades"]
        combined[save_cols].to_parquet(file_path, index=False, compression="snappy")
        total_saved += new_rows
    
    return total_saved


def load_klines_parquet(
    symbol: str,
    interval: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load klines from Parquet files.
    """
    data_dir = DATA_DIR / symbol / interval
    if not data_dir.exists():
        return pd.DataFrame()
    
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    
    # Filter by date range
    if start_date or end_date:
        filtered_files = []
        for f in files:
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d").date()
                if start_date and file_date < start_date.date():
                    continue
                if end_date and file_date > end_date.date():
                    continue
                filtered_files.append(f)
            except ValueError:
                continue
        files = filtered_files
    
    if not files:
        return pd.DataFrame()
    
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"], keep="last")
    df = df.sort_values("open_time").reset_index(drop=True)
    
    # Add datetime column for convenience
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    
    return df


def get_last_collected_time(symbol: str, interval: str) -> Optional[int]:
    """Get the last collected candle timestamp for a symbol/interval."""
    data_dir = DATA_DIR / symbol / interval
    if not data_dir.exists():
        return None
    
    files = sorted(data_dir.glob("*.parquet"), reverse=True)
    if not files:
        return None
    
    try:
        df = pd.read_parquet(files[0])
        if df.empty:
            return None
        return int(df["open_time"].max())
    except Exception:
        return None


def find_gaps(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> List[Tuple[int, int]]:
    """
    Find gaps in collected data.
    Returns list of (gap_start, gap_end) tuples.
    """
    df = load_klines_parquet(
        symbol,
        interval,
        datetime.fromtimestamp(start_time / 1000, tz=timezone.utc),
        datetime.fromtimestamp(end_time / 1000, tz=timezone.utc),
    )
    
    if df.empty:
        return [(start_time, end_time)]
    
    interval_ms = INTERVAL_MS.get(interval, 900_000)
    gaps = []
    
    # Check for gap at start
    first_time = int(df["open_time"].min())
    if first_time > start_time + interval_ms:
        gaps.append((start_time, first_time - interval_ms))
    
    # Check for gaps in middle
    df = df.sort_values("open_time")
    times = df["open_time"].values
    for i in range(1, len(times)):
        expected_next = times[i-1] + interval_ms
        actual_next = times[i]
        if actual_next > expected_next + interval_ms:
            gaps.append((int(expected_next), int(actual_next - interval_ms)))
    
    # Check for gap at end
    last_time = int(df["open_time"].max())
    if last_time < end_time - interval_ms:
        gaps.append((last_time + interval_ms, end_time))
    
    return gaps


def collect_symbol(
    symbol: str,
    intervals: List[str],
    days: int = 30,
    backfill: bool = False,
) -> Dict[str, int]:
    """
    Collect OHLCV data for a single symbol.
    Returns dict of {interval: rows_saved}.
    """
    results = {}
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    
    for interval in intervals:
        LOG.info(f"Collecting {symbol} {interval}...")
        
        if backfill:
            # Find and fill gaps
            gaps = find_gaps(symbol, interval, start_ms, now_ms)
            total_saved = 0
            for gap_start, gap_end in gaps:
                LOG.info(f"  Filling gap: {datetime.fromtimestamp(gap_start/1000, tz=timezone.utc)} - "
                        f"{datetime.fromtimestamp(gap_end/1000, tz=timezone.utc)}")
                klines = fetch_all_klines(symbol, interval, gap_start, gap_end)
                saved = save_klines_parquet(symbol, interval, klines)
                total_saved += saved
            results[interval] = total_saved
        else:
            # Full collection
            klines = fetch_all_klines(symbol, interval, start_ms, now_ms)
            results[interval] = save_klines_parquet(symbol, interval, klines)
        
        LOG.info(f"  {symbol} {interval}: saved {results[interval]} rows")
    
    return results


def collect_incremental(
    symbol: str,
    interval: str,
) -> int:
    """
    Collect only new candles since last collection.
    Returns number of new rows saved.
    """
    last_time = get_last_collected_time(symbol, interval)
    now_ms = int(time.time() * 1000)
    
    if last_time is None:
        # No existing data, fetch last 24h
        start_time = now_ms - (24 * 60 * 60 * 1000)
    else:
        # Start from the last candle (will be deduplicated)
        start_time = last_time
    
    klines = fetch_all_klines(symbol, interval, start_time, now_ms)
    return save_klines_parquet(symbol, interval, klines)


def daemon_loop(
    symbols: List[str],
    interval: str,
    poll_seconds: int = 60,
):
    """
    Run as a continuous daemon, collecting new candles periodically.
    """
    LOG.info(f"Starting OHLCV collector daemon for {len(symbols)} symbols, interval={interval}")
    LOG.info(f"Poll interval: {poll_seconds}s")
    
    while True:
        try:
            for symbol in symbols:
                saved = collect_incremental(symbol, interval)
                if saved > 0:
                    LOG.info(f"{symbol} {interval}: +{saved} new candles")
            
            time.sleep(poll_seconds)
            
        except KeyboardInterrupt:
            LOG.info("Daemon stopped by user")
            break
        except Exception as e:
            LOG.error(f"Daemon error: {e}")
            time.sleep(poll_seconds)


def get_data_summary() -> Dict[str, Any]:
    """Get summary of collected data."""
    summary = {"symbols": {}, "total_rows": 0, "total_size_mb": 0}
    
    if not DATA_DIR.exists():
        return summary
    
    for symbol_dir in DATA_DIR.iterdir():
        if not symbol_dir.is_dir() or symbol_dir.name.startswith("."):
            continue
        
        symbol = symbol_dir.name
        summary["symbols"][symbol] = {}
        
        for interval_dir in symbol_dir.iterdir():
            if not interval_dir.is_dir():
                continue
            
            interval = interval_dir.name
            files = list(interval_dir.glob("*.parquet"))
            
            total_rows = 0
            total_size = 0
            min_date = None
            max_date = None
            
            for f in files:
                try:
                    df = pd.read_parquet(f)
                    total_rows += len(df)
                    total_size += f.stat().st_size
                    
                    file_date = datetime.strptime(f.stem, "%Y-%m-%d").date()
                    if min_date is None or file_date < min_date:
                        min_date = file_date
                    if max_date is None or file_date > max_date:
                        max_date = file_date
                except Exception:
                    continue
            
            summary["symbols"][symbol][interval] = {
                "files": len(files),
                "rows": total_rows,
                "size_mb": round(total_size / 1_000_000, 2),
                "date_range": f"{min_date} - {max_date}" if min_date else "N/A",
            }
            summary["total_rows"] += total_rows
            summary["total_size_mb"] += total_size / 1_000_000
    
    summary["total_size_mb"] = round(summary["total_size_mb"], 2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="OHLCV Data Collector for Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated list of symbols (default: from pairs_universe.json)",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="15m",
        help=f"Comma-separated list of intervals (default: 15m). Options: {', '.join(INTERVAL_MS.keys())}",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of history to collect (default: 30)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Only fill gaps in existing data",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as continuous daemon collecting new candles",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Poll interval in seconds for daemon mode (default: 60)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of collected data and exit",
    )
    
    args = parser.parse_args()
    
    # Summary mode
    if args.summary:
        summary = get_data_summary()
        print("\n=== OHLCV Data Summary ===\n")
        print(f"Total rows: {summary['total_rows']:,}")
        print(f"Total size: {summary['total_size_mb']:.2f} MB")
        print(f"\nBy symbol:")
        for symbol, intervals in summary["symbols"].items():
            print(f"\n  {symbol}:")
            for interval, stats in intervals.items():
                print(f"    {interval}: {stats['rows']:,} rows, {stats['size_mb']:.2f} MB ({stats['date_range']})")
        return
    
    # Parse symbols and intervals
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = load_universe()
    
    intervals = [i.strip() for i in args.intervals.split(",")]
    
    # Validate intervals
    for interval in intervals:
        if interval not in INTERVAL_MS:
            LOG.error(f"Invalid interval: {interval}. Valid options: {', '.join(INTERVAL_MS.keys())}")
            sys.exit(1)
    
    LOG.info(f"Symbols: {symbols}")
    LOG.info(f"Intervals: {intervals}")
    LOG.info(f"Data directory: {DATA_DIR}")
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Daemon mode
    if args.daemon:
        if len(intervals) > 1:
            LOG.warning("Daemon mode only supports one interval, using first: " + intervals[0])
        daemon_loop(symbols, intervals[0], args.poll_seconds)
        return
    
    # Collection mode
    LOG.info(f"Collecting {args.days} days of history...")
    start_time = time.time()
    
    total_saved = 0
    for symbol in symbols:
        results = collect_symbol(
            symbol=symbol,
            intervals=intervals,
            days=args.days,
            backfill=args.backfill,
        )
        total_saved += sum(results.values())
    
    elapsed = time.time() - start_time
    LOG.info(f"Collection complete: {total_saved:,} rows saved in {elapsed:.1f}s")
    
    # Print summary
    summary = get_data_summary()
    print(f"\n=== Collection Summary ===")
    print(f"Total rows in storage: {summary['total_rows']:,}")
    print(f"Total size: {summary['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
