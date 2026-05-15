#!/usr/bin/env python3
"""Export collected OHLCV parquet files into flat replay-seed CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
TIMESTAMP_COLUMNS = ["timestamp", "ts", "open_time", "datetime"]


def _parse_utc_datetime(value: str) -> pd.Timestamp:
    """Parse a date or timestamp string to a UTC pandas timestamp."""
    parsed = pd.to_datetime(value, utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Invalid datetime value: {value!r}")
    return pd.Timestamp(parsed)


def _timestamp_to_utc_iso(value: Any) -> str:
    """Normalize epoch seconds/ms or ISO timestamps to UTC ISO-8601."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("missing timestamp value")

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("empty timestamp value")
        if stripped.isdigit():
            value = int(stripped)
        else:
            parsed = pd.to_datetime(stripped, utc=True)
            if pd.isna(parsed):
                raise ValueError(f"Invalid timestamp value: {value!r}")
            return pd.Timestamp(parsed).isoformat().replace("+00:00", "Z")

    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1_000_000_000_000:
            parsed = pd.to_datetime(int(numeric), unit="ms", utc=True)
        else:
            parsed = pd.to_datetime(numeric, unit="s", utc=True)
        return pd.Timestamp(parsed).isoformat().replace("+00:00", "Z")

    parsed = pd.to_datetime(value, utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return pd.Timestamp(parsed).isoformat().replace("+00:00", "Z")


def _detect_timestamp_column(columns: list[str]) -> str:
    for candidate in TIMESTAMP_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Required timestamp column missing. Expected one of: "
        + ", ".join(TIMESTAMP_COLUMNS)
    )


def _load_ohlcv_parquet_files(symbol_dir: Path) -> list[pd.DataFrame]:
    parquet_files = sorted(symbol_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {symbol_dir}")

    frames: list[pd.DataFrame] = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file)
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        raise ValueError(f"No rows found in parquet files under {symbol_dir}")

    return frames


def export_replay_seed_from_ohlcv(
    *,
    ohlcv_root: str | Path,
    out_dir: str | Path,
    symbols: list[str],
    interval: str,
    start: str,
    end: str,
) -> list[Path]:
    """Export one replay-seed CSV per symbol from OHLCV parquet files."""
    ohlcv_root_path = Path(ohlcv_root)
    out_dir_path = Path(out_dir)
    start_ts = _parse_utc_datetime(start)
    end_ts = _parse_utc_datetime(end)
    if end_ts <= start_ts:
        raise ValueError("end must be greater than start")

    out_dir_path.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []

    for symbol in symbols:
        symbol_dir = ohlcv_root_path / symbol / interval
        frames = _load_ohlcv_parquet_files(symbol_dir)
        frame = pd.concat(frames, ignore_index=True)

        timestamp_column = _detect_timestamp_column(list(frame.columns))
        missing_price_columns = [col for col in REQUIRED_PRICE_COLUMNS if col not in frame.columns]
        if missing_price_columns:
            raise ValueError(
                f"Missing required columns in {symbol_dir}: {missing_price_columns}"
            )

        normalized = frame[[timestamp_column, *REQUIRED_PRICE_COLUMNS]].copy()
        normalized["timestamp"] = normalized[timestamp_column].map(_timestamp_to_utc_iso)
        normalized["timestamp_dt"] = pd.to_datetime(normalized["timestamp"], utc=True)

        filtered = normalized[(normalized["timestamp_dt"] >= start_ts) & (normalized["timestamp_dt"] < end_ts)]
        if filtered.empty:
            raise ValueError(
                f"Filtered window is empty for {symbol} {interval} in {symbol_dir}"
            )

        filtered = filtered.sort_values("timestamp_dt").drop_duplicates(subset=["timestamp"], keep="last")
        export_frame = filtered[["timestamp", *REQUIRED_PRICE_COLUMNS]].reset_index(drop=True)

        output_file = out_dir_path / f"{symbol}_{interval}.csv"
        export_frame.to_csv(output_file, index=False, columns=OUTPUT_COLUMNS)
        written_files.append(output_file)

    return written_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export OHLCV parquet files to replay-seed CSVs")
    parser.add_argument("--ohlcv-root", default="data/ohlcv")
    parser.add_argument("--out-dir", default="data/replay_seed")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    export_replay_seed_from_ohlcv(
        ohlcv_root=args.ohlcv_root,
        out_dir=args.out_dir,
        symbols=args.symbols,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()