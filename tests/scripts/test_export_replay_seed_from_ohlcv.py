from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.export_replay_seed_from_ohlcv import export_replay_seed_from_ohlcv


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    return pd.read_csv(path).to_dict(orient="records")


def test_exports_one_symbol_from_parquet_to_csv(tmp_path: Path) -> None:
    ohlcv_root = tmp_path / "ohlcv"
    out_dir = tmp_path / "replay_seed"
    frame = pd.DataFrame(
        [
            {"timestamp": 1713398400000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
            {"timestamp": 1713399300000, "open": 1.5, "high": 2.5, "low": 1.25, "close": 2.0, "volume": 11},
        ]
    )
    _write_parquet(ohlcv_root / "BTCUSDT" / "15m" / "2024-04-18.parquet", frame)

    written = export_replay_seed_from_ohlcv(
        ohlcv_root=ohlcv_root,
        out_dir=out_dir,
        symbols=["BTCUSDT"],
        interval="15m",
        start="2024-04-17",
        end="2024-04-19",
    )

    assert written == [out_dir / "BTCUSDT_15m.csv"]
    rows = _read_csv_rows(out_dir / "BTCUSDT_15m.csv")
    assert len(rows) == 2


def test_filters_by_date_window(tmp_path: Path) -> None:
    ohlcv_root = tmp_path / "ohlcv"
    out_dir = tmp_path / "replay_seed"
    frame = pd.DataFrame(
        [
            {"timestamp": 1713312000000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
            {"timestamp": 1713398400000, "open": 2, "high": 3, "low": 1.5, "close": 2.5, "volume": 11},
            {"timestamp": 1713484800000, "open": 3, "high": 4, "low": 2.5, "close": 3.5, "volume": 12},
        ]
    )
    _write_parquet(ohlcv_root / "BTCUSDT" / "15m" / "2024-04-17.parquet", frame)

    export_replay_seed_from_ohlcv(
        ohlcv_root=ohlcv_root,
        out_dir=out_dir,
        symbols=["BTCUSDT"],
        interval="15m",
        start="2024-04-18",
        end="2024-04-19",
    )

    rows = _read_csv_rows(out_dir / "BTCUSDT_15m.csv")
    assert len(rows) == 1
    assert rows[0]["timestamp"] == "2024-04-18T00:00:00Z"


def test_sorts_timestamps_ascending(tmp_path: Path) -> None:
    ohlcv_root = tmp_path / "ohlcv"
    out_dir = tmp_path / "replay_seed"
    frame = pd.DataFrame(
        [
            {"timestamp": "2024-04-18T00:15:00Z", "open": 2, "high": 3, "low": 1.5, "close": 2.5, "volume": 11},
            {"timestamp": "2024-04-18T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        ]
    )
    _write_parquet(ohlcv_root / "BTCUSDT" / "15m" / "2024-04-18.parquet", frame)

    export_replay_seed_from_ohlcv(
        ohlcv_root=ohlcv_root,
        out_dir=out_dir,
        symbols=["BTCUSDT"],
        interval="15m",
        start="2024-04-18",
        end="2024-04-19",
    )

    rows = _read_csv_rows(out_dir / "BTCUSDT_15m.csv")
    assert [row["timestamp"] for row in rows] == ["2024-04-18T00:00:00Z", "2024-04-18T00:15:00Z"]


def test_emits_exact_expected_columns(tmp_path: Path) -> None:
    ohlcv_root = tmp_path / "ohlcv"
    out_dir = tmp_path / "replay_seed"
    frame = pd.DataFrame(
        [
            {"timestamp": 1713398400000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        ]
    )
    _write_parquet(ohlcv_root / "BTCUSDT" / "15m" / "2024-04-18.parquet", frame)

    export_replay_seed_from_ohlcv(
        ohlcv_root=ohlcv_root,
        out_dir=out_dir,
        symbols=["BTCUSDT"],
        interval="15m",
        start="2024-04-17",
        end="2024-04-19",
    )

    csv_path = out_dir / "BTCUSDT_15m.csv"
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "timestamp,open,high,low,close,volume"


def test_fails_on_missing_parquet_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No parquet files found"):
        export_replay_seed_from_ohlcv(
            ohlcv_root=tmp_path / "ohlcv",
            out_dir=tmp_path / "replay_seed",
            symbols=["BTCUSDT"],
            interval="15m",
            start="2024-04-17",
            end="2024-04-19",
        )


def test_fails_on_empty_filtered_window(tmp_path: Path) -> None:
    ohlcv_root = tmp_path / "ohlcv"
    out_dir = tmp_path / "replay_seed"
    frame = pd.DataFrame(
        [
            {"timestamp": 1713312000000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        ]
    )
    _write_parquet(ohlcv_root / "BTCUSDT" / "15m" / "2024-04-17.parquet", frame)

    with pytest.raises(ValueError, match="Filtered window is empty"):
        export_replay_seed_from_ohlcv(
            ohlcv_root=ohlcv_root,
            out_dir=out_dir,
            symbols=["BTCUSDT"],
            interval="15m",
            start="2024-04-18",
            end="2024-04-19",
        )


def test_handles_epoch_ms_timestamps(tmp_path: Path) -> None:
    ohlcv_root = tmp_path / "ohlcv"
    out_dir = tmp_path / "replay_seed"
    frame = pd.DataFrame(
        [
            {"timestamp": 1713398400000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        ]
    )
    _write_parquet(ohlcv_root / "BTCUSDT" / "15m" / "2024-04-18.parquet", frame)

    export_replay_seed_from_ohlcv(
        ohlcv_root=ohlcv_root,
        out_dir=out_dir,
        symbols=["BTCUSDT"],
        interval="15m",
        start="2024-04-17",
        end="2024-04-19",
    )

    rows = _read_csv_rows(out_dir / "BTCUSDT_15m.csv")
    assert rows[0]["timestamp"] == "2024-04-18T00:00:00Z"


def test_handles_iso_timestamps(tmp_path: Path) -> None:
    ohlcv_root = tmp_path / "ohlcv"
    out_dir = tmp_path / "replay_seed"
    frame = pd.DataFrame(
        [
            {"timestamp": "2024-04-18T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        ]
    )
    _write_parquet(ohlcv_root / "BTCUSDT" / "15m" / "2024-04-18.parquet", frame)

    export_replay_seed_from_ohlcv(
        ohlcv_root=ohlcv_root,
        out_dir=out_dir,
        symbols=["BTCUSDT"],
        interval="15m",
        start="2024-04-17",
        end="2024-04-19",
    )

    rows = _read_csv_rows(out_dir / "BTCUSDT_15m.csv")
    assert rows[0]["timestamp"] == "2024-04-18T00:00:00Z"