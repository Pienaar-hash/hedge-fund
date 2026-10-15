"""Deterministic, disk-bounded ingestion of Binance USD-M futures historical archives.

Downloads monthly aggTrades and 1m kline archives from the public Binance
historical data repository, hashes each source archive before any further
processing, extracts only the columns H-LIQ-001 requires, persists a compact
binary cache, and then discards the raw archive. Every attempt (success or
failure) is recorded as one line in an append-only JSONL manifest so the
experiment has a full, reproducible provenance trail even though the raw
multi-gigabyte source files are not retained on disk.

No forward-filling. No fabrication. A month that cannot be ingested (network
failure, corrupt archive, or the disk-safety margin below) is recorded with an
explicit gap_status and excluded from event construction.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date
from typing import Iterator, Literal, Optional

import numpy as np
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/monthly"

GapStatus = Literal[
    "OK",
    "MISSING_404",
    "DOWNLOAD_FAILED",
    "CORRUPT_ARCHIVE",
    "NOT_INGESTED_DISK_BUDGET",
]

DEFAULT_MIN_FREE_BYTES = 3 * 1024 * 1024 * 1024  # stop before disk fills up


@dataclass(frozen=True)
class ManifestEntry:
    source_url: str
    symbol: str
    data_type: str  # "aggTrades" | "klines_1m"
    year: int
    month: int
    coverage_start: str
    coverage_end: str
    filename: str
    byte_size: Optional[int]
    sha256: Optional[str]
    validation_status: str  # "VALID" | "INVALID" | "NOT_ATTEMPTED"
    gap_status: GapStatus
    cache_path: Optional[str]
    row_count: Optional[int]
    attempted_at: str
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


def month_range(start: date, end: date) -> Iterator[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def _coverage_bounds(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    end_y, end_m = (year + 1, 1) if month == 12 else (year, month + 1)
    end = date(end_y, end_m, 1)
    return start.isoformat() + "T00:00:00Z", end.isoformat() + "T00:00:00Z"


def aggtrades_url(symbol: str, year: int, month: int) -> str:
    return f"{BASE_URL}/aggTrades/{symbol}/{symbol}-aggTrades-{year:04d}-{month:02d}.zip"


def klines_url(symbol: str, year: int, month: int, interval: str = "1m") -> str:
    return f"{BASE_URL}/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"


def _free_bytes(path: str) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def _download_with_hash(url: str, dest_path: str, timeout: int = 180, retries: int = 3) -> tuple[int, str]:
    """Stream a URL to disk while hashing it incrementally. Returns (byte_size, sha256_hex)."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            hasher = hashlib.sha256()
            size = 0
            with requests.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code == 404:
                    raise FileNotFoundError(f"404 {url}")
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        hasher.update(chunk)
                        size += len(chunk)
            return size, hasher.hexdigest()
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - network errors of many types
            last_exc = exc
            time.sleep(min(2 ** attempt, 10))
    assert last_exc is not None
    raise last_exc


def _first_data_row(zip_path: str, member: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as raw:
            first_line = raw.readline().decode("utf-8")
    return first_line.strip().split(",")


def _csv_member(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("no CSV member found in archive")
        return names[0]


def _extract_aggtrades(zip_path: str, chunksize: int = 2_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (transact_time_ms int64, price float64, qty float64) from a Binance
    aggTrades monthly zip.

    Reads the inner CSV in bounded-size chunks via pandas' C parser (usecols
    restricts parsing to the 3 needed columns) and spills each chunk straight
    to a small on-disk raw buffer instead of accumulating Python lists/objects
    in memory -- a naive per-row list .append() approach OOMs on this
    environment's 3.7GB RAM for a multi-ten-million-row monthly file.
    """
    import pandas as pd

    member = _csv_member(zip_path)
    first_row = _first_data_row(zip_path, member)
    has_header = _looks_like_header(first_row)

    with tempfile.TemporaryDirectory() as work_dir:
        t_path = os.path.join(work_dir, "t.raw")
        p_path = os.path.join(work_dir, "p.raw")
        q_path = os.path.join(work_dir, "q.raw")
        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            reader = pd.read_csv(text, header=0 if has_header else None, usecols=[1, 2, 5], chunksize=chunksize)
            with open(t_path, "ab") as tf, open(p_path, "ab") as pf, open(q_path, "ab") as qf:
                for chunk in reader:
                    chunk.iloc[:, 0].to_numpy(dtype=np.float64).tofile(pf)
                    chunk.iloc[:, 1].to_numpy(dtype=np.float64).tofile(qf)
                    chunk.iloc[:, 2].to_numpy(dtype=np.int64).tofile(tf)
        t = np.fromfile(t_path, dtype=np.int64)
        p = np.fromfile(p_path, dtype=np.float64)
        q = np.fromfile(q_path, dtype=np.float64)
    return t, p, q


def _looks_like_header(row: list[str]) -> bool:
    try:
        float(row[1])
        return False
    except (ValueError, IndexError):
        return True


def _extract_klines(zip_path: str, chunksize: int = 500_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (open_time_ms int64, high float64, low float64, close float64)
    from a Binance 1m klines monthly zip, using the same chunked/disk-buffered
    approach as _extract_aggtrades (klines files are much smaller, but this
    keeps one parsing strategy for both)."""
    import pandas as pd

    member = _csv_member(zip_path)
    first_row = _first_data_row(zip_path, member)
    has_header = _looks_like_header_kline(first_row)

    with tempfile.TemporaryDirectory() as work_dir:
        ot_path = os.path.join(work_dir, "ot.raw")
        h_path = os.path.join(work_dir, "h.raw")
        l_path = os.path.join(work_dir, "l.raw")
        c_path = os.path.join(work_dir, "c.raw")
        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            reader = pd.read_csv(text, header=0 if has_header else None, usecols=[0, 2, 3, 4], chunksize=chunksize)
            with open(ot_path, "ab") as otf, open(h_path, "ab") as hf, open(l_path, "ab") as lf, open(c_path, "ab") as cf:
                for chunk in reader:
                    chunk.iloc[:, 0].to_numpy(dtype=np.int64).tofile(otf)
                    chunk.iloc[:, 1].to_numpy(dtype=np.float64).tofile(hf)
                    chunk.iloc[:, 2].to_numpy(dtype=np.float64).tofile(lf)
                    chunk.iloc[:, 3].to_numpy(dtype=np.float64).tofile(cf)
        ot = np.fromfile(ot_path, dtype=np.int64)
        h = np.fromfile(h_path, dtype=np.float64)
        l = np.fromfile(l_path, dtype=np.float64)
        c = np.fromfile(c_path, dtype=np.float64)
    return ot, h, l, c


def _looks_like_header_kline(row: list[str]) -> bool:
    try:
        int(row[0])
        return False
    except (ValueError, IndexError):
        return True


def ingest_aggtrades_month(
    symbol: str,
    year: int,
    month: int,
    cache_dir: str,
    tmp_dir: str,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> ManifestEntry:
    url = aggtrades_url(symbol, year, month)
    cov_start, cov_end = _coverage_bounds(year, month)
    attempted_at = _utc_now_iso()
    filename = f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"
    out_dir = os.path.join(cache_dir, "aggtrades", symbol)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, f"{year:04d}-{month:02d}.npz")

    if _free_bytes(cache_dir) < min_free_bytes:
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="aggTrades", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=None,
            sha256=None, validation_status="NOT_ATTEMPTED", gap_status="NOT_INGESTED_DISK_BUDGET",
            cache_path=None, row_count=None, attempted_at=attempted_at,
        )

    os.makedirs(tmp_dir, exist_ok=True)
    tmp_zip = os.path.join(tmp_dir, filename)
    try:
        size, sha = _download_with_hash(url, tmp_zip)
    except FileNotFoundError:
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="aggTrades", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=None,
            sha256=None, validation_status="INVALID", gap_status="MISSING_404",
            cache_path=None, row_count=None, attempted_at=attempted_at,
        )
    except Exception as exc:  # noqa: BLE001
        _safe_remove(tmp_zip)
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="aggTrades", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=None,
            sha256=None, validation_status="INVALID", gap_status="DOWNLOAD_FAILED",
            cache_path=None, row_count=None, attempted_at=attempted_at, error=str(exc),
        )

    try:
        ts, price, qty = _extract_aggtrades(tmp_zip)
        if ts.shape[0] > 1 and not bool(np.all(np.diff(ts) >= 0)):
            order = np.argsort(ts, kind="stable")
            ts, price, qty = ts[order], price[order], qty[order]
        _save_compact_columns(cache_path, t=ts, p=price, q=qty)
        row_count = int(ts.shape[0])
    except Exception as exc:  # noqa: BLE001
        _safe_remove(tmp_zip)
        _safe_remove(cache_path)
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="aggTrades", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=size,
            sha256=sha, validation_status="INVALID", gap_status="CORRUPT_ARCHIVE",
            cache_path=None, row_count=None, attempted_at=attempted_at, error=str(exc),
        )
    finally:
        _safe_remove(tmp_zip)

    return ManifestEntry(
        source_url=url, symbol=symbol, data_type="aggTrades", year=year, month=month,
        coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=size,
        sha256=sha, validation_status="VALID", gap_status="OK",
        cache_path=cache_path, row_count=row_count, attempted_at=attempted_at,
    )


def ingest_klines_month(
    symbol: str,
    year: int,
    month: int,
    cache_dir: str,
    tmp_dir: str,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> ManifestEntry:
    url = klines_url(symbol, year, month)
    cov_start, cov_end = _coverage_bounds(year, month)
    attempted_at = _utc_now_iso()
    filename = f"{symbol}-1m-{year:04d}-{month:02d}.zip"
    out_dir = os.path.join(cache_dir, "klines", symbol)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, f"{year:04d}-{month:02d}.npz")

    if _free_bytes(cache_dir) < min_free_bytes:
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="klines_1m", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=None,
            sha256=None, validation_status="NOT_ATTEMPTED", gap_status="NOT_INGESTED_DISK_BUDGET",
            cache_path=None, row_count=None, attempted_at=attempted_at,
        )

    os.makedirs(tmp_dir, exist_ok=True)
    tmp_zip = os.path.join(tmp_dir, filename)
    try:
        size, sha = _download_with_hash(url, tmp_zip)
    except FileNotFoundError:
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="klines_1m", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=None,
            sha256=None, validation_status="INVALID", gap_status="MISSING_404",
            cache_path=None, row_count=None, attempted_at=attempted_at,
        )
    except Exception as exc:  # noqa: BLE001
        _safe_remove(tmp_zip)
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="klines_1m", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=None,
            sha256=None, validation_status="INVALID", gap_status="DOWNLOAD_FAILED",
            cache_path=None, row_count=None, attempted_at=attempted_at, error=str(exc),
        )

    try:
        ot, high, low, close = _extract_klines(tmp_zip)
        if ot.shape[0] > 1 and not bool(np.all(np.diff(ot) >= 0)):
            order = np.argsort(ot, kind="stable")
            ot, high, low, close = ot[order], high[order], low[order], close[order]
        _save_compact_columns(cache_path, t=ot, h=high, l=low, c=close)
        row_count = int(ot.shape[0])
    except Exception as exc:  # noqa: BLE001
        _safe_remove(tmp_zip)
        _safe_remove(cache_path)
        return ManifestEntry(
            source_url=url, symbol=symbol, data_type="klines_1m", year=year, month=month,
            coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=size,
            sha256=sha, validation_status="INVALID", gap_status="CORRUPT_ARCHIVE",
            cache_path=None, row_count=None, attempted_at=attempted_at, error=str(exc),
        )
    finally:
        _safe_remove(tmp_zip)

    return ManifestEntry(
        source_url=url, symbol=symbol, data_type="klines_1m", year=year, month=month,
        coverage_start=cov_start, coverage_end=cov_end, filename=filename, byte_size=size,
        sha256=sha, validation_status="VALID", gap_status="OK",
        cache_path=cache_path, row_count=row_count, attempted_at=attempted_at,
    )


def _save_compact_columns(path: str, **arrays: np.ndarray) -> None:
    """Write each named array to `path` (an .npz) one at a time so peak memory
    is bounded by the single largest column, not the sum of all columns."""
    import zipfile as _zipfile

    tmp_path = path + ".tmp"
    with _zipfile.ZipFile(tmp_path, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(f"{name}.npy", buf.getvalue())
            del buf
    os.replace(tmp_path, path)


def _safe_remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_manifest(manifest_path: str, entry: ManifestEntry) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(entry.to_json() + "\n")


def load_manifest(manifest_path: str) -> list[dict]:
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_aggtrades_month(cache_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    return data["t"], data["p"], data["q"]


def load_klines_month(cache_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    return data["t"], data["h"], data["l"], data["c"]
