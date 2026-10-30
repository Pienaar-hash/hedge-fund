"""Bounded daily USD-M archive acquisition; no event or outcome construction.

The legacy monthly APIs are deliberately not used. Live memory consists of one
fixed-size column partition, one capped CSV line and one download/hash block.
All outputs must go to a fresh run directory. Successful manifests are published
only after source checksum, ZIP CRC and streaming validation have completed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import struct
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import requests

Kind = Literal["aggTrades", "klines_1m"]
BASE = "https://data.binance.vision/data/futures/um/daily"
AGG_HEADER = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"
KLINE_HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
SCHEMAS = {
    "aggTrades": {"id": "<i8", "p": "<f8", "q": "<f8", "first_id": "<i8",
                  "last_id": "<i8", "t": "<i8"},
    "klines_1m": {"t": "<i8", "o": "<f8", "h": "<f8", "l": "<f8", "c": "<f8"},
}


class LimitError(RuntimeError):
    """An explicit resource/admission limit was reached."""


class IntegrityError(ValueError):
    """A source failed validation; never use its partitions as complete data."""


@dataclass(frozen=True)
class Limits:
    chunk_rows: int = 100_000
    max_rows: int = 10_000_000
    max_archive_bytes: int = 512 * 1024**2
    max_csv_bytes: int = 2 * 1024**3
    max_output_bytes: int = 4 * 1024**3
    max_line_bytes: int = 4096
    min_free_bytes: int = 3 * 1024**3
    min_available_ram: int = 1024**3
    timeout_seconds: float = 900

    def __post_init__(self):
        if not 1 <= self.chunk_rows <= 100_000:
            raise ValueError("chunk_rows must be within 1..100000")
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("limits must be positive")


def daily_url(symbol: str, day: date, kind: Kind) -> str:
    if symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
        raise ValueError("symbol outside frozen universe")
    if not date(2022, 1, 1) <= day <= date(2025, 12, 31):
        raise ValueError("day outside frozen period")
    if kind == "aggTrades":
        return f"{BASE}/aggTrades/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
    if kind == "klines_1m":
        return f"{BASE}/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"
    raise ValueError("unsupported data type")


def host_health(root: Path) -> dict:
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal"}:
            mem[key + "_bytes"] = int(value.split()[0]) * 1024
    return {**mem, "disk_available_bytes": shutil.disk_usage(root).free,
            "memory_pressure": Path("/proc/pressure/memory").read_text(),
            "io_pressure": Path("/proc/pressure/io").read_text(),
            "loadavg": Path("/proc/loadavg").read_text().strip()}


class Budget:
    """Shared across all three sources; accounts cumulative actual writes."""

    def __init__(self, root: Path, limits: Limits):
        self.root, self.limits = root, limits
        self.started = time.monotonic()
        self.written = 0
        self.health_checks = 0
        self.min_available_ram_seen = None
        self.min_free_disk_seen = None
        self.max_memory_full_avg10 = 0.0

    def check(self, pending_bytes: int = 0):
        if time.monotonic() - self.started > self.limits.timeout_seconds:
            raise LimitError("elapsed-time limit")
        if self.written + pending_bytes > self.limits.max_output_bytes:
            raise LimitError("total write-byte limit")
        health = host_health(self.root)
        pressure = health.get("memory_pressure", "")
        full = next((line for line in pressure.splitlines() if line.startswith("full ")), "")
        avg10 = float(dict(item.split("=") for item in full.split()[1:]).get("avg10", "0"))
        self.max_memory_full_avg10 = max(self.max_memory_full_avg10, avg10)
        if avg10 > 1.0:
            raise LimitError("host full memory-pressure avg10 exceeds 1 percent")
        ram, disk = health["MemAvailable_bytes"], health["disk_available_bytes"]
        self.health_checks += 1
        self.min_available_ram_seen = min(ram, self.min_available_ram_seen or ram)
        self.min_free_disk_seen = min(disk, self.min_free_disk_seen or disk)
        if ram < self.limits.min_available_ram:
            raise LimitError("host available-memory floor")
        if disk - pending_bytes < self.limits.min_free_bytes:
            raise LimitError("available-disk floor")

    def charge(self, amount: int):
        self.check(amount)
        self.written += amount


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict, budget: Budget):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    budget.charge(len(payload))
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("xb") as handle:
        handle.write(payload)
    temp.replace(path)


def download(url: str, dest: Path, cap: int, budget: Budget) -> tuple[int, str]:
    """One attempt, capped streaming response, no automatic redirects or retries."""
    budget.check()
    digest, size = hashlib.sha256(), 0
    partial = dest.with_suffix(dest.suffix + ".partial")
    # Identity encoding makes the source-byte cap independent of HTTP compression.
    with requests.get(url, stream=True, timeout=(10, 30), allow_redirects=False,
                      headers={"Accept-Encoding": "identity"}) as response:
        if response.status_code != 200:
            raise IntegrityError(f"HTTP {response.status_code}: {url}")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise IntegrityError("unexpected HTTP content encoding")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > cap:
            raise LimitError("source Content-Length exceeds cap")
        with partial.open("xb") as handle:
            for block in response.iter_content(chunk_size=1024**2):
                if size + len(block) > cap:
                    raise LimitError("source download-byte limit")
                budget.charge(len(block))
                handle.write(block)
                digest.update(block)
                size += len(block)
        if length is not None and size != int(length):
            raise IntegrityError("truncated response")
    partial.replace(dest)
    return size, digest.hexdigest()


def verify_checksum(checksum_path: Path, filename: str, actual: str) -> str:
    # Checksum responses are capped to 4 KiB before this read.
    if checksum_path.stat().st_size > 4096:
        raise LimitError("checksum-byte limit")
    text = checksum_path.read_text(encoding="ascii").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?([^\s]+)", text)
    if not match or match[2] != filename:
        raise IntegrityError("invalid checksum identity")
    expected = match[1].lower()
    if expected != actual:
        raise IntegrityError("source checksum mismatch")
    return expected


def _int(value: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise IntegrityError("invalid unsigned integer")
    parsed = int(value)
    if parsed > np.iinfo(np.int64).max:
        raise IntegrityError("integer outside int64")
    return parsed


def _float(value: str, *, zero: bool = False) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed < 0 if zero else parsed <= 0):
        raise IntegrityError("nonfinite or invalid numeric value")
    return parsed


def _parse(fields: list[str], kind: Kind) -> tuple[int | float, ...]:
    if kind == "aggTrades":
        if len(fields) != 7 or fields[6].lower() not in {"true", "false"}:
            raise IntegrityError("invalid aggTrades schema")
        aid, p, q, first, last, t = (_int(fields[0]), _float(fields[1]),
                                    _float(fields[2]), _int(fields[3]),
                                    _int(fields[4]), _int(fields[5]))
        if first > last:
            raise IntegrityError("invalid underlying trade ID range")
        return aid, p, q, first, last, t
    if len(fields) != 12:
        raise IntegrityError("invalid kline schema")
    t, o, h, low, close = (_int(fields[0]), *(_float(v) for v in fields[1:5]))
    if not low <= min(o, close) <= max(o, close) <= h:
        raise IntegrityError("invalid OHLC bounds")
    if _int(fields[6]) != t + 59_999:
        raise IntegrityError("invalid 1m close timestamp")
    for index in (5, 7, 9, 10):
        _float(fields[index], zero=True)
    _int(fields[8])
    _float(fields[11], zero=True)
    return t, o, h, low, close


def partition_archive(archive: Path, day: date, kind: Kind, dest: Path,
                      budget: Budget) -> dict:
    """Validate to EOF (including ZIP CRC), preserving continuity across chunks.

    On failure no COMPLETE/manifest marker is published. Partial partitions are
    retained for diagnosis, and consumers must require the successful manifest.
    """
    limits = budget.limits
    start = int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp() * 1000)
    end = start + 86_400_000
    dest.mkdir()  # exclusive: never overwrite previous output
    columns = {key: np.empty(limits.chunk_rows, dtype=dtype)
               for key, dtype in SCHEMAS[kind].items()}
    parts, rows, used, csv_bytes = [], 0, 0, 0
    prior_t = prior_id = prior_last = first_t = first_id = None
    missing_ids = unrepresented_raw_ids = raw_discontinuities = max_silence_ms = 0
    gap_samples = []

    def flush():
        nonlocal used
        index = len(parts)
        files = {}
        # Explicit caps on rows/chunk size also bound this manifest's size.
        if index >= math.ceil(limits.max_rows / limits.chunk_rows):
            raise LimitError("partition-count limit")
        for key, array in columns.items():
            path = dest / f"part-{index:06d}-{key}.npy"
            # .npy v1 header is smaller than this conservative 256-byte allowance.
            budget.check(used * array.dtype.itemsize + 256)
            with path.open("xb") as handle:
                np.save(handle, array[:used], allow_pickle=False)
            budget.written += path.stat().st_size
            files[key] = {"path": path.name, "bytes": path.stat().st_size,
                          "sha256": file_hash(path), "dtype": array.dtype.str}
        parts.append({"rows": used, "columns": files})
        used = 0

    # The compressed archive itself is capped before ZipFile builds its index.
    if archive.stat().st_size > limits.max_archive_bytes:
        raise LimitError("compressed archive-byte limit")
    # Bound ZIP's central-directory allocation before ZipFile constructs objects.
    # This deliberately rejects ZIP64/multi-volume archives in this daily tranche.
    with archive.open("rb") as handle:
        handle.seek(max(0, archive.stat().st_size - 65_557))
        tail = handle.read(65_557)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or len(tail) - eocd < 22:
        raise IntegrityError("missing ZIP end record")
    _, disk, cd_disk, disk_entries, entries, cd_bytes, _, comment = struct.unpack(
        "<4s4H2LH", tail[eocd:eocd + 22])
    if (disk or cd_disk or entries != 1 or disk_entries != 1 or cd_bytes > 65_536
            or len(tail) - eocd != 22 + comment):
        raise IntegrityError("ZIP directory exceeds single-member bounds")
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if len(members) != 1 or not members[0].filename.endswith(".csv"):
            raise IntegrityError("expected exactly one CSV member")
        member = members[0]
        if member.file_size > limits.max_csv_bytes:
            raise LimitError("declared decompressed-byte limit")
        if member.flag_bits & 1:
            raise IntegrityError("encrypted source")
        with zf.open(member) as raw:
            line_number = 0
            while True:
                line = raw.readline(limits.max_line_bytes + 1)
                if not line:
                    break
                line_number += 1
                csv_bytes += len(line)
                if len(line) > limits.max_line_bytes or csv_bytes > limits.max_csv_bytes:
                    raise LimitError("CSV line/decompressed-byte limit")
                text = line.decode("ascii").rstrip("\r\n")
                if line_number == 1 and text == (AGG_HEADER if kind == "aggTrades" else KLINE_HEADER):
                    continue
                if rows >= limits.max_rows:
                    raise LimitError("source row limit")
                values = _parse(text.split(","), kind)
                t = int(values[-1] if kind == "aggTrades" else values[0])
                if not start <= t < end:
                    raise IntegrityError("timestamp outside UTC day / incorrect timestamp units")
                if prior_t is not None and t < prior_t:
                    raise IntegrityError("nonmonotonic timestamp")
                if kind == "aggTrades":
                    aid, first, last = int(values[0]), int(values[3]), int(values[4])
                    if prior_id is not None:
                        if aid <= prior_id or first <= prior_last:
                            raise IntegrityError("duplicate/nonmonotonic trade IDs")
                        gap = aid - prior_id - 1
                        raw_gap = first - prior_last - 1
                        missing_ids += gap
                        # USD-M aggregates exclude insurance-fund/ADL trades.
                        # A discontinuity in f/l is not proof of a missing a.
                        # Its cause remains unresolved, so it still blocks acceptance.
                        unrepresented_raw_ids += raw_gap
                        raw_discontinuities += int(raw_gap > 0)
                        if (gap or raw_gap) and len(gap_samples) < 20:
                            gap_samples.append({"after_id": prior_id, "missing_agg_ids": gap,
                                                "unrepresented_underlying_ids": raw_gap})
                        max_silence_ms = max(max_silence_ms, t - prior_t)
                    prior_id, prior_last = aid, last
                    if first_id is None:
                        first_id = aid
                elif t % 60_000 or (prior_t is not None and t <= prior_t):
                    raise IntegrityError("duplicate/unaligned kline timestamp")
                if first_t is None:
                    first_t = t
                for array, value in zip(columns.values(), values):
                    array[used] = value
                used += 1
                rows += 1
                prior_t = t
                if used == limits.chunk_rows:
                    flush()
                elif rows % 4096 == 0:
                    budget.check()
        # Reading the member to EOF checks CRC; never use testzip (second full pass).
        if csv_bytes != member.file_size:
            raise IntegrityError("decompressed size mismatch")
    if used:
        flush()
    if not rows:
        raise IntegrityError("empty source")
    gaps = {"missing_aggregate_ids": missing_ids,
            "unrepresented_underlying_ids": unrepresented_raw_ids,
            "underlying_id_discontinuities": raw_discontinuities,
            "samples": gap_samples, "max_observed_trade_silence_ms": max_silence_ms,
            "missing_1m_intervals": 1440 - rows if kind == "klines_1m" else None,
            "leading_trade_silence_ms": first_t - start if kind == "aggTrades" else None,
            "trailing_trade_silence_ms": end - 1 - prior_t if kind == "aggTrades" else None,
            "cross_day_trade_id_continuity": "NOT_ASSESSED",
            "trade_silence_interpretation": "inactivity, not proof of missing records"}
    if missing_ids or (kind == "klines_1m" and rows != 1440):
        validation_status, gap_status = "INVALID", "GAPS_DETECTED"
    elif raw_discontinuities:
        validation_status = "REVIEW_REQUIRED"
        gap_status = "UNDERLYING_ID_DISCONTINUITY_UNRESOLVED"
    else:
        validation_status, gap_status = "VALID", "OK_WITHIN_DAY"
    return {"rows": rows, "csv_bytes": csv_bytes, "first_timestamp_ms": first_t,
            "last_timestamp_ms": prior_t, "first_aggregate_id": first_id,
            "last_aggregate_id": prior_id, "gaps": gaps, "partitions": parts,
            "partition_bytes": sum(f["bytes"] for p in parts for f in p["columns"].values()),
            "validation_rule_version": 2,
            "validation_status": validation_status, "gap_status": gap_status,
            "zip_crc": "VERIFIED"}


def ingest_day(symbol: str, day: date, kind: Kind, root: Path, budget: Budget) -> dict:
    """Acquire exactly one source; fail closed with a retained failure record."""
    url = daily_url(symbol, day, kind)
    source = root / f"{symbol}-{day.isoformat()}-{kind}"
    source.mkdir()
    archive = source / url.rsplit("/", 1)[1]
    started = time.monotonic()
    entry = {"source_url": url, "checksum_url": url + ".CHECKSUM", "symbol": symbol,
             "data_type": kind, "coverage_start": day.isoformat() + "T00:00:00Z",
             "coverage_end": (day + timedelta(days=1)).isoformat() + "T00:00:00Z",
             "filename": archive.name, "limits": asdict(budget.limits)}
    try:
        checksum = archive.with_suffix(".zip.CHECKSUM")
        checksum_size, checksum_hash = download(url + ".CHECKSUM", checksum, 4096, budget)
        size, digest = download(url, archive, budget.limits.max_archive_bytes, budget)
        entry.update({"byte_size": size, "sha256": digest,
                      "checksum_file_bytes": checksum_size, "checksum_file_sha256": checksum_hash})
        entry["source_expected_sha256"] = verify_checksum(checksum, archive.name, digest)
        entry["source_checksum"] = "VERIFIED"
        entry.update(partition_archive(archive, day, kind, source / "partitions", budget))
        entry["elapsed_seconds"] = time.monotonic() - started
        write_json(source / "manifest.json", entry, budget)
        if entry["validation_status"] != "VALID":
            raise IntegrityError(f"source validation blocked: {entry['gap_status']}; see manifest")
        return entry
    except Exception as exc:
        status = "REVIEW_REQUIRED" if entry.get("validation_status") == "REVIEW_REQUIRED" else "INVALID"
        entry.update({"validation_status": status, "error": f"{type(exc).__name__}: {exc}",
                      "elapsed_seconds": time.monotonic() - started})
        # Diagnostic evidence only. Keep this small and don't allocate an archive-sized buffer.
        (source / "FAILURE.json").write_text(json.dumps(entry, sort_keys=True, indent=2) + "\n")
        raise
