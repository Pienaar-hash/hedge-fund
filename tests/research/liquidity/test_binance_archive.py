"""Tests for research/liquidity/binance_archive.py — no network calls.

Only the deterministic, filesystem-local pieces are exercised here: URL
construction, month iteration, manifest serialization, and column-pruning
extraction against a locally constructed fixture zip (never downloaded).
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import date

from research.liquidity.binance_archive import (
    ManifestEntry,
    _extract_aggtrades,
    _extract_klines,
    aggtrades_url,
    append_manifest,
    klines_url,
    load_manifest,
    month_range,
)


def test_month_range_inclusive():
    months = list(month_range(date(2022, 11, 1), date(2023, 2, 1)))
    assert months == [(2022, 11), (2022, 12), (2023, 1), (2023, 2)]


def test_aggtrades_url_shape():
    url = aggtrades_url("BTCUSDT", 2022, 1)
    assert url == "https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2022-01.zip"


def test_klines_url_shape():
    url = klines_url("ETHUSDT", 2023, 6)
    assert url == "https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/1m/ETHUSDT-1m-2023-06.zip"


def _make_fixture_zip(tmp_path, name: str, csv_body: str) -> str:
    zip_path = str(tmp_path / name)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(name.replace(".zip", ".csv"), csv_body)
    return zip_path


def test_extract_aggtrades_no_header(tmp_path):
    csv_body = (
        "1,100.5,0.01,1,1,1700000000123,True\n"
        "2,101.2,0.02,2,2,1700000000456,False\n"
    )
    zip_path = _make_fixture_zip(tmp_path, "fixture.zip", csv_body)
    t, p, q = _extract_aggtrades(zip_path)
    assert list(t) == [1700000000123, 1700000000456]
    assert list(p) == [100.5, 101.2]
    assert list(q) == [0.01, 0.02]


def test_extract_aggtrades_with_header(tmp_path):
    csv_body = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,100.5,0.01,1,1,1700000000123,True\n"
    )
    zip_path = _make_fixture_zip(tmp_path, "fixture_h.zip", csv_body)
    t, p, q = _extract_aggtrades(zip_path)
    assert list(t) == [1700000000123]
    assert list(p) == [100.5]


def test_extract_klines_no_header(tmp_path):
    csv_body = (
        "1700000000000,100.0,105.0,99.0,102.0,10.5,1700000059999,0,0,0,0,0\n"
        "1700000060000,102.0,103.0,101.0,101.5,8.2,1700000119999,0,0,0,0,0\n"
    )
    zip_path = _make_fixture_zip(tmp_path, "klines.zip", csv_body)
    ot, hi, lo, cl = _extract_klines(zip_path)
    assert list(ot) == [1700000000000, 1700000060000]
    assert list(hi) == [105.0, 103.0]
    assert list(lo) == [99.0, 101.0]
    assert list(cl) == [102.0, 101.5]


def test_manifest_roundtrip(tmp_path):
    manifest_path = str(tmp_path / "manifest.jsonl")
    entry = ManifestEntry(
        source_url="https://example.invalid/x.zip", symbol="BTCUSDT", data_type="aggTrades", year=2022, month=1,
        coverage_start="2022-01-01T00:00:00Z", coverage_end="2022-02-01T00:00:00Z", filename="x.zip",
        byte_size=123, sha256="deadbeef", validation_status="VALID", gap_status="OK", cache_path="/tmp/x.npz",
        row_count=42, attempted_at="2026-09-03T00:00:00Z",
    )
    append_manifest(manifest_path, entry)
    loaded = load_manifest(manifest_path)
    assert len(loaded) == 1
    assert loaded[0]["sha256"] == "deadbeef"
    assert loaded[0]["row_count"] == 42
