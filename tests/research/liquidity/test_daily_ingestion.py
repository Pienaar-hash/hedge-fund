"""Offline resource/integrity tests for the bounded daily path."""
import dataclasses
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from research.liquidity import daily_ingestion as daily

DAY = date(2022, 1, 2)
START = 1641081600000


def fixture_archive(tmp_path, body):
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("input.csv", body)
    return path


def budget(tmp_path, **kwargs):
    return daily.Budget(tmp_path, daily.Limits(chunk_rows=2, min_free_bytes=1,
                                             min_available_ram=1, **kwargs))


def agg(aid, t=START, first=None):
    first = aid if first is None else first
    return f"{aid},47000.5,0.01,{first},{first},{t},true\n"


def parse(tmp_path, body, **kwargs):
    return daily.partition_archive(fixture_archive(tmp_path, body), DAY,
                                   "aggTrades", tmp_path / "parts", budget(tmp_path, **kwargs))


def test_daily_identities_and_scope():
    assert daily.daily_url("BTCUSDT", DAY, "aggTrades").endswith(
        "/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2022-01-02.zip")
    assert daily.daily_url("BTCUSDT", DAY, "klines_1m").endswith(
        "/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2022-01-02.zip")
    for symbol, day, kind in [("OTHER", DAY, "aggTrades"),
                              ("BTCUSDT", date(2026, 1, 1), "aggTrades"),
                              ("BTCUSDT", DAY, "trades")]:
        with pytest.raises(ValueError):
            daily.daily_url(symbol, day, kind)


def test_partition_bound_and_deterministic_replay(tmp_path):
    archive = fixture_archive(tmp_path, daily.AGG_HEADER + "\n" + "".join(agg(i) for i in range(1, 8)))
    first = daily.partition_archive(archive, DAY, "aggTrades", tmp_path / "a", budget(tmp_path))
    second = daily.partition_archive(archive, DAY, "aggTrades", tmp_path / "b", budget(tmp_path))
    assert first == second
    assert [p["rows"] for p in first["partitions"]] == [2, 2, 2, 1]
    assert first["gaps"]["missing_aggregate_ids"] == 0
    for part in first["partitions"]:
        for column in part["columns"].values():
            path = tmp_path / "a" / column["path"]
            assert len(np.load(path, allow_pickle=False)) <= 2
            assert daily.file_hash(path) == column["sha256"]


def test_gap_across_partition_boundary_is_not_inactivity(tmp_path):
    result = parse(tmp_path, agg(1) + agg(2) + agg(4, START + 1000))
    assert result["validation_status"] == "INVALID"
    assert result["gaps"]["missing_aggregate_ids"] == 1
    assert result["gaps"]["unrepresented_underlying_ids"] == 1
    assert result["gaps"]["max_observed_trade_silence_ms"] == 1000


@pytest.mark.parametrize("body", [agg(1) + agg(2) + agg(2),
                                  agg(1, START + 2) + agg(2, START + 3) + agg(3, START + 1),
                                  agg(1, START - 1), agg(1, START * 1000),
                                  agg(1).replace("47000.5", "nan"),
                                  agg(1).replace("0.01", "-1"),
                                  "unknown,header\n" + agg(1), ""])
def test_invalid_source_rejected(tmp_path, body):
    with pytest.raises((daily.IntegrityError, ValueError)):
        parse(tmp_path, body)


@pytest.mark.parametrize("kwargs,body", [({"max_rows": 2}, agg(1) + agg(2) + agg(3)),
                                        ({"max_csv_bytes": 8}, agg(1)),
                                        ({"max_line_bytes": 8}, agg(1)),
                                        ({"max_archive_bytes": 8}, agg(1)),
                                        ({"max_output_bytes": 8}, agg(1))])
def test_explicit_size_limits(tmp_path, kwargs, body):
    with pytest.raises(daily.LimitError):
        parse(tmp_path, body, **kwargs)


def test_health_and_time_limits(tmp_path, monkeypatch):
    b = budget(tmp_path)
    monkeypatch.setattr(daily, "host_health", lambda _: {"MemAvailable_bytes": 0, "disk_available_bytes": 10**12})
    with pytest.raises(daily.LimitError, match="memory"):
        b.check()
    monkeypatch.setattr(daily, "host_health", lambda _: {"MemAvailable_bytes": 10**12, "disk_available_bytes": 0})
    with pytest.raises(daily.LimitError, match="disk"):
        b.check()
    monkeypatch.setattr(daily.time, "monotonic", lambda: b.started + 1000)
    with pytest.raises(daily.LimitError, match="time"):
        b.check()


def test_checksum_identity_and_digest(tmp_path):
    path = tmp_path / "fixture.CHECKSUM"
    digest = "a" * 64
    path.write_text(digest + "  input.zip\n")
    assert daily.verify_checksum(path, "input.zip", digest) == digest
    with pytest.raises(daily.IntegrityError, match="mismatch"):
        daily.verify_checksum(path, "input.zip", "b" * 64)
    with pytest.raises(daily.IntegrityError, match="identity"):
        daily.verify_checksum(path, "other.zip", digest)


def test_kline_full_day_and_missing_interval(tmp_path):
    body = "".join(f"{t},10,12,8,11,1,{t+59999},10,1,0,0,0\n"
                   for t in range(START, START + 86_400_000, 60_000))
    archive = fixture_archive(tmp_path, daily.KLINE_HEADER + "\n" + body)
    b = daily.Budget(tmp_path, dataclasses.replace(budget(tmp_path).limits, chunk_rows=100))
    result = daily.partition_archive(archive, DAY, "klines_1m", tmp_path / "full", b)
    assert result["rows"] == 1440 and result["validation_status"] == "VALID"
    archive = fixture_archive(tmp_path, body.split("\n", 1)[1])
    result = daily.partition_archive(archive, DAY, "klines_1m", tmp_path / "missing", b)
    assert result["gaps"]["missing_1m_intervals"] == 1
    assert result["validation_status"] == "INVALID"


def test_zip_crc_and_directory_bounds(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("input.csv", agg(1))
    payload = bytearray(archive.read_bytes())
    pos = payload.index(b"47000.5")
    payload[pos] = ord("5")
    archive.write_bytes(payload)
    with pytest.raises(zipfile.BadZipFile):
        daily.partition_archive(archive, DAY, "aggTrades", tmp_path / "bad", budget(tmp_path))
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.csv", agg(1))
        zf.writestr("b.csv", agg(2))
    with pytest.raises(daily.IntegrityError, match="directory"):
        daily.partition_archive(archive, DAY, "aggTrades", tmp_path / "multi", budget(tmp_path))


def test_download_cap_without_content_length(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, chunk_size):
            yield b"1234"
            yield b"5678"

    monkeypatch.setattr(daily.requests, "get", lambda *a, **kw: Response())
    with pytest.raises(daily.LimitError):
        daily.download("https://example.invalid", tmp_path / "input.zip", 5, budget(tmp_path))
    assert not (tmp_path / "input.zip").exists()
    assert (tmp_path / "input.zip.partial").read_bytes() == b"1234"


def test_checksum_failure_never_starts_parser(tmp_path, monkeypatch):
    def fake_download(url, dest, cap, b):
        payload = (("a" * 64 + "  " + dest.name.removesuffix(".CHECKSUM") + "\n").encode()
                   if url.endswith(".CHECKSUM") else b"wrong archive")
        dest.write_bytes(payload)
        return len(payload), hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(daily, "download", fake_download)
    with pytest.raises(daily.IntegrityError, match="checksum mismatch"):
        daily.ingest_day("BTCUSDT", DAY, "aggTrades", tmp_path, budget(tmp_path))
    assert not list(tmp_path.rglob("partitions"))
    assert not list(tmp_path.rglob("manifest.json"))
    assert len(list(tmp_path.rglob("FAILURE.json"))) == 1


def test_aggregate_memory_allocations_are_chunk_sized(tmp_path, monkeypatch):
    original = daily.np.empty
    allocations = []

    def bounded_empty(shape, *args, **kwargs):
        assert shape == 2
        allocations.append(shape)
        return original(shape, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("whole-tape reconstruction is forbidden")

    monkeypatch.setattr(daily.np, "empty", bounded_empty)
    monkeypatch.setattr(daily.np, "fromfile", forbidden)
    monkeypatch.setattr(daily.np, "concatenate", forbidden)
    result = parse(tmp_path, "".join(agg(i) for i in range(1, 101)))
    assert result["rows"] == 100 and len(result["partitions"]) == 50
    assert len(allocations) == 6


def test_full_memory_pressure_rejects_job(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "host_health", lambda _: {
        "MemAvailable_bytes": 10**12, "disk_available_bytes": 10**12,
        "memory_pressure": "some avg10=2.00\nfull avg10=1.01 avg60=0.00 total=1\n"})
    with pytest.raises(daily.LimitError, match="pressure"):
        budget(tmp_path).check()


CONTEXT = json.loads((Path(__file__).parent / "fixtures" /
                      "h_liq_001_underlying_id_context.json").read_text())


@pytest.mark.parametrize("window", CONTEXT["windows"],
                         ids=lambda window: str(window["after_aggregate_id"]))
@pytest.mark.parametrize("chunk_rows", [4, 100_000])
def test_original_anomaly_context_requires_review(tmp_path, window, chunk_rows):
    # Eight original rows only, never a replay of the daily benchmark.
    # Chunk size 4 puts the original anomaly across a synthetic chunk boundary.
    assert CONTEXT["source_archive_sha256"] == "87f3d9e800bf9abec643ce0e9add98eedc927bea4c8006da652235ea779cc429"
    assert hashlib.sha256(window["csv"].encode()).hexdigest() == window["csv_sha256"]
    archive = fixture_archive(tmp_path, window["csv"])
    limits = dataclasses.replace(budget(tmp_path).limits, chunk_rows=chunk_rows)
    result = daily.partition_archive(archive, DAY, "aggTrades", tmp_path / "context",
                                     daily.Budget(tmp_path, limits))
    assert result["rows"] == 8
    assert result["validation_rule_version"] == 2
    assert result["validation_status"] == "REVIEW_REQUIRED"
    assert result["gap_status"] == "UNDERLYING_ID_DISCONTINUITY_UNRESOLVED"
    assert result["gaps"]["missing_aggregate_ids"] == 0
    assert result["gaps"]["unrepresented_underlying_ids"] == 1
    assert result["gaps"]["underlying_id_discontinuities"] == 1
    assert result["gaps"]["samples"] == [{
        "after_id": window["after_aggregate_id"], "missing_agg_ids": 0,
        "unrepresented_underlying_ids": 1}]
    assert "missing_underlying_trade_ids" not in result["gaps"]


@pytest.mark.parametrize("aggregate_gap,raw_gap,status", [
    (False, False, "VALID"), (True, False, "INVALID"),
    (False, True, "REVIEW_REQUIRED"), (True, True, "INVALID")])
def test_aggregate_and_underlying_gaps_have_separate_classification(
        tmp_path, aggregate_gap, raw_gap, status):
    # Third row lies across the chunk boundary, independent of gap type.
    third_id = 4 if aggregate_gap else 3
    third_first = 6 if raw_gap else 3
    result = parse(tmp_path, agg(1) + agg(2) + agg(third_id, first=third_first))
    assert result["validation_status"] == status
    assert result["gaps"]["missing_aggregate_ids"] == int(aggregate_gap)
    assert result["gaps"]["unrepresented_underlying_ids"] == (3 if raw_gap else 0)
    assert result["gaps"]["underlying_id_discontinuities"] == int(raw_gap)


@pytest.mark.parametrize("body", [agg(1, first=10) + agg(2, first=10),
                                  agg(1, first=10) + agg(2, first=9),
                                  agg(2) + agg(1)])
def test_id_overlap_and_ordering_still_fail(tmp_path, body):
    with pytest.raises(daily.IntegrityError, match="trade IDs"):
        parse(tmp_path, body)


def test_unresolved_underlying_gap_still_blocks_ingestion(tmp_path, monkeypatch):
    # The only permitted downloads in this fixture are mocked local bytes.
    archive = fixture_archive(tmp_path, agg(1) + agg(2, first=3))
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    def fake_download(url, dest, cap, b):
        data = ((digest + "  " + dest.name.removesuffix(".CHECKSUM") + "\n").encode()
                if url.endswith(".CHECKSUM") else payload)
        dest.write_bytes(data)
        return len(data), hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(daily, "download", fake_download)
    with pytest.raises(daily.IntegrityError, match="UNDERLYING_ID_DISCONTINUITY_UNRESOLVED"):
        daily.ingest_day("BTCUSDT", DAY, "aggTrades", tmp_path, budget(tmp_path))
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())
    failure = json.loads(next(tmp_path.rglob("FAILURE.json")).read_text())
    assert manifest["validation_status"] == failure["validation_status"] == "REVIEW_REQUIRED"
    assert manifest["gaps"]["missing_aggregate_ids"] == 0
    assert failure["gaps"]["unrepresented_underlying_ids"] == 1
