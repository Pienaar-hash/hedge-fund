from __future__ import annotations

import json

import pytest

from execution import executor_live
from execution.intel import pipeline_v6_compare


def test_pipeline_compare_scheduler(monkeypatch):
    calls = []

    def fake_compare():
        calls.append(1)

    monkeypatch.setattr(pipeline_v6_compare, "compare_pipeline_v6", fake_compare)
    monkeypatch.setattr(executor_live, "_PIPELINE_V6_COMPARE_INTERVAL_S", 0.0, raising=False)
    monkeypatch.setattr(executor_live, "_LAST_PIPELINE_V6_COMPARE", 0.0, raising=False)
    executor_live._maybe_run_pipeline_v6_compare()
    assert len(calls) == 1
    monkeypatch.setattr(executor_live.time, "time", lambda: 100.0)
    monkeypatch.setattr(executor_live, "_PIPELINE_V6_COMPARE_INTERVAL_S", 60.0, raising=False)
    monkeypatch.setattr(executor_live, "_LAST_PIPELINE_V6_COMPARE", 100.0, raising=False)
    executor_live._maybe_run_pipeline_v6_compare()
    assert len(calls) == 1


def test_pipeline_compare_emits_jsonl_records(tmp_path, monkeypatch):
    entries: list[dict] = []
    shadow_rows = [
        {"symbol": "BTCUSDT", "timestamp": 1.0, "risk_decision": {"allowed": True}, "sizing": {"sized_gross_usd": 100.0}},
        {"symbol": "ETHUSDT", "timestamp": 2.0, "risk_decision": {"allowed": True}, "sizing": {"sized_gross_usd": 200.0}},
    ]
    orders_rows = [
        {"symbol": "BTCUSDT", "ts": 1.1, "notional": 100.0},
    ]

    class DummyLogger:
        def write(self, payload):
            entries.append(payload)

    monkeypatch.setattr(pipeline_v6_compare, "get_logger", lambda _path: DummyLogger())
    monkeypatch.setattr(pipeline_v6_compare, "MIN_SAMPLE_SIZE_FOR_STEADY_STATE", 3)
    orig_shadow = pipeline_v6_compare.PIPELINE_SHADOW_LOG
    orig_log = pipeline_v6_compare.COMPARE_LOG_PATH
    orig_state = pipeline_v6_compare.COMPARE_STATE_PATH
    pipeline_v6_compare.PIPELINE_SHADOW_LOG = tmp_path / "shadow.jsonl"
    pipeline_v6_compare.COMPARE_LOG_PATH = tmp_path / "compare.jsonl"
    pipeline_v6_compare.COMPARE_STATE_PATH = tmp_path / "state/summary.json"
    shadow_lines = "\n".join(json.dumps(row) for row in shadow_rows) + "\n"
    order_lines = "\n".join(json.dumps(row) for row in orders_rows) + "\n"
    pipeline_v6_compare.PIPELINE_SHADOW_LOG.write_text(shadow_lines)
    (tmp_path / "orders.jsonl").write_text(order_lines)
    try:
        summary = pipeline_v6_compare.compare_pipeline_v6(
            shadow_limit=10,
            orders_path=tmp_path / "orders.jsonl",
            metrics_path=tmp_path / "metrics.jsonl",
        )
    finally:
        pipeline_v6_compare.PIPELINE_SHADOW_LOG = orig_shadow
        pipeline_v6_compare.COMPARE_LOG_PATH = orig_log
        pipeline_v6_compare.COMPARE_STATE_PATH = orig_state
    assert entries, "expected compare to emit at least one JSONL payload"
    assert summary["is_warmup"] is True
    assert summary["warmup_reason"] == "sample_size_below_min"


def test_pipeline_compare_steady_state(tmp_path, monkeypatch):
    entries: list[dict] = []
    shadow_rows = [
        {"symbol": "BTCUSDT", "timestamp": 1.0, "risk_decision": {"allowed": True}, "sizing": {"sized_gross_usd": 10.0}},
        {"symbol": "ETHUSDT", "timestamp": 2.0, "risk_decision": {"allowed": True}, "sizing": {"sized_gross_usd": 20.0}},
        {"symbol": "SOLUSDT", "timestamp": 3.0, "risk_decision": {"allowed": True}, "sizing": {"sized_gross_usd": 30.0}},
    ]
    orders_rows = [
        {"symbol": row["symbol"], "ts": row["timestamp"] + 0.1, "notional": row["sizing"]["sized_gross_usd"]}
        for row in shadow_rows
    ]

    class DummyLogger:
        def write(self, payload):
            entries.append(payload)

    monkeypatch.setattr(pipeline_v6_compare, "get_logger", lambda _path: DummyLogger())
    monkeypatch.setattr(pipeline_v6_compare, "MIN_SAMPLE_SIZE_FOR_STEADY_STATE", 3)
    orig_shadow = pipeline_v6_compare.PIPELINE_SHADOW_LOG
    orig_log = pipeline_v6_compare.COMPARE_LOG_PATH
    orig_state = pipeline_v6_compare.COMPARE_STATE_PATH
    pipeline_v6_compare.PIPELINE_SHADOW_LOG = tmp_path / "shadow.jsonl"
    pipeline_v6_compare.COMPARE_LOG_PATH = tmp_path / "compare.jsonl"
    pipeline_v6_compare.COMPARE_STATE_PATH = tmp_path / "state/summary.json"
    shadow_lines = "\n".join(json.dumps(row) for row in shadow_rows) + "\n"
    order_lines = "\n".join(json.dumps(row) for row in orders_rows) + "\n"
    pipeline_v6_compare.PIPELINE_SHADOW_LOG.write_text(shadow_lines)
    (tmp_path / "orders.jsonl").write_text(order_lines)
    try:
        summary = pipeline_v6_compare.compare_pipeline_v6(
            shadow_limit=10,
            orders_path=tmp_path / "orders.jsonl",
            metrics_path=tmp_path / "metrics.jsonl",
        )
    finally:
        pipeline_v6_compare.PIPELINE_SHADOW_LOG = orig_shadow
        pipeline_v6_compare.COMPARE_LOG_PATH = orig_log
        pipeline_v6_compare.COMPARE_STATE_PATH = orig_state
    assert entries
    assert summary["is_warmup"] is False
    assert summary["warmup_reason"] in ("", None)


def test_sizing_diff_stats(tmp_path, monkeypatch):
    shadow_rows = [
        {"symbol": "BTCUSDT", "timestamp": 1.0, "risk_decision": {"allowed": True}, "sizing": {"sized_gross_usd": 100.0}},
        {"symbol": "ETHUSDT", "timestamp": 2.0, "risk_decision": {"allowed": True}, "sizing": {"sized_gross_usd": 200.0}},
    ]
    orders_rows = [
        {"symbol": "BTCUSDT", "ts": 1.1, "notional": 110.0},
        {"symbol": "ETHUSDT", "ts": 2.1, "notional": 190.0},
    ]

    class DummyLogger:
        def __init__(self):
            self.payloads = []

        def write(self, payload):
            self.payloads.append(payload)

    logger = DummyLogger()
    monkeypatch.setattr(pipeline_v6_compare, "get_logger", lambda _path: logger)
    monkeypatch.setattr(pipeline_v6_compare, "MIN_SAMPLE_SIZE_FOR_STEADY_STATE", 1)
    orig_shadow = pipeline_v6_compare.PIPELINE_SHADOW_LOG
    orig_log = pipeline_v6_compare.COMPARE_LOG_PATH
    orig_state = pipeline_v6_compare.COMPARE_STATE_PATH
    pipeline_v6_compare.PIPELINE_SHADOW_LOG = tmp_path / "shadow.jsonl"
    pipeline_v6_compare.COMPARE_LOG_PATH = tmp_path / "compare.jsonl"
    pipeline_v6_compare.COMPARE_STATE_PATH = tmp_path / "state/summary.json"
    shadow_lines = "\n".join(json.dumps(row) for row in shadow_rows) + "\n"
    order_lines = "\n".join(json.dumps(row) for row in orders_rows) + "\n"
    pipeline_v6_compare.PIPELINE_SHADOW_LOG.write_text(shadow_lines)
    (tmp_path / "orders.jsonl").write_text(order_lines)
    try:
        summary = pipeline_v6_compare.compare_pipeline_v6(
            shadow_limit=10,
            orders_path=tmp_path / "orders.jsonl",
            metrics_path=tmp_path / "metrics.jsonl",
        )
    finally:
        pipeline_v6_compare.PIPELINE_SHADOW_LOG = orig_shadow
        pipeline_v6_compare.COMPARE_LOG_PATH = orig_log
        pipeline_v6_compare.COMPARE_STATE_PATH = orig_state
    sizing_stats = summary.get("sizing_diff_stats", {})
    assert sizing_stats.get("sample_size") == 2
    # live 110 vs shadow 100 => 0.1, live 190 vs shadow 200 => 0.05
    assert sizing_stats.get("upsize_count") == 1
    assert sizing_stats.get("p50") == pytest.approx(0.075, rel=0, abs=1e-6)
    assert sizing_stats.get("p95") == pytest.approx(0.05, rel=0, abs=1e-6)
