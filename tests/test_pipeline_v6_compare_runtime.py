from __future__ import annotations

import json

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
        {"symbol": "BTCUSDT", "timestamp": 1.0, "risk_decision": {"allowed": True}},
        {"symbol": "ETHUSDT", "timestamp": 2.0, "risk_decision": {"allowed": True}},
    ]
    orders_rows = [
        {"symbol": "BTCUSDT", "ts": 1.1},
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
        {"symbol": "BTCUSDT", "timestamp": 1.0, "risk_decision": {"allowed": True}},
        {"symbol": "ETHUSDT", "timestamp": 2.0, "risk_decision": {"allowed": True}},
        {"symbol": "SOLUSDT", "timestamp": 3.0, "risk_decision": {"allowed": True}},
    ]
    orders_rows = [{"symbol": row["symbol"], "ts": row["timestamp"] + 0.1} for row in shadow_rows]

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
