from __future__ import annotations

import json
from pathlib import Path

from execution.intel import pipeline_v6_compare as compare


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_compare_pipeline_summary(tmp_path, monkeypatch):
    shadow = [
        {"symbol": "BTCUSDT", "timestamp": 1, "risk_decision": {"allowed": True}},
        {"symbol": "ETHUSDT", "timestamp": 2, "risk_decision": {"allowed": False}},
    ]
    orders = [
        {"symbol": "BTCUSDT"},
    ]
    shadow_log = tmp_path / "pipeline_v6_shadow.jsonl"
    compare.PIPELINE_SHADOW_LOG = shadow_log
    compare.COMPARE_LOG_PATH = tmp_path / "pipeline_v6_compare.jsonl"
    compare.COMPARE_STATE_PATH = tmp_path / "state/pipeline_v6_compare_summary.json"
    _write_jsonl(shadow_log, shadow)
    _write_jsonl(tmp_path / "orders.jsonl", orders)
    summary = compare.compare_pipeline_v6(
        shadow_limit=10,
        orders_path=tmp_path / "orders.jsonl",
        metrics_path=tmp_path / "metrics.jsonl",
    )
    assert summary["sample_size"] == 2
    assert "veto_mismatch_pct" in summary
