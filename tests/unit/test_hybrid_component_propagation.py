from __future__ import annotations

from typing import Any, Dict


def test_trace_writes_expected_fields(monkeypatch):
    from execution import hybrid_component_propagation as hcp

    captured: Dict[str, Any] = {}

    def _fake_append(path, rec):
        captured["path"] = str(path)
        captured["rec"] = rec

    monkeypatch.setattr(hcp, "append_jsonl", _fake_append)
    monkeypatch.setattr(hcp.time, "time", lambda: 1234.5)

    intent = {
        "symbol": "ethusdt",
        "intent_id": "ord_abc",
        "strategy": "hydra",
        "source_head": "TREND",
        "hybrid_components": {
            "expectancy": 0.42,
            "router": 0.7,
        },
        "type": "intent",
    }

    hcp.trace_hybrid_component_propagation(
        origin_stage="executor_received",
        intent=intent,
        merge_path="hydra_only",
    )

    rec = captured["rec"]
    assert captured["path"].endswith("logs/execution/hybrid_component_propagation.jsonl")
    assert rec["ts"] == 1234.5
    assert rec["origin_stage"] == "executor_received"
    assert rec["symbol"] == "ETHUSDT"
    assert rec["intent_id"] == "ord_abc"
    assert rec["intent_id_present"] is True
    assert rec["correlation_key"] == "ord_abc"
    assert rec["strategy"] == "hydra"
    assert rec["source_head"] == "TREND"
    assert rec["has_hybrid_components"] is True
    assert rec["hybrid_component_keys"] == ["expectancy", "router"]
    assert rec["hybrid_expectancy"] == 0.42
    assert rec["merge_path"] == "hydra_only"
    assert rec["intent_type"] == "intent"


def test_trace_is_fail_open(monkeypatch):
    from execution import hybrid_component_propagation as hcp

    def _boom(*args, **kwargs):
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(hcp, "append_jsonl", _boom)

    # Must not raise
    hcp.trace_hybrid_component_propagation(
        origin_stage="fee_gate_received",
        intent={"symbol": "BTCUSDT", "attempt_id": "sig_1"},
    )


def test_trace_emits_hashed_correlation_key_when_intent_id_missing(monkeypatch):
    from execution import hybrid_component_propagation as hcp

    captured: Dict[str, Any] = {}

    def _fake_append(path, rec):
        captured["rec"] = rec

    monkeypatch.setattr(hcp, "append_jsonl", _fake_append)
    # Fixed fallback clock for deterministic ts_bucket
    monkeypatch.setattr(hcp.time, "time", lambda: 1778429500.0)

    # No intent_id / attempt_id provided.
    intent = {
        "symbol": "ETHUSDT",
        "direction": "LONG",
        "strategy": "hydra",
        "source_head": "TREND",
        "score": 0.456789,
        "nav_pct": 0.012345,
    }

    hcp.trace_hybrid_component_propagation(
        origin_stage="hydra_merged",
        intent=intent,
    )

    rec = captured["rec"]
    assert rec["intent_id"] == ""
    assert rec["intent_id_present"] is False
    assert isinstance(rec["correlation_key"], str)
    assert rec["correlation_key"].startswith("corr_")
    assert len(rec["correlation_key"]) > 8
