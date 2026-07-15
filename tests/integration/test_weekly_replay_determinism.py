from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from execution.weekly_audit_ledger import append_decision_record, verify_decision_chain
from execution.weekly_market_data import build_weekly_input_manifest, load_completed_weekly_closes, validate_weekly_history
from execution.weekly_momentum_engine import classify_conviction, compute_weekly_features, rank_weekly_features


def _history():
    payload = {}
    start_dt = datetime(2025, 1, 6, 12, tzinfo=timezone.utc)
    for symbol, rows in (
        ("ETHUSDT", [{"ts": (start_dt + timedelta(weeks=week)).isoformat(), "close": 100 + 2 * week} for week in range(60)]),
        ("BTCUSDT", [{"ts": (start_dt + timedelta(weeks=week)).isoformat(), "close": 100 + 2 * week} for week in range(60)]),
    ):
        payload[symbol] = rows
    return payload


def test_shuffled_universe_produces_identical_ranking():
    ordered = _history()
    shuffled = {"BTCUSDT": ordered["BTCUSDT"], "ETHUSDT": ordered["ETHUSDT"]}
    features_a = classify_conviction(rank_weekly_features(compute_weekly_features(validate_weekly_history(load_completed_weekly_closes(ordered)))))
    features_b = classify_conviction(rank_weekly_features(compute_weekly_features(validate_weekly_history(load_completed_weekly_closes(shuffled)))))
    assert json.dumps(features_a, sort_keys=True) == json.dumps(features_b, sort_keys=True)


def test_same_input_produces_byte_identical_manifest_and_record(tmp_path: Path):
    validated = validate_weekly_history(load_completed_weekly_closes(_history()))
    manifest_a = build_weekly_input_manifest(validated, price_source="fixture", completed_week_end="2026-07-13T00:00:00+00:00")
    manifest_b = build_weekly_input_manifest(validated, price_source="fixture", completed_week_end="2026-07-13T00:00:00+00:00")
    assert json.dumps(manifest_a, sort_keys=True, separators=(",", ":")) == json.dumps(manifest_b, sort_keys=True, separators=(",", ":"))

    ledger = tmp_path / "weekly.jsonl"
    record = {
        "schema_version": "weekly_momentum_v1",
        "cycle_id": "WM_2026_29",
        "decision_ts": "2026-07-15T00:00:00+00:00",
        "completed_week_end": "2026-07-13T00:00:00+00:00",
        "engine_commit": "test",
        "module_sha256": "test",
        "config_sha256": "test",
        "universe_sha256": "test",
        "price_source": "fixture",
        "input_manifest_sha256": manifest_a["input_manifest_sha256"],
        "nav_usd": 10000.0,
        "risk_mode": "ACTIVE",
        "candidates": [],
        "selected": [],
        "rejected": [],
        "existing_positions": [],
        "target_positions": [],
        "orders": [],
        "risk_checks": []
    }
    first = append_decision_record(ledger, record)
    ok, errors = verify_decision_chain(ledger)
    assert ok
    assert errors == []
    assert first["decision_hash"]


def test_duplicate_cycle_execution_rejected(tmp_path: Path):
    ledger = tmp_path / "weekly.jsonl"
    record = {
        "schema_version": "weekly_momentum_v1",
        "cycle_id": "WM_2026_29",
        "decision_ts": "2026-07-15T00:00:00+00:00",
        "completed_week_end": "2026-07-13T00:00:00+00:00",
        "engine_commit": "test",
        "module_sha256": "test",
        "config_sha256": "test",
        "universe_sha256": "test",
        "price_source": "fixture",
        "input_manifest_sha256": "abc",
        "nav_usd": 10000.0,
        "risk_mode": "ACTIVE",
        "candidates": [],
        "selected": [],
        "rejected": [],
        "existing_positions": [],
        "target_positions": [],
        "orders": [],
        "risk_checks": []
    }
    append_decision_record(ledger, record)
    append_decision_record(ledger, record)
    ok, errors = verify_decision_chain(ledger)
    assert not ok
    assert any("duplicate cycle_id" in error for error in errors)
