"""Checkpointed episode-ledger ingestion contract tests."""

import json
from pathlib import Path

import pytest

import execution.episode_ledger as ledger_module


def _fill(ts: str, side: str, reduce: bool, *, price: float = 100.0, order_id: str = "") -> dict:
    return {
        "event_type": "order_fill", "ts": ts, "ts_fill_first": ts,
        "symbol": "BTCUSDT", "positionSide": "LONG", "side": side,
        "reduceOnly": reduce, "executedQty": 1.0, "avgPrice": price,
        "fee_total": 0.1, "orderId": order_id or ts,
        "metadata": {"strategy": "fixture", "exit": {"reason": "tp"}},
    }


@pytest.fixture
def ledger_paths(tmp_path, monkeypatch):
    execution = tmp_path / "execution"
    state = tmp_path / "state"
    execution.mkdir()
    monkeypatch.setattr(ledger_module, "EXECUTION_LOG_DIR", execution)
    monkeypatch.setattr(ledger_module, "EXECUTION_LOG_PATH", execution / "orders_executed.jsonl")
    monkeypatch.setattr(ledger_module, "EPISODE_LEDGER_PATH", state / "episode_ledger.json")
    monkeypatch.setattr(ledger_module, "EPISODE_LEDGER_CHECKPOINT_PATH", state / "episode_ledger_checkpoint.json")
    monkeypatch.setattr(ledger_module, "EPISODE_LEDGER_REBUILD_LOG_PATH", execution / "episode_ledger_rebuild.jsonl")
    monkeypatch.setattr(ledger_module, "DLE_SHADOW_LOG_PATH", execution / "no_shadow.jsonl")
    return execution, state


def _append(path: Path, *events: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def _telemetry(execution: Path) -> list[dict]:
    return [json.loads(line) for line in (execution / "episode_ledger_rebuild.jsonl").read_text().splitlines()]


def test_initial_full_then_noop_incremental(ledger_paths):
    execution, state = ledger_paths
    source = execution / "orders_executed.jsonl"
    _append(source, _fill("2026-01-01T00:00:00+00:00", "BUY", False), _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=110))
    first = ledger_module.rebuild_and_save()
    assert len(first.episodes) == 1
    checkpoint = json.loads((state / "episode_ledger_checkpoint.json").read_text())
    assert checkpoint["schema_version"] == ledger_module.EPISODE_LEDGER_CHECKPOINT_VERSION
    assert _telemetry(execution)[-1]["mode"] == "full_fallback"
    second = ledger_module.rebuild_and_save()
    assert ledger_module._canonical_ledger_hash(second) == ledger_module._canonical_ledger_hash(first)
    row = _telemetry(execution)[-1]
    assert row["mode"] == "incremental"
    assert row["bytes_read"] == 0 and row["new_events"] == 0
    assert row["git_sha"] != "unknown"


def test_append_open_then_close_matches_clean_full(ledger_paths):
    execution, _ = ledger_paths
    source = execution / "orders_executed.jsonl"
    _append(source, _fill("2026-01-01T00:00:00+00:00", "BUY", False))
    ledger_module.rebuild_and_save()
    _append(source, _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=110))
    incremental = ledger_module.rebuild_and_save()
    full = ledger_module.build_episode_ledger()
    assert ledger_module._canonical_ledger_hash(incremental) == ledger_module._canonical_ledger_hash(full)
    row = _telemetry(execution)[-1]
    assert row["mode"] == "incremental"
    assert row["new_events"] == 1 and row["episodes_created"] == 1


def test_rotation_and_multiple_rotations_do_not_duplicate(ledger_paths):
    execution, _ = ledger_paths
    current = execution / "orders_executed.jsonl"
    _append(current, _fill("2026-01-01T00:00:00+00:00", "BUY", False))
    ledger_module.rebuild_and_save()
    current.rename(execution / "orders_executed.1.jsonl")
    _append(current, _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=110))
    one_rotation = ledger_module.rebuild_and_save()
    (execution / "orders_executed.1.jsonl").rename(execution / "orders_executed.2.jsonl")
    current.rename(execution / "orders_executed.1.jsonl")
    _append(current)
    multiple = ledger_module.rebuild_and_save()
    full = ledger_module.build_episode_ledger()
    assert len(multiple.episodes) == 1
    assert ledger_module._canonical_ledger_hash(multiple) == ledger_module._canonical_ledger_hash(full)
    assert one_rotation.episodes[0].episode_id == multiple.episodes[0].episode_id
    assert _telemetry(execution)[-1]["mode"] == "incremental"


def test_atomic_active_file_replacement_with_proven_prefix_is_incremental(ledger_paths):
    execution, _ = ledger_paths
    current = execution / "orders_executed.jsonl"
    entry = _fill("2026-01-01T00:00:00+00:00", "BUY", False)
    exit_fill = _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=110)
    _append(current, entry)
    ledger_module.rebuild_and_save()
    replacement = execution / "replacement.jsonl"
    _append(replacement, entry, exit_fill)
    replacement.replace(current)
    incremental = ledger_module.rebuild_and_save()
    assert len(incremental.episodes) == 1
    assert _telemetry(execution)[-1]["mode"] == "incremental"
    assert ledger_module._canonical_ledger_hash(incremental) == ledger_module._canonical_ledger_hash(ledger_module.build_episode_ledger())


@pytest.mark.parametrize("mutation,reason", [("truncate", "source_truncated"), ("replace", "source_missing_or_replaced")])
def test_unsafe_sources_force_visible_full_fallback(ledger_paths, mutation, reason):
    execution, _ = ledger_paths
    current = execution / "orders_executed.jsonl"
    _append(current, _fill("2026-01-01T00:00:00+00:00", "BUY", False))
    ledger_module.rebuild_and_save()
    if mutation == "truncate":
        current.write_text("")
    else:
        replacement = execution / "replacement.jsonl"
        _append(replacement, _fill("2026-01-02T00:00:00+00:00", "BUY", False))
        replacement.replace(current)
    ledger_module.rebuild_and_save()
    row = _telemetry(execution)[-1]
    assert row["mode"] == "full_fallback"
    assert row["fallback_reason"] == reason


def test_duplicate_malformed_checkpoint_and_forced_full(ledger_paths):
    execution, state = ledger_paths
    current = execution / "orders_executed.jsonl"
    entry = _fill("2026-01-01T00:00:00+00:00", "BUY", False)
    _append(current, entry)
    ledger_module.rebuild_and_save()
    _append(current, entry)
    with current.open("a") as handle:
        handle.write("{not-json}\n")
    same = ledger_module.rebuild_and_save()
    assert len(same.episodes) == 0
    assert _telemetry(execution)[-1]["new_events"] == 0
    (state / "episode_ledger_checkpoint.json").write_text("not json")
    ledger_module.rebuild_and_save()
    assert _telemetry(execution)[-1]["fallback_reason"] == "checkpoint_missing_or_corrupt"
    ledger_module.rebuild_and_save(force_full=True)
    assert _telemetry(execution)[-1]["mode"] == "forced_full"


@pytest.mark.parametrize("field, value, reason", [
    ("schema_version", "obsolete", "checkpoint_version_mismatch"),
    ("ledger_hash", "not-the-ledger", "ledger_checkpoint_hash_mismatch"),
])
def test_checkpoint_incompatibility_and_divergence_fallback(ledger_paths, field, value, reason):
    execution, state = ledger_paths
    _append(execution / "orders_executed.jsonl", _fill("2026-01-01T00:00:00+00:00", "BUY", False))
    ledger_module.rebuild_and_save()
    checkpoint_path = state / "episode_ledger_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint[field] = value
    checkpoint_path.write_text(json.dumps(checkpoint))
    ledger_module.rebuild_and_save()
    assert _telemetry(execution)[-1]["fallback_reason"] == reason


def test_telemetry_failure_is_fail_open(ledger_paths, monkeypatch):
    execution, _ = ledger_paths
    _append(execution / "orders_executed.jsonl", _fill("2026-01-01T00:00:00+00:00", "BUY", False))
    # The telemetry wrapper is fail-open when its append dependency fails.
    monkeypatch.setattr("execution.log_utils.append_jsonl", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("sink down")))
    result = ledger_module.rebuild_and_save()
    assert result.stats["total_fills"] == 1


@pytest.mark.parametrize("historical_episodes", [10, 100, 500])
def test_noop_and_append_read_only_new_bytes_at_scale(ledger_paths, historical_episodes):
    execution, _ = ledger_paths
    source = execution / "orders_executed.jsonl"
    events = []
    for index in range(historical_episodes):
        events.extend([
            _fill(f"2026-01-01T00:{index:02d}:00+00:00", "BUY", False, order_id=f"e{index}"),
            _fill(f"2026-01-01T01:{index:02d}:00+00:00", "SELL", True, price=101, order_id=f"x{index}"),
        ])
    _append(source, *events)
    ledger_module.rebuild_and_save()
    full_bytes = _telemetry(execution)[-1]["bytes_read"]
    ledger_module.rebuild_and_save()
    assert _telemetry(execution)[-1]["bytes_read"] == 0
    appended = _fill(f"2026-02-01T00:00:{historical_episodes % 60:02d}+00:00", "BUY", False, order_id=f"tail{historical_episodes}")
    _append(source, appended)
    ledger_module.rebuild_and_save()
    row = _telemetry(execution)[-1]
    assert row["mode"] == "incremental"
    assert 0 < row["bytes_read"] < full_bytes
