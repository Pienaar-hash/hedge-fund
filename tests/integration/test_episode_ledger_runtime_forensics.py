from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution import episode_ledger as ledger_module
from execution import episode_ledger_runtime_forensics as forensics


pytestmark = [pytest.mark.integration, pytest.mark.runtime]


def _append(path: Path, *events: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def _fill(ts: str, side: str, reduce_only: bool, *, price: float = 100.0, order_id: str = "") -> dict:
    metadata = {"strategy": "fixture", "exit": {"reason": "tp"}}
    if reduce_only:
        metadata["entry_price"] = 100.0
    return {
        "event_type": "order_fill",
        "ts": ts,
        "ts_fill_first": ts,
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "side": side,
        "reduceOnly": reduce_only,
        "executedQty": 1.0,
        "avgPrice": price,
        "fee_total": 0.1,
        "orderId": order_id or ts,
        "metadata": metadata,
    }


def test_execute_run_reads_only_private_snapshot_paths(tmp_path):
    snapshot_root = tmp_path / "snapshot"
    execution = snapshot_root / "sources" / "execution"
    state = snapshot_root / "sources" / "state"
    execution.mkdir(parents=True)
    state.mkdir(parents=True)
    _append(
        execution / "orders_executed.jsonl",
        _fill("2026-01-01T00:00:00+00:00", "BUY", False, order_id="entry"),
        _fill("2026-01-01T01:00:00+00:00", "SELL", True, price=101.0, order_id="exit"),
    )
    (execution / "dle_shadow_events.jsonl").write_text("", encoding="utf-8")
    (state / "nav_state.json").write_text(json.dumps({"total_equity": 10000.0}), encoding="utf-8")
    snapshot_manifest = [
        {"role": "execution_log_1", "snapshot": str((execution / "orders_executed.jsonl").resolve()), "identity": forensics.file_identity(execution / "orders_executed.jsonl")},
        {"role": "dle_authority_log", "snapshot": str((execution / "dle_shadow_events.jsonl").resolve()), "identity": forensics.file_identity(execution / "dle_shadow_events.jsonl")},
        {"role": "nav_input", "snapshot": str((state / "nav_state.json").resolve()), "identity": forensics.file_identity(state / "nav_state.json")},
    ]
    runtime_root = tmp_path / "run" / "runtime"
    forensics.stage_runtime_root(snapshot_manifest, runtime_root, include_resume_state=False)
    input_manifest = {
        "snapshot_manifest": snapshot_manifest,
        "module_content_hash": "module-sha",
        "resolved_paths_hash": "resolved-sha",
        "process_provenance_hash": "proc-sha",
    }
    attestation = forensics.attest_module(Path.cwd())
    result = forensics.execute_run(
        "initial",
        runtime_root,
        forensics.read_roles_for_runtime(runtime_root),
        attestation,
        input_manifest,
        ledger_module.rebuild_and_save,
    )
    assert result["summary"]["canonical_ledger_hash"]
    assert result["summary"]["input_manifest_hash"] == forensics.stable_json_hash(input_manifest)
    assert result["read_records"], "read records should not be empty"
    assert all(entry["within_runtime_root"] for entry in result["read_records"])
    assert all(str(snapshot_root.resolve()) not in entry["resolved_path"] for entry in result["read_records"])
    assert any(entry["logical_role"] == "nav_input" for entry in result["read_records"])
