from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

import execution.episode_ledger_runtime_forensics as forensics


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attestation_hash_changes_without_git_head_change(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    module_path = repo / "synthetic_module.py"
    module_path.write_text("def build_episode_ledger():\n    return 1\n", encoding="utf-8")
    subprocess.check_call(["git", "init"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)
    subprocess.check_call(["git", "add", "synthetic_module.py"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo)
    module = _import_module(module_path, "synthetic_module")
    monkeypatch.setattr(forensics, "ledger_module", module)
    before = forensics.attest_module(repo)
    module_path.write_text("def build_episode_ledger():\n    return 2\n", encoding="utf-8")
    after = forensics.attest_module(repo)
    assert before["repository_provenance"]["head"] == after["repository_provenance"]["head"]
    assert before["loaded_module_provenance"]["sha256"] != after["loaded_module_provenance"]["sha256"]


def test_ensure_safe_output_root_rejects_live_overlap(tmp_path):
    live_root = tmp_path / "live"
    live_root.mkdir()
    safe = tmp_path / "evidence"
    assert forensics.ensure_safe_output_root(safe, [live_root]) == safe.resolve()
    with pytest.raises(ValueError, match="unsafe output root"):
        forensics.ensure_safe_output_root(live_root / "nested", [live_root])


def test_resolve_live_paths_reports_symlink_target_and_outside_root(tmp_path):
    live_root = tmp_path / "live"
    execution = live_root / "logs" / "execution"
    state = live_root / "logs" / "state"
    execution.mkdir(parents=True)
    state.mkdir(parents=True)
    external = tmp_path / "external-orders.jsonl"
    external.write_text('{"event_type":"order_fill","executedQty":1}\n', encoding="utf-8")
    (execution / "orders_executed.jsonl").symlink_to(external)
    (execution / "dle_shadow_events.jsonl").write_text("", encoding="utf-8")
    (state / "nav_state.json").write_text(json.dumps({"total_equity": 10000.0}), encoding="utf-8")
    (state / "episode_ledger.json").write_text(json.dumps({"episodes": [], "stats": {}, "last_rebuild_ts": ""}), encoding="utf-8")
    (state / "episode_ledger_checkpoint.json").write_text(json.dumps({"schema_version": "x"}), encoding="utf-8")
    resolved = forensics.resolve_live_paths(live_root, live_root)
    active = resolved["active_execution_log"]
    assert active["symlink"] is True
    assert active["resolved_target"] == str(external.resolve())
    assert active["outside_expected_root"] is True


def test_enrich_read_records_marks_partial_final_line(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    path = runtime_root / "partial.jsonl"
    path.write_text('{"event_type":"order_fill","executedQty":1}', encoding="utf-8")
    records = [
        {
            "logical_role": "execution_log_1",
            "resolved_path": str(path.resolve()),
            "mode": "r",
            "sequence": 1,
            "open_ts": "2026-01-01T00:00:00+00:00",
            "close_ts": "2026-01-01T00:00:01+00:00",
            "device": path.stat().st_dev,
            "inode": path.stat().st_ino,
            "size_at_open": path.stat().st_size,
            "size_at_close": path.stat().st_size,
            "start_offset": 0,
            "end_offset": path.stat().st_size,
        }
    ]
    enriched = forensics.enrich_read_records(records, runtime_root, None)
    assert enriched[0]["partial_final_line"] is True
    assert enriched[0]["rejected_malformed_count"] == 1
