from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution import episode_ledger_deployment_protocol as protocol


pytestmark = [pytest.mark.integration, pytest.mark.runtime]


def test_run_rehearsal_executes_all_protocol_scenarios(tmp_path: Path):
    report = protocol.run_rehearsal(tmp_path / "rehearsal")
    assert report["passed"] is True
    assert report["scenarios"]["happy_path"]["final_state"] == "ACCEPTED"
    assert report["scenarios"]["reject_prohibited_candidate"]["failure_code"] == "FAILED_CLOSED_PROHIBITED_CANDIDATE"
    assert report["scenarios"]["reject_dirty_worktree"]["failure_code"] == "FAILED_CLOSED_DIRTY_WORKTREE"
    assert report["scenarios"]["reject_module_hash_mismatch"]["failure_code"] == "FAILED_CLOSED_MODULE_HASH_MISMATCH"
    assert report["scenarios"]["reject_preexisting_inconsistency"]["failure_code"] == "FAILED_CLOSED_PREEXISTING_INCONSISTENCY"
    assert report["scenarios"]["reject_snapshot_failure"]["failure_code"] == "FAILED_CLOSED_SNAPSHOT_FAILED"
    assert report["scenarios"]["reject_equivalence_failure"]["failure_code"] == "FAILED_CLOSED_EQUIVALENCE_FAILED"
    assert report["scenarios"]["rollback_unauthorized_mutation"]["final_state"] == "ROLLED_BACK"


def test_rehearsal_bundle_manifest_includes_nav_and_post_start_attestation(tmp_path: Path):
    report = protocol.run_rehearsal(tmp_path / "rehearsal")
    happy_root = tmp_path / "rehearsal" / "happy_path"
    bundle_path = happy_root / "workspace" / "evidence" / "bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["nav_semantic_input_declaration"]["enabled"] is True
    assert bundle["nav_semantic_input_declaration"]["path"] == "logs/state/nav_state.json"
    post_start_gate = report["scenarios"]["happy_path"]["result"]["post_start_gate"]
    loaded_hash = post_start_gate["preflight"]["loaded_module_attestation"]["loaded_module_provenance"]["sha256"]
    assert loaded_hash == bundle["candidate_module_content_hash"]


def test_rehearsal_outputs_remain_under_private_root(tmp_path: Path):
    report = protocol.run_rehearsal(tmp_path / "rehearsal")
    for scenario in report["scenarios"]:
        scenario_root = (tmp_path / "rehearsal" / scenario).resolve()
        for file_path in scenario_root.rglob("*"):
            assert tmp_path.resolve() == file_path.resolve() or tmp_path.resolve() in file_path.resolve().parents
