from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from execution import episode_ledger_deployment_protocol as protocol


def test_validate_state_model_exposes_required_states():
    model = protocol.validate_state_model()
    assert "UNPREPARED" in model["states"]
    assert "ACCEPTED" in model["terminal_states"]
    assert "FAILED_CLOSED" in model["terminal_states"]


def test_validate_transition_requires_authority_and_evidence():
    with pytest.raises(ValueError, match="authority"):
        protocol.validate_transition("UNPREPARED", "PREFLIGHT_CAPTURED", "deployment_authorizer", {"preflight_gate": {}, "resolved_paths": {}})
    with pytest.raises(ValueError, match="missing transition evidence"):
        protocol.validate_transition("UNPREPARED", "PREFLIGHT_CAPTURED", "protocol_runner", {"preflight_gate": {}})


def test_schema_validation_rejects_missing_required_bundle_field():
    bundle = {
        "candidate_commit_sha": "abc",
        "candidate_worktree_clean": True,
    }
    with pytest.raises(ValueError, match="missing required property"):
        protocol.verify_bundle_schema(bundle)


def test_detect_unauthorized_mutations_ignores_allowlisted_outputs():
    before = {
        "logs/state/nav_state.json": {"sha256": "a", "size": 1},
        "logs/state/episode_ledger.json": {"sha256": "a", "size": 1},
    }
    after = {
        "logs/state/nav_state.json": {"sha256": "b", "size": 1},
        "logs/state/episode_ledger.json": {"sha256": "c", "size": 2},
    }
    diff = protocol.detect_unauthorized_mutations(before, after)
    assert diff == [{"path": "logs/state/nav_state.json", "before": {"sha256": "a", "size": 1}, "after": {"sha256": "b", "size": 1}}]


def test_validate_candidate_freshness_rejects_dirty_worktree(tmp_path: Path):
    repo = tmp_path / "repo"
    metadata = protocol.make_candidate_repo(protocol.repo_root(), repo, comment_suffix="dirty-test")
    assert metadata["candidate_commit"]
    (repo / "DIRTY.txt").write_text("dirty\n", encoding="utf-8")
    freshness = protocol.validate_candidate_freshness(repo, protocol.git_module_hash(protocol.repo_root()))
    assert freshness["passed"] is False
    assert freshness["failure_code"] == "FAILED_CLOSED_DIRTY_WORKTREE"


def test_validate_candidate_freshness_rejects_prohibited_module_hash(tmp_path: Path):
    repo = tmp_path / "repo"
    protocol.copy_required_tree(protocol.repo_root(), repo)
    protocol.init_git_repo(repo, baseline_marker=protocol.PROTOCOL_BASELINE_COMMIT)
    freshness = protocol.validate_candidate_freshness(repo, protocol.git_module_hash(protocol.repo_root()))
    assert freshness["passed"] is False
    assert freshness["failure_code"] == "FAILED_CLOSED_PROHIBITED_CANDIDATE"


def test_setup_rehearsal_roots_stays_within_private_root(tmp_path: Path):
    roots = protocol.setup_rehearsal_roots(tmp_path)
    for path in roots.values():
        assert tmp_path.resolve() == path.resolve() or tmp_path.resolve() in path.resolve().parents
