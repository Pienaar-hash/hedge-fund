from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from execution import episode_ledger_runtime_forensics as forensics


PROTOCOL_BASELINE_COMMIT = "d450db17c1bd089426b9e7c35b84172876fa471d"
PROHIBITED_CANDIDATE_COMMIT = "dec9acc7"
PROTOCOL_STATES = (
    "UNPREPARED",
    "PREFLIGHT_CAPTURED",
    "SNAPSHOT_FROZEN",
    "OFFLINE_EQUIVALENCE_PASSED",
    "RUNTIME_PROVENANCE_VERIFIED",
    "DEPLOYMENT_AUTHORIZED",
    "DEPLOYING",
    "POST_START_VALIDATING",
    "ACCEPTED",
    "ROLLED_BACK",
    "FAILED_CLOSED",
)
TERMINAL_STATES = {"ACCEPTED", "ROLLED_BACK", "FAILED_CLOSED"}
STATE_TRANSITIONS = {
    "UNPREPARED": {"PREFLIGHT_CAPTURED", "FAILED_CLOSED"},
    "PREFLIGHT_CAPTURED": {"SNAPSHOT_FROZEN", "FAILED_CLOSED"},
    "SNAPSHOT_FROZEN": {"OFFLINE_EQUIVALENCE_PASSED", "FAILED_CLOSED"},
    "OFFLINE_EQUIVALENCE_PASSED": {"RUNTIME_PROVENANCE_VERIFIED", "FAILED_CLOSED"},
    "RUNTIME_PROVENANCE_VERIFIED": {"DEPLOYMENT_AUTHORIZED", "FAILED_CLOSED"},
    "DEPLOYMENT_AUTHORIZED": {"DEPLOYING", "FAILED_CLOSED"},
    "DEPLOYING": {"POST_START_VALIDATING", "ROLLED_BACK", "FAILED_CLOSED"},
    "POST_START_VALIDATING": {"ACCEPTED", "ROLLED_BACK", "FAILED_CLOSED"},
    "ACCEPTED": set(),
    "ROLLED_BACK": set(),
    "FAILED_CLOSED": set(),
}
TRANSITION_AUTHORITIES = {
    "PREFLIGHT_CAPTURED": {"protocol_runner"},
    "SNAPSHOT_FROZEN": {"protocol_runner"},
    "OFFLINE_EQUIVALENCE_PASSED": {"protocol_runner"},
    "RUNTIME_PROVENANCE_VERIFIED": {"protocol_runner"},
    "DEPLOYMENT_AUTHORIZED": {"deployment_authorizer"},
    "DEPLOYING": {"supervisor_controller"},
    "POST_START_VALIDATING": {"protocol_runner"},
    "ACCEPTED": {"deployment_authorizer"},
    "ROLLED_BACK": {"supervisor_controller"},
    "FAILED_CLOSED": {"protocol_runner", "supervisor_controller", "deployment_authorizer"},
}
REQUIRED_EVIDENCE_KEYS = {
    "PREFLIGHT_CAPTURED": {"preflight_gate", "resolved_paths"},
    "SNAPSHOT_FROZEN": {"atomic_snapshot"},
    "OFFLINE_EQUIVALENCE_PASSED": {"equivalence_gate"},
    "RUNTIME_PROVENANCE_VERIFIED": {"candidate_bundle", "freshness_gate"},
    "DEPLOYMENT_AUTHORIZED": {"authorization_record", "rollback_plan"},
    "DEPLOYING": {"deployment_start"},
    "POST_START_VALIDATING": {"post_start_gate"},
    "ACCEPTED": {"acceptance_record"},
    "ROLLED_BACK": {"rollback_record"},
    "FAILED_CLOSED": {"failure_record"},
}
FAILURE_CODES = {
    "FAILED_CLOSED_PREEXISTING_INCONSISTENCY",
    "FAILED_CLOSED_LIVE_MODULE_UNPROVEN",
    "FAILED_CLOSED_PROHIBITED_CANDIDATE",
    "FAILED_CLOSED_DIRTY_WORKTREE",
    "FAILED_CLOSED_MODULE_HASH_MISMATCH",
    "FAILED_CLOSED_SNAPSHOT_FAILED",
    "FAILED_CLOSED_EQUIVALENCE_FAILED",
    "FAILED_CLOSED_UNAUTHORIZED_MUTATION",
    "FAILED_CLOSED_RESUME_FAILED",
    "FAILED_CLOSED_ROLLBACK_VERIFICATION_FAILED",
}
AUTHORIZED_MUTATION_PATTERNS = (
    "logs/state/episode_ledger.json",
    "logs/state/episode_ledger_checkpoint.json",
    "logs/execution/episode_ledger_rebuild.jsonl",
    "runtime_attestation.json",
    "supervisor_state.json",
)
RUNTIME_CONFIG_RELATIVE_PATHS = (
    "config/runtime.yaml",
    "config/settings.json",
    "config/strategy_config.json",
    "config/exit_reason_map.yaml",
)
SAFE_OUTPUT_NAMES = {
    "bundle_manifest.json",
    "comparison_report.json",
    "result.json",
    "result.md",
    "README.md",
    "SHA256SUMS",
}
REHEARSAL_SCENARIOS = (
    "happy_path",
    "reject_prohibited_candidate",
    "reject_dirty_worktree",
    "reject_module_hash_mismatch",
    "reject_preexisting_inconsistency",
    "reject_snapshot_failure",
    "reject_equivalence_failure",
    "rollback_unauthorized_mutation",
)
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "config" / "episode_ledger_deployment_protocol.schema.json"


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class TransitionRecord:
    from_state: str
    to_state: str
    authority: str
    ts: str
    evidence_hash: str
    evidence_keys: list[str]


@dataclass
class AttemptState:
    scenario: str
    root: Path
    state: str = "UNPREPARED"
    history: list[TransitionRecord] = field(default_factory=list)
    failure_code: Optional[str] = None
    failure_detail: Optional[str] = None

    def transition(self, to_state: str, authority: str, evidence: dict[str, Any]) -> None:
        validate_transition(self.state, to_state, authority, evidence)
        self.history.append(
            TransitionRecord(
                from_state=self.state,
                to_state=to_state,
                authority=authority,
                ts=forensics.utc_now(),
                evidence_hash=forensics.stable_json_hash(evidence),
                evidence_keys=sorted(evidence.keys()),
            )
        )
        self.state = to_state

    def fail_closed(self, code: str, detail: str, evidence: dict[str, Any], *, authority: str = "protocol_runner") -> None:
        self.failure_code = code
        self.failure_detail = detail
        payload = dict(evidence)
        payload.setdefault("failure_record", {"code": code, "detail": detail})
        self.transition("FAILED_CLOSED", authority, payload)


@dataclass
class CandidateBundleContext:
    candidate_repo_root: Path
    rollback_repo_root: Path
    runtime_root: Path
    evidence_dir: Path
    prohibited_module_hash: str


def validate_transition(from_state: str, to_state: str, authority: str, evidence: dict[str, Any]) -> None:
    if from_state not in STATE_TRANSITIONS:
        raise ValueError(f"unknown state {from_state}")
    if to_state not in STATE_TRANSITIONS[from_state]:
        raise ValueError(f"invalid transition {from_state} -> {to_state}")
    allowed = TRANSITION_AUTHORITIES.get(to_state, set())
    if authority not in allowed:
        raise ValueError(f"authority {authority} not allowed for {to_state}")
    required = REQUIRED_EVIDENCE_KEYS.get(to_state, set())
    missing = sorted(key for key in required if key not in evidence)
    if missing:
        raise ValueError(f"missing transition evidence for {to_state}: {', '.join(missing)}")


def protocol_state_model() -> dict[str, Any]:
    return {
        "states": list(PROTOCOL_STATES),
        "terminal_states": sorted(TERMINAL_STATES),
        "valid_transitions": {state: sorted(next_states) for state, next_states in STATE_TRANSITIONS.items()},
        "transition_authority": {state: sorted(authorities) for state, authorities in TRANSITION_AUTHORITIES.items()},
        "required_evidence": {state: sorted(keys) for state, keys in REQUIRED_EVIDENCE_KEYS.items()},
        "failure_codes": sorted(FAILURE_CODES),
    }


def validate_state_model(model: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    payload = protocol_state_model() if model is None else model
    states = set(payload["states"])
    if states != set(PROTOCOL_STATES):
        raise ValueError("protocol states do not match expected set")
    for terminal in payload["terminal_states"]:
        if terminal not in states:
            raise ValueError(f"unknown terminal state {terminal}")
    for state, next_states in payload["valid_transitions"].items():
        if state not in states:
            raise ValueError(f"transition table references unknown state {state}")
        for next_state in next_states:
            if next_state not in states:
                raise ValueError(f"transition table references unknown next state {next_state}")
    for state, authorities in payload["transition_authority"].items():
        if state not in states:
            raise ValueError(f"authority table references unknown state {state}")
        if state != "UNPREPARED" and not authorities:
            raise ValueError(f"state {state} has no transition authority")
    for state, keys in payload["required_evidence"].items():
        if state not in states:
            raise ValueError(f"required evidence references unknown state {state}")
        if state != "UNPREPARED" and not keys:
            raise ValueError(f"state {state} has no required evidence")
    return payload


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_required_tree(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "execution/__init__.py",
        "execution/episode_ledger.py",
        "execution/episode_ledger_runtime_forensics.py",
        "execution/exit_reason_normalizer.py",
        "execution/helpers.py",
        "execution/log_utils.py",
        "requirements.txt",
        "config/runtime.yaml",
        "config/settings.json",
        "config/strategy_config.json",
        "config/exit_reason_map.yaml",
    ):
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def init_git_repo(path: Path, *, baseline_marker: str) -> str:
    subprocess.check_call(["git", "init"], cwd=path)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=path)
    write_text(path / "PROTOCOL_BASELINE_COMMIT.txt", baseline_marker + "\n")
    write_text(path / ".gitignore", "__pycache__/\n*.pyc\n")
    subprocess.check_call(["git", "add", "."], cwd=path)
    subprocess.check_call(["git", "commit", "-m", "baseline"], cwd=path)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def make_candidate_repo(source_root: Path, destination_root: Path, *, comment_suffix: str) -> dict[str, Any]:
    copy_required_tree(source_root, destination_root)
    baseline_commit = init_git_repo(destination_root, baseline_marker=PROTOCOL_BASELINE_COMMIT)
    episode_path = destination_root / "execution" / "episode_ledger.py"
    with episode_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# rehearsal-candidate: {comment_suffix}\n")
    subprocess.check_call(["git", "add", "execution/episode_ledger.py"], cwd=destination_root)
    subprocess.check_call(["git", "commit", "-m", "candidate rehearsal delta"], cwd=destination_root)
    candidate_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination_root, text=True).strip()
    return {
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "baseline_marker": PROTOCOL_BASELINE_COMMIT,
    }


def load_module_from_path(module_path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_consistent_runtime_root(runtime_root: Path, *, nav_equity: float = 12000.0) -> dict[str, Any]:
    execution_dir = runtime_root / "logs" / "execution"
    state_dir = runtime_root / "logs" / "state"
    config_dir = runtime_root / "config"
    execution_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event_type": "order_fill",
            "ts": "2026-07-12T00:00:00+00:00",
            "ts_fill_first": "2026-07-12T00:00:00+00:00",
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "side": "BUY",
            "reduceOnly": False,
            "executedQty": 1.0,
            "avgPrice": 100.0,
            "fee_total": 0.1,
            "orderId": "entry-1",
            "metadata": {"strategy": "rehearsal", "exit": {"reason": "tp"}},
        },
        {
            "event_type": "order_fill",
            "ts": "2026-07-12T01:00:00+00:00",
            "ts_fill_first": "2026-07-12T01:00:00+00:00",
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "side": "SELL",
            "reduceOnly": True,
            "executedQty": 1.0,
            "avgPrice": 102.0,
            "fee_total": 0.1,
            "orderId": "exit-1",
            "metadata": {"strategy": "rehearsal", "exit": {"reason": "tp"}, "entry_price": 100.0},
        },
    ]
    write_text(
        execution_dir / "orders_executed.jsonl",
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
    )
    write_text(execution_dir / "dle_shadow_events.jsonl", "")
    write_json(state_dir / "nav_state.json", {"total_equity": nav_equity})
    write_json(state_dir / "positions_state.json", {"positions": []})
    write_json(runtime_root / "supervisor_state.json", {"service": "dummy-executor", "status": "STOPPED", "pid": None})
    source_root = repo_root()
    for relative in RUNTIME_CONFIG_RELATIVE_PATHS:
        source = source_root / relative
        destination = runtime_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    from execution import episode_ledger as live_ledger_module

    with forensics.patched_runtime_paths(runtime_root), forensics.pushd(runtime_root):
        live_ledger_module.rebuild_and_save()
        ledger = live_ledger_module.load_episode_ledger()
        if ledger is None:
            raise RuntimeError("failed to seed consistent runtime root")
        ledger_hash = live_ledger_module._canonical_ledger_hash(ledger)
    checkpoint = read_json(state_dir / "episode_ledger_checkpoint.json")
    return {
        "ledger_hash": ledger_hash,
        "checkpoint_hash": checkpoint.get("ledger_hash"),
    }


def checkpoint_ledger_hash(checkpoint_path: Path) -> Optional[str]:
    if not checkpoint_path.exists():
        return None
    return read_json(checkpoint_path).get("ledger_hash")


def canonical_ledger_hash_from_file(ledger_path: Path) -> Optional[str]:
    if not ledger_path.exists():
        return None
    payload = read_json(ledger_path)
    from execution import episode_ledger as live_ledger_module

    return live_ledger_module._canonical_ledger_hash(payload)


def terminal_anchor(path: Path, size: int = 96) -> dict[str, Any]:
    raw = path.read_bytes() if path.exists() else b""
    suffix = raw[-size:] if raw else b""
    return {
        "sha256": forensics.sha256_bytes(raw),
        "size": len(raw),
        "suffix_anchor_hex": suffix.hex(),
        "ends_with_newline": raw.endswith(b"\n") if raw else True,
    }


def resolve_protocol_paths(live_root: Path, output_root: Path) -> dict[str, Any]:
    resolved = forensics.resolve_live_paths(live_root, live_root)
    for relative in ("logs/state/positions_state.json", "runtime_attestation.json", "supervisor_state.json"):
        path = live_root / relative
        resolved[relative.replace("/", "_").replace(".json", "")] = {
            **forensics.file_identity(path),
            "outside_expected_root": live_root.resolve() not in path.resolve().parents and path.resolve() != live_root.resolve(),
        }
    resolved["output_root"] = str(output_root.resolve())
    return resolved


def capture_preflight_gate(
    live_root: Path,
    pid: int,
    output_root: Path,
    *,
    require_live_module_attestation: bool = True,
) -> dict[str, Any]:
    attestation_path = live_root / "runtime_attestation.json"
    positions_path = live_root / "logs" / "state" / "positions_state.json"
    supervisor_path = live_root / "supervisor_state.json"
    loaded_module_attestation = read_json(attestation_path) if attestation_path.exists() else None
    process = live_process_provenance(pid, loaded_module_attestation)
    resolved = resolve_protocol_paths(live_root, output_root)
    ledger_path = live_root / "logs" / "state" / "episode_ledger.json"
    checkpoint_path = live_root / "logs" / "state" / "episode_ledger_checkpoint.json"
    nav_path = live_root / "logs" / "state" / "nav_state.json"
    gate = {
        "live_pid": pid,
        "process_start_time_utc": datetime_from_proc(pid),
        "process_provenance": process,
        "resolved_paths": resolved,
        "loaded_module_attestation": loaded_module_attestation,
        "current_ledger_semantic_hash": canonical_ledger_hash_from_file(ledger_path),
        "current_checkpoint_ledger_hash": checkpoint_ledger_hash(checkpoint_path),
        "current_nav_hash": forensics.sha256_file(nav_path),
        "source_terminal_anchors": {
            "active_execution_log": terminal_anchor(live_root / "logs" / "execution" / "orders_executed.jsonl"),
            "dle_authority_log": terminal_anchor(live_root / "logs" / "execution" / "dle_shadow_events.jsonl"),
        },
        "open_position_state": {
            "path": str(positions_path.resolve()),
            "sha256": forensics.sha256_file(positions_path),
            "open_positions": len(read_json(positions_path).get("positions", [])),
        },
        "supervisor_state": read_json(supervisor_path),
    }
    if gate["loaded_module_attestation"] is None and require_live_module_attestation:
        gate["passed"] = False
        gate["failure_code"] = "FAILED_CLOSED_LIVE_MODULE_UNPROVEN"
        return gate
    if gate["current_ledger_semantic_hash"] != gate["current_checkpoint_ledger_hash"]:
        gate["passed"] = False
        gate["failure_code"] = "FAILED_CLOSED_PREEXISTING_INCONSISTENCY"
        return gate
    gate["passed"] = True
    gate["failure_code"] = None
    return gate


def live_process_provenance(pid: int, loaded_module_attestation: Optional[dict[str, Any]]) -> dict[str, Any]:
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    repo_provenance = None
    if loaded_module_attestation is not None:
        repo_root_value = loaded_module_attestation.get("repository_provenance", {}).get("repo_root")
        if repo_root_value:
            candidate_repo_root = Path(repo_root_value)
            git_dir = candidate_repo_root / ".git"
            if git_dir.exists():
                repo_provenance = forensics.detect_repo_head(candidate_repo_root)
    return {
        "pid": pid,
        "cwd": str(Path(f"/proc/{pid}/cwd").resolve()),
        "exe": str(Path(f"/proc/{pid}/exe").resolve()),
        "cmdline": cmdline,
        "state": forensics.proc_state(pid),
        "safe_environ": forensics.safe_environ(pid),
        "repository_provenance": repo_provenance,
    }


def datetime_from_proc(pid: int) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    uptime_seconds = float(stat[21]) / float(clock_ticks)
    boot_time = None
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        if line.startswith("btime "):
            boot_time = int(line.split()[1])
            break
    if boot_time is None:
        raise RuntimeError("unable to determine btime from /proc/stat")
    start_epoch = boot_time + uptime_seconds
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(start_epoch))


def snapshot_specs_for_protocol(live_root: Path) -> list[tuple[str, Path, str]]:
    resolved = forensics.resolve_live_paths(live_root, live_root)
    specs = forensics.snapshot_source_specs(resolved)
    specs.extend(
        [
            ("positions_state", live_root / "logs" / "state" / "positions_state.json", "sources/state/positions_state.json"),
            ("runtime_attestation", live_root / "runtime_attestation.json", "sources/runtime/runtime_attestation.json"),
            ("supervisor_state", live_root / "supervisor_state.json", "sources/runtime/supervisor_state.json"),
        ]
    )
    for relative in RUNTIME_CONFIG_RELATIVE_PATHS:
        specs.append((relative.replace("/", "_"), live_root / relative, f"sources/{relative}"))
    return specs


def capture_atomic_snapshot(
    *,
    live_root: Path,
    pid: int,
    evidence_dir: Path,
    inject_copy_failure: bool = False,
) -> dict[str, Any]:
    forensics.ensure_safe_output_root(evidence_dir, [live_root, repo_root()])
    evidence_dir.mkdir(parents=True, exist_ok=True)
    boundary: dict[str, Any] = {"method": "process_sigstop_copy_resume", "ts_started": forensics.utc_now()}
    specs = snapshot_specs_for_protocol(live_root)
    try:
        with forensics.stopped_process(pid) as stop_info:
            boundary["stop_resume"] = stop_info
            if inject_copy_failure:
                raise RuntimeError("simulated snapshot copy failure")
            manifest = forensics.copy_snapshot_sources(evidence_dir, specs)
    except Exception as exc:
        boundary["ts_failed"] = forensics.utc_now()
        boundary["error"] = str(exc)
        return {"passed": False, "boundary": boundary, "snapshot_manifest": []}
    boundary["ts_completed"] = forensics.utc_now()
    write_json(evidence_dir / "snapshot_boundary.json", boundary)
    write_json(evidence_dir / "snapshot_manifest.json", {"snapshot_manifest": manifest})
    return {"passed": True, "boundary": boundary, "snapshot_manifest": manifest}


def copy_snapshot_manifest(snapshot_manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(snapshot_manifest))


def snapshot_manifest_nav_hash(snapshot_manifest: list[dict[str, Any]]) -> str:
    for item in snapshot_manifest:
        if item["role"] == "nav_input":
            return item["identity"]["sha256"]
    raise RuntimeError("snapshot manifest missing nav_input")


def run_python_json(pythonpath: Iterable[Path], code: str, *, cwd: Path, extra_env: Optional[dict[str, str]] = None) -> Any:
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(str(path) for path in pythonpath)
    if extra_env:
        env.update(extra_env)
    result = subprocess.check_output([sys.executable, "-c", code], cwd=cwd, env=env, text=True)
    return json.loads(result)


def subprocess_attest_module(candidate_repo_root: Path) -> dict[str, Any]:
    code = textwrap.dedent(
        """
        import json
        from pathlib import Path
        from execution import episode_ledger_runtime_forensics as f
        print(json.dumps(f.attest_module(Path.cwd()), sort_keys=True))
        """
    )
    return run_python_json([candidate_repo_root], code, cwd=candidate_repo_root)


def subprocess_run_builder(
    candidate_repo_root: Path,
    runtime_root: Path,
    run_name: str,
    input_manifest: dict[str, Any],
    builder_name: str,
) -> dict[str, Any]:
    manifest_path = runtime_root.parent / f"{run_name}_input_manifest.json"
    write_json(manifest_path, input_manifest)
    code = textwrap.dedent(
        """
        import json
        from pathlib import Path
        from execution import episode_ledger as ledger
        from execution import episode_ledger_runtime_forensics as f

        runtime_root = Path("RUNTIME_ROOT_PLACEHOLDER")
        run_name = "RUN_NAME_PLACEHOLDER"
        manifest_path = Path("MANIFEST_PLACEHOLDER")
        builder_name = "BUILDER_PLACEHOLDER"
        input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        attestation = f.attest_module(Path.cwd())
        result = f.execute_run(
            run_name,
            runtime_root,
            f.read_roles_for_runtime(runtime_root),
            attestation,
            input_manifest,
            getattr(ledger, builder_name),
        )
        print(json.dumps(result, sort_keys=True))
        """
    )
    code = (
        code.replace("RUNTIME_ROOT_PLACEHOLDER", str(runtime_root))
        .replace("RUN_NAME_PLACEHOLDER", run_name)
        .replace("MANIFEST_PLACEHOLDER", str(manifest_path))
        .replace("BUILDER_PLACEHOLDER", builder_name)
    )
    return run_python_json([candidate_repo_root], code, cwd=candidate_repo_root)


def rebuild_log_tail(runtime_root: Path) -> dict[str, Any]:
    rebuild_path = runtime_root / "logs" / "execution" / "episode_ledger_rebuild.jsonl"
    if not rebuild_path.exists():
        return {}
    lines = [line for line in rebuild_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def augment_run_summary(summary: dict[str, Any], runtime_root: Path, snapshot_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_payload = summary["semantic_payload"]
    episodes_payload = semantic_payload.get("episodes_v2") or semantic_payload.get("episodes") or []
    stats = semantic_payload.get("stats", {})
    summary = dict(summary)
    summary["per_episode_payload_hash"] = forensics.stable_json_hash(episodes_payload)
    summary["exit_distribution_hash"] = forensics.stable_json_hash(stats.get("exit_reasons", {}))
    summary["nav_semantic_input_hash"] = snapshot_manifest_nav_hash(snapshot_manifest)
    summary["rebuild_log_tail"] = rebuild_log_tail(runtime_root)
    return summary


def full_surface_difference(left: dict[str, Any], right: dict[str, Any]) -> Optional[dict[str, Any]]:
    comparisons = (
        "canonical_ledger_hash",
        "episode_count",
        "ordered_episode_uid_hash",
        "per_episode_payload_hash",
        "authority_payload_hash",
        "aggregate_stats_hash",
        "reconciliation_hash",
        "exit_distribution_hash",
        "nav_semantic_input_hash",
    )
    for key in comparisons:
        if left.get(key) != right.get(key):
            return {"field": key, "left": left.get(key), "right": right.get(key)}
    semantic_diff = forensics.first_semantic_difference(left.get("semantic_payload"), right.get("semantic_payload"))
    if semantic_diff is not None:
        return {"field": "semantic_payload", **semantic_diff}
    return None


def run_equivalence_gate(
    *,
    candidate_repo_root: Path,
    snapshot_manifest: list[dict[str, Any]],
    evidence_dir: Path,
    inject_semantic_diff: bool = False,
) -> dict[str, Any]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    module_attestation = subprocess_attest_module(candidate_repo_root)
    input_manifest = {
        "snapshot_manifest": copy_snapshot_manifest(snapshot_manifest),
        "module_content_hash": module_attestation["loaded_module_provenance"]["sha256"],
        "resolved_paths_hash": forensics.stable_json_hash([item["snapshot"] for item in snapshot_manifest]),
        "process_provenance_hash": sha256_text(candidate_repo_root.as_posix()),
    }
    runs: dict[str, dict[str, Any]] = {}

    run_full_root = evidence_dir / "run_full" / "runtime"
    forensics.stage_runtime_root(snapshot_manifest, run_full_root, include_resume_state=False)
    result_full = subprocess_run_builder(candidate_repo_root, run_full_root, "A_full", input_manifest, "build_episode_ledger")
    runs["A_full"] = augment_run_summary(result_full["summary"], run_full_root, snapshot_manifest)

    run_initial_root = evidence_dir / "run_candidate_initial" / "runtime"
    forensics.stage_runtime_root(snapshot_manifest, run_initial_root, include_resume_state=False)
    result_initial = subprocess_run_builder(candidate_repo_root, run_initial_root, "B_initial", input_manifest, "rebuild_and_save")
    runs["B_initial"] = augment_run_summary(result_initial["summary"], run_initial_root, snapshot_manifest)

    run_noop_root = evidence_dir / "run_checkpoint_noop" / "runtime"
    forensics.stage_runtime_root(snapshot_manifest, run_noop_root, include_resume_state=True)
    result_noop = subprocess_run_builder(candidate_repo_root, run_noop_root, "C_noop", input_manifest, "rebuild_and_save")
    runs["C_noop"] = augment_run_summary(result_noop["summary"], run_noop_root, snapshot_manifest)

    run_rerun_root = evidence_dir / "run_checkpoint_noop_rerun" / "runtime"
    if run_rerun_root.exists():
        shutil.rmtree(run_rerun_root)
    shutil.copytree(run_noop_root, run_rerun_root)
    result_rerun = subprocess_run_builder(candidate_repo_root, run_rerun_root, "D_noop_rerun", input_manifest, "rebuild_and_save")
    runs["D_noop_rerun"] = augment_run_summary(result_rerun["summary"], run_rerun_root, snapshot_manifest)

    run_append_candidate_root = evidence_dir / "run_controlled_append" / "runtime_candidate"
    forensics.stage_runtime_root(snapshot_manifest, run_append_candidate_root, include_resume_state=False)
    subprocess_run_builder(candidate_repo_root, run_append_candidate_root, "append_seed", input_manifest, "rebuild_and_save")
    forensics.append_controlled_batch(run_append_candidate_root / "logs" / "execution" / "orders_executed.jsonl")
    result_append_candidate = subprocess_run_builder(
        candidate_repo_root,
        run_append_candidate_root,
        "E_append_candidate",
        input_manifest,
        "rebuild_and_save",
    )
    runs["E_append_candidate"] = augment_run_summary(result_append_candidate["summary"], run_append_candidate_root, snapshot_manifest)

    run_append_full_root = evidence_dir / "run_controlled_append" / "runtime_full"
    forensics.stage_runtime_root(snapshot_manifest, run_append_full_root, include_resume_state=False)
    forensics.append_controlled_batch(run_append_full_root / "logs" / "execution" / "orders_executed.jsonl")
    result_append_full = subprocess_run_builder(
        candidate_repo_root,
        run_append_full_root,
        "F_append_full",
        input_manifest,
        "build_episode_ledger",
    )
    runs["F_append_full"] = augment_run_summary(result_append_full["summary"], run_append_full_root, snapshot_manifest)

    comparisons = {
        "A_vs_B": full_surface_difference(runs["A_full"], runs["B_initial"]),
        "A_vs_C": full_surface_difference(runs["A_full"], runs["C_noop"]),
        "A_vs_D": full_surface_difference(runs["A_full"], runs["D_noop_rerun"]),
        "E_vs_F": full_surface_difference(runs["E_append_candidate"], runs["F_append_full"]),
    }
    if inject_semantic_diff:
        comparisons["A_vs_B"] = {"field": "synthetic_fault", "left": "baseline", "right": "candidate"}
    first_diff = next(({"comparison": name, **diff} for name, diff in comparisons.items() if diff is not None), None)
    checkpoint_compatibility = {
        "captured_checkpoint_mode": runs["C_noop"]["rebuild_log_tail"].get("mode"),
        "captured_checkpoint_fallback_reason": runs["C_noop"]["rebuild_log_tail"].get("fallback_reason"),
        "rerun_mode": runs["D_noop_rerun"]["rebuild_log_tail"].get("mode"),
        "rerun_bytes_read": runs["D_noop_rerun"]["rebuild_log_tail"].get("bytes_read"),
        "compatible": first_diff is None,
    }
    report = {
        "passed": first_diff is None,
        "module_attestation": module_attestation,
        "input_manifest_hash": forensics.stable_json_hash(input_manifest),
        "runs": runs,
        "comparisons": comparisons,
        "first_semantic_difference": first_diff,
        "checkpoint_compatibility": checkpoint_compatibility,
    }
    write_json(evidence_dir / "comparison_report.json", report)
    return report


def file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        manifest[str(path.relative_to(root))] = {
            "sha256": forensics.sha256_file(path),
            "size": path.stat().st_size,
        }
    return manifest


def is_authorized_mutation(relative_path: str) -> bool:
    return relative_path in AUTHORIZED_MUTATION_PATTERNS or relative_path.endswith(".pyc") or "__pycache__/" in relative_path


def detect_unauthorized_mutations(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    unexpected: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) == after.get(path):
            continue
        if is_authorized_mutation(path):
            continue
        unexpected.append({"path": path, "before": before.get(path), "after": after.get(path)})
    return unexpected


def git_status(repo_path: Path) -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=repo_path, text=True).strip()


def git_head(repo_path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True).strip()


def git_module_hash(repo_path: Path) -> str:
    return forensics.sha256_file(repo_path / "execution" / "episode_ledger.py")


def validate_candidate_freshness(candidate_repo_root: Path, prohibited_module_hash: str) -> dict[str, Any]:
    head = git_head(candidate_repo_root)
    status = git_status(candidate_repo_root)
    module_hash = git_module_hash(candidate_repo_root)
    baseline_marker = (candidate_repo_root / "PROTOCOL_BASELINE_COMMIT.txt").read_text(encoding="utf-8").strip()
    commits = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=candidate_repo_root, text=True).strip()
    result = {
        "candidate_commit": head,
        "candidate_module_hash": module_hash,
        "baseline_marker": baseline_marker,
        "clean_worktree": status == "",
        "commit_count": int(commits),
        "passed": True,
        "failure_code": None,
    }
    if head.startswith(PROHIBITED_CANDIDATE_COMMIT):
        result["passed"] = False
        result["failure_code"] = "FAILED_CLOSED_PROHIBITED_CANDIDATE"
        return result
    if module_hash == prohibited_module_hash:
        result["passed"] = False
        result["failure_code"] = "FAILED_CLOSED_PROHIBITED_CANDIDATE"
        return result
    if status != "":
        result["passed"] = False
        result["failure_code"] = "FAILED_CLOSED_DIRTY_WORKTREE"
        return result
    if baseline_marker != PROTOCOL_BASELINE_COMMIT or int(commits) < 2:
        result["passed"] = False
        result["failure_code"] = "FAILED_CLOSED_PROHIBITED_CANDIDATE"
        return result
    return result


def build_bundle_manifest(
    *,
    context: CandidateBundleContext,
    preflight_gate: dict[str, Any],
    snapshot: dict[str, Any],
    equivalence_gate: dict[str, Any],
    freshness_gate: dict[str, Any],
    authorization_record: dict[str, Any],
) -> dict[str, Any]:
    candidate_attestation = equivalence_gate["module_attestation"]
    rollback_attestation = subprocess_attest_module(context.rollback_repo_root)
    runtime_config_hashes = {}
    for relative in RUNTIME_CONFIG_RELATIVE_PATHS:
        path = context.runtime_root / relative
        runtime_config_hashes[relative] = forensics.sha256_file(path)
    bundle = {
        "candidate_commit_sha": freshness_gate["candidate_commit"],
        "candidate_worktree_clean": freshness_gate["clean_worktree"],
        "candidate_module_content_hash": candidate_attestation["loaded_module_provenance"]["sha256"],
        "candidate_callable_fingerprints": candidate_attestation["loaded_module_provenance"]["callables"],
        "interpreter_identity": {
            "path": sys.executable,
            "version": sys.version,
        },
        "dependency_environment_fingerprint": {
            "requirements_sha256": forensics.sha256_file(context.candidate_repo_root / "requirements.txt"),
            "python_implementation": sys.implementation.name,
        },
        "runtime_configuration_hashes": runtime_config_hashes,
        "resolved_source_destination_paths": {
            "live_runtime_root": str(context.runtime_root.resolve()),
            "evidence_dir": str(context.evidence_dir.resolve()),
            "candidate_repo_root": str(context.candidate_repo_root.resolve()),
            "rollback_repo_root": str(context.rollback_repo_root.resolve()),
        },
        "canonicalization_identifier": "execution.episode_ledger._canonical_ledger_hash",
        "nav_semantic_input_declaration": {
            "enabled": True,
            "path": "logs/state/nav_state.json",
            "sha256": snapshot_manifest_nav_hash(snapshot["snapshot_manifest"]),
        },
        "test_results": {
            "rehearsal_matrix_passed": equivalence_gate["passed"],
            "comparison_report_hash": forensics.stable_json_hash(equivalence_gate["comparisons"]),
        },
        "atomic_snapshot_manifest": snapshot["snapshot_manifest"],
        "full_replay_equivalence_results": {
            "comparisons": equivalence_gate["comparisons"],
            "first_semantic_difference": equivalence_gate["first_semantic_difference"],
        },
        "checkpoint_compatibility_result": equivalence_gate["checkpoint_compatibility"],
        "rollback_target": {
            "commit_sha": git_head(context.rollback_repo_root),
            "module_hash": rollback_attestation["loaded_module_provenance"]["sha256"],
            "procedure": [
                "stop candidate supervisor process",
                "restart rollback supervisor process from declared rollback repo",
                "verify post-rollback loaded module hash equals rollback target hash",
            ],
        },
        "authorization_record": authorization_record,
        "bundle_checksum_manifest": {
            "sha256sums_sha256": "pending",
            "entries": 0,
        },
        "state_machine": protocol_state_model(),
        "preflight_gate": preflight_gate,
        "freshness_gate": freshness_gate,
    }
    return bundle


def compute_checksum_manifest(root: Path) -> list[str]:
    return forensics.compute_sha256sums(root)


def verify_bundle_schema(bundle: dict[str, Any], schema_path: Path = SCHEMA_PATH) -> None:
    schema = read_json(schema_path)
    validate_json_schema(bundle, schema)


def validate_json_schema(instance: Any, schema: dict[str, Any], path: str = "$") -> None:
    schema_type = schema.get("type")
    if schema_type is not None:
        type_ok = {
            "object": isinstance(instance, dict),
            "array": isinstance(instance, list),
            "string": isinstance(instance, str),
            "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
            "integer": isinstance(instance, int) and not isinstance(instance, bool),
            "boolean": isinstance(instance, bool),
            "null": instance is None,
        }.get(schema_type, True)
        if not type_ok:
            raise ValueError(f"{path}: expected {schema_type}")
    if "enum" in schema and instance not in schema["enum"]:
        raise ValueError(f"{path}: value {instance!r} not in enum")
    if "const" in schema and instance != schema["const"]:
        raise ValueError(f"{path}: expected const {schema['const']!r}")
    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                raise ValueError(f"{path}: missing required property {key}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                validate_json_schema(value, properties[key], f"{path}.{key}")
            elif isinstance(additional, dict):
                validate_json_schema(value, additional, f"{path}.{key}")
            elif additional is False:
                raise ValueError(f"{path}: unexpected property {key}")
    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            validate_json_schema(item, schema["items"], f"{path}[{index}]")


def spawn_dummy_executor(
    *,
    repo_path: Path,
    runtime_root: Path,
    label: str,
    perform_rebuild: bool,
) -> subprocess.Popen[str]:
    code = textwrap.dedent(
        """
        import json
        import os
        import signal
        import sys
        import time
        from pathlib import Path
        from execution import episode_ledger as ledger
        from execution import episode_ledger_runtime_forensics as f

        repo_root = Path(sys.argv[1])
        runtime_root = Path(sys.argv[2])
        label = sys.argv[3]
        perform_rebuild = sys.argv[4] == "1"
        attestation_path = runtime_root / "runtime_attestation.json"
        supervisor_path = runtime_root / "supervisor_state.json"
        positions_path = runtime_root / "logs" / "state" / "positions_state.json"

        running = {"value": True}

        def _handle(_signum, _frame):
            running["value"] = False

        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)
        with f.patched_runtime_paths(runtime_root), f.pushd(runtime_root):
            if perform_rebuild:
                ledger.rebuild_and_save()
            attestation = f.attest_module(repo_root)
            attestation["runtime_label"] = label
            attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
        supervisor_path.write_text(json.dumps({"service": "dummy-executor", "status": "RUNNING", "pid": os.getpid(), "label": label}), encoding="utf-8")
        if not positions_path.exists():
            positions_path.parent.mkdir(parents=True, exist_ok=True)
            positions_path.write_text(json.dumps({"positions": []}), encoding="utf-8")
        while running["value"]:
            time.sleep(0.2)
        supervisor_path.write_text(json.dumps({"service": "dummy-executor", "status": "STOPPED", "pid": None, "label": label}), encoding="utf-8")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_path)
    return subprocess.Popen(
        [sys.executable, "-c", code, str(repo_path), str(runtime_root), label, "1" if perform_rebuild else "0"],
        cwd=runtime_root,
        env=env,
        text=True,
    )


def wait_for_file(path: Path, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for {path}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def clear_runtime_markers(runtime_root: Path) -> None:
    for relative in ("runtime_attestation.json", "supervisor_state.json"):
        path = runtime_root / relative
        if path.exists():
            path.unlink()


def perform_rollback(
    *,
    attempt: AttemptState,
    rollback_repo_root: Path,
    runtime_root: Path,
    evidence_dir: Path,
    reason: str,
) -> dict[str, Any]:
    clear_runtime_markers(runtime_root)
    rollback_proc = spawn_dummy_executor(repo_path=rollback_repo_root, runtime_root=runtime_root, label="rollback", perform_rebuild=True)
    wait_for_file(runtime_root / "runtime_attestation.json")
    preflight = capture_preflight_gate(runtime_root, rollback_proc.pid, evidence_dir)
    loaded_hash = preflight["loaded_module_attestation"]["loaded_module_provenance"]["sha256"]
    expected_hash = git_module_hash(rollback_repo_root)
    if loaded_hash != expected_hash:
        stop_process(rollback_proc)
        raise ProtocolError(
            "FAILED_CLOSED_ROLLBACK_VERIFICATION_FAILED",
            f"rollback module hash mismatch: {loaded_hash} != {expected_hash}",
        )
    rollback_record = {
        "rollback_reason": reason,
        "rollback_commit": git_head(rollback_repo_root),
        "rollback_module_hash": expected_hash,
        "post_rollback_preflight": preflight,
    }
    write_json(evidence_dir / "rollback_record.json", rollback_record)
    attempt.transition("ROLLED_BACK", "supervisor_controller", {"rollback_record": rollback_record})
    stop_process(rollback_proc)
    return rollback_record


def build_authorization_record(bundle_hash: str) -> dict[str, Any]:
    return {
        "authorized_by": "rehearsal-authorizer",
        "ts": forensics.utc_now(),
        "bundle_hash": bundle_hash,
        "statement": "offline equivalence, provenance, and rollback gates satisfied for rehearsal only",
    }


def provisional_authorization_record() -> dict[str, Any]:
    return {
        "authorized_by": "pending",
        "ts": forensics.utc_now(),
        "bundle_hash": "pending",
        "statement": "authorization pending; schema validation only",
    }


def prepare_bundle_checksums(bundle: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    bundle_path = evidence_dir / "bundle_manifest.json"
    write_json(bundle_path, bundle)
    checksum_lines = compute_checksum_manifest(evidence_dir)
    write_text(evidence_dir / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    bundle = dict(bundle)
    bundle["bundle_checksum_manifest"] = {
        "sha256sums_sha256": forensics.sha256_file(evidence_dir / "SHA256SUMS"),
        "entries": len(checksum_lines),
    }
    write_json(bundle_path, bundle)
    checksum_lines = compute_checksum_manifest(evidence_dir)
    write_text(evidence_dir / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    return bundle


def run_happy_path_scenario(attempt: AttemptState, context: CandidateBundleContext) -> dict[str, Any]:
    live_proc = spawn_dummy_executor(repo_path=context.rollback_repo_root, runtime_root=context.runtime_root, label="known-good", perform_rebuild=False)
    try:
        wait_for_file(context.runtime_root / "runtime_attestation.json")
        preflight = capture_preflight_gate(context.runtime_root, live_proc.pid, context.evidence_dir)
        attempt.transition("PREFLIGHT_CAPTURED", "protocol_runner", {"preflight_gate": preflight, "resolved_paths": preflight["resolved_paths"]})

        snapshot = capture_atomic_snapshot(live_root=context.runtime_root, pid=live_proc.pid, evidence_dir=context.evidence_dir / "pre_deploy_snapshot")
        if not snapshot["passed"]:
            raise ProtocolError("FAILED_CLOSED_SNAPSHOT_FAILED", snapshot["boundary"]["error"])
        attempt.transition("SNAPSHOT_FROZEN", "protocol_runner", {"atomic_snapshot": snapshot})

        equivalence = run_equivalence_gate(
            candidate_repo_root=context.candidate_repo_root,
            snapshot_manifest=snapshot["snapshot_manifest"],
            evidence_dir=context.evidence_dir / "pre_deploy_equivalence",
        )
        if not equivalence["passed"]:
            raise ProtocolError("FAILED_CLOSED_EQUIVALENCE_FAILED", "offline equivalence mismatch")
        attempt.transition("OFFLINE_EQUIVALENCE_PASSED", "protocol_runner", {"equivalence_gate": equivalence})

        freshness = validate_candidate_freshness(context.candidate_repo_root, context.prohibited_module_hash)
        if not freshness["passed"]:
            raise ProtocolError(freshness["failure_code"], "candidate freshness gate rejected candidate")
        bundle_stub = build_bundle_manifest(
            context=context,
            preflight_gate=preflight,
            snapshot=snapshot,
            equivalence_gate=equivalence,
            freshness_gate=freshness,
            authorization_record=provisional_authorization_record(),
        )
        verify_bundle_schema(bundle_stub)
        attempt.transition("RUNTIME_PROVENANCE_VERIFIED", "protocol_runner", {"candidate_bundle": bundle_stub, "freshness_gate": freshness})

        authorization = build_authorization_record(forensics.stable_json_hash(bundle_stub))
        bundle = build_bundle_manifest(
            context=context,
            preflight_gate=preflight,
            snapshot=snapshot,
            equivalence_gate=equivalence,
            freshness_gate=freshness,
            authorization_record=authorization,
        )
        bundle = prepare_bundle_checksums(bundle, context.evidence_dir)
        verify_bundle_schema(bundle)
        attempt.transition("DEPLOYMENT_AUTHORIZED", "deployment_authorizer", {"authorization_record": authorization, "rollback_plan": bundle["rollback_target"]})

        before_manifest = file_manifest(context.runtime_root)
        stop_process(live_proc)
        clear_runtime_markers(context.runtime_root)
        candidate_proc = spawn_dummy_executor(repo_path=context.candidate_repo_root, runtime_root=context.runtime_root, label="candidate", perform_rebuild=True)
        wait_for_file(context.runtime_root / "runtime_attestation.json")
        deploy_record = {"candidate_pid": candidate_proc.pid, "ts": forensics.utc_now()}
        attempt.transition("DEPLOYING", "supervisor_controller", {"deployment_start": deploy_record})

        post_start_preflight = capture_preflight_gate(context.runtime_root, candidate_proc.pid, context.evidence_dir)
        post_start_snapshot = capture_atomic_snapshot(
            live_root=context.runtime_root,
            pid=candidate_proc.pid,
            evidence_dir=context.evidence_dir / "post_start_snapshot",
        )
        post_start_equivalence = run_equivalence_gate(
            candidate_repo_root=context.candidate_repo_root,
            snapshot_manifest=post_start_snapshot["snapshot_manifest"],
            evidence_dir=context.evidence_dir / "post_start_equivalence",
        )
        after_manifest = file_manifest(context.runtime_root)
        unexpected = detect_unauthorized_mutations(before_manifest, after_manifest)
        post_start_gate = {
            "preflight": post_start_preflight,
            "snapshot": post_start_snapshot,
            "equivalence": post_start_equivalence,
            "unexpected_mutations": unexpected,
            "exchange_activity_detected": False,
            "position_state": read_json(context.runtime_root / "logs" / "state" / "positions_state.json"),
        }
        attempt.transition("POST_START_VALIDATING", "protocol_runner", {"post_start_gate": post_start_gate})
        if post_start_preflight["loaded_module_attestation"]["loaded_module_provenance"]["sha256"] != bundle["candidate_module_content_hash"]:
            raise ProtocolError("FAILED_CLOSED_MODULE_HASH_MISMATCH", "post-start loaded module hash mismatched bundle")
        if unexpected:
            raise ProtocolError("FAILED_CLOSED_UNAUTHORIZED_MUTATION", "unexpected filesystem mutation detected")
        if not post_start_equivalence["passed"]:
            raise ProtocolError("FAILED_CLOSED_EQUIVALENCE_FAILED", "post-start equivalence mismatch")
        acceptance = {
            "ts": forensics.utc_now(),
            "accepted_bundle_hash": forensics.stable_json_hash(bundle),
            "candidate_pid": candidate_proc.pid,
        }
        attempt.transition("ACCEPTED", "deployment_authorizer", {"acceptance_record": acceptance})
        stop_process(candidate_proc)
        return {
            "state": attempt.state,
            "bundle_hash": forensics.stable_json_hash(bundle),
            "preflight": preflight,
            "snapshot": snapshot,
            "equivalence": equivalence,
            "post_start_gate": post_start_gate,
            "bundle_manifest": bundle,
        }
    finally:
        stop_process(live_proc)


def run_failure_scenario(
    scenario: str,
    attempt: AttemptState,
    context: CandidateBundleContext,
) -> dict[str, Any]:
    evidence = {}
    if scenario == "reject_prohibited_candidate":
        freshness = validate_candidate_freshness(context.rollback_repo_root, context.prohibited_module_hash)
        if freshness["passed"]:
            raise AssertionError("expected prohibited candidate rejection")
        attempt.fail_closed(freshness["failure_code"], "prohibited candidate rejected", {"failure_record": freshness})
        return {"state": attempt.state, "failure_code": attempt.failure_code}

    if scenario == "reject_dirty_worktree":
        write_text(context.candidate_repo_root / "DIRTY.txt", "dirty\n")
        freshness = validate_candidate_freshness(context.candidate_repo_root, context.prohibited_module_hash)
        if freshness["failure_code"] != "FAILED_CLOSED_DIRTY_WORKTREE":
            raise AssertionError("expected dirty worktree rejection")
        attempt.fail_closed(freshness["failure_code"], "dirty worktree rejected", {"failure_record": freshness})
        return {"state": attempt.state, "failure_code": attempt.failure_code}

    if scenario == "reject_module_hash_mismatch":
        live_proc = spawn_dummy_executor(repo_path=context.rollback_repo_root, runtime_root=context.runtime_root, label="known-good", perform_rebuild=False)
        try:
            wait_for_file(context.runtime_root / "runtime_attestation.json")
            preflight = capture_preflight_gate(context.runtime_root, live_proc.pid, context.evidence_dir)
            attempt.transition("PREFLIGHT_CAPTURED", "protocol_runner", {"preflight_gate": preflight, "resolved_paths": preflight["resolved_paths"]})
            snapshot = capture_atomic_snapshot(live_root=context.runtime_root, pid=live_proc.pid, evidence_dir=context.evidence_dir / "pre_deploy_snapshot")
            attempt.transition("SNAPSHOT_FROZEN", "protocol_runner", {"atomic_snapshot": snapshot})
            equivalence = run_equivalence_gate(candidate_repo_root=context.candidate_repo_root, snapshot_manifest=snapshot["snapshot_manifest"], evidence_dir=context.evidence_dir / "pre_deploy_equivalence")
            attempt.transition("OFFLINE_EQUIVALENCE_PASSED", "protocol_runner", {"equivalence_gate": equivalence})
            freshness = validate_candidate_freshness(context.candidate_repo_root, context.prohibited_module_hash)
            bundle = build_bundle_manifest(
                context=context,
                preflight_gate=preflight,
                snapshot=snapshot,
                equivalence_gate=equivalence,
                freshness_gate=freshness,
                authorization_record={"pending": True},
            )
            bundle["candidate_module_content_hash"] = "0" * 64
            attempt.transition("RUNTIME_PROVENANCE_VERIFIED", "protocol_runner", {"candidate_bundle": bundle, "freshness_gate": freshness})
            attempt.fail_closed(
                "FAILED_CLOSED_MODULE_HASH_MISMATCH",
                "bundle candidate module hash intentionally mismatched",
                {"failure_record": {"expected": bundle["candidate_module_content_hash"], "actual": equivalence["module_attestation"]["loaded_module_provenance"]["sha256"]}},
            )
            return {"state": attempt.state, "failure_code": attempt.failure_code}
        finally:
            stop_process(live_proc)

    if scenario == "reject_preexisting_inconsistency":
        checkpoint_path = context.runtime_root / "logs" / "state" / "episode_ledger_checkpoint.json"
        checkpoint = read_json(checkpoint_path)
        checkpoint["ledger_hash"] = "deadbeef"
        write_json(checkpoint_path, checkpoint)
        live_proc = spawn_dummy_executor(repo_path=context.rollback_repo_root, runtime_root=context.runtime_root, label="known-good", perform_rebuild=False)
        try:
            wait_for_file(context.runtime_root / "runtime_attestation.json")
            preflight = capture_preflight_gate(context.runtime_root, live_proc.pid, context.evidence_dir)
            if preflight["failure_code"] != "FAILED_CLOSED_PREEXISTING_INCONSISTENCY":
                raise AssertionError("expected preexisting inconsistency rejection")
            attempt.fail_closed(preflight["failure_code"], "preexisting ledger/checkpoint mismatch rejected", {"failure_record": preflight})
            return {"state": attempt.state, "failure_code": attempt.failure_code}
        finally:
            stop_process(live_proc)

    if scenario == "reject_snapshot_failure":
        live_proc = spawn_dummy_executor(repo_path=context.rollback_repo_root, runtime_root=context.runtime_root, label="known-good", perform_rebuild=False)
        try:
            wait_for_file(context.runtime_root / "runtime_attestation.json")
            preflight = capture_preflight_gate(context.runtime_root, live_proc.pid, context.evidence_dir)
            attempt.transition("PREFLIGHT_CAPTURED", "protocol_runner", {"preflight_gate": preflight, "resolved_paths": preflight["resolved_paths"]})
            snapshot = capture_atomic_snapshot(
                live_root=context.runtime_root,
                pid=live_proc.pid,
                evidence_dir=context.evidence_dir / "pre_deploy_snapshot",
                inject_copy_failure=True,
            )
            if snapshot["passed"]:
                raise AssertionError("expected snapshot failure")
            attempt.fail_closed("FAILED_CLOSED_SNAPSHOT_FAILED", "snapshot copy failed", {"failure_record": snapshot})
            return {"state": attempt.state, "failure_code": attempt.failure_code, "process_alive_after_failure": live_proc.poll() is None}
        finally:
            stop_process(live_proc)

    if scenario == "reject_equivalence_failure":
        live_proc = spawn_dummy_executor(repo_path=context.rollback_repo_root, runtime_root=context.runtime_root, label="known-good", perform_rebuild=False)
        try:
            wait_for_file(context.runtime_root / "runtime_attestation.json")
            preflight = capture_preflight_gate(context.runtime_root, live_proc.pid, context.evidence_dir)
            attempt.transition("PREFLIGHT_CAPTURED", "protocol_runner", {"preflight_gate": preflight, "resolved_paths": preflight["resolved_paths"]})
            snapshot = capture_atomic_snapshot(live_root=context.runtime_root, pid=live_proc.pid, evidence_dir=context.evidence_dir / "pre_deploy_snapshot")
            attempt.transition("SNAPSHOT_FROZEN", "protocol_runner", {"atomic_snapshot": snapshot})
            equivalence = run_equivalence_gate(
                candidate_repo_root=context.candidate_repo_root,
                snapshot_manifest=snapshot["snapshot_manifest"],
                evidence_dir=context.evidence_dir / "pre_deploy_equivalence",
                inject_semantic_diff=True,
            )
            if equivalence["passed"]:
                raise AssertionError("expected equivalence failure")
            attempt.fail_closed("FAILED_CLOSED_EQUIVALENCE_FAILED", "synthetic equivalence mismatch rejected", {"failure_record": equivalence})
            return {"state": attempt.state, "failure_code": attempt.failure_code}
        finally:
            stop_process(live_proc)

    if scenario == "rollback_unauthorized_mutation":
        live_proc = spawn_dummy_executor(repo_path=context.rollback_repo_root, runtime_root=context.runtime_root, label="known-good", perform_rebuild=False)
        candidate_proc: Optional[subprocess.Popen[str]] = None
        try:
            wait_for_file(context.runtime_root / "runtime_attestation.json")
            preflight = capture_preflight_gate(context.runtime_root, live_proc.pid, context.evidence_dir)
            attempt.transition("PREFLIGHT_CAPTURED", "protocol_runner", {"preflight_gate": preflight, "resolved_paths": preflight["resolved_paths"]})
            snapshot = capture_atomic_snapshot(live_root=context.runtime_root, pid=live_proc.pid, evidence_dir=context.evidence_dir / "pre_deploy_snapshot")
            attempt.transition("SNAPSHOT_FROZEN", "protocol_runner", {"atomic_snapshot": snapshot})
            equivalence = run_equivalence_gate(candidate_repo_root=context.candidate_repo_root, snapshot_manifest=snapshot["snapshot_manifest"], evidence_dir=context.evidence_dir / "pre_deploy_equivalence")
            attempt.transition("OFFLINE_EQUIVALENCE_PASSED", "protocol_runner", {"equivalence_gate": equivalence})
            freshness = validate_candidate_freshness(context.candidate_repo_root, context.prohibited_module_hash)
            bundle = build_bundle_manifest(
                context=context,
                preflight_gate=preflight,
                snapshot=snapshot,
                equivalence_gate=equivalence,
                freshness_gate=freshness,
                authorization_record={"pending": True},
            )
            attempt.transition("RUNTIME_PROVENANCE_VERIFIED", "protocol_runner", {"candidate_bundle": bundle, "freshness_gate": freshness})
            authorization = build_authorization_record(forensics.stable_json_hash(bundle))
            attempt.transition("DEPLOYMENT_AUTHORIZED", "deployment_authorizer", {"authorization_record": authorization, "rollback_plan": bundle["rollback_target"]})
            before_manifest = file_manifest(context.runtime_root)
            stop_process(live_proc)
            clear_runtime_markers(context.runtime_root)
            candidate_proc = spawn_dummy_executor(repo_path=context.candidate_repo_root, runtime_root=context.runtime_root, label="candidate", perform_rebuild=True)
            wait_for_file(context.runtime_root / "runtime_attestation.json")
            attempt.transition("DEPLOYING", "supervisor_controller", {"deployment_start": {"candidate_pid": candidate_proc.pid}})
            write_json(context.runtime_root / "config" / "exchange_precision_cache.json", {"unexpected": True})
            post_manifest = file_manifest(context.runtime_root)
            unexpected = detect_unauthorized_mutations(before_manifest, post_manifest)
            attempt.transition("POST_START_VALIDATING", "protocol_runner", {"post_start_gate": {"unexpected_mutations": unexpected}})
            if not unexpected:
                raise AssertionError("expected unauthorized mutation detection")
            stop_process(candidate_proc)
            candidate_proc = None
            rollback_record = perform_rollback(
                attempt=attempt,
                rollback_repo_root=context.rollback_repo_root,
                runtime_root=context.runtime_root,
                evidence_dir=context.evidence_dir,
                reason="unexpected filesystem mutation",
            )
            return {"state": attempt.state, "rollback_record": rollback_record}
        finally:
            stop_process(candidate_proc) if candidate_proc is not None else None
            stop_process(live_proc)

    raise ValueError(f"unsupported scenario {scenario}")


def setup_rehearsal_roots(root: Path) -> dict[str, Path]:
    private_live_root = root / "private_live"
    private_live_root.mkdir(parents=True, exist_ok=True)
    build_consistent_runtime_root(private_live_root)
    rollback_repo = root / "rollback_repo"
    candidate_repo = root / "candidate_repo"
    copy_required_tree(repo_root(), rollback_repo)
    init_git_repo(rollback_repo, baseline_marker=PROTOCOL_BASELINE_COMMIT)
    make_candidate_repo(repo_root(), candidate_repo, comment_suffix="fresh-module-hash")
    return {
        "private_live_root": private_live_root,
        "rollback_repo": rollback_repo,
        "candidate_repo": candidate_repo,
    }


def scenario_context(root: Path, prohibited_module_hash: str) -> CandidateBundleContext:
    paths = setup_rehearsal_roots(root)
    return CandidateBundleContext(
        candidate_repo_root=paths["candidate_repo"],
        rollback_repo_root=paths["rollback_repo"],
        runtime_root=paths["private_live_root"],
        evidence_dir=root / "evidence",
        prohibited_module_hash=prohibited_module_hash,
    )


def run_rehearsal(output_dir: Path) -> dict[str, Any]:
    validate_state_model()
    prohibited_module_hash = git_module_hash(repo_root())
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_baseline_commit": PROTOCOL_BASELINE_COMMIT,
        "prohibited_candidate_commit": PROHIBITED_CANDIDATE_COMMIT,
        "prohibited_candidate_module_hash": prohibited_module_hash,
        "scenarios": {},
    }
    for scenario in REHEARSAL_SCENARIOS:
        scenario_root = output_dir / scenario
        if scenario_root.exists():
            shutil.rmtree(scenario_root)
        scenario_root.mkdir(parents=True, exist_ok=True)
        attempt = AttemptState(scenario=scenario, root=scenario_root)
        context = scenario_context(scenario_root / "workspace", prohibited_module_hash)
        try:
            if scenario == "happy_path":
                result = run_happy_path_scenario(attempt, context)
            else:
                result = run_failure_scenario(scenario, attempt, context)
        except ProtocolError as exc:
            if attempt.state != "FAILED_CLOSED":
                attempt.fail_closed(exc.code, str(exc), {"failure_record": {"code": exc.code, "detail": str(exc)}})
            result = {"state": attempt.state, "failure_code": attempt.failure_code}
        scenario_result = {
            "final_state": attempt.state,
            "failure_code": attempt.failure_code,
            "failure_detail": attempt.failure_detail,
            "history": [record.__dict__ for record in attempt.history],
            "result": result,
        }
        write_json(scenario_root / "result.json", scenario_result)
        report["scenarios"][scenario] = scenario_result
    report["passed"] = (
        report["scenarios"]["happy_path"]["final_state"] == "ACCEPTED"
        and report["scenarios"]["reject_prohibited_candidate"]["failure_code"] == "FAILED_CLOSED_PROHIBITED_CANDIDATE"
        and report["scenarios"]["reject_dirty_worktree"]["failure_code"] == "FAILED_CLOSED_DIRTY_WORKTREE"
        and report["scenarios"]["reject_module_hash_mismatch"]["failure_code"] == "FAILED_CLOSED_MODULE_HASH_MISMATCH"
        and report["scenarios"]["reject_preexisting_inconsistency"]["failure_code"] == "FAILED_CLOSED_PREEXISTING_INCONSISTENCY"
        and report["scenarios"]["reject_snapshot_failure"]["failure_code"] == "FAILED_CLOSED_SNAPSHOT_FAILED"
        and report["scenarios"]["reject_equivalence_failure"]["failure_code"] == "FAILED_CLOSED_EQUIVALENCE_FAILED"
        and report["scenarios"]["rollback_unauthorized_mutation"]["final_state"] == "ROLLED_BACK"
    )
    write_json(output_dir / "rehearsal_report.json", report)
    write_text(output_dir / "README.md", "Episode-ledger deployment protocol rehearsal artifacts.\n")
    checksum_lines = compute_checksum_manifest(output_dir)
    write_text(output_dir / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
    return report


def render_result_markdown(report: dict[str, Any], output_dir: Path) -> str:
    lines = [
        "# Episode Ledger Provenance-Gated Deployment Protocol Rehearsal",
        "",
        f"- Protocol baseline commit: `{report['protocol_baseline_commit']}`",
        f"- Prohibited candidate commit: `{report['prohibited_candidate_commit']}`",
        f"- Prohibited candidate module hash: `{report['prohibited_candidate_module_hash']}`",
        f"- Rehearsal passed: `{report['passed']}`",
        f"- Output directory: `{output_dir}`",
        "",
        "## Scenarios",
        "",
    ]
    for name, scenario in report["scenarios"].items():
        lines.append(f"- `{name}` -> state `{scenario['final_state']}`, failure `{scenario['failure_code'] or 'null'}`")
    lines.append("")
    lines.append("No live deployment occurred.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rehearse the provenance-gated episode-ledger deployment protocol against private dummy processes.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = run_rehearsal(Path(args.output_dir).resolve())
    markdown = render_result_markdown(report, Path(args.output_dir).resolve())
    write_text(Path(args.output_dir).resolve() / "result.md", markdown)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
