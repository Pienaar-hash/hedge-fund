from __future__ import annotations

import argparse
import builtins
import contextlib
import copy
import hashlib
import inspect
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from execution import episode_ledger as ledger_module
from execution import exit_reason_normalizer as exit_reason_normalizer_module


PREFIX_BYTES = 96
SAFE_ENV_KEYS = {
    "ENV",
    "PYTHONPATH",
    "PWD",
    "SUPERVISOR_GROUP_NAME",
    "SUPERVISOR_PROCESS_NAME",
    "NAV_WRITER_INTERVAL_SEC",
    "LOOP_SLEEP",
    "DRY_RUN",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        "symlink": path.is_symlink(),
        "resolved_target": str(path.resolve()),
    }


def detect_repo_head(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        return subprocess.check_output(list(args), cwd=str(repo_root), text=True).strip()

    return {
        "repo_root": str(repo_root.resolve()),
        "head": _run("git", "rev-parse", "HEAD"),
        "status": _run("git", "status", "--short"),
    }


def safe_environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    values: dict[str, str] = {}
    for item in raw:
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        name = key.decode("utf-8", errors="replace")
        if name in SAFE_ENV_KEYS:
            values[name] = value.decode("utf-8", errors="replace")
    return values


def proc_state(pid: int) -> str:
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("State:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def wait_for_state(pid: int, *, expect_stopped: bool, timeout_s: float = 5.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = proc_state(pid)
        if expect_stopped and state.startswith("T"):
            return state
        if not expect_stopped and not state.startswith("T"):
            return state
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for pid {pid} expect_stopped={expect_stopped}")


@contextlib.contextmanager
def stopped_process(pid: int) -> Iterator[dict[str, Any]]:
    before = {"ts": utc_now(), "state": proc_state(pid)}
    os.kill(pid, signal.SIGSTOP)
    stopped = {"ts": utc_now(), "state": wait_for_state(pid, expect_stopped=True)}
    try:
        yield {"before": before, "stopped": stopped}
    finally:
        os.kill(pid, signal.SIGCONT)
        resumed = {"ts": utc_now(), "state": wait_for_state(pid, expect_stopped=False)}
        before["resumed"] = resumed


def attest_module(repo_root: Path) -> dict[str, Any]:
    module_path = Path(inspect.getsourcefile(ledger_module) or ledger_module.__file__).resolve()
    raw = module_path.read_bytes()
    stat = module_path.stat()
    callables: dict[str, Any] = {}
    for name in ("build_episode_ledger", "rebuild_and_save", "_load_execution_log", "_read_new_execution_events", "_canonical_ledger_hash"):
        if not hasattr(ledger_module, name):
            continue
        obj = getattr(ledger_module, name)
        try:
            payload = inspect.getsource(obj).encode("utf-8")
            kind = "source"
        except OSError:
            payload = obj.__code__.co_code
            kind = "bytecode"
        callables[name] = {"kind": kind, "sha256": sha256_bytes(payload)}
    return {
        "repository_provenance": detect_repo_head(repo_root),
        "loaded_module_provenance": {
            "module_name": ledger_module.__name__,
            "module_file": str(module_path),
            "module_resolved_path": str(module_path.resolve()),
            "sha256": sha256_bytes(raw),
            "size": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "callables": callables,
        },
        "runtime_process_provenance": {
            "pid": os.getpid(),
            "interpreter": sys.executable,
            "python_version": sys.version,
            "cwd": str(Path.cwd().resolve()),
        },
    }


def live_process_provenance(pid: int, live_root: Path) -> dict[str, Any]:
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    return {
        "pid": pid,
        "cwd": str(Path(f"/proc/{pid}/cwd").resolve()),
        "exe": str(Path(f"/proc/{pid}/exe").resolve()),
        "cmdline": cmdline,
        "state": proc_state(pid),
        "safe_environ": safe_environ(pid),
        "repository_provenance": detect_repo_head(live_root),
    }


def resolve_live_paths(live_root: Path, expected_root: Path) -> dict[str, Any]:
    execution_dir = live_root / "logs" / "execution"
    state_dir = live_root / "logs" / "state"
    selected_logs = sorted(execution_dir.glob("orders_executed.*.jsonl"), key=lambda path: path.name, reverse=True)
    active = execution_dir / "orders_executed.jsonl"
    if active.exists():
        selected_logs.append(active)
    paths = {
        "execution_logs": selected_logs,
        "active_execution_log": active,
        "dle_authority_log": execution_dir / "dle_shadow_events.jsonl",
        "nav_input": state_dir / "nav_state.json",
        "preexisting_ledger": state_dir / "episode_ledger.json",
        "preexisting_checkpoint": state_dir / "episode_ledger_checkpoint.json",
    }
    records: dict[str, Any] = {
        "process_cwd": str(live_root.resolve()),
        "repository_root": str(live_root.resolve()),
        "expected_root": str(expected_root.resolve()),
    }
    for key, value in paths.items():
        if isinstance(value, list):
            items = []
            for path in value:
                items.append({
                    **file_identity(path),
                    "outside_expected_root": expected_root.resolve() not in path.resolve().parents and path.resolve() != expected_root.resolve(),
                })
            records[key] = items
        else:
            records[key] = {
                **file_identity(value),
                "outside_expected_root": expected_root.resolve() not in value.resolve().parents and value.resolve() != expected_root.resolve(),
            }
    return records


def snapshot_source_specs(resolved_paths: dict[str, Any]) -> list[tuple[str, Path, str]]:
    specs: list[tuple[str, Path, str]] = []
    for index, entry in enumerate(resolved_paths["execution_logs"], 1):
        specs.append((f"execution_log_{index}", Path(entry["resolved_target"]), f"sources/execution/{Path(entry['resolved_target']).name}"))
    for role_key, relative in (
        ("dle_authority_log", "sources/dle/dle_shadow_events.jsonl"),
        ("nav_input", "sources/state/nav_state.json"),
        ("preexisting_ledger", "sources/state/episode_ledger.json"),
        ("preexisting_checkpoint", "sources/state/episode_ledger_checkpoint.json"),
    ):
        specs.append((role_key, Path(resolved_paths[role_key]["resolved_target"]), relative))
    return specs


def copy_snapshot_sources(evidence_dir: Path, specs: Iterable[tuple[str, Path, str]]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for role, source, relative in specs:
        destination = evidence_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest.append({"role": role, "source": str(source), "snapshot": str(destination), "identity": file_identity(destination)})
    return manifest


def ensure_safe_output_root(output_root: Path, forbidden_roots: Iterable[Path]) -> Path:
    resolved = output_root.resolve()
    for forbidden in forbidden_roots:
        root = forbidden.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"unsafe output root {resolved} overlaps forbidden root {root}")
    return resolved


@dataclass
class ReadRecord:
    logical_role: str
    resolved_path: str
    mode: str
    sequence: int
    open_ts: str
    close_ts: Optional[str]
    device: int
    inode: int
    size_at_open: int
    size_at_close: Optional[int]
    start_offset: int
    end_offset: int


class RecordedFile:
    def __init__(self, handle: Any, path: Path, role: str, mode: str, recorder: "OpenRecorder") -> None:
        self._handle = handle
        self._path = path
        self._mode = mode
        self._recorder = recorder
        stat = path.stat()
        self._record = ReadRecord(
            logical_role=role,
            resolved_path=str(path),
            mode=mode,
            sequence=recorder.next_sequence(),
            open_ts=utc_now(),
            close_ts=None,
            device=stat.st_dev,
            inode=stat.st_ino,
            size_at_open=stat.st_size,
            size_at_close=None,
            start_offset=0,
            end_offset=0,
        )
        self._seek_before_read: Optional[int] = None

    def __enter__(self) -> "RecordedFile":
        self._handle.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
        return self._handle.__exit__(exc_type, exc, tb)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        value = self.readline()
        if value == "" or value == b"":
            raise StopIteration
        return value

    def read(self, *args: Any, **kwargs: Any) -> Any:
        value = self._handle.read(*args, **kwargs)
        self._record.end_offset = self.tell()
        return value

    def readline(self, *args: Any, **kwargs: Any) -> Any:
        value = self._handle.readline(*args, **kwargs)
        self._record.end_offset = self.tell()
        return value

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        result = self._handle.seek(offset, whence)
        if self._record.end_offset == 0:
            self._record.start_offset = self.tell()
        self._seek_before_read = self.tell()
        return result

    def tell(self) -> int:
        return int(self._handle.tell())

    def close(self) -> None:
        if self._record.close_ts is not None:
            return
        try:
            self._record.end_offset = self.tell()
        except Exception:
            pass
        self._handle.close()
        self._record.close_ts = utc_now()
        try:
            stat = self._path.stat()
            self._record.size_at_close = stat.st_size
        except FileNotFoundError:
            self._record.size_at_close = None
        self._recorder.records.append(asdict(copy.deepcopy(self._record)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


class OpenRecorder:
    def __init__(self, role_by_path: dict[str, str]) -> None:
        self._role_by_path = role_by_path
        self._counter = 0
        self.records: list[dict[str, Any]] = []
        self._orig_open = builtins.open
        self._orig_path_open = Path.open
        self._patched_path_open: Optional[Callable[..., Any]] = None

    def next_sequence(self) -> int:
        self._counter += 1
        return self._counter

    def _wrap(self, file: Any, mode: str, opener: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        handle = opener(file, mode, *args, **kwargs)
        if "r" not in mode or any(flag in mode for flag in ("w", "a", "+")):
            return handle
        try:
            resolved = Path(os.fspath(file)).resolve()
        except TypeError:
            return handle
        role = self._role_by_path.get(str(resolved))
        if role is None:
            return handle
        return RecordedFile(handle, resolved, role, mode, self)

    def open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        return self._wrap(file, mode, self._orig_open, *args, **kwargs)

    def path_open(self, path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        return self._wrap(path, mode, lambda target, m, *a, **k: self._orig_path_open(target, mode=m, *a, **k), *args, **kwargs)

    @contextlib.contextmanager
    def install(self) -> Iterator["OpenRecorder"]:
        builtins.open = self.open  # type: ignore[assignment]
        def _patched(path_obj: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            return self.path_open(path_obj, mode, *args, **kwargs)

        self._patched_path_open = _patched
        Path.open = _patched  # type: ignore[assignment]
        try:
            yield self
        finally:
            builtins.open = self._orig_open  # type: ignore[assignment]
            Path.open = self._orig_path_open  # type: ignore[assignment]


def _event_identity(event: dict[str, Any]) -> str:
    return json.dumps(
        [
            event.get("symbol", ""),
            event.get("positionSide", ""),
            event.get("side", ""),
            str(event.get("ts_fill_first", "")),
            str(event.get("orderId", "")),
        ],
        separators=(",", ":"),
    )


def consumed_bytes(path: Path, start: int, end: int) -> bytes:
    if end <= start:
        return b""
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(end - start)


def parser_summary(role: str, payload: bytes, seen: set[str]) -> dict[str, Any]:
    summary = {
        "parser_record_count": 0,
        "accepted_event_count": 0,
        "rejected_malformed_count": 0,
        "duplicate_count": 0,
        "partial_final_line": bool(payload) and not payload.endswith(b"\n"),
    }
    if not payload:
        return summary
    if role.startswith("execution_log_"):
        for raw in payload.splitlines(keepends=True):
            if not raw.strip():
                continue
            if not raw.endswith(b"\n"):
                summary["rejected_malformed_count"] += 1
                continue
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                summary["rejected_malformed_count"] += 1
                continue
            summary["parser_record_count"] += 1
            if event.get("event_type") != "order_fill":
                continue
            if float(event.get("executedQty", 0) or 0) <= 0:
                continue
            identity = _event_identity(event)
            if identity in seen:
                summary["duplicate_count"] += 1
                continue
            seen.add(identity)
            summary["accepted_event_count"] += 1
        return summary
    if role == "dle_authority_log":
        for raw in payload.splitlines(keepends=True):
            if not raw.strip():
                continue
            if not raw.endswith(b"\n"):
                summary["rejected_malformed_count"] += 1
                continue
            try:
                json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                summary["rejected_malformed_count"] += 1
                continue
            summary["parser_record_count"] += 1
            summary["accepted_event_count"] += 1
        return summary
    try:
        json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        summary["rejected_malformed_count"] = 1
        return summary
    summary["parser_record_count"] = 1
    summary["accepted_event_count"] = 1
    return summary


def checkpoint_expectations(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    if not checkpoint_path.exists():
        return {}
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for entry in payload.get("sources", []):
        result[entry["path"]] = entry
    return result


def enrich_read_records(records: list[dict[str, Any]], runtime_root: Path, initial_checkpoint: Optional[Path]) -> list[dict[str, Any]]:
    seen_execution: set[str] = set()
    expectations = checkpoint_expectations(initial_checkpoint) if initial_checkpoint else {}
    enriched: list[dict[str, Any]] = []
    for record in records:
        path = Path(record["resolved_path"])
        payload = consumed_bytes(path, int(record["start_offset"]), int(record["end_offset"]))
        prefix = payload[:PREFIX_BYTES].hex()
        suffix = payload[-PREFIX_BYTES:].hex() if payload else ""
        summary = parser_summary(record["logical_role"], payload, seen_execution)
        item = dict(record)
        item.update(
            {
                "bytes_consumed": len(payload),
                "sha256": sha256_bytes(payload),
                "prefix_anchor_hex": prefix,
                "suffix_anchor_hex": suffix,
                "within_runtime_root": runtime_root.resolve() == path.resolve() or runtime_root.resolve() in path.resolve().parents,
                "replacement_observed": record["size_at_close"] is None,
                "truncation_observed": record["size_at_close"] is not None and record["size_at_close"] < record["size_at_open"],
                "growth_observed": record["size_at_close"] is not None and record["size_at_close"] > record["size_at_open"],
                "rotation_observed": False,
                **summary,
            }
        )
        expectation = expectations.get(path.name)
        if expectation:
            item["checkpoint_expected"] = {
                "identity": expectation.get("identity"),
                "offset": expectation.get("offset"),
                "content_anchor": expectation.get("content_anchor"),
                "identity_match": expectation.get("identity") == f"{record['device']}:{record['inode']}",
                "offset_match": int(expectation.get("offset", 0)) == int(record["start_offset"]),
                "anchor_match": expectation.get("content_anchor") == sha256_bytes(consumed_bytes(path, 0, int(record["start_offset"]))[:0]) if False else None,
            }
        enriched.append(item)
    return enriched


@contextlib.contextmanager
def patched_runtime_paths(runtime_root: Path) -> Iterator[None]:
    originals = {
        "EXECUTION_LOG_DIR": ledger_module.EXECUTION_LOG_DIR,
        "EXECUTION_LOG_PATH": ledger_module.EXECUTION_LOG_PATH,
        "DLE_SHADOW_LOG_PATH": ledger_module.DLE_SHADOW_LOG_PATH,
        "NAV_STATE_PATH": ledger_module.NAV_STATE_PATH,
        "EPISODE_LEDGER_PATH": ledger_module.EPISODE_LEDGER_PATH,
        "EPISODE_LEDGER_CHECKPOINT_PATH": ledger_module.EPISODE_LEDGER_CHECKPOINT_PATH,
        "EPISODE_LEDGER_REBUILD_LOG_PATH": ledger_module.EPISODE_LEDGER_REBUILD_LOG_PATH,
        "_git_commit": ledger_module._git_commit,
        "exit_reason_normalizer_level": exit_reason_normalizer_module.LOG.level,
    }
    try:
        ledger_module.EXECUTION_LOG_DIR = runtime_root / "logs" / "execution"
        ledger_module.EXECUTION_LOG_PATH = runtime_root / "logs" / "execution" / "orders_executed.jsonl"
        ledger_module.DLE_SHADOW_LOG_PATH = runtime_root / "logs" / "execution" / "dle_shadow_events.jsonl"
        ledger_module.NAV_STATE_PATH = runtime_root / "logs" / "state" / "nav_state.json"
        ledger_module.EPISODE_LEDGER_PATH = runtime_root / "logs" / "state" / "episode_ledger.json"
        ledger_module.EPISODE_LEDGER_CHECKPOINT_PATH = runtime_root / "logs" / "state" / "episode_ledger_checkpoint.json"
        ledger_module.EPISODE_LEDGER_REBUILD_LOG_PATH = runtime_root / "logs" / "execution" / "episode_ledger_rebuild.jsonl"
        ledger_module._git_commit = lambda: "forensic-run"
        exit_reason_normalizer_module.LOG.setLevel(logging.ERROR)
        yield
    finally:
        exit_reason_normalizer_module.LOG.setLevel(originals["exit_reason_normalizer_level"])
        for name, value in originals.items():
            if name == "exit_reason_normalizer_level":
                continue
            setattr(ledger_module, name, value)


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def stage_runtime_root(snapshot_manifest: list[dict[str, Any]], runtime_root: Path, *, include_resume_state: bool) -> dict[str, Path]:
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for item in snapshot_manifest:
        role = item["role"]
        source = Path(item["snapshot"])
        if role in {"preexisting_ledger", "preexisting_checkpoint"} and not include_resume_state:
            continue
        if role.startswith("execution_log_") or role == "dle_authority_log":
            relative = Path("logs/execution") / source.name
        else:
            relative = Path("logs/state") / source.name
        destination = runtime_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        mapping[role] = destination
    return mapping


def first_semantic_difference(left: Any, right: Any, prefix: str = "") -> Optional[dict[str, Any]]:
    if type(left) != type(right):
        return {"path": prefix or "$", "left": left, "right": right}
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                return {"path": f"{prefix}.{key}" if prefix else key, "left": left.get(key), "right": right.get(key)}
            diff = first_semantic_difference(left[key], right[key], f"{prefix}.{key}" if prefix else key)
            if diff is not None:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": prefix or "$", "left": len(left), "right": len(right)}
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            diff = first_semantic_difference(lhs, rhs, f"{prefix}[{index}]")
            if diff is not None:
                return diff
        return None
    if left != right:
        return {"path": prefix or "$", "left": left, "right": right}
    return None


def ordered_episode_uid_hash(ledger: ledger_module.EpisodeLedger) -> str:
    uids = []
    if ledger.episodes_v2:
        uids = [episode.episode_uid for episode in ledger.episodes_v2]
    else:
        uids = [
            ledger_module._compute_episode_uid(
                episode.symbol,
                episode.side,
                episode.entry_ts,
                episode.exit_ts,
                episode.total_qty,
                episode.avg_entry_price,
                episode.avg_exit_price,
            )
            for episode in ledger.episodes
        ]
    return stable_json_hash(uids)


def authority_payload_hash(ledger: ledger_module.EpisodeLedger) -> str:
    payload = []
    for episode in ledger.episodes_v2:
        payload.append(
            {
                "episode_uid": episode.episode_uid,
                "entry": episode.authority_entry.to_dict(),
                "exit": episode.authority_exit.to_dict(),
                "flags": episode.authority_flags.to_dict(),
            }
        )
    return stable_json_hash(payload)


def ledger_semantic_payload(ledger: ledger_module.EpisodeLedger) -> dict[str, Any]:
    payload = copy.deepcopy(ledger.to_dict())
    payload.pop("last_rebuild_ts", None)
    return payload


def run_summary(
    name: str,
    ledger: ledger_module.EpisodeLedger,
    runtime_root: Path,
    module_attestation: dict[str, Any],
    input_manifest: dict[str, Any],
    read_records: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint_path = runtime_root / "logs" / "state" / "episode_ledger_checkpoint.json"
    summary = {
        "name": name,
        "module_content_hash": module_attestation["loaded_module_provenance"]["sha256"],
        "input_manifest_hash": stable_json_hash(input_manifest),
        "canonical_ledger_hash": ledger_module._canonical_ledger_hash(ledger),
        "episode_count": len(ledger.episodes),
        "ordered_episode_uid_hash": ordered_episode_uid_hash(ledger),
        "authority_payload_hash": authority_payload_hash(ledger),
        "aggregate_stats_hash": stable_json_hash(ledger.stats),
        "reconciliation_hash": stable_json_hash(ledger.stats.get("reconciliation", {})),
        "checkpoint_hash": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        "read_fingerprint_manifest_hash": stable_json_hash(read_records),
        "semantic_payload": ledger_semantic_payload(ledger),
    }
    return summary


def execute_run(
    name: str,
    runtime_root: Path,
    read_roles: dict[str, str],
    module_attestation: dict[str, Any],
    input_manifest: dict[str, Any],
    builder: Callable[[], ledger_module.EpisodeLedger],
) -> dict[str, Any]:
    recorder = OpenRecorder(read_roles)
    initial_checkpoint = runtime_root / "logs" / "state" / "episode_ledger_checkpoint.json"
    initial_checkpoint_path = initial_checkpoint if initial_checkpoint.exists() else None
    with patched_runtime_paths(runtime_root), pushd(runtime_root), recorder.install():
        ledger = builder()
    enriched = enrich_read_records(recorder.records, runtime_root, initial_checkpoint_path)
    write_json(runtime_root.parent / "read_fingerprints" / f"{name}.json", enriched)
    summary = run_summary(name, ledger, runtime_root, module_attestation, input_manifest, enriched)
    write_json(runtime_root.parent / f"{name}.json", summary)
    return {"summary": summary, "read_records": enriched}


def read_roles_for_runtime(runtime_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    execution_dir = runtime_root / "logs" / "execution"
    for index, path in enumerate(sorted(execution_dir.glob("orders_executed.*.jsonl"), key=lambda item: item.name, reverse=True), 1):
        result[str(path.resolve())] = f"execution_log_{index}"
    active = execution_dir / "orders_executed.jsonl"
    if active.exists():
        result[str(active.resolve())] = f"execution_log_{len([key for key in result if 'orders_executed' in key]) + 1}"
    for role, path in (
        ("dle_authority_log", execution_dir / "dle_shadow_events.jsonl"),
        ("nav_input", runtime_root / "logs" / "state" / "nav_state.json"),
        ("starting_ledger", runtime_root / "logs" / "state" / "episode_ledger.json"),
        ("starting_checkpoint", runtime_root / "logs" / "state" / "episode_ledger_checkpoint.json"),
    ):
        if path.exists():
            result[str(path.resolve())] = role
    return result


def append_controlled_batch(active_log: Path) -> list[dict[str, Any]]:
    base_ts = "2026-07-12T16:20:00+00:00"
    events = [
        {
            "event_type": "order_fill",
            "ts": base_ts,
            "ts_fill_first": base_ts,
            "symbol": "ZZTESTUSDT",
            "positionSide": "LONG",
            "side": "BUY",
            "reduceOnly": False,
            "executedQty": 1.0,
            "avgPrice": 100.0,
            "fee_total": 0.1,
            "orderId": "controlled-entry",
            "metadata": {"strategy": "forensics", "exit": {"reason": "tp"}},
        },
        {
            "event_type": "order_fill",
            "ts": "2026-07-12T16:21:00+00:00",
            "ts_fill_first": "2026-07-12T16:21:00+00:00",
            "symbol": "ZZTESTUSDT",
            "positionSide": "LONG",
            "side": "SELL",
            "reduceOnly": True,
            "executedQty": 1.0,
            "avgPrice": 101.0,
            "fee_total": 0.1,
            "orderId": "controlled-exit",
            "metadata": {"strategy": "forensics", "exit": {"reason": "tp"}, "entry_price": 100.0},
        },
    ]
    with active_log.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    return events


def compute_sha256sums(root: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    return lines


def run_card(
    live_root: Path,
    live_pid: int,
    evidence_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    ensure_safe_output_root(evidence_dir, [live_root, repo_root])
    evidence_dir.mkdir(parents=True, exist_ok=True)
    module_attestation = attest_module(repo_root)
    live_process = live_process_provenance(live_pid, live_root)
    resolved_paths = resolve_live_paths(live_root, live_root)
    write_json(evidence_dir / "module_attestation.json", module_attestation)
    write_json(evidence_dir / "process_provenance.json", live_process)
    write_json(evidence_dir / "resolved_paths.json", resolved_paths)
    boundary: dict[str, Any] = {"method": "process_sigstop_copy_resume", "ts_started": utc_now()}
    specs = snapshot_source_specs(resolved_paths)
    with stopped_process(live_pid) as stop_info:
        boundary["stop_resume"] = stop_info
        snapshot_manifest = copy_snapshot_sources(evidence_dir, specs)
    boundary["ts_completed"] = utc_now()
    write_json(evidence_dir / "snapshot_boundary.json", boundary)
    write_json(evidence_dir / "manifest.json", {"snapshot_manifest": snapshot_manifest})

    metadata_lines = ["role\tdevice\tinode\tsize\tsha256\tsnapshot"]
    line_count_lines = ["role\tlines\tsnapshot"]
    for item in snapshot_manifest:
        identity = item["identity"]
        metadata_lines.append(
            f"{item['role']}\t{identity['device']}\t{identity['inode']}\t{identity['size']}\t{identity['sha256']}\t{Path(item['snapshot']).relative_to(evidence_dir)}"
        )
        raw = Path(item["snapshot"]).read_bytes()
        line_count = raw.count(b"\n")
        line_count_lines.append(f"{item['role']}\t{line_count}\t{Path(item['snapshot']).relative_to(evidence_dir)}")
    write_text(evidence_dir / "source_metadata.tsv", "\n".join(metadata_lines) + "\n")
    write_text(evidence_dir / "source_line_counts.tsv", "\n".join(line_count_lines) + "\n")
    write_text(evidence_dir / "README.md", "Episode-ledger runtime provenance evidence.\n")
    (evidence_dir / "read_fingerprints").mkdir(exist_ok=True)

    input_manifest = {
        "snapshot_manifest": snapshot_manifest,
        "module_content_hash": module_attestation["loaded_module_provenance"]["sha256"],
        "resolved_paths_hash": stable_json_hash(resolved_paths),
        "process_provenance_hash": stable_json_hash(live_process),
    }

    runs: dict[str, Any] = {}
    matrix_dir = evidence_dir

    run_full_root = matrix_dir / "run_full" / "runtime"
    stage_runtime_root(snapshot_manifest, run_full_root, include_resume_state=False)
    runs["A_full"] = execute_run(
        "A_full",
        run_full_root,
        read_roles_for_runtime(run_full_root),
        module_attestation,
        input_manifest,
        ledger_module.build_episode_ledger,
    )

    run_initial_root = matrix_dir / "run_candidate_initial" / "runtime"
    stage_runtime_root(snapshot_manifest, run_initial_root, include_resume_state=False)
    runs["B_initial"] = execute_run(
        "B_initial",
        run_initial_root,
        read_roles_for_runtime(run_initial_root),
        module_attestation,
        input_manifest,
        ledger_module.rebuild_and_save,
    )

    run_noop_root = matrix_dir / "run_checkpoint_noop" / "runtime"
    stage_runtime_root(snapshot_manifest, run_noop_root, include_resume_state=True)
    runs["C_noop"] = execute_run(
        "C_noop",
        run_noop_root,
        read_roles_for_runtime(run_noop_root),
        module_attestation,
        input_manifest,
        ledger_module.rebuild_and_save,
    )

    run_noop_rerun_root = matrix_dir / "run_checkpoint_noop_rerun" / "runtime"
    shutil.copytree(run_noop_root, run_noop_rerun_root)
    runs["D_noop_rerun"] = execute_run(
        "D_noop_rerun",
        run_noop_rerun_root,
        read_roles_for_runtime(run_noop_rerun_root),
        module_attestation,
        input_manifest,
        ledger_module.rebuild_and_save,
    )

    run_append_root = matrix_dir / "run_controlled_append" / "runtime_candidate"
    stage_runtime_root(snapshot_manifest, run_append_root, include_resume_state=False)
    with patched_runtime_paths(run_append_root), pushd(run_append_root):
        ledger_module.rebuild_and_save()
    append_controlled_batch(run_append_root / "logs" / "execution" / "orders_executed.jsonl")
    runs["E_append_candidate"] = execute_run(
        "E_append_candidate",
        run_append_root,
        read_roles_for_runtime(run_append_root),
        module_attestation,
        input_manifest,
        ledger_module.rebuild_and_save,
    )
    run_append_full_root = matrix_dir / "run_controlled_append" / "runtime_full"
    stage_runtime_root(snapshot_manifest, run_append_full_root, include_resume_state=False)
    append_controlled_batch(run_append_full_root / "logs" / "execution" / "orders_executed.jsonl")
    runs["E_append_full"] = execute_run(
        "E_append_full",
        run_append_full_root,
        read_roles_for_runtime(run_append_full_root),
        module_attestation,
        input_manifest,
        ledger_module.build_episode_ledger,
    )

    comparisons = {
        "A_vs_B": first_semantic_difference(runs["A_full"]["summary"]["semantic_payload"], runs["B_initial"]["summary"]["semantic_payload"]),
        "A_vs_C": first_semantic_difference(runs["A_full"]["summary"]["semantic_payload"], runs["C_noop"]["summary"]["semantic_payload"]),
        "A_vs_D": first_semantic_difference(runs["A_full"]["summary"]["semantic_payload"], runs["D_noop_rerun"]["summary"]["semantic_payload"]),
        "E_full_vs_candidate": first_semantic_difference(runs["E_append_full"]["summary"]["semantic_payload"], runs["E_append_candidate"]["summary"]["semantic_payload"]),
    }
    divergence = next(({"comparison": key, **value} for key, value in comparisons.items() if value is not None), None)
    if divergence is not None:
        disposition = "runtime_divergence_reproduced"
    elif boundary.get("stop_resume"):
        disposition = "runtime_divergence_eliminated_under_atomic_provenance"
    else:
        disposition = "inconclusive_runtime_boundary_not_proven"
    comparison_report = {
        "module_attestation_hash": stable_json_hash(module_attestation),
        "input_manifest_hash": stable_json_hash(input_manifest),
        "runs": {name: value["summary"] for name, value in runs.items()},
        "comparisons": comparisons,
        "disposition": disposition,
        "first_semantic_difference": divergence,
    }
    write_json(evidence_dir / "comparison_report.json", comparison_report)
    result_md = [
        "# Episode Ledger Runtime Provenance Result",
        "",
        f"- Disposition: `{disposition}`",
        f"- First semantic difference: `{json.dumps(divergence, sort_keys=True) if divergence else 'null'}`",
        f"- Live executor PID: `{live_pid}`",
        f"- Live executor repository HEAD: `{live_process['repository_provenance']['head']}`",
        f"- Forensic module SHA-256: `{module_attestation['loaded_module_provenance']['sha256']}`",
        f"- Redeployment remains prohibited: `dec9acc7`",
    ]
    write_text(evidence_dir / "result.md", "\n".join(result_md) + "\n")
    sha_lines = compute_sha256sums(evidence_dir)
    write_text(evidence_dir / "SHA256SUMS", "\n".join(sha_lines) + "\n")
    return {
        "evidence_dir": str(evidence_dir),
        "manifest_hash": sha256_file(evidence_dir / "SHA256SUMS"),
        "comparison_report": comparison_report,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture an atomic runtime provenance set and replay episode-ledger builders offline.")
    parser.add_argument("--live-root", required=True)
    parser.add_argument("--live-pid", required=True, type=int)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    result = run_card(
        live_root=Path(args.live_root).resolve(),
        live_pid=args.live_pid,
        evidence_dir=Path(args.evidence_dir).resolve(),
        repo_root=Path(args.repo_root).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
