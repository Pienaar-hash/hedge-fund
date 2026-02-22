from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from execution.binary_lab_executor import BinaryLabEventType, BinaryLabMode
from execution.binary_lab_runtime import BinaryLabRuntimeWriter, RuntimeLoopContext

pytestmark = pytest.mark.integration


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _limits_payload() -> dict:
    return {
        "_meta": {
            "sleeve_id": "binary_lab_s1",
            "time_horizon_minutes": 15,
        },
        "capital": {
            "sleeve_total_usd": 2000,
            "per_round_usd": 20,
        },
        "position_rules": {
            "max_concurrent": 3,
            "same_direction_stacking": False,
            "martingale": False,
            "size_escalation_after_wins": False,
        },
        "kill_conditions": {
            "sleeve_drawdown_usd": 300,
            "sleeve_drawdown_pct": 0.15,
            "kill_nav_usd": 1700,
        },
        "time_horizon": {"round_minutes": 15},
    }


def _admission_payload() -> dict:
    return {
        "datasets": {
            "polymarket_snapshot": {"state": "OBSERVE_ONLY"},
            "prediction_polymarket_feed": {"state": "OBSERVE_ONLY"},
        }
    }


def _run_sequence(base: Path) -> list[bytes]:
    limits_path = base / "config" / "binary_lab_limits.json"
    admission_path = base / "config" / "dataset_admission.json"
    state_path = base / "logs" / "state" / "binary_lab_state.json"
    _write_json(limits_path, _limits_payload())
    _write_json(admission_path, _admission_payload())
    expected_hash = _sha256_file(limits_path)

    writer = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=expected_hash,
    )

    snapshots: list[bytes] = []
    writer.boot("2026-02-19T00:00:00+00:00")
    snapshots.append(state_path.read_bytes())

    writer.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:00:10+00:00",
            activate=True,
            activation_gate_go=True,
            mode=BinaryLabMode.PAPER,
            prediction_phase="P1_ADVISORY",
            horizon_minutes=15,
        )
    )
    snapshots.append(state_path.read_bytes())

    writer.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:01:00+00:00",
            event_type_override=BinaryLabEventType.ROUND_CLOSED,
            trade_taken=True,
            outcome="WIN",
            conviction_band="high",
            pnl_usd=8.0,
            size_usd=20.0,
            open_positions=1,
        )
    )
    snapshots.append(state_path.read_bytes())

    writer.tick(
        RuntimeLoopContext(
            now_ts="2026-02-20T00:00:00+00:00",
            event_type_override=BinaryLabEventType.DAILY_CHECKPOINT,
            open_positions=0,
        )
    )
    snapshots.append(state_path.read_bytes())

    return snapshots


def test_replay_is_byte_identical_for_same_event_sequence(tmp_path: Path) -> None:
    run_a = _run_sequence(tmp_path / "run_a")
    run_b = _run_sequence(tmp_path / "run_b")

    assert len(run_a) == len(run_b)
    for idx, (a, b) in enumerate(zip(run_a, run_b, strict=True)):
        assert a == b, f"state mismatch at replay step {idx}"


def test_replay_hashes_match_for_same_event_sequence(tmp_path: Path) -> None:
    run_a = _run_sequence(tmp_path / "run_hash_a")
    run_b = _run_sequence(tmp_path / "run_hash_b")

    hashes_a = [hashlib.sha256(x).hexdigest() for x in run_a]
    hashes_b = [hashlib.sha256(x).hexdigest() for x in run_b]
    assert hashes_a == hashes_b
