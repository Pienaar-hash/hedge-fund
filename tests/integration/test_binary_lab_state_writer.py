from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from execution.binary_lab_executor import BinaryLabMode
from execution.binary_lab_runtime import BinaryLabRuntimeWriter, RuntimeLoopContext

pytestmark = pytest.mark.integration


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _hash_file(path: Path) -> str:
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


def test_boot_writes_binary_lab_state_surface(tmp_path: Path) -> None:
    limits_path = tmp_path / "config" / "binary_lab_limits.json"
    admission_path = tmp_path / "config" / "dataset_admission.json"
    state_path = tmp_path / "logs" / "state" / "binary_lab_state.json"

    _write_json(limits_path, _limits_payload())
    _write_json(admission_path, _admission_payload())
    expected_hash = _hash_file(limits_path)

    writer = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=expected_hash,
    )
    writer.boot("2026-02-19T00:00:00+00:00")
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_DEPLOYED"
    assert payload["sleeve_id"] == "binary_lab_s1"
    assert payload["config_hash"] == expected_hash


def test_tick_updates_state_deterministically(tmp_path: Path) -> None:
    limits_path = tmp_path / "config" / "binary_lab_limits.json"
    admission_path = tmp_path / "config" / "dataset_admission.json"
    state_path = tmp_path / "logs" / "state" / "binary_lab_state.json"

    _write_json(limits_path, _limits_payload())
    _write_json(admission_path, _admission_payload())
    expected_hash = _hash_file(limits_path)

    writer = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=expected_hash,
    )
    writer.boot("2026-02-19T00:00:00+00:00")

    activate = writer.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:01:00+00:00",
            activate=True,
            activation_gate_go=True,
            horizon_minutes=15,
        )
    )
    assert activate.accepted is True

    round_closed = writer.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:02:00+00:00",
            trade_taken=True,
            outcome="WIN",
            conviction_band="high",
            pnl_usd=10.0,
            size_usd=20.0,
            open_positions=1,
        )
    )
    assert round_closed.accepted is True

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ACTIVE"
    assert payload["metrics"]["total_trades"] == 1
    assert payload["metrics"]["wins"] == 1
    assert payload["capital"]["current_nav_usd"] == 2010.0


def test_atomic_write_no_partial_json_or_temp_leftover(tmp_path: Path) -> None:
    limits_path = tmp_path / "config" / "binary_lab_limits.json"
    admission_path = tmp_path / "config" / "dataset_admission.json"
    state_path = tmp_path / "logs" / "state" / "binary_lab_state.json"

    _write_json(limits_path, _limits_payload())
    _write_json(admission_path, _admission_payload())
    expected_hash = _hash_file(limits_path)

    writer = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=expected_hash,
    )
    writer.boot("2026-02-19T00:00:00+00:00")
    writer.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:01:00+00:00",
            activate=True,
            activation_gate_go=True,
            horizon_minutes=15,
        )
    )

    # File content is always valid JSON (no partial writes)
    raw = state_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)

    # Atomic writer should not leave temp artifacts behind
    tmp_files = list(state_path.parent.glob("binary_lab_state.json.tmp"))
    assert tmp_files == []


def test_fail_closed_when_expected_hash_is_missing(tmp_path: Path) -> None:
    limits_path = tmp_path / "config" / "binary_lab_limits.json"
    admission_path = tmp_path / "config" / "dataset_admission.json"
    state_path = tmp_path / "logs" / "state" / "binary_lab_state.json"

    _write_json(limits_path, _limits_payload())
    _write_json(admission_path, _admission_payload())

    writer = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=None,
    )
    state = writer.boot("2026-02-19T00:00:00+00:00")
    assert state.status.value == "DISABLED"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "DISABLED"


def test_observe_only_datasets_cannot_activate_live_mode(tmp_path: Path) -> None:
    limits_path = tmp_path / "config" / "binary_lab_limits.json"
    admission_path = tmp_path / "config" / "dataset_admission.json"
    state_path = tmp_path / "logs" / "state" / "binary_lab_state.json"

    _write_json(limits_path, _limits_payload())
    _write_json(admission_path, _admission_payload())  # OBSERVE_ONLY by design
    expected_hash = _hash_file(limits_path)

    writer = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=expected_hash,
    )
    writer.boot("2026-02-19T00:00:00+00:00")
    result = writer.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:01:00+00:00",
            activate=True,
            activation_gate_go=True,
            mode=BinaryLabMode.LIVE,
            prediction_phase="P1_ADVISORY",
            horizon_minutes=15,
        )
    )
    assert result.accepted is False
    assert result.deny_reason == "prediction_phase_not_p2"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "NOT_DEPLOYED"


def test_restart_same_day_does_not_double_increment_daily_checkpoint(tmp_path: Path) -> None:
    limits_path = tmp_path / "config" / "binary_lab_limits.json"
    admission_path = tmp_path / "config" / "dataset_admission.json"
    state_path = tmp_path / "logs" / "state" / "binary_lab_state.json"

    _write_json(limits_path, _limits_payload())
    _write_json(admission_path, _admission_payload())
    expected_hash = _hash_file(limits_path)

    writer1 = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=expected_hash,
    )
    writer1.boot("2026-02-19T00:00:00+00:00")
    writer1.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:01:00+00:00",
            activate=True,
            activation_gate_go=True,
            horizon_minutes=15,
        )
    )
    writer1.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:02:00+00:00",
            event_type_override=None,  # default DAILY_CHECKPOINT when no trade/activate
        )
    )

    payload_after_first = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload_after_first["day"] == 1
    assert payload_after_first["last_checkpoint_utc_date"] == "2026-02-19"

    # Simulate process restart: new writer instance reading persisted state
    writer2 = BinaryLabRuntimeWriter(
        limits_path=limits_path,
        dataset_admission_path=admission_path,
        state_path=state_path,
        expected_limits_hash=expected_hash,
    )
    writer2.boot("2026-02-19T00:10:00+00:00")
    writer2.tick(
        RuntimeLoopContext(
            now_ts="2026-02-19T00:11:00+00:00",
            event_type_override=None,
        )
    )

    payload_after_restart = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload_after_restart["day"] == 1
    assert payload_after_restart["last_checkpoint_utc_date"] == "2026-02-19"
