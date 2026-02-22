from __future__ import annotations

from execution.binary_lab_executor import (
    BinaryLabEvent,
    BinaryLabEventType,
    BinaryLabMode,
    BinaryLabStatus,
    initialize_state,
    reduce_event,
    state_from_payload,
)


def _limits() -> dict:
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
        "time_horizon": {
            "round_minutes": 15,
        },
    }


def test_activate_paper_mode_allowed_with_observe_only_datasets() -> None:
    state = initialize_state(_limits())
    event = BinaryLabEvent(
        event_type=BinaryLabEventType.ACTIVATE,
        ts="2026-02-19T00:00:00+00:00",
        activation_gate_go=True,
        mode=BinaryLabMode.PAPER,
        horizon_minutes=15,
        config_hash="sha256:test",
        prediction_phase="P1_ADVISORY",
        dataset_states={
            "polymarket_snapshot": "OBSERVE_ONLY",
            "prediction_polymarket_feed": "OBSERVE_ONLY",
        },
    )
    result = reduce_event(state, event, _limits())
    assert result.accepted is True
    assert result.state.status == BinaryLabStatus.ACTIVE
    assert result.state.mode == BinaryLabMode.PAPER
    assert result.state.config_hash == "sha256:test"


def test_activate_live_mode_blocked_without_p2_and_production_datasets() -> None:
    state = initialize_state(_limits())
    event = BinaryLabEvent(
        event_type=BinaryLabEventType.ACTIVATE,
        activation_gate_go=True,
        mode=BinaryLabMode.LIVE,
        horizon_minutes=15,
        prediction_phase="P1_ADVISORY",
        dataset_states={
            "polymarket_snapshot": "OBSERVE_ONLY",
            "prediction_polymarket_feed": "OBSERVE_ONLY",
        },
    )
    result = reduce_event(state, event, _limits())
    assert result.accepted is False
    assert result.deny_reason == "prediction_phase_not_p2"
    assert result.state.status == BinaryLabStatus.NOT_DEPLOYED


def test_horizon_mismatch_blocks_activation() -> None:
    state = initialize_state(_limits())
    event = BinaryLabEvent(
        event_type=BinaryLabEventType.ACTIVATE,
        activation_gate_go=True,
        mode=BinaryLabMode.PAPER,
        horizon_minutes=5,
        prediction_phase="P1_ADVISORY",
        dataset_states={
            "polymarket_snapshot": "OBSERVE_ONLY",
            "prediction_polymarket_feed": "OBSERVE_ONLY",
        },
    )
    result = reduce_event(state, event, _limits())
    assert result.accepted is False
    assert result.deny_reason == "horizon_mismatch"


def test_round_close_updates_metrics_and_nav() -> None:
    state = initialize_state(_limits())
    activated = reduce_event(
        state,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ACTIVATE,
            activation_gate_go=True,
            mode=BinaryLabMode.PAPER,
            horizon_minutes=15,
            config_hash="sha256:test",
            prediction_phase="P1_ADVISORY",
            dataset_states={
                "polymarket_snapshot": "OBSERVE_ONLY",
                "prediction_polymarket_feed": "OBSERVE_ONLY",
            },
        ),
        _limits(),
    ).state

    result = reduce_event(
        activated,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ROUND_CLOSED,
            trade_taken=True,
            outcome="WIN",
            conviction_band="high",
            pnl_usd=12.5,
            size_usd=20,
            open_positions=1,
        ),
        _limits(),
    )
    assert result.accepted is True
    assert result.state.total_trades == 1
    assert result.state.wins == 1
    assert result.state.losses == 0
    assert result.state.current_nav_usd == 2012.5
    assert result.state.pnl_usd == 12.5
    assert "high" in result.state.by_conviction_band


def test_kill_line_breach_terminates_immediately() -> None:
    state = initialize_state(_limits())
    activated = reduce_event(
        state,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ACTIVATE,
            activation_gate_go=True,
            mode=BinaryLabMode.PAPER,
            horizon_minutes=15,
            config_hash="sha256:test",
            prediction_phase="P1_ADVISORY",
            dataset_states={
                "polymarket_snapshot": "OBSERVE_ONLY",
                "prediction_polymarket_feed": "OBSERVE_ONLY",
            },
        ),
        _limits(),
    ).state

    result = reduce_event(
        activated,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ROUND_CLOSED,
            trade_taken=True,
            outcome="LOSS",
            conviction_band="medium",
            pnl_usd=-310.0,
            size_usd=20,
            open_positions=1,
        ),
        _limits(),
    )
    assert result.accepted is True
    assert result.state.kill_breached is True
    assert result.state.status == BinaryLabStatus.TERMINATED
    assert result.state.termination_reason == "kill_line_breached"
    assert "TERMINATE_IMMEDIATELY" in result.actions


def test_rule_violation_max_concurrent_terminates() -> None:
    state = initialize_state(_limits())
    activated = reduce_event(
        state,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ACTIVATE,
            activation_gate_go=True,
            mode=BinaryLabMode.PAPER,
            horizon_minutes=15,
            config_hash="sha256:test",
            prediction_phase="P1_ADVISORY",
            dataset_states={
                "polymarket_snapshot": "OBSERVE_ONLY",
                "prediction_polymarket_feed": "OBSERVE_ONLY",
            },
        ),
        _limits(),
    ).state

    result = reduce_event(
        activated,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ROUND_CLOSED,
            trade_taken=True,
            outcome="WIN",
            conviction_band="very_high",
            pnl_usd=5.0,
            size_usd=20,
            open_positions=4,
        ),
        _limits(),
    )
    assert result.state.status == BinaryLabStatus.TERMINATED
    assert result.state.rule_violations >= 1
    assert result.state.termination_reason == "max_concurrent_breach"


def test_daily_checkpoint_hash_mismatch_breaks_freeze_and_terminates() -> None:
    state = initialize_state(_limits())
    activated = reduce_event(
        state,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ACTIVATE,
            activation_gate_go=True,
            mode=BinaryLabMode.PAPER,
            horizon_minutes=15,
            config_hash="sha256:locked",
            prediction_phase="P1_ADVISORY",
            dataset_states={
                "polymarket_snapshot": "OBSERVE_ONLY",
                "prediction_polymarket_feed": "OBSERVE_ONLY",
            },
        ),
        _limits(),
    ).state

    result = reduce_event(
        activated,
        BinaryLabEvent(
            event_type=BinaryLabEventType.DAILY_CHECKPOINT,
            config_hash="sha256:changed",
            open_positions=0,
        ),
        _limits(),
    )
    assert result.accepted is True
    assert result.state.freeze_intact is False
    assert result.state.status == BinaryLabStatus.TERMINATED
    assert result.state.termination_reason == "config_hash_mismatch"


def test_daily_checkpoint_day_30_marks_completed() -> None:
    state = initialize_state(_limits(), day_total=2)
    activated = reduce_event(
        state,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ACTIVATE,
            activation_gate_go=True,
            mode=BinaryLabMode.PAPER,
            horizon_minutes=15,
            config_hash="sha256:locked",
            prediction_phase="P1_ADVISORY",
            dataset_states={
                "polymarket_snapshot": "OBSERVE_ONLY",
                "prediction_polymarket_feed": "OBSERVE_ONLY",
            },
        ),
        _limits(),
    ).state

    d1 = reduce_event(
        activated,
        BinaryLabEvent(event_type=BinaryLabEventType.DAILY_CHECKPOINT, config_hash="sha256:locked", open_positions=0),
        _limits(),
    ).state
    d2 = reduce_event(
        d1,
        BinaryLabEvent(event_type=BinaryLabEventType.DAILY_CHECKPOINT, config_hash="sha256:locked", open_positions=0),
        _limits(),
    )
    assert d2.state.day == 2
    assert d2.state.status == BinaryLabStatus.COMPLETED
    assert "WINDOW_COMPLETE" in d2.actions


def test_duplicate_daily_checkpoint_same_utc_day_is_noop_for_day_counter() -> None:
    state = initialize_state(_limits(), day_total=30)
    activated = reduce_event(
        state,
        BinaryLabEvent(
            event_type=BinaryLabEventType.ACTIVATE,
            activation_gate_go=True,
            mode=BinaryLabMode.PAPER,
            horizon_minutes=15,
            config_hash="sha256:locked",
            prediction_phase="P1_ADVISORY",
            dataset_states={
                "polymarket_snapshot": "OBSERVE_ONLY",
                "prediction_polymarket_feed": "OBSERVE_ONLY",
            },
        ),
        _limits(),
    ).state

    first = reduce_event(
        activated,
        BinaryLabEvent(
            event_type=BinaryLabEventType.DAILY_CHECKPOINT,
            ts="2026-02-19T00:00:00+00:00",
            config_hash="sha256:locked",
            open_positions=0,
        ),
        _limits(),
    )
    second = reduce_event(
        first.state,
        BinaryLabEvent(
            event_type=BinaryLabEventType.DAILY_CHECKPOINT,
            ts="2026-02-19T12:00:00+00:00",
            config_hash="sha256:locked",
            open_positions=0,
        ),
        _limits(),
    )
    assert first.state.day == 1
    assert second.state.day == 1
    assert second.state.last_checkpoint_utc_date == "2026-02-19"
    assert "CHECKPOINT_DUPLICATE_NOOP" in second.actions


def test_state_payload_roundtrip_includes_checkpoint_marker() -> None:
    state = initialize_state(_limits())
    state.last_checkpoint_utc_date = "2026-02-19"
    state.config_hash = "sha256:locked"
    payload = state.to_payload()
    loaded = state_from_payload(payload)
    assert loaded.last_checkpoint_utc_date == "2026-02-19"
    assert loaded.config_hash == "sha256:locked"
