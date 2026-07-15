from __future__ import annotations

from execution.weekly_risk_gate import (
    classify_weekly_risk_mode,
    evaluate_hard_stop,
    validate_entry,
    validate_portfolio_targets,
)


def test_weekly_loss_halt_blocks_entries():
    assert classify_weekly_risk_mode(
        weekly_loss_pct=0.08,
        drawdown_pct=0.0,
        data_valid=True,
        audit_state_valid=True,
    ) == "HALTED"


def test_drawdown_halt_blocks_entries():
    assert classify_weekly_risk_mode(
        weekly_loss_pct=0.0,
        drawdown_pct=0.20,
        data_valid=True,
        audit_state_valid=True,
    ) == "HALTED"


def test_hard_stop_remains_executable_while_halted():
    assert evaluate_hard_stop(entry_price=100.0, current_price=89.99)


def test_validate_portfolio_targets_rejects_excess_risk():
    ok, reasons = validate_portfolio_targets(
        [
            {"symbol": "BTCUSDT", "target_pct": 0.4, "hard_stop_pct": 0.1},
            {"symbol": "SOLUSDT", "target_pct": 0.4, "hard_stop_pct": 0.1},
            {"symbol": "DOGEUSDT", "target_pct": 0.3, "hard_stop_pct": 0.1},
        ]
    )
    assert not ok
    assert "gross_exposure" in reasons


def test_validate_entry_respects_selection_and_group_rules():
    allowed, reasons = validate_entry(
        {"symbol": "BTCUSDT", "conviction": "HIGH", "correlation_group": "majors"},
        risk_mode="ACTIVE",
        selected_symbols={"BTCUSDT"},
        existing_groups=set(),
        unresolved_orders=False,
        audit_chain_ok=True,
    )
    assert allowed
    assert reasons == []
