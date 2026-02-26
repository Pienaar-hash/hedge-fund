"""
Tests for ops/fund_ops_monthly.py — Fund-Ops Monthly Template Generator.

Covers:
  - All 7 section builders produce correctly labeled blocks
  - Section content contains expected fields
  - Graceful handling of missing state files
  - Full report structure
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ops.fund_ops_monthly import (
    generate_monthly_report,
    section_1_capital_state,
    section_2_trade_activity,
    section_3_regime_context,
    section_4_risk_discipline,
    section_5_binary_lab,
    section_6_structural,
    section_7_measurement_focus,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def isolate_state(tmp_path, monkeypatch):
    """Isolate all state file reads to tmp_path with realistic data."""
    import ops.fund_ops_monthly as mod

    nav_state = tmp_path / "nav_state.json"
    nav_state.write_text(json.dumps({
        "total_equity": 10000.0,
        "peak_equity": 10050.0,
        "drawdown_pct": 0.0,
    }))

    nav_log = tmp_path / "nav_log.json"
    import time
    now = time.time()
    # Create entries with a visible drawdown: peak at 10200, trough at 9700
    nav_log.write_text(json.dumps([
        {"t": now - 35 * 86400, "nav": 9500.0},
        {"t": now - 30 * 86400, "nav": 9800.0},
        {"t": now - 20 * 86400, "nav": 10200.0},
        {"t": now - 15 * 86400, "nav": 9700.0},
        {"t": now - 1 * 86400, "nav": 9950.0},
        {"t": now, "nav": 10000.0},
    ]))

    episode_ledger = tmp_path / "episode_ledger.json"
    episode_ledger.write_text(json.dumps({
        "episode_count": 50,
        "episodes": [
            {
                "entry_ts": f"2026-02-{i:02d}T10:00:00+00:00",
                "exit_ts": f"2026-02-{i:02d}T12:00:00+00:00",
                "conviction_band": "medium" if i % 3 == 0 else "low",
                "hybrid_score": 0.45 + (i % 10) * 0.01,
            }
            for i in range(1, 51)
        ],
        "stats": {
            "total_net_pnl": -120.50,
            "winners": 12,
            "losers": 38,
            "win_rate": 24.0,
        },
    }))

    risk_snapshot = tmp_path / "risk_snapshot.json"
    risk_snapshot.write_text(json.dumps({
        "portfolio_dd_pct": 0.0012,
        "circuit_breaker": {"active": False},
    }))

    sentinel = tmp_path / "sentinel_x.json"
    sentinel.write_text(json.dumps({
        "primary_regime": "TREND_UP",
        "secondary_regime": "MEAN_REVERT",
        "smoothed_probs": {
            "TREND_UP": 0.65,
            "TREND_DOWN": 0.05,
            "MEAN_REVERT": 0.20,
            "BREAKOUT": 0.05,
            "CHOPPY": 0.04,
            "CRISIS": 0.01,
        },
        "history_meta": {
            "consecutive_count": 120,
            "last_n_labels": ["TREND_UP"] * 10,
        },
    }))

    binary_lab = tmp_path / "binary_lab_state.json"
    binary_lab.write_text(json.dumps({
        "status": "ACTIVE",
        "mode": "SHADOW",
        "day": 5,
        "day_total": 30,
        "capital": {"current_nav_usd": 2000.0, "pnl_usd": 15.0},
        "metrics": {"total_trades": 10, "wins": 6, "losses": 4, "win_rate": 60.0},
        "kill_line": {"breached": False, "distance_usd": 285.0},
        "freeze_intact": True,
        "config_hash": "abc123",
        "rule_violations": 0,
        "termination_reason": None,
    }))

    aw_state = tmp_path / "aw_state.json"
    aw_state.write_text(json.dumps({
        "active": True,
        "manifest_intact": True,
        "config_intact": True,
        "boot_manifest_hash": "aabbccdd",
        "boot_config_hash": "11223344",
        "remaining_days": 10,
    }))

    vetoes = tmp_path / "risk_vetoes.jsonl"
    vetoes.write_text("")

    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("")

    monkeypatch.setattr(mod, "_NAV_STATE", nav_state)
    monkeypatch.setattr(mod, "_NAV_LOG", nav_log)
    monkeypatch.setattr(mod, "_EPISODE_LEDGER", episode_ledger)
    monkeypatch.setattr(mod, "_RISK_SNAPSHOT", risk_snapshot)
    monkeypatch.setattr(mod, "_SENTINEL_X", sentinel)
    monkeypatch.setattr(mod, "_BINARY_LAB", binary_lab)
    monkeypatch.setattr(mod, "_AW_STATE", aw_state)
    monkeypatch.setattr(mod, "_RISK_VETOES", vetoes)
    monkeypatch.setattr(mod, "_RUNTIME_YAML", runtime)


# ── Section label tests ─────────────────────────────────────────────


class TestSectionLabels:
    def test_section_1_label(self, isolate_state):
        result = section_1_capital_state()
        assert result.startswith("[SECTION_1]")

    def test_section_2_label(self, isolate_state):
        result = section_2_trade_activity()
        assert result.startswith("[SECTION_2]")

    def test_section_3_label(self, isolate_state):
        result = section_3_regime_context()
        assert result.startswith("[SECTION_3]")

    def test_section_4_label(self, isolate_state):
        result = section_4_risk_discipline()
        assert result.startswith("[SECTION_4]")

    def test_section_5_label(self, isolate_state):
        result = section_5_binary_lab()
        assert result.startswith("[SECTION_5]")

    def test_section_6_label(self, isolate_state):
        result = section_6_structural()
        assert result.startswith("[SECTION_6]")

    def test_section_7_label(self, isolate_state):
        result = section_7_measurement_focus()
        assert result.startswith("[SECTION_7]")


# ── Section content tests ───────────────────────────────────────────


class TestSectionContent:
    def test_section_1_has_nav_fields(self, isolate_state):
        result = section_1_capital_state()
        assert "Starting NAV:" in result
        assert "Ending NAV:" in result
        assert "$10,000.00" in result
        assert "Net Return:" in result
        assert "Max Drawdown:" in result
        # Drawdown should reflect the 10200->9700 trough (4.9%)
        assert "4.9" in result
        assert "Risk Cap Breaches:" in result

    def test_section_2_has_trade_fields(self, isolate_state):
        result = section_2_trade_activity()
        assert "Episodes Closed:" in result
        assert "Win Rate:" in result
        assert "Realised PnL:" in result
        assert "Avg Trades/Day:" in result
        assert "Acceptance Rate:" in result
        assert "Conviction Distribution" in result
        assert "scored episodes" in result

    def test_section_3_has_regime_dist(self, isolate_state):
        result = section_3_regime_context()
        assert "TREND_UP" in result
        assert "65.0%" in result
        assert "120 consecutive" in result

    def test_section_4_has_integrity(self, isolate_state):
        result = section_4_risk_discipline()
        assert "INTACT" in result
        assert "aabbccdd" in result
        assert "Circuit Breaker:" in result

    def test_section_5_has_binary_metrics(self, isolate_state):
        result = section_5_binary_lab()
        assert "ACTIVE (SHADOW)" in result
        assert "5/30" in result
        assert "$2,000.00" in result
        assert "$15.00" in result
        assert "60.0%" in result
        assert "Freeze Intact:" in result

    def test_section_7_has_measurement_goals(self, isolate_state):
        result = section_7_measurement_focus()
        assert "certification window" in result
        assert "No parameter changes" in result


# ── Missing state graceful handling ─────────────────────────────────


class TestMissingState:
    def test_section_1_missing_nav(self, tmp_path, monkeypatch):
        import ops.fund_ops_monthly as mod
        monkeypatch.setattr(mod, "_NAV_STATE", tmp_path / "missing.json")
        monkeypatch.setattr(mod, "_NAV_LOG", tmp_path / "missing2.json")
        monkeypatch.setattr(mod, "_RISK_SNAPSHOT", tmp_path / "missing3.json")
        monkeypatch.setattr(mod, "_RISK_VETOES", tmp_path / "missing4.jsonl")
        result = section_1_capital_state()
        assert "[SECTION_1]" in result
        assert "DATA_UNAVAILABLE" in result

    def test_section_3_missing_sentinel(self, tmp_path, monkeypatch):
        import ops.fund_ops_monthly as mod
        monkeypatch.setattr(mod, "_SENTINEL_X", tmp_path / "missing.json")
        result = section_3_regime_context()
        assert "[SECTION_3]" in result
        assert "DATA_UNAVAILABLE" in result

    def test_section_5_missing_binary(self, tmp_path, monkeypatch):
        import ops.fund_ops_monthly as mod
        monkeypatch.setattr(mod, "_BINARY_LAB", tmp_path / "missing.json")
        result = section_5_binary_lab()
        assert "[SECTION_5]" in result
        assert "Not deployed" in result


# ── Full report structure ───────────────────────────────────────────


class TestFullReport:
    def test_all_7_sections_present(self, isolate_state):
        report = generate_monthly_report()
        for i in range(1, 8):
            assert f"[SECTION_{i}]" in report, f"Missing [SECTION_{i}]"

    def test_no_extra_sections(self, isolate_state):
        report = generate_monthly_report()
        assert "[SECTION_8]" not in report

    def test_sections_in_order(self, isolate_state):
        report = generate_monthly_report()
        positions = []
        for i in range(1, 8):
            pos = report.index(f"[SECTION_{i}]")
            positions.append(pos)
        assert positions == sorted(positions), "Sections out of order"

    def test_no_headings(self, isolate_state):
        report = generate_monthly_report()
        for line in report.split("\n"):
            assert not line.startswith("#"), f"Heading found: {line}"

    def test_no_commentary(self, isolate_state):
        report = generate_monthly_report()
        lower = report.lower()
        for word in ("conclusion", "summary:", "in summary", "overall"):
            assert word not in lower, f"Commentary word found: {word}"
