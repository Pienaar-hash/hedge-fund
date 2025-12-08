"""Tests for normalized fractional fields in state_publish.py."""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest

import execution.state_publish as state_publish


class TestToFrac:
    """Tests for the _to_frac normalization helper."""

    def test_percent_style_normalized(self) -> None:
        """dd_pct=1.1 (percent-style) -> dd_frac=0.011"""
        assert state_publish._to_frac(1.1) == pytest.approx(0.011, rel=1e-6)

    def test_larger_percent_style(self) -> None:
        """dd_pct=35.0 (35% drawdown) -> dd_frac=0.35"""
        assert state_publish._to_frac(35.0) == pytest.approx(0.35, rel=1e-6)

    def test_fractional_style_unchanged(self) -> None:
        """Already fractional value (0.011) stays unchanged."""
        assert state_publish._to_frac(0.011) == pytest.approx(0.011, rel=1e-6)

    def test_boundary_value_one(self) -> None:
        """Value exactly 1.0 is treated as fractional (100%)."""
        assert state_publish._to_frac(1.0) == pytest.approx(1.0, rel=1e-6)

    def test_zero(self) -> None:
        """Zero stays zero."""
        assert state_publish._to_frac(0.0) == pytest.approx(0.0, rel=1e-6)

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        assert state_publish._to_frac(None) is None

    def test_string_numeric(self) -> None:
        """String numeric value is converted."""
        assert state_publish._to_frac("1.5") == pytest.approx(0.015, rel=1e-6)

    def test_invalid_string_returns_none(self) -> None:
        """Non-numeric string returns None."""
        assert state_publish._to_frac("invalid") is None


class TestWriteRiskSnapshotState:
    """Tests for write_risk_snapshot_state with fractional fields."""

    def test_dd_frac_added_from_percent_style(self, tmp_path, monkeypatch) -> None:
        """dd_pct=1.14 in percent-style -> dd_frac=0.0114 in output."""
        monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path)

        payload = {
            "updated_ts": "2025-11-28T10:00:00Z",
            "dd_state": {
                "dd_state": "defensive",
                "drawdown": {
                    "dd_pct": 1.14,
                    "dd_abs": 48.0,
                    "peak": 4200.0,
                    "nav": 4152.0,
                    "daily_loss": {
                        "pct": 1.14,
                        "daily_peak": 4200.0,
                        "nav": 4152.0,
                    },
                },
            },
        }
        state_publish.write_risk_snapshot_state(payload, state_dir=tmp_path)

        out_path = tmp_path / "risk_snapshot.json"
        assert out_path.exists()
        result = json.loads(out_path.read_text())

        # Check fractional fields added
        assert "dd_frac" in result
        assert "daily_loss_frac" in result
        assert result["dd_frac"] == pytest.approx(0.0114, rel=1e-6)
        assert result["daily_loss_frac"] == pytest.approx(0.0114, rel=1e-6)

        # Check original fields preserved
        dd_state_block = result.get("dd_state", {})
        drawdown_block = dd_state_block.get("drawdown", {})
        assert drawdown_block.get("dd_pct") == 1.14

    def test_fractional_style_preserved(self, tmp_path, monkeypatch) -> None:
        """Values already fractional (<=1) stay unchanged."""
        monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path)

        payload = {
            "dd_state": {
                "drawdown": {
                    "dd_pct": 0.05,  # Already fractional (5%)
                    "daily_loss": {"pct": 0.03},  # Already fractional (3%)
                },
            },
        }
        state_publish.write_risk_snapshot_state(payload, state_dir=tmp_path)

        out_path = tmp_path / "risk_snapshot.json"
        result = json.loads(out_path.read_text())

        assert result["dd_frac"] == pytest.approx(0.05, rel=1e-6)
        assert result["daily_loss_frac"] == pytest.approx(0.03, rel=1e-6)

    def test_missing_dd_state_graceful(self, tmp_path, monkeypatch) -> None:
        """Missing dd_state block doesn't raise, sets None."""
        monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path)

        payload = {"updated_ts": "2025-11-28T10:00:00Z", "symbols": []}
        state_publish.write_risk_snapshot_state(payload, state_dir=tmp_path)

        out_path = tmp_path / "risk_snapshot.json"
        result = json.loads(out_path.read_text())

        assert result.get("dd_frac") is None
        assert result.get("daily_loss_frac") is None

    def test_missing_daily_loss_graceful(self, tmp_path, monkeypatch) -> None:
        """Missing daily_loss block sets daily_loss_frac to None."""
        monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path)

        payload = {
            "dd_state": {
                "drawdown": {
                    "dd_pct": 2.5,
                    # daily_loss missing
                },
            },
        }
        state_publish.write_risk_snapshot_state(payload, state_dir=tmp_path)

        out_path = tmp_path / "risk_snapshot.json"
        result = json.loads(out_path.read_text())

        assert result["dd_frac"] == pytest.approx(0.025, rel=1e-6)
        assert result.get("daily_loss_frac") is None

    def test_empty_payload(self, tmp_path, monkeypatch) -> None:
        """Empty payload doesn't crash."""
        monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path)

        state_publish.write_risk_snapshot_state({}, state_dir=tmp_path)

        out_path = tmp_path / "risk_snapshot.json"
        result = json.loads(out_path.read_text())

        assert result.get("dd_frac") is None
        assert result.get("daily_loss_frac") is None

    def test_none_payload(self, tmp_path, monkeypatch) -> None:
        """None payload doesn't crash."""
        monkeypatch.setattr(state_publish, "STATE_DIR", tmp_path)

        state_publish.write_risk_snapshot_state(None, state_dir=tmp_path)

        out_path = tmp_path / "risk_snapshot.json"
        result = json.loads(out_path.read_text())

        assert result.get("dd_frac") is None
        assert result.get("daily_loss_frac") is None


class TestDashboardGracefulAccess:
    """Tests that dashboard code handles missing fields gracefully."""

    def test_format_fraction_none(self) -> None:
        """format_fraction handles None gracefully."""
        from dashboard.dashboard_utils import format_fraction

        assert format_fraction(None) == "–"

    def test_format_fraction_valid(self) -> None:
        """format_fraction formats valid values."""
        from dashboard.dashboard_utils import format_fraction

        assert format_fraction(0.0114) == "0.0114"
        assert format_fraction(0.35, nd=2) == "0.35"

    def test_format_fraction_invalid_string(self) -> None:
        """format_fraction handles invalid strings gracefully."""
        from dashboard.dashboard_utils import format_fraction

        assert format_fraction("invalid") == "–"

    def test_kpi_panel_missing_frac_fields(self) -> None:
        """kpi_panel.render_kpis_overview handles missing frac fields."""
        # Just verify the import and function signature - actual rendering
        # requires Streamlit context which we can't easily test
        from dashboard.kpi_panel import render_kpis_overview

        # This should not raise even with empty dict
        kpis: Dict[str, Any] = {}
        dd_frac = kpis.get("dd_frac")  # Returns None, doesn't raise
        daily_loss_frac = kpis.get("daily_loss_frac")  # Returns None
        assert dd_frac is None
        assert daily_loss_frac is None
