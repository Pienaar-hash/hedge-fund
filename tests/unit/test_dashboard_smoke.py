"""Smoke tests for dashboard entry point and state loading.

Ensures the consolidated dashboard architecture doesn't regress:
- app.py imports are valid
- State loader is callable and returns a dict
- Layout functions are importable
- Components directory is populated
"""
from __future__ import annotations

import pytest


@pytest.mark.unit
class TestDashboardSmoke:
    """Smoke tests for dashboard consolidation."""

    def test_dashboard_state_loader_importable(self):
        from dashboard.app import _load_dashboard_state
        assert callable(_load_dashboard_state)

    def test_dashboard_state_returns_dict(self):
        from dashboard.app import _load_dashboard_state
        state = _load_dashboard_state()
        assert isinstance(state, dict)
        # Core keys must exist
        for key in ("nav_state", "risk_snapshot", "positions"):
            assert key in state, f"missing state key: {key}"

    def test_layout_functions_importable(self):
        from dashboard.layout import (
            render_header_block,
            render_kpi_strip,
            render_positions_block,
            render_strategy_block,
            render_runtime_block,
            render_diagnostics_block,
        )
        for fn in (
            render_header_block,
            render_kpi_strip,
            render_positions_block,
            render_strategy_block,
            render_runtime_block,
            render_diagnostics_block,
        ):
            assert callable(fn)

    def test_optional_panel_imports_are_guarded(self):
        from pathlib import Path
        source = Path("dashboard/app.py").read_text()
        assert "try:\n    from dashboard.components.edge_calibration_panel import" in source
        assert "try:\n    from dashboard.components.engine_lift_panel import" in source
        assert "try:\n    from dashboard.components.hydra_monotonicity_panel import" in source

    def test_css_file_exists(self):
        from pathlib import Path
        css = Path("dashboard/static/quant_theme.css")
        assert css.exists(), "quant_theme.css missing"
        assert css.stat().st_size > 100, "quant_theme.css appears empty"

    def test_no_legacy_entry_points(self):
        from pathlib import Path
        assert not Path("dashboard/app_v7_6.py").exists(), "app_v7_6.py should be archived"
        assert not Path("dashboard/layout_v7_6.py").exists(), "layout_v7_6.py should be archived"
        assert not Path("dashboard/main.py").exists(), "main.py should not exist"
