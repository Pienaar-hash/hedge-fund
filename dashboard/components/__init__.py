"""
Dashboard Components — Institutional-grade UI modules.

Each component uses st.html() for raw HTML rendering (Streamlit ≥1.33).
"""
from dashboard.components.kpi_strip import render_kpi_strip
from dashboard.components.aum import render_aum_block
from dashboard.components.runtime_health import render_runtime_health_block
from dashboard.components.positions import render_positions_block
from dashboard.components.performance import render_performance_block
from dashboard.components.treasury import render_treasury_block
from dashboard.components.diagnostics import render_diagnostics_block

__all__ = [
    "render_kpi_strip",
    "render_aum_block",
    "render_runtime_health_block",
    "render_positions_block",
    "render_performance_block",
    "render_treasury_block",
    "render_diagnostics_block",
]
