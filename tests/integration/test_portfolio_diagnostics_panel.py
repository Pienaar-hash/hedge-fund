from __future__ import annotations

import time
from unittest.mock import MagicMock

from dashboard.diagnostics_panel import (
    render_correlation_heatmap,
    render_daily_return_strip,
    render_diagnostics_panel,
    render_regime_pnl_bars,
    render_symbol_drawdown_snapshot,
)
from dashboard.utils import diagnostics_loaders as loaders


def _patch_streamlit(monkeypatch):
    import dashboard.diagnostics_panel as panel

    st = MagicMock()
    st.bar_chart = MagicMock()
    st.dataframe = MagicMock()
    st.info = MagicMock()
    st.warning = MagicMock()
    st.subheader = MagicMock()
    st.caption = MagicMock()
    st.markdown = MagicMock()
    st.metric = MagicMock()
    st.header = MagicMock()
    st.divider = MagicMock()

    def _columns(n=1):
        parent = st

        class _Col:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def __getattr__(self_inner, name):
                attr = getattr(parent, name, None)
                if callable(attr):
                    return attr
                return lambda *a, **k: None

        return [_Col() for _ in range(n)]

    st.columns = _columns
    panel.st = st
    return st


def test_load_equity_state_missing_file_returns_empty_dict(tmp_path):
    path = tmp_path / "equity.json"
    result = loaders.load_equity_state(path)
    assert result == {}


def test_load_positions_state_missing_file_returns_empty_list(tmp_path):
    path = tmp_path / "positions.json"
    result = loaders.load_positions_state(path)
    assert result == []


def test_load_pnl_attribution_missing_file_returns_empty_dict(tmp_path):
    path = tmp_path / "pnl_attribution.json"
    result = loaders.load_pnl_attribution_state(path)
    assert result == {}


def test_symbol_drawdown_snapshot_handles_empty_equity_and_positions(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_symbol_drawdown_snapshot({}, [])
    assert st.metric.called
    assert st.info.called


def test_symbol_drawdown_snapshot_uses_last_drawdown_value(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_symbol_drawdown_snapshot({"drawdown": [0.1, 0.2]}, [])
    args, _ = st.metric.call_args
    assert "20.00%" in args[1]


def test_daily_return_strip_with_two_days_of_data(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    now = time.time()
    next_day = now + 86400
    equity = {"timestamps": [now, next_day], "equity": [100.0, 105.0]}
    render_daily_return_strip(equity)
    assert st.markdown.called


def test_regime_pnl_bars_handles_missing_regime_section(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_regime_pnl_bars({})
    assert st.info.called


def test_regime_pnl_bars_with_simple_regime_data(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    pnl_attr = {"per_regime": {"atr": {"0": {"total": 1}, "1": {"total": 2}, "2": {"total": 0}, "3": {"total": -1}}}}
    render_regime_pnl_bars(pnl_attr)
    assert st.bar_chart.called


def test_correlation_heatmap_handles_no_pnl_attribution(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_correlation_heatmap({}, {})
    assert st.info.called


def test_correlation_heatmap_handles_sparse_data_without_crash(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    pnl_attr = {"per_symbol": {"BTCUSDT": {"total_pnl": 1.0}}}
    render_correlation_heatmap({}, pnl_attr)
    assert st.dataframe.called


def test_diagnostics_panel_renders_with_minimal_snapshot(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    equity = {"timestamps": [time.time()], "equity": [100]}
    positions = [{"symbol": "BTCUSDT", "unrealized_pnl": -1.0}]
    pnl_attr = {"per_regime": {"atr": {"0": {"total": 0}, "1": {"total": 0}, "2": {"total": 0}, "3": {"total": 0}}}}
    render_diagnostics_panel(equity, positions, pnl_attr)
    assert st.header.called


def test_diagnostics_panel_renders_with_all_sections_missing(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_diagnostics_panel({}, [], {})
    assert st.header.called
