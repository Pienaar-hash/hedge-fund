from __future__ import annotations

from unittest.mock import MagicMock

from dashboard.pnl_attribution_panel import (
    render_daily_pnl_strip,
    render_pnl_attribution_panel,
    render_regime_attribution,
    render_risk_mode_attribution,
    render_strategy_attribution,
    render_summary_block,
    render_symbol_attribution,
)
from dashboard.utils.attribution_loaders import load_pnl_attribution


def _patch_streamlit(monkeypatch):
    import dashboard.pnl_attribution_panel as panel

    st = MagicMock()
    st.bar_chart = MagicMock()
    st.altair_chart = MagicMock()
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

            def metric(self_inner, *args, **kwargs):
                return parent.metric(*args, **kwargs)

            def __getattr__(self_inner, name):
                attr = getattr(parent, name, None)
                if callable(attr):
                    return attr
                return lambda *a, **k: None

        return [_Col() for _ in range(n)]

    st.columns = _columns
    panel.st = st
    return st


def test_load_pnl_attribution_missing_file_returns_empty_dict(tmp_path):
    missing = tmp_path / "pnl_attribution.json"
    result = load_pnl_attribution(missing)
    assert result == {}


def test_render_symbol_attribution_with_minimal_snapshot(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"per_symbol": {"BTCUSDT": {"realized_pnl": 1.0, "unrealized_pnl": 2.0, "total_pnl": 3.0, "trade_count": 2}}}
    render_symbol_attribution(snapshot)
    assert st.dataframe.called
    assert st.bar_chart.called


def test_render_strategy_attribution_with_minimal_snapshot(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"per_strategy": {"momentum": {"realized_pnl": 1.0, "unrealized_pnl": 0.5, "total_pnl": 1.5, "trade_count": 1}}}
    render_strategy_attribution(snapshot)
    assert st.dataframe.called


def test_render_regime_attribution_handles_missing_fields(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {}
    render_regime_attribution(snapshot)
    assert st.info.called


def test_render_risk_mode_attribution_handles_missing_fields(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_risk_mode_attribution({})
    assert st.markdown.call_count == 4


def test_render_daily_pnl_strip_handles_ordering(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"per_day": {"2025-01-02": {"total": -1, "realized": -1, "unrealized": 0}, "2025-01-01": {"total": 2, "realized": 2, "unrealized": 0}}}
    render_daily_pnl_strip(snapshot)
    assert st.markdown.called


def test_summary_block_parses_numbers_correctly(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"summary": {"total_pnl": 3.0, "total_realized": 2.0, "total_unrealized": 1.0, "win_rate": 0.5, "record_count": 4, "ts": 0}}
    render_summary_block(snapshot)
    assert st.metric.call_count == 3


def test_panel_renders_without_error_on_empty_snapshot(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_pnl_attribution_panel({})
    assert st.info.called


def test_panel_renders_without_error_when_per_regime_missing(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"summary": {"total_pnl": 0, "total_realized": 0, "total_unrealized": 0, "win_rate": 0, "record_count": 0, "ts": 0}}
    render_pnl_attribution_panel(snapshot)
    assert st.subheader.called


def test_panel_renders_without_error_when_per_risk_mode_missing(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"summary": {"total_pnl": 0, "total_realized": 0, "total_unrealized": 0, "win_rate": 0, "record_count": 0, "ts": 0}, "per_regime": {"atr": {}, "dd": {}}}
    render_pnl_attribution_panel(snapshot)
    assert st.header.called


def test_panel_respects_color_language(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"per_risk_mode": {"OK": {"realized": 1, "unrealized": 0, "total": 1, "trade_count": 1}}}
    render_risk_mode_attribution(snapshot)
    # OK tint should appear in at least one markdown call
    ok_calls = [call for call in st.markdown.call_args_list if "#21c35420" in str(call)]
    assert ok_calls


def test_no_exceptions_on_zero_data(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"per_symbol": {}, "per_strategy": {}, "summary": {"total_pnl": 0, "total_realized": 0, "total_unrealized": 0, "win_rate": 0, "record_count": 0, "ts": 0}}
    render_symbol_attribution(snapshot)
    render_strategy_attribution(snapshot)
    render_regime_attribution(snapshot)
    render_risk_mode_attribution(snapshot)
    render_daily_pnl_strip({"per_day": {}})
    assert st.info.called
