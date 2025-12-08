from __future__ import annotations

from unittest.mock import MagicMock

from dashboard.exit_attribution_panel import (
    render_exit_attribution_panel,
    render_exit_by_strategy,
    render_exit_by_symbol,
    render_exit_regime_heatmaps,
    render_exit_summary_card,
)


def _patch_streamlit(monkeypatch):
    import dashboard.exit_attribution_panel as panel

    st = MagicMock()
    st.info = MagicMock()
    st.header = MagicMock()
    st.subheader = MagicMock()
    st.metric = MagicMock()
    st.markdown = MagicMock()
    st.dataframe = MagicMock()

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


def test_exit_panel_handles_missing_block(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_exit_attribution_panel({})
    assert st.info.called


def test_exit_summary_metrics_render(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {
        "exits": {
            "summary": {
                "total_exits": 2,
                "tp_hits": 1,
                "sl_hits": 1,
                "tp_ratio": 0.5,
                "avg_rr_tp": 1.5,
                "avg_rr_sl": 0.4,
                "avg_exit_pnl": 3.0,
                "total_exit_pnl": 6.0,
            }
        }
    }
    render_exit_summary_card(snapshot)
    assert st.metric.call_count >= 4


def test_exit_tables_render(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {
        "exits": {
            "by_strategy": {"vol_target": {"total_exits": 1, "tp_hits": 1, "sl_hits": 0, "tp_ratio": 1.0, "avg_exit_pnl": 2.0}},
            "by_symbol": {"BTCUSDT": {"total_exits": 1, "tp_hits": 1, "sl_hits": 0, "tp_ratio": 1.0, "avg_exit_pnl": 2.0}},
            "regimes": {"atr": {"0": {"total_exits": 1, "tp_hits": 1, "sl_hits": 0, "tp_ratio": 1.0, "total_exit_pnl": 2.0}}},
        }
    }
    render_exit_by_strategy(snapshot)
    render_exit_by_symbol(snapshot)
    render_exit_regime_heatmaps(snapshot)
    assert st.dataframe.called
