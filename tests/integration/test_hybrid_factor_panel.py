from __future__ import annotations

from unittest.mock import MagicMock

from dashboard.hybrid_factor_panel import render_hybrid_factor_panel


def _patch_streamlit(monkeypatch):
    import dashboard.hybrid_factor_panel as panel

    st = MagicMock()
    st.header = MagicMock()
    st.subheader = MagicMock()
    st.info = MagicMock()
    st.dataframe = MagicMock()

    def _columns(n=1):
        parent = st

        class _Col:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

            def dataframe(self_inner, *args, **kwargs):
                return parent.dataframe(*args, **kwargs)

            def subheader(self_inner, *args, **kwargs):
                return parent.subheader(*args, **kwargs)

        return [_Col() for _ in range(n)]

    st.columns = _columns
    panel.st = st
    return st


def test_panel_handles_empty_snapshot(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    render_hybrid_factor_panel({})
    assert st.info.called


def test_panel_renders_deciles(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {"regimes": {"factors": {"hybrid_score_decile": {"0": {"trade_count": 1, "total_pnl": 1.0, "avg_pnl": 1.0}}}}}
    render_hybrid_factor_panel(snapshot)
    assert st.dataframe.called


def test_panel_renders_trend_and_carry(monkeypatch):
    st = _patch_streamlit(monkeypatch)
    snapshot = {
        "regimes": {
            "factors": {
                "trend_strength_bucket": {"weak": {"trade_count": 1, "total_pnl": 0.0, "avg_pnl": 0.0}},
                "carry_regime": {"neutral": {"trade_count": 1, "total_pnl": 0.0, "avg_pnl": 0.0}},
            }
        }
    }
    render_hybrid_factor_panel(snapshot)
    assert st.subheader.called
