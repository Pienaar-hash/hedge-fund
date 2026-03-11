"""
Architecture Health Strip — Compact soak readiness surface.

Single horizontal bar showing MRI, CEL, SDD, RDD at a glance.
Answers: "Is the architecture behaving?" without opening panels.

Data source: logs/state/fallback_metrics.json
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

_STATE_PATH = Path("logs/state/fallback_metrics.json")
_EPISODE_PATH = Path("logs/state/episode_ledger.json")
_ERR_PATH = Path("logs/state/edge_calibration.json")
_MONO_PATH = Path("logs/state/hydra_monotonicity.json")
_FUNNEL_PATH = Path("logs/state/hydra_funnel.json")


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_architecture_health(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _STATE_PATH
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}
    # Merge live trade count and score coverage from episode ledger
    try:
        if _EPISODE_PATH.exists():
            ep = json.loads(_EPISODE_PATH.read_text())
            total = int(ep.get("episode_count", 0))
            data["live_trade_count"] = total
            # Score coverage: fraction of episodes with hybrid_score > 0
            scored = sum(
                1 for e in (ep.get("episodes") or [])
                if (e.get("hybrid_score") or 0) > 0
            )
            data["score_coverage"] = scored / total if total > 0 else 0.0
            # Score spread: P90 - P10 of hybrid_score (compression detector)
            scores = sorted(
                float(e["hybrid_score"])
                for e in (ep.get("episodes") or [])
                if (e.get("hybrid_score") or 0) > 0
            )
            if len(scores) >= 10:
                p10 = scores[len(scores) // 10]
                p90 = scores[len(scores) * 9 // 10]
                data["score_spread"] = round(p90 - p10, 4)
    except (json.JSONDecodeError, IOError, ValueError):
        pass
    # Merge ERR from edge calibration
    try:
        if _ERR_PATH.exists():
            ec = json.loads(_ERR_PATH.read_text())
            if ec.get("err") is not None:
                data["err"] = float(ec["err"])
                data["err_count"] = int(ec.get("count", 0))
    except (json.JSONDecodeError, IOError, ValueError):
        pass
    # Merge Spearman ρ from hydra monotonicity
    try:
        if _MONO_PATH.exists():
            mono = json.loads(_MONO_PATH.read_text())
            if mono.get("spearman") is not None:
                data["spearman"] = float(mono["spearman"])
                data["spearman_n"] = int(mono.get("n", 0))
                data["spearman_slope"] = str(mono.get("slope", "unknown"))
                data["head_contamination"] = bool(mono.get("head_contamination", False))
            if mono.get("q5_q1_spread") is not None:
                data["q5_q1_spread"] = float(mono["q5_q1_spread"])
                data["q5_q1_n"] = int(mono.get("n", 0))
    except (json.JSONDecodeError, IOError, ValueError):
        pass
    # Merge Hydra visibility funnel
    try:
        if _FUNNEL_PATH.exists():
            funnel = json.loads(_FUNNEL_PATH.read_text())
            data["hydra_funnel"] = funnel
            data["hydra_visibility_rate"] = float(funnel.get("visibility_rate", 0))
    except (json.JSONDecodeError, IOError, ValueError):
        pass
    return data


# ---------------------------------------------------------------------------
# Color logic
# ---------------------------------------------------------------------------

def _mri_color(v: float) -> str:
    if v >= 0.75:
        return "#21ba45"
    if v >= 0.50:
        return "#f2c037"
    return "#db2828"


def _cel_color(v: float) -> str:
    if v > 0.01:
        return "#21ba45"
    if v >= 0:
        return "#f2c037"
    return "#db2828"


def _sdd_color(v: float) -> str:
    av = abs(v)
    if av < 0.02:
        return "#21ba45"
    if av <= 0.05:
        return "#f2c037"
    return "#db2828"


def _rdd_color(v: float) -> str:
    if v < 0.06:
        return "#21ba45"
    if v <= 0.15:
        return "#f2c037"
    return "#db2828"


def _err_color(v: Optional[float]) -> str:
    if v is None:
        return "#666"
    if 0.8 <= v <= 1.2:
        return "#21ba45"
    if 0.6 <= v <= 1.4:
        return "#f2c037"
    return "#db2828"


def _spearman_color(v: Optional[float]) -> str:
    if v is None:
        return "#666"
    if v > 0.2:
        return "#21ba45"
    if v >= 0:
        return "#f2c037"
    return "#db2828"


def _coverage_color(v: float) -> str:
    if v >= 0.60:
        return "#21ba45"
    if v >= 0.20:
        return "#f2c037"
    return "#db2828"


def _spread_color(v: Optional[float]) -> str:
    if v is None:
        return "#666"
    if v >= 0.15:
        return "#21ba45"
    if v >= 0.08:
        return "#f2c037"
    return "#db2828"


def _q5q1_color(v: Optional[float]) -> str:
    if v is None:
        return "#666"
    if v > 0.004:
        return "#21ba45"
    if v >= 0:
        return "#f2c037"
    return "#db2828"


def _visibility_color(v: float) -> str:
    if v >= 0.50:
        return "#21ba45"
    if v >= 0.20:
        return "#f2c037"
    return "#db2828"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_architecture_strip(metrics: Dict[str, Any]) -> None:
    """Render a compact architecture health strip."""
    if not metrics:
        return

    mri = float(metrics.get("ecs_readiness_score", 0))
    cel = float(metrics.get("conflict_edge_lift", 0))
    sdd = float(metrics.get("score_scale_delta", 0))
    rdd = float(metrics.get("regime_dependence_spread", 0))
    ecs_ready = bool(metrics.get("ecs_ready", False))

    cel_n = int(metrics.get("cel_count", 0))
    sdd_n = int(metrics.get("sdd_count", 0))
    rdd_n = len(metrics.get("regime_rsd", {}) or {})
    live_trades = int(metrics.get("live_trade_count", 0))
    err_val = metrics.get("err")  # None if not yet computed
    err_count = int(metrics.get("err_count", 0))
    spearman_val = metrics.get("spearman")  # None if not yet computed
    spearman_n = int(metrics.get("spearman_n", 0))
    head_contamination = bool(metrics.get("head_contamination", False))
    score_coverage = float(metrics.get("score_coverage", 0))
    score_spread = metrics.get("score_spread")  # None if <10 scored trades
    hydra_vis = float(metrics.get("hydra_visibility_rate", 0))
    q5_q1_spread = metrics.get("q5_q1_spread")  # None if <10 scored trades
    q5_q1_n = int(metrics.get("q5_q1_n", 0))
    hydra_funnel = metrics.get("hydra_funnel") or {}
    funnel_stages = hydra_funnel.get("stages") or {}
    regime_vis = hydra_funnel.get("regime_visibility") or {}
    merge_win_rate = float(hydra_funnel.get("merge_win_rate", 0))

    bar_full = int(min(max(mri, 0), 1) * 10)
    bar_empty = 10 - bar_full

    ecs_badge = ""
    if ecs_ready:
        ecs_badge = (
            '&nbsp;<span style="background:#21ba45;color:#fff;'
            'padding:1px 6px;border-radius:3px;font-weight:600;'
            'font-size:0.7rem;">ECS READY</span>'
        )

    html = f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;
                padding:10px 16px;margin-bottom:12px;font-family:monospace;">
      <div style="color:#888;font-size:0.7rem;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:6px;">
        Architecture Health{ecs_badge}
      </div>
      <div style="display:flex;gap:24px;align-items:baseline;flex-wrap:wrap;">
        <span>
          <span style="color:#aaa;font-size:0.75rem;">MRI</span>&nbsp;
          <span style="color:{_mri_color(mri)};">{'█' * bar_full}{'░' * bar_empty}</span>
          <span style="color:{_mri_color(mri)};font-weight:600;">&nbsp;{mri:.2f}</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">CEL</span>&nbsp;
          <span style="color:{_cel_color(cel)};font-weight:600;">{cel:+.3f}</span>
          <span style="color:#666;font-size:0.65rem;">&nbsp;(n={cel_n})</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">SDD</span>&nbsp;
          <span style="color:{_sdd_color(sdd)};font-weight:600;">{sdd:+.3f}</span>
          <span style="color:#666;font-size:0.65rem;">&nbsp;(n={sdd_n})</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">RDD</span>&nbsp;
          <span style="color:{_rdd_color(rdd)};font-weight:600;">{rdd:.3f}</span>
          <span style="color:#666;font-size:0.65rem;">&nbsp;(n={rdd_n})</span>
        </span>
        <span style="margin-left:auto;">
          <span style="color:#aaa;font-size:0.75rem;">Live Trades</span>&nbsp;
          <span style="color:#21ba45;font-weight:700;font-size:0.95rem;">{live_trades}</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">ERR</span>&nbsp;
          <span style="color:{_err_color(err_val)};font-weight:600;">{f'{err_val:.2f}' if err_val is not None else '—'}</span>
          <span style="color:#666;font-size:0.65rem;">&nbsp;(n={err_count})</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">ρ</span>&nbsp;
          <span style="color:{_spearman_color(spearman_val)};font-weight:600;">{f'{spearman_val:+.3f}' if spearman_val is not None else '—'}</span>
          <span style="color:#666;font-size:0.65rem;">&nbsp;(n={spearman_n})</span>
        </span>
{'        <span><span style="background:#db2828;color:#fff;padding:1px 6px;border-radius:3px;font-weight:600;font-size:0.65rem;">HEAD MIX</span></span>' if head_contamination else ''}
        <span>
          <span style="color:#aaa;font-size:0.75rem;">ScoreCov</span>&nbsp;
          <span style="color:{_coverage_color(score_coverage)};font-weight:600;">{score_coverage:.0%}</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">Spread</span>&nbsp;
          <span style="color:{_spread_color(score_spread)};font-weight:600;">{f'{score_spread:.2f}' if score_spread is not None else '—'}</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">HydraVis</span>&nbsp;
          <span style="color:{_visibility_color(hydra_vis)};font-weight:600;">{hydra_vis:.0%}</span>
        </span>
        <span>
          <span style="color:#aaa;font-size:0.75rem;">Q5−Q1</span>&nbsp;
          <span style="color:{_q5q1_color(q5_q1_spread)};font-weight:600;">{f'{q5_q1_spread * 100:+.2f}%' if q5_q1_spread is not None else '—'}</span>
          <span style="color:#666;font-size:0.65rem;">&nbsp;(n={q5_q1_n})</span>
        </span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # --- Hydra Funnel detail row (only if data exists) ---
    if funnel_stages.get("generated", 0) > 0:
        gen = funnel_stages.get("generated", 0)
        pm = funnel_stages.get("post_merge", 0)
        sub = funnel_stages.get("submitted", 0)
        pd_ = funnel_stages.get("post_doctrine", 0)
        exe = funnel_stages.get("executed", 0)
        mwr = merge_win_rate

        # Build per-regime mini table
        regime_rows = ""
        for regime, rv in sorted(regime_vis.items()):
            rv_rate = float(rv.get("visibility_rate", 0))
            rv_gen = (rv.get("stages") or {}).get("generated", 0)
            rv_exe = (rv.get("stages") or {}).get("executed", 0)
            safe_regime = _html.escape(str(regime))
            regime_rows += (
                f'<span style="margin-right:16px;">'
                f'<span style="color:#888;font-size:0.65rem;">{safe_regime}</span>&nbsp;'
                f'<span style="color:{_visibility_color(rv_rate)};font-size:0.7rem;">{rv_rate:.0%}</span>'
                f'<span style="color:#555;font-size:0.6rem;">&nbsp;({rv_exe}/{rv_gen})</span>'
                f'</span>'
            )

        funnel_html = f"""
        <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:6px;
                    padding:8px 16px;margin-bottom:12px;font-family:monospace;">
          <div style="color:#888;font-size:0.65rem;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:4px;">
            Hydra Funnel
          </div>
          <div style="display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;font-size:0.75rem;">
            <span style="color:#aaa;">Generated</span>
            <span style="color:#ccc;font-weight:600;">{gen}</span>
            <span style="color:#555;">&rarr;</span>
            <span style="color:#aaa;">Merged</span>
            <span style="color:#ccc;font-weight:600;">{pm}</span>
            <span style="color:#555;">&rarr;</span>
            <span style="color:#aaa;">Submitted</span>
            <span style="color:#ccc;font-weight:600;">{sub}</span>
            <span style="color:#555;">&rarr;</span>
            <span style="color:#aaa;">Doctrine</span>
            <span style="color:#ccc;font-weight:600;">{pd_}</span>
            <span style="color:#555;">&rarr;</span>
            <span style="color:#aaa;">Executed</span>
            <span style="color:{_visibility_color(hydra_vis)};font-weight:700;">{exe}</span>
            <span style="color:#555;">&nbsp;|&nbsp;</span>
            <span style="color:#aaa;">MergeWin</span>
            <span style="color:{_visibility_color(mwr)};font-weight:600;">{mwr:.0%}</span>
          </div>
{'          <div style="margin-top:4px;display:flex;flex-wrap:wrap;">' + regime_rows + '</div>' if regime_rows else ''}
        </div>
        """
        st.markdown(funnel_html, unsafe_allow_html=True)
