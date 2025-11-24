## 1. `dashboard/live_helpers.py` — enforce v6 paths + router policy fix 

### a) Kill `nav.json` fallback and point router policy at v6 state

At the top of the file, update the constants block:

```diff
 LOG = logging.getLogger("dash.live_helpers")

 STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD"}
 STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
 NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
-NAV_STATE_FALLBACK_PATH = Path(os.getenv("NAV_STATE_FALLBACK_PATH") or (STATE_DIR / "nav.json"))
 SYNCED_STATE_PATH = Path(os.getenv("SYNCED_STATE_PATH") or (STATE_DIR / "synced_state.json"))
 UNIVERSE_STATE_PATH = Path(os.getenv("UNIVERSE_STATE_PATH") or (STATE_DIR / "universe.json"))
 ROUTER_HEALTH_STATE_PATH = Path(os.getenv("ROUTER_HEALTH_STATE_PATH") or (STATE_DIR / "router_health.json"))
 EXPECTANCY_STATE_PATH = Path(os.getenv("EXPECTANCY_STATE_PATH") or (STATE_DIR / "expectancy_v6.json"))
 SYMBOL_SCORES_STATE_PATH = Path(os.getenv("SYMBOL_SCORES_STATE_PATH") or (STATE_DIR / "symbol_scores_v6.json"))
 PIPELINE_COMPARE_STATE_PATH = Path(os.getenv("PIPELINE_V6_COMPARE_STATE_PATH") or (STATE_DIR / "pipeline_v6_compare_summary.json"))
 PIPELINE_COMPARE_LOG_PATH = Path(os.getenv("PIPELINE_V6_COMPARE_LOG_PATH") or "logs/pipeline_v6_compare.jsonl")
 RISK_ALLOCATOR_STATE_PATH = Path(os.getenv("RISK_ALLOC_STATE_PATH") or (STATE_DIR / "risk_allocation_suggestions_v6.json"))
-ROUTER_POLICY_STATE_PATH = Path(os.getenv("ROUTER_POLICY_STATE_PATH") or (STATE_DIR / "router_health.json"))
-ROUTER_SUGGESTIONS_STATE_PATH = Path(os.getenv("ROUTER_SUGGESTIONS_STATE_PATH") or (STATE_DIR / "router_policy_suggestions_v6.json"))
+ROUTER_POLICY_STATE_PATH = Path(
+    os.getenv("ROUTER_POLICY_STATE_PATH") or (STATE_DIR / "router_policy_state_v6.json")
+)
+ROUTER_SUGGESTIONS_STATE_PATH = Path(
+    os.getenv("ROUTER_SUGGESTIONS_STATE_PATH") or (STATE_DIR / "router_policy_suggestions_v6.json")
+)
 PIPELINE_SHADOW_HEAD_STATE_PATH = Path(os.getenv("PIPELINE_SHADOW_HEAD_STATE_PATH") or (STATE_DIR / "pipeline_v6_shadow_head.json"))
 RUNTIME_PROBE_STATE_PATH = Path(os.getenv("RUNTIME_PROBE_STATE_PATH") or (STATE_DIR / "v6_runtime_probe.json"))
```

### b) `_resolve_nav_state_path` must be v6-only

```diff
-def _resolve_nav_state_path() -> Path:
-    if NAV_STATE_PATH.exists():
-        return NAV_STATE_PATH
-    if NAV_STATE_FALLBACK_PATH.exists():
-        return NAV_STATE_FALLBACK_PATH
-    return NAV_STATE_PATH
+def _resolve_nav_state_path() -> Path:
+    """
+    Resolve the canonical v6 NAV state path.
+    No legacy nav.json fallback is allowed in v6.
+    """
+    return NAV_STATE_PATH
```

This forces the dashboard to only ever read `logs/state/nav_state.json`. If it’s missing/empty, the header will show `0 / N/A` instead of silently reading `nav.json`.

---

## 2. `dashboard/app.py` — v6-only state + nav wiring cleanup 

### a) Stop using `nav.json` and `positions.json` as defaults

Near the top of `main()` where the state paths are defined:

```diff
     STATE_DIR = Path(os.getenv("STATE_DIR") or (PROJECT_ROOT / "logs/state"))
     NAV_STATE_PRIMARY = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
-    NAV_STATE_FALLBACK = STATE_DIR / "nav.json"
-    NAV_STATE_CANDIDATES = [NAV_STATE_PRIMARY]
-    if NAV_STATE_FALLBACK not in NAV_STATE_CANDIDATES:
-        NAV_STATE_CANDIDATES.append(NAV_STATE_FALLBACK)
-    NAV_STATE_PATH = NAV_STATE_PRIMARY
-    POSITIONS_STATE_PATH = Path(os.getenv("POSITIONS_STATE_PATH") or (STATE_DIR / "positions.json"))
+    # v6: nav_state.json is the only canonical NAV state, no legacy fallback
+    NAV_STATE_PATH = NAV_STATE_PRIMARY
+    # v6: positions_state.json is the canonical positions snapshot
+    POSITIONS_STATE_PATH = Path(
+        os.getenv("POSITIONS_STATE_PATH") or (STATE_DIR / "positions_state.json")
+    )
```

### b) Use `nav_helpers.load_nav_state` instead of a local fallback shim

Right now this file defines its own `load_nav_state_payload()` + `load_nav_state()` that re-introduce the v5 candidate list. We want the imported v6 helper to be the single source of truth.

Find and **delete** these two helpers entirely:

```python
def load_nav_state_payload() -> Tuple[Dict[str, Any], str]:
    for path in NAV_STATE_CANDIDATES:
        payload = load_json(path, default={}) or {}
        if isinstance(payload, dict) and payload:
            return payload, path.name
    fallback_path = NAV_STATE_CANDIDATES[0]
    payload = load_json(fallback_path, default={}) or {}
    return payload if isinstance(payload, dict) else {}, fallback_path.name


def load_nav_state() -> Tuple[Dict[str, Any], str]:
    payload, source = load_nav_state_payload()
    return payload, source
```

These shadow the imported `load_nav_state` and keep the v5 fallback alive; removing them makes us use the v6 implementation from `dashboard.nav_helpers`.

### c) Make `load_local_nav_doc` call the shared v6 loader

Update `load_local_nav_doc` to delegate to the imported helper:

```diff
     @st.cache_data(ttl=30, show_spinner=False)
     def load_local_nav_doc() -> Dict[str, Any]:
-        payload, _ = load_nav_state_payload()
-        return payload if isinstance(payload, dict) else {}
+        """
+        Thin wrapper around dashboard.nav_helpers.load_nav_state().
+        Ensures we always read the v6 nav_state.json contract.
+        """
+        payload, _ = load_nav_state()
+        return payload if isinstance(payload, dict) else {}
```

The rest of the file (runtime strip, NAV header, tabs, etc.) can stay as-is; it will now be fed purely by v6 state.

---

## 3. `dashboard/nav_helpers.py` — v6-only nav/synced helpers

Replace the **entire** file with this v6-only implementation:

```python
# dashboard/nav_helpers.py — v6-only helpers

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
SYNCED_STATE_PATH = Path(os.getenv("SYNCED_STATE_PATH") or (STATE_DIR / "synced_state.json"))


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _to_epoch_seconds(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        if val > 1e12:
            val /= 1000.0
        return val
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        # try int/float first
        try:
            if txt.isdigit():
                return _to_epoch_seconds(float(txt))
        except Exception:
            pass
        # ISO-ish timestamp
        try:
            if txt.endswith("Z"):
                txt = txt[:-1] + "+00:00"
            return datetime.fromisoformat(txt).astimezone(timezone.utc).timestamp()
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def load_nav_state() -> Tuple[Dict[str, Any], str]:
    """
    v6: single canonical NAV state file: logs/state/nav_state.json
    Returns (payload, source_name).
    """
    payload = _load_json(NAV_STATE_PATH)
    return payload, NAV_STATE_PATH.name


def load_synced_state() -> Dict[str, Any]:
    """
    v6: synced_state.json mirrors executor state (nav, positions, caps).
    """
    return _load_json(SYNCED_STATE_PATH)


def nav_state_age_seconds(nav_state: Dict[str, Any]) -> Optional[float]:
    """
    Compute age of the nav_state snapshot in seconds, based on the freshest
    timestamp we can find in the document.
    """
    if not isinstance(nav_state, dict) or not nav_state:
        return None

    candidates: List[Any] = []

    # top-level updated_at/ts first
    for key in ("updated_at", "ts", "updated_ts"):
        if key in nav_state:
            candidates.append(nav_state.get(key))

    # then latest series entry if present
    series = nav_state.get("series")
    if isinstance(series, list) and series:
        for entry in reversed(series):
            if not isinstance(entry, dict):
                continue
            for key in ("t", "ts"):
                if key in entry:
                    candidates.append(entry.get(key))
            break

    now = time.time()
    for raw in candidates:
        ts_val = _to_epoch_seconds(raw)
        if ts_val is not None:
            return max(0.0, now - float(ts_val))

    return None


def signal_attempts_summary(lines: List[str]) -> str:
    """
    Compact screener tail summary for the Signals tab.

    We deliberately keep this logic simple and robust:
    it just counts occurrences of key tags rather than
    trying to fully parse log formats.
    """
    if not lines:
        return "No screener attempts recorded yet."

    attempted = 0
    emitted = 0
    submitted = 0

    for line in lines:
        if "attempted=" in line:
            attempted += 1
        if "emitted=" in line:
            emitted += 1
        if "submitted=" in line:
            submitted += 1

    parts: List[str] = []
    if attempted:
        parts.append(f"attempt lines={attempted}")
    if emitted:
        parts.append(f"emitted lines={emitted}")
    if submitted:
        parts.append(f"submitted lines={submitted}")

    if not parts:
        return f"{len(lines)} screener log lines."

    return " · ".join(parts)
```

This gives us:

* v6-only `load_nav_state()` that always reads `nav_state.json`
* v6-only `load_synced_state()`
* a generic but stable `nav_state_age_seconds`
* a simple, resilient `signal_attempts_summary` used in the Signals tab

---

## 4. `dashboard/intel_panel.py` — clamp scores/expectancy to [0, 1]

Replace `dashboard/intel_panel.py` with:

```python
# dashboard/intel_panel.py — v6 intel view

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def _clamp_01(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def render_intel_panel(
    expectancy_v6: Dict[str, Any],
    symbol_scores_v6: Dict[str, Any],
    risk_allocator_v6: Dict[str, Any],
) -> None:
    """
    v6 Intel panel:

    - symbol_scores_v6["symbols"]: list of {symbol, score, components}
    - expectancy_v6["symbols"]:    map symbol -> {expectancy, hit_rate, max_drawdown}
    - risk_allocator_v6:           v6 allocator suggestions (optional, schema tolerant)

    All scores/expectancy are normalized/clamped into [0, 1] for display.
    """
    st.subheader("Intel v6 — Symbol Scores & Expectancy")

    scores_list = symbol_scores_v6.get("symbols") if isinstance(symbol_scores_v6, dict) else None
    if not isinstance(scores_list, list):
        scores_list = []

    exp_map = expectancy_v6.get("symbols") if isinstance(expectancy_v6, dict) else {}
    if not isinstance(exp_map, dict):
        exp_map = {}

    rows: List[Dict[str, Any]] = []
    for entry in scores_list:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        sym = str(symbol).upper()

        exp_entry = exp_map.get(sym) if isinstance(exp_map.get(sym), dict) else {}
        expectancy_raw = (exp_entry or {}).get("expectancy")
        hit_raw = (exp_entry or {}).get("hit_rate")
        dd_raw = (exp_entry or {}).get("max_drawdown")

        rows.append(
            {
                "symbol": sym,
                "score_raw": _safe_float(entry.get("score")),
                "score": _clamp_01(entry.get("score")),
                "expectancy_raw": _safe_float(expectancy_raw),
                "expectancy": _clamp_01(expectancy_raw),
                "hit_rate_raw": _safe_float(hit_raw),
                "hit_rate": _clamp_01(hit_raw),
                "max_drawdown": _safe_float(dd_raw),
            }
        )

    if not rows:
        st.info("No v6 intel snapshots yet — wait for INTEL_V6 to publish.")
        return

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True, ignore_index=True)

    # Display the normalized scores but keep raw columns for debugging
    display_cols = [
        "symbol",
        "score",
        "expectancy",
        "hit_rate",
        "max_drawdown",
        "score_raw",
        "expectancy_raw",
        "hit_rate_raw",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.markdown("### Top Symbols (normalized to 0–1)")
    st.dataframe(df[display_cols].head(50), use_container_width=True, height=360)

    # Allocator suggestions (schema-tolerant)
    st.markdown("---")
    st.subheader("Risk Allocator v6 — Suggestions")

    if not isinstance(risk_allocator_v6, dict) or not risk_allocator_v6:
        st.caption("No v6 allocator suggestions yet.")
        return

    alloc_symbols = risk_allocator_v6.get("symbols")
    allocations = risk_allocator_v6.get("allocations")
    generated_ts = risk_allocator_v6.get("generated_ts") or risk_allocator_v6.get("ts")

    if isinstance(alloc_symbols, list):
        alloc_rows: List[Dict[str, Any]] = []
        for entry in alloc_symbols:
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("symbol") or "").upper()
            if not sym:
                continue
            row = {"symbol": sym}
            for key in ("weight", "target_notional", "max_exposure_pct", "dd_state"):
                if key in entry:
                    row[key] = entry.get(key)
            alloc_rows.append(row)
        if alloc_rows:
            st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, height=260)
        else:
            st.json(risk_allocator_v6)
    elif isinstance(allocations, dict):
        st.json(allocations)
    else:
        st.json(risk_allocator_v6)

    if generated_ts:
        st.caption(f"Allocator snapshot ts={generated_ts}")
```

This keeps everything schema-tolerant, but **all scores and expectancy that are conceptually 0–1 are now explicitly clamped into that range**.

---

## 5. `dashboard/pipeline_panel.py` — v6 parity with green / amber / red

New v6 implementation that reads the summary head + compare summary and emits proper status:

```python
# dashboard/pipeline_panel.py — v6 pipeline parity view

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _classify_parity(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map v6 compare summary into a discrete status.

    Inputs (from pipeline_v6_compare_summary.json):
      - sample_size
      - min_sample_size
      - is_warmup
      - veto_mismatch_pct
      - size_diff_stats.{mean,p50,p95}
      - sizing_diff_stats.{p50,p95,upsize_count,sample_size}
    """
    sample_size = int(summary.get("sample_size") or 0)
    min_sample_size = int(summary.get("min_sample_size") or 0)
    is_warmup = bool(summary.get("is_warmup"))
    warmup_reason = summary.get("warmup_reason")

    sizing = summary.get("sizing_diff_stats") or {}
    p95 = abs(_safe_float(sizing.get("p95")))
    upsize_count = int(sizing.get("upsize_count") or 0)

    # Warmup / insufficient sample
    if is_warmup or sample_size < max(1, min_sample_size):
        return {
            "level": "warmup",
            "label": "WARMUP",
            "color": "#9ca3af",
            "reason": warmup_reason
            or f"Shadow compare warming up (sample={sample_size}, min={min_sample_size}).",
        }

    # Hard red: any upsizes or large sizing p95
    if upsize_count > 0 or p95 > 0.10:
        return {
            "level": "red",
            "label": "RED",
            "color": "#b91c1c",
            "reason": f"Upsized trades detected (count={upsize_count}) or p95 sizing diff={p95:.3f} > 0.10.",
        }

    # Amber: moderate sizing deviation but no upsizes
    if p95 > 0.05:
        return {
            "level": "amber",
            "label": "AMBER",
            "color": "#d97706",
            "reason": f"Sizing p95={p95:.3f} in [0.05, 0.10] — check parity before enabling v6.",
        }

    # Everything within tight bounds
    return {
        "level": "green",
        "label": "GREEN",
        "color": "#16a34a",
        "reason": f"Sizing p95={p95:.3f} and no upsizes detected.",
    }


def render_pipeline_parity(
    shadow_head: Dict[str, Any],
    compare_summary: Dict[str, Any],
) -> None:
    """
    Render the v6 pipeline shadow compare status in the Overview tab.
    Called from app.py as:

        render_pipeline_parity(pipeline_shadow_head, pipeline_compare_summary)
    """
    st.subheader("Pipeline v6 — Shadow Compare")

    if not isinstance(compare_summary, dict) or not compare_summary:
        st.caption("No pipeline v6 compare summary yet.")
        return

    status = _classify_parity(compare_summary)

    color = status["color"]
    label = status["label"]
    reason = status["reason"]

    st.markdown(
        f"""
        <div style="
            padding:0.6rem 0.9rem;
            border-radius:0.75rem;
            border:1px solid rgba(15,23,42,0.08);
            background:rgba(15,23,42,0.02);
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:0.75rem;
        ">
          <div style="font-weight:600;">
             Pipeline parity status:
             <span style="
                 display:inline-flex;
                 align-items:center;
                 justify-content:center;
                 padding:0.15rem 0.6rem;
                 border-radius:999px;
                 background:{color};
                 color:#f9fafb;
                 font-size:0.85rem;
                 font-weight:700;
                 margin-left:0.35rem;
             ">{label}</span>
          </div>
          <div style="font-size:0.85rem;color:#4b5563;max-width:520px;">
             {reason}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # High-level stats table
    basics = {
        "sample_size": int(compare_summary.get("sample_size") or 0),
        "min_sample_size": int(compare_summary.get("min_sample_size") or 0),
        "veto_mismatch_pct": _safe_float(compare_summary.get("veto_mismatch_pct")),
    }
    size_diff = compare_summary.get("size_diff_stats") or {}
    sizing_diff = compare_summary.get("sizing_diff_stats") or {}
    slip_diff = compare_summary.get("slippage_diff_bps") or {}

    df_rows = [
        {
            "metric": "sample_size",
            "value": basics["sample_size"],
        },
        {
            "metric": "min_sample_size",
            "value": basics["min_sample_size"],
        },
        {
            "metric": "veto_mismatch_pct",
            "value": f"{basics['veto_mismatch_pct']:.2f}",
        },
        {
            "metric": "size_diff_p95",
            "value": f"{_safe_float(size_diff.get('p95')):.4f}",
        },
        {
            "metric": "sizing_diff_p95",
            "value": f"{_safe_float(sizing_diff.get('p95')):.4f}",
        },
        {
            "metric": "sizing_upsize_count",
            "value": int(sizing_diff.get("upsize_count") or 0),
        },
        {
            "metric": "slippage_diff_p95_bps",
            "value": f"{_safe_float(slip_diff.get('p95')):.2f}",
        },
    ]
    st.table(pd.DataFrame(df_rows))

    if shadow_head:
        st.caption("Pipeline shadow head (raw)")
        st.json(shadow_head, expanded=False)
```

This gives you:

* deterministic **green / amber / red** based on `sizing_diff_stats.p95` and `upsize_count`
* a small stats table keyed off the contract you showed

---

## 6. `dashboard/router_health.py` — include policy quality / maker-first / bias

Here’s a schema-tolerant router health module that:

* uses the optional `router` execution snapshot if provided
* merges in any v6 router policy state (`maker_first`, `bias`, `quality`, etc.) when available
* exposes everything via `RouterHealthData` + `is_empty_router_health`

```python
# dashboard/router_health.py — v6 router health + policy overlays

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from dashboard.live_helpers import (
    load_router_policy_v6,
    load_router_suggestions_v6,
)


@dataclass
class RouterHealthData:
    per_symbol: pd.DataFrame
    trades: pd.DataFrame
    pnl_curve: pd.DataFrame
    summary: Dict[str, Any]
    overlays: Dict[str, Any]


def is_empty_router_health(data: RouterHealthData) -> bool:
    if data is None:
        return True
    if not isinstance(data.summary, dict):
        return True
    if data.summary.get("count", 0) == 0 and data.per_symbol.empty and data.pnl_curve.empty:
        return True
    return False


def _to_dataframe(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, list):
        return pd.DataFrame([row for row in payload if isinstance(row, dict)])
    return pd.DataFrame()


def load_router_health(
    window: int = 300,
    snapshot: Optional[Dict[str, Any]] = None,
    trades_snapshot: Optional[Dict[str, Any]] = None,
) -> RouterHealthData:
    """
    v6 router health loader.

    - snapshot: optional cached router exec snapshot (from load_exec_snapshot("router"))
      expected keys (best-effort):
        - summary: dict
        - per_symbol: list[dict]
        - pnl_curve: list[dict] with "time"
    - trades_snapshot: optional trades mirror, unused here but kept for signature parity.
    """
    summary: Dict[str, Any] = {}
    per_symbol = pd.DataFrame()
    pnl_curve = pd.DataFrame()
    overlays: Dict[str, Any] = {}

    if isinstance(snapshot, dict):
        if isinstance(snapshot.get("summary"), dict):
            summary = dict(snapshot["summary"])
        per_symbol = _to_dataframe(snapshot.get("per_symbol"))
        pnl_curve = _to_dataframe(snapshot.get("pnl_curve"))

    # Limit pnl_curve to the most recent chunk
    if not pnl_curve.empty:
        if "time" in pnl_curve.columns:
            try:
                pnl_curve["time"] = pd.to_datetime(pnl_curve["time"], utc=True, errors="coerce")
                pnl_curve = pnl_curve.sort_values("time")
                if window > 0:
                    pnl_curve = pnl_curve.tail(window)
            except Exception:
                pnl_curve = pnl_curve.tail(window)
        else:
            pnl_curve = pnl_curve.tail(window)

    # Merge router policy quality / maker-first / bias from v6 state
    policy_state = load_router_policy_v6() or {}
    policy_symbols = policy_state.get("symbols")
    policy_df = pd.DataFrame()

    if isinstance(policy_symbols, list):
        policy_df = pd.DataFrame(
            [row for row in policy_symbols if isinstance(row, dict) and row.get("symbol")]
        )
    elif isinstance(policy_state, dict) and "per_symbol" in policy_state:
        policy_df = _to_dataframe(policy_state.get("per_symbol"))

    if not per_symbol.empty and not policy_df.empty:
        # Normalize columns
        policy_df = policy_df.copy()
        policy_df["symbol"] = policy_df["symbol"].astype(str).str.upper()
        per_symbol = per_symbol.copy()
        if "symbol" in per_symbol.columns:
            per_symbol["symbol"] = per_symbol["symbol"].astype(str).str.upper()

        merge_cols = [c for c in ("maker_first", "bias", "quality", "allocator_state") if c in policy_df.columns]
        if merge_cols:
            right = policy_df[["symbol"] + merge_cols].drop_duplicates("symbol")
            per_symbol = per_symbol.merge(right, on="symbol", how="left")

    # Router policy suggestions v6: used to expose freshness / allocator state
    suggestions = load_router_suggestions_v6()
    overlays["policy_suggestions"] = suggestions

    if isinstance(suggestions, dict):
        summary["policy_stale"] = bool(suggestions.get("stale"))
        if "generated_at" in suggestions:
            summary["policy_generated_at"] = suggestions.get("generated_at")

    # Default fields for summary if missing
    defaults = {
        "count": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "cum_pnl": 0.0,
        "fill_rate_pct": 0.0,
        "fees_total": 0.0,
        "realized_pnl": 0.0,
        "confidence_weighted_cum_pnl": 0.0,
        "rolling_sharpe_last": 0.0,
        "normalized_sharpe": 0.0,
        "volatility_scale": 1.0,
    }
    for key, val in defaults.items():
        summary.setdefault(key, val)

    return RouterHealthData(
        per_symbol=per_symbol,
        trades=_to_dataframe(trades_snapshot.get("items")) if isinstance(trades_snapshot, dict) else pd.DataFrame(),
        pnl_curve=pnl_curve,
        summary=summary,
        overlays=overlays,
    )
```

This keeps `app.py`’s existing Router Health tab working, but adds the v6 policy overlays so we can display them later if we want to enrich the UI further.
