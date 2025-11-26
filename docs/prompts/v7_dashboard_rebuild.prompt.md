# v7 Dashboard Rebuild Prompt
# Mode: Codex IDE / Codex CLI
# Branch: v7-risk-tuning

## High-level intent

The current dashboard is a v6/v6.5 holdover with bolted-on v7 patches.  
Result: lots of empty KPIs, confusing layout, and duplicated ingestion logic.

We want to **rebuild the v7 dashboard ingestion + layout** so that it is:

- **Data-correct**: driven purely by the *actual* v7 state files and KPIs being written now.
- **Simple**: a single, unified loader for all state; no legacy v6 readers.
- **Stable**: empty UI only when data truly doesn’t exist, not because of schema mismatch.
- **Minimal**: we don’t need fancy styling; just clear, correct information.

Do NOT touch execution risk/router logic; this is a **dashboard-only** rebuild.

---

## 0. Ground truth for data

Treat the **actual code & state schemas** in the repo as the only source of truth:

- `execution/state_publish.py`
- `execution/sync_state.py`
- `execution/utils/execution_health.py`
- `execution/utils/metrics.py`
- `execution/router_metrics.py`

Plus any **sample state files** available under `logs/state/` on this box (if present), e.g.:

- `logs/state/nav_state.json`
- `logs/state/positions_state.json` / `positions.json`
- `logs/state/router.json` / `router_health.json`
- `logs/state/risk_snapshot.json`
- `logs/state/kpis_v7.json`
- any AUM- or leaderboard-related state files

Use these to infer the *real* v7 schema; do not rely on old comments.

---

## 1. New data model: one unified state loader

### Create a new module

Create a new file:

- `dashboard/state_v7.py`

This module will own all state ingestion for the dashboard.

### Implement `load_all_state()`

Design a function:

```python
def load_all_state() -> Dict[str, Any]:
    """
    Load all v7 state surfaces (nav, aum, kpis, router, positions, symbols, meta)
    and return a single normalized dictionary.

    Never raises; returns sane defaults when files are missing.
    """
````

**Responsibilities:**

1. Read **nav** from (in this priority order):

   * `logs/state/nav_state.json`
   * if not present, fall back to any existing `nav.json` or other nav file produced by sync_state.

   Normalize to:

   ```python
   state["nav"] = {
       "nav_usd": float(...),
       "nav_zar": float(... or 0),
       "age_s": float(...) or None,
       "updated_at": float(...) or None,
       "dd_state": <string or None>,
       "atr_regime": <string or None>,
   }
   ```

2. Read **AUM** from the same nav payload:

   * Prefer any explicit AUM structure (e.g. `nav_payload["aum"]["assets"]`, `nav_payload["nav_detail"]["asset_breakdown"]`, etc.).
   * Build a list of slices:

   ```python
   state["aum"] = {
       "slices": [
           {"label": "Futures", "usd": ..., "zar": ...},
           {"label": "BTC", "usd": ..., "zar": ...},
           {"label": "XAUT", "usd": ..., "zar": ...},
           {"label": "USDC", "usd": ..., "zar": ...},
           ...
       ],
       "total_usd": ...,  # sum of slice.usd
       "total_zar": ...,  # sum of slice.zar
   }
   ```

   * Derive ZAR using any FX already present in nav state (e.g. `usd_zar`).

3. Read **KPIs v7** from:

   * `logs/state/kpis_v7.json` (if present)

   Normalize:

   ```python
   kpis_raw = ...  # file contents or {}

   state["kpis"] = {
       "nav": kpis_raw.get("nav", {}),
       "risk": kpis_raw.get("risk", {}),
       "router": kpis_raw.get("router", {}),
       "symbols": kpis_raw.get("symbols", {}),
   }
   ```

   Do *not* invent fields; just expose what’s really there.

4. Read **positions** from:

   * `logs/state/positions_state.json` or `logs/state/positions.json`, depending on what sync_state writes.

   Normalize each position row minimally to:

   ```python
   {
       "symbol": ...,
       "side": ...,
       "qty": float(...),
       "notional": float(...) or 0.0,
       "pnl": float(...) or 0.0,
       "notional_fmt": "12,345.67",
       "pnl_fmt": "123.45",
       "dd_state": <from symbol kpis, if available>,
       "dd_today_pct": <from symbol kpis, if available>,
       "atr_ratio": <from symbol kpis, if available>,
       "atr_regime": <from symbol kpis, if available>,
   }
   ```

   * Reuse any PnL/notional fields that already exist.
   * Add `_fmt_usd()` helper in this module to avoid duplicating formatting across app.

5. Read **router / router_health** from:

   * `logs/state/router.json` / `router_health.json` / whatever the repo uses.

   Keep this simple for v7 Overview:

   ```python
   state["router"] = {
       "quality": ...,            # e.g. "ok" / "degraded"
       "maker_fill_rate": float(...) or None,
       "fallback_ratio": float(...) or None,
       "slip_q50_bps": float(...) or None,
       # you can carry more, but at least these
   }
   ```

6. Read **meta**:

   At minimum:

   ```python
   state["meta"] = {
       "data_age_s": ... ,        # some global age or max of important ages
       "testnet": bool(...),      # if you can infer from env or state
   }
   ```

### Error-handling:

* Use small helpers like `_safe_load_json(path, default)` and return default on error.
* Never raise; always return a dict with at least:

  ```python
  {"nav": {}, "aum": {"slices": []}, "kpis": {}, "positions": [], "router": {}, "meta": {}}
  ```

---

## 2. Rewire dashboard/app.py to the new data model

In `dashboard/app.py`:

1. **Remove** direct imports of:

   * `load_nav_state`
   * `nav_state_age_seconds`
   * any other per-file readers in `nav_helpers` / `live_helpers` that pull from individual JSON files.

2. **Import** only:

   ```python
   from dashboard.state_v7 import load_all_state
   ```

3. At the start of the main app:

   ```python
   state = load_all_state()
   nav = state.get("nav", {})
   aum = state.get("aum", {})
   kpis = state.get("kpis", {})
   positions = state.get("positions", [])
   router = state.get("router", {})
   meta = state.get("meta", {})
   ```

Use only these variables downstream.

---

## 3. Simplified layout using the new state

### A. Header

Keep the existing “Hedge — Portfolio Dashboard (v7)” header.
Subheader should show:

* `NAV: {nav['nav_usd']:.2f} USD`
* `Source: nav_state.json • Age: {nav['age_s']:.1f}s` when available.

### B. v7 Risk KPIs row

Under “v7 Risk KPIs”, render 4 big stats using `kpis["risk"]` and `router`:

* Drawdown State:

  * From `kpis["risk"].get("dd_state") or "normal"`
* ATR Regime:

  * From `kpis["risk"].get("atr_regime") or "unknown"`
* Fee / PnL:

  * From `kpis["risk"].get("fee_pnl_ratio")` → unicode string like `"0.23"` or `"23%"`; if missing → `"n/a"`
* Router Quality:

  * From `router.get("quality") or "unknown"`

Below that, render a table for per-symbol risk using `kpis["kpis"].get("symbols", {})`, not by recomputing anything.

### C. Overview (v7) cards

Create a simple 3-column `st.columns([1,1,1])` block:

1. Column 1: **NAV & AUM**

   * NAV (USD/ZAR) from `nav`
   * AUM (USD/ZAR) from `aum["total_*"]`

2. Column 2: **DD / ATR**

   * `drawdown_pct = kpis["risk"].get("drawdown_pct")`
   * `atr_ratio = kpis["risk"].get("atr_ratio")`
   * Show `"normal"` / `"unknown"` when missing.

3. Column 3: **Router Fill / Fallback**

   * `maker_fill_rate = router.get("maker_fill_rate")`
   * `fallback_ratio = router.get("fallback_ratio")`
   * Display a human-readable label, e.g. `"maker 0.84 / fallback 0.16"` or `"fallback 1.00"` if that’s all we have.

### D. AUM panel

Under “AUM Breakdown (USD)”:

* Use `aum["slices"]` directly for the donut.
* Legend labels = `slice["label"]`.
* Hover should show `slice["usd"]` and `slice["zar"]`.

No more coupling to nav_helpers; the donut gets everything from `state["aum"]`.

### E. Positions panel

Below AUM:

* “Open Positions”
* DataFrame built from `positions` list.

Include columns:

* `symbol`, `side`, `qty`, `notional_fmt`, `pnl_fmt`, `dd_state`, `dd_today_pct`, `atr_ratio`, `atr_regime`.

No extra calls to execution health; just use what `load_all_state()` already injected.

---

## 4. Remove legacy ingestion helpers (keep only what’s still needed)

In:

* `dashboard/nav_helpers.py`
* `dashboard/live_helpers.py`
* other dashboard modules

1. **Delete** or stop exporting:

   * `load_nav_state`
   * `nav_state_age_seconds`
   * any `load_*_state` functions that now duplicate `load_all_state`.

2. Keep only generic utilities (e.g. formatting helpers or caching wrappers) if they are reused.

3. Ensure nothing tries to read state files directly except `state_v7.py`.

---

## 5. Testnet / environment behaviour

Ensure `load_all_state()`:

* Does NOT suppress data merely because `ENV=testnet` or similar; if the state files exist, load and show them.
* Only gate behaviour on testnet if there is a *clear, existing* convention (e.g. certain keys omitted); do **not** introduce new gating.

---

## 6. Validation

After implementing the rebuild:

1. Syntax check:

```bash
python -m py_compile dashboard/state_v7.py dashboard/app.py \
    dashboard/live_helpers.py dashboard/nav_helpers.py
```

2. Restart dashboard:

```bash
sudo supervisorctl restart hedge:dashboard
```

3. Manually verify:

* The header NAV and age match the executor logs/nav_state.
* The v7 Risk KPIs row has **real** values where state supports them.
* Overview (v7) shows:

  * correct NAV / AUM
  * correct DD / ATR when risk_snapshot has values
  * router fill / fallback numbers matching router stats
* AUM donut shows all slices (Futures, BTC, XAUT, USDC, etc) with sane proportions.
* Positions table shows non-zero `notional_fmt` and `pnl_fmt` where appropriate, plus symbol-level ATR/DD when available.
* No UI component reads from individual JSON files anymore; everything comes from `state_v7.load_all_state()`.

---

## 7. Final summary

When done, produce a short internal summary (as comments or a notes file) covering:

* New data model (`state["nav"]`, `state["aum"]`, `state["kpis"]`, `state["positions"]`, `state["router"]`, `state["meta"]`).
* Legacy helpers removed.
* Any empty fields that still appear and why (e.g. truly missing upstream data vs not yet wired).

# END OF PROMPT