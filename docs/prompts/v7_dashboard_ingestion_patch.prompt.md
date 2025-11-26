# v7 Dashboard Ingestion Patch Prompt
# Mode: Codex IDE / CLI
# Branch: v7-risk-tuning

Goal:
Implement the v7 dashboard ingestion fixes from the latest audit so that:

- KPIs show drawdown %, ATR ratio, router fill/fallback, fee/PnL (where available).
- AUM donut shows all portfolio slices (Futures + BTC + XAUT + USDC) rather than a single “Futures” slice.
- NAV age in KPIs is aligned with nav_state.json (age_s / updated_at).
- Positions table has notional_fmt and pnl_fmt populated and can reuse per-symbol ATR/DD metrics from telemetry instead of ad-hoc recomputation.

Use the discrepancy report + patch plan (already agreed):

1) Enrich KPI builder in `execution/state_publish.py`.
2) Fix/extend AUM ingestion in `dashboard/nav_helpers.py` (and, if needed, in `execution/sync_state.py` write path).
3) Align router KPIs between telemetry and dashboard.
4) Add safe formatting for positions notional/pnl.
5) Validate via py_compile and by inspecting `logs/state/kpis_v7.json` and `logs/state/nav_state.json`.

---

## Files in scope (primary)

Please keep changes local to these unless absolutely required:

- `execution/state_publish.py`   (KPI builder / build_kpis_v7)
- `execution/sync_state.py`      (nav_state / AUM / positions payload)
- `dashboard/nav_helpers.py`     (build_aum_slices, NAV/AUM helpers)
- `dashboard/app.py`             (KPI and positions consumption)
- `dashboard/kpi_panel.py`       (if present, KPI panel wiring)
- `dashboard/live_helpers.py`    (positions enrichment)

Do not modify:
- `execution/risk_limits.py`
- `execution/risk_engine_v6.py`
- Any router/execution core logic (beyond reading their telemetry).

---

## 1. Enrich KPI builder in execution/state_publish.py

Locate the v7 KPI builder (e.g. `build_kpis_v7` or equivalent) in `execution/state_publish.py` (the audit referenced ~lines 356–488).

### A. Router stats (fill/fallback + slip)

Currently:
- It averages maker_fill_rate and slippage but never aggregates fallback or emits a router_stats block, so “Router Fill / Fallback” in the Overview is blank.

Patch:

1. Read router stats primarily from `risk_snapshot` if present:
   - Expected schema (adjust to actual names you see):
     - `risk_snapshot["router_stats"]["maker_fill_rate"]`
     - `risk_snapshot["router_stats"]["fallback_ratio"]`
     - `risk_snapshot["router_stats"]["slip_q25_bps"]`
     - `risk_snapshot["router_stats"]["slip_q50_bps"]`
     - `risk_snapshot["router_stats"]["slip_q75_bps"]`
     - `risk_snapshot["router_stats"]["quality"]` (or similar, e.g. policy_quality).

2. If `risk_snapshot["router_stats"]` is missing:
   - Fall back to aggregating from router health / per-symbol metrics (e.g. `router_health.symbols[*].maker_fill`, `fallback_rate`, `slippage_bps`).
   - Compute simple averages for fallback_ratio and slip quartiles; do this cheaply.

3. Emit a dedicated block in the KPI payload, e.g.:

```python
kpis["router"] = {
    "maker_fill_rate": float(...),
    "fallback_ratio": float(...),
    "slip_q25_bps": float(...),
    "slip_q50_bps": float(...),
    "slip_q75_bps": float(...),
    "quality": router_quality_str_or_score,
}
````

Ensure all numeric fields are floats and handle missing data gracefully (set to `None` or 0.0, but be explicit).

### B. Drawdown % and ATR ratio

Currently:

* `logs/state/risk_snapshot.json` carries `drawdown.dd_pct` and `atr.median_ratio`, but `build_kpis_v7` ignores them so the Overview “Drawdown %” and “ATR Ratio” cards render “–”.

Patch:

1. From `risk_snapshot`, read:

   * `risk_snapshot["drawdown"]["dd_pct"]` (or equivalent key)
   * `risk_snapshot["atr"]["median_ratio"]` (or equivalent)

2. Attach them into the KPI payload, e.g.:

```python
kpis["risk"] = {
    "drawdown_pct": float(dd_pct) if dd_pct is not None else None,
    "atr_ratio": float(atr_ratio) if atr_ratio is not None else None,
    # keep any existing fields you already emit
}
```

3. If the keys are missing or invalid, leave them as `None` so the dashboard can display “–” consistently rather than crash.

### C. NAV age in KPIs

Currently:

* `build_kpis_v7` reads `nav_age_s/age_s` but mirrored payload shows `nav.nav_age_s: null` even though `nav_state.json` has `age_s=0.0`, so the KPI NAV age defaults to a stale fallback.

Patch:

1. When building KPIs, take `nav_payload` (or equivalent) and pull:

   * `age_s = nav_payload.get("age_s") or nav_payload.get("nav_age_s")`
   * `updated_at = nav_payload.get("updated_at")`

2. Ensure the KPI block has:

```python
kpis["nav"] = {
    "nav_usd": float(nav_payload.get("nav_usd") or nav_payload.get("total_equity") or 0.0),
    "age_s": float(age_s) if age_s is not None else None,
    "updated_at": float(updated_at) if updated_at is not None else None,
}
```

3. Do not treat missing age_s as stale; simply leave it `None`. The dashboard should then not mislabel it.

### D. Per-symbol ATR/DD mirrors for positions

Currently:

* Per-symbol ATR/DD metrics are emitted in `risk_snapshot["atr"]["symbols"]` and `risk_snapshot["symbols"]` (e.g. dd_today_pct), but KPIs ignore them; positions then rely on `compute_execution_health` and go blank when that fails.

Patch:

1. From `risk_snapshot`, extract a mapping:

```python
symbol_risk = {
    sym: {
        "atr_ratio": <float>,
        "dd_state": <string or enum>,
        "dd_today_pct": <float>,
    },
    ...
}
```

Use whatever actual keys exist (e.g. `atr["symbols"][sym]["ratio"]`, `symbols[sym]["dd_today_pct"]`, etc).

2. Add this map into the KPI payload:

```python
kpis["symbols"] = symbol_risk  # or a more specific key, e.g. "symbol_risk"
```

Keep it reasonably small; only include currently open or recently active symbols if the structure is large.

---

## 2. Fix / extend AUM ingestion

### A. AUM slices from nav snapshot

Currently:

* AUM donut is limited to a single “Futures” slice because `build_aum_slices` only reads `aum.futures` and `aum.offexchange` and ignores richer asset breakdown (BTC/USDC/XAUT) present in nav snapshot.

In `dashboard/nav_helpers.py`, find `build_aum_slices` (audit referenced ~lines 116–154):

1. Inspect the current expected schema for `nav_snapshot`. It likely has:

* `nav_snapshot["aum"]["futures"]["usd"]` etc, and/or
* `nav_snapshot["nav_detail"]["asset_breakdown"]` or similar asset buckets.

2. Extend `build_aum_slices`:

* If `nav_snapshot["nav_detail"]["asset_breakdown"]` (or equivalent) exists, iterate that structure to produce per-asset slices for:

  * `BTC`, `XAUT`, `USDC`, and any other non-futures assets you find.
* Always keep the current “Futures” slice as one of the slices.

Example structure you can target for the donut:

```python
[
    {"label": "Futures", "usd": ..., "zar": ...},
    {"label": "BTC", "usd": ..., "zar": ...},
    {"label": "XAUT", "usd": ..., "zar": ...},
    {"label": "USDC", "usd": ..., "zar": ...},
]
```

Use the FX rate already present in the nav snapshot to compute ZAR if needed.

3. Ensure the function handles missing breakdowns gracefully:

* If asset breakdown is missing, keep the existing “Futures-only” behavior instead of failing.

### B. Optional: normalize asset breakdown in sync_state

If you find that the only place asset breakdown lives is `nav_payload["nav_detail"]["asset_breakdown"]` inside `execution/sync_state.py`:

1. When writing `nav_state.json`, mirror or normalize a simple AUM structure:

```python
nav_payload.setdefault("aum", {})
nav_payload["aum"]["assets"] = [
    {"asset": "BTC", "usd": ..., "pnl": ...},
    {"asset": "XAUT", "usd": ..., "pnl": ...},
    {"asset": "USDC", "usd": ..., "pnl": ...},
    # keep futures under a "Futures" or similar label
]
```

2. Keep any existing `aum["futures"]`/`aum["offexchange"]` fields so you don’t break old consumers; `build_aum_slices` should prefer `aum["assets"]` (or breakdown) but fall back if missing.

---

## 3. Router KPI alignment

In `execution/state_publish.py` (same KPI builder):

* Ensure router KPIs use `risk_snapshot["router_stats"]` as the primary source, as described above.
* Router-related fields in KPI payload should cover:

  * maker fill
  * fallback ratio
  * slippage quartiles
  * router quality / policy state

In `dashboard/app.py` / `dashboard/kpi_panel.py`:

* Ensure the router KPI panel reads from the new `kpis["router"]` block instead of recomputing or leaving blanks.
* Where KPI values are used to render “Router Fill / Fallback” and slippage, wire them to the new fields:

  * `kpis["router"]["maker_fill_rate"]`
  * `kpis["router"]["fallback_ratio"]`
  * `kpis["router"]["slip_q50_bps"]` (and others as needed)

Handle missing values with a clean `"–"` rather than errors.

---

## 4. Positions formatting and per-symbol risk

In `execution/sync_state.py`, locate the positions payload assembly (audit referenced ~lines 1329–1434, where nav/positions/leaderboard are mirrored).

Patch:

1. Ensure each position dict has numeric `notional` and `pnl` fields populated (if they aren’t already).
2. Add safe formatting helpers:

```python
def _fmt_usd(v: float) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "-"
```

3. For each position, attach:

```python
pos["notional_fmt"] = _fmt_usd(pos.get("notional", 0.0))
pos["pnl_fmt"] = _fmt_usd(pos.get("pnl", 0.0))
```

4. Also, if `kpis["symbols"]` (from step 1D) exists, enrich positions with symbol-level ATR/DD:

* For symbol `sym`, look up `symbol_risk.get(sym, {})` and set:

```python
pos["atr_ratio"] = symbol_risk[sym].get("atr_ratio")
pos["dd_today_pct"] = symbol_risk[sym].get("dd_today_pct")
pos["dd_state"] = symbol_risk[sym].get("dd_state")
```

5. In `dashboard/app.py` / positions table rendering, use these fields directly (no need to recompute ATR/DD there). If the front-end already expects them, you may only need to ensure they are present.

---

## 5. Dashboard wiring for KPIs and AUM

In `dashboard/app.py` and `dashboard/kpi_panel.py` (or wherever the v7 KPI panel is rendered):

1. Ensure KPI panel reads:

* Drawdown % → `kpis["risk"]["drawdown_pct"]`
* ATR Ratio → `kpis["risk"]["atr_ratio"]`
* Router Fill / Fallback → `kpis["router"]["maker_fill_rate"]` / `kpis["router"]["fallback_ratio"]`
* NAV age → `kpis["nav"]["age_s"]`
* AUM summary → aggregated from `build_aum_slices(...)`

2. Where any of these values are `None`, render `"–"` or similar instead of 0.0, unless 0.0 is truly correct.

3. Ensure the AUM donut uses the extended slice output from `build_aum_slices` so BTC/XAUT/USDC appear as separate slices.

---

## 6. Validation

After applying patches:

1. Syntax check:

```bash
python -m py_compile execution/state_publish.py execution/sync_state.py \
    dashboard/nav_helpers.py dashboard/app.py
```

2. Restart services (or just dashboard + sync as needed):

```bash
sudo supervisorctl restart hedge:sync
sudo supervisorctl restart hedge:dashboard
```

3. Inspect new KPI state and nav_state:

```bash
cat logs/state/kpis_v7.json | head
cat logs/state/nav_state.json | head
```

Confirm:

* `kpis_v7.json` now has:

  * nav.age_s
  * risk.drawdown_pct
  * risk.atr_ratio
  * router.maker_fill_rate, router.fallback_ratio, router.slip_q* fields
  * symbols map (per-symbol ATR/DD) if included
* `nav_state.json` still has `age_s`, `updated_at`, and any new AUM structure you added.

4. Open dashboard and confirm:

* NAV and AUM cards align with executor.
* AUM donut shows BTC/XAUT/USDC slices.
* KPI panel shows Drawdown %, ATR Ratio, Router Fill/Fallback.
* Positions table shows notional_fmt and pnl_fmt, and ATR/DD data where available.

5. Report back a short summary of changes + any follow-ups you think we should tackle next (e.g. symbol-level KPI views, more detailed router charts).

---

# END OF PROMPT

```
```
# ---------------------------------------------------------------------------
# 4) ADDITIONAL DASHBOARD PATCH NOTES