## `v7_kpi_layout_fix.prompt.md`

````markdown
# v7 KPI + Layout Fix Patch
# Mode: Codex IDE / CLI
# Branch: v7-risk-tuning

Goal:
1. Wire the dashboard KPIs to the **v7 KPI state** (`logs/state/kpis_v7.json`) so cards are populated:
   - Drawdown %
   - ATR Ratio
   - Router Fill / Fallback
   - Fee / PnL (if available)
   - Per-symbol ATR / DD in tables and positions.
2. Keep the new AUM donut slices, but **clean up the layout** so:
   - Overview + AUM donut + Router KPIs are laid out clearly in one section.
   - The donut doesn’t feel “detached” or misaligned versus the KPI cards.

We already:
- Have `logs/state/kpis_v7.json` written by `execution/state_publish.py` (build_kpis_v7) with nav/risk/router/symbols blocks.
- Have `nav_state.json` and AUM slices working.
- See the AUM donut + router fallback in the UI, but most KPI cards are still “unknown/n/a/–”.

Work only in the **dashboard** and read-only in execution; do not change risk or router logic.

---

## 1. KPI ingestion: add a single v7 loader + plumb it through

### Files to touch

- `dashboard/live_helpers.py`
- `dashboard/app.py`
- `dashboard/kpi_panel.py` (if present)
- (optionally) `dashboard/dashboard_utils.py` if you want helper functions there

### A. Add a canonical KPI_v7 loader

In `dashboard/live_helpers.py`:

1. Near the top where `STATE_DIR` and paths are defined, add:

```python
KPI_V7_STATE_PATH = Path(
    os.getenv("KPI_V7_STATE_PATH") or (STATE_DIR / "kpis_v7.json")
)
````

2. Implement a safe loader:

```python
def load_kpis_v7(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return the v7 KPI snapshot from logs/state/kpis_v7.json; never raises.
    """
    try:
        return _load_state_json(KPI_V7_STATE_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}
```

Re-use whatever `_load_state_json` helper already exists in this module.

3. Export it in `__all__` / exported names list if such a list exists at the bottom (e.g. the list that includes `"load_nav_state"`, `"load_positions_state"`, etc.).

### B. Replace any ad-hoc KPI loading

Search in `dashboard/app.py` and `dashboard/kpi_panel.py` for any code that:

* reads **risk_snapshot.json**, **router_health.json**, or
* manually opens `kpis_v7.json`.

Replace that logic with a call to `load_kpis_v7()` from `live_helpers`.

Example in `dashboard/app.py`:

```python
from dashboard.live_helpers import ..., load_kpis_v7
...
kpis_v7 = load_kpis_v7({})
```

Use this **one object** (`kpis_v7`) for all top-level v7 KPI cards.

---

## 2. Map KPI fields correctly into the UI

Using the schema written by `build_kpis_v7` (nav / risk / router / symbols), wire up the dashboard.

In `dashboard/app.py` or `dashboard/kpi_panel.py`:

### A. Top “v7 Risk KPIs” row

For the four headline fields at the top:

* **Drawdown State**

  * use `kpis_v7.get("risk", {}).get("dd_state")` if present
  * fallback to `"normal"` when risk block exists but dd_state missing.

* **ATR Regime**

  * use `kpis_v7.get("risk", {}).get("atr_regime")`
  * fallback to `"unknown"`.

* **Fee / PnL**

  * use `kpis_v7.get("risk", {}).get("fee_pnl_ratio")`
  * if non-null, format as e.g. `"0.23"` or `"23%"` depending on contract; if null, show `"n/a"`.

* **Router Quality**

  * use `kpis_v7.get("router", {}).get("quality")`
  * if missing, show `"unknown"`.

Make sure these no longer derive from `risk_snapshot.json` directly; **v7 KPIs are the contract**.

### B. Per-symbol ATR regime table

The table that currently shows:

* symbol
* dd_state
* dd_today_pct
* atr_ratio
* atr_regime (currently “unknown” / `None`)

Use:

```python
symbol_kpis = (kpis_v7.get("symbols") or {})  # mapping: symbol -> dict
```

Then for each symbol row:

```python
row["dd_state"] = symbol_kpis.get(sym, {}).get("dd_state") or row.get("dd_state")
row["dd_today_pct"] = symbol_kpis.get(sym, {}).get("dd_today_pct", row.get("dd_today_pct"))
row["atr_ratio"] = symbol_kpis.get(sym, {}).get("atr_ratio", row.get("atr_ratio"))
row["atr_regime"] = symbol_kpis.get(sym, {}).get("atr_regime", row.get("atr_regime"))
```

Do NOT crash if `kpis_v7["symbols"]` is missing; just keep legacy values.

### C. Overview KPI cards (NAV, AUM, DD/ATR + Router)

In the “Overview (v7)” row where we show:

* NAV (USD)
* AUM (USD)
* DD / ATR
* Router Fill / Fallback

Wire:

* **NAV (USD)** → from `nav_state` as today (unchanged).

* **AUM (USD)** → from nav_state / AUM slices as today (unchanged).

* **DD / ATR** (single card) → display:

  ```python
  dd_pct = kpis_v7.get("risk", {}).get("drawdown_pct")
  atr_ratio = kpis_v7.get("risk", {}).get("atr_ratio")
  dd_label = f"{dd_pct:.2f}%" if dd_pct is not None else "normal"
  atr_label = f"{atr_ratio:.2f}" if atr_ratio is not None else "unknown"
  st.metric("DD / ATR", f"{dd_label} / {atr_label}")
  ```

* **Router Fill / Fallback** card:

  ```python
  router_block = kpis_v7.get("router", {}) or {}
  maker_fill = router_block.get("maker_fill_rate")
  fallback_ratio = router_block.get("fallback_ratio")
  label = ""
  if maker_fill is not None:
      label += f"maker {maker_fill:.2f}"
  if fallback_ratio is not None:
      label += f" fallback {fallback_ratio:.2f}" if label else f"fallback {fallback_ratio:.2f}"
  if not label:
      label = "unknown"
  st.metric("Router Fill / Fallback", label)
  ```

This removes the current “fallback 1.00” only display and shows both when available.

### D. KPI tile block (bottom-right small KPIs)

In whatever panel shows mini KPIs (NAV / AUM / Drawdown / Fee / Router):

* Use `kpis_v7["nav"]`, `kpis_v7["risk"]`, `kpis_v7["router"]` for these values.
* If `age_s` is in `kpis_v7["nav"]`, show it as “NAV Age: Xs”.

---

## 3. Positions table: reuse symbol KPIs and format notional/pnl

In `dashboard/live_helpers.py` and/or `dashboard/app.py` (where positions table is built):

1. Make sure positions ingest `kpis_v7` (or just symbol map):

```python
kpis = load_kpis_v7({})
symbol_kpis = kpis.get("symbols") or {}
```

2. For each position row `pos`:

* Add or overwrite:

```python
sk = symbol_kpis.get(pos["symbol"], {})
pos.setdefault("dd_state", sk.get("dd_state"))
pos.setdefault("dd_today_pct", sk.get("dd_today_pct"))
pos.setdefault("atr_ratio", sk.get("atr_ratio"))
pos.setdefault("atr_regime", sk.get("atr_regime"))
```

3. Ensure `notional_fmt` / `pnl_fmt` are derived from numeric fields:

If sync_state already writes numeric `notional`/`pnl`, then:

```python
def _fmt_usd(v):
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "0.00"

pos["notional_fmt"] = pos.get("notional_fmt") or _fmt_usd(pos.get("notional", 0.0))
pos["pnl_fmt"] = pos.get("pnl_fmt") or _fmt_usd(pos.get("pnl", 0.0))
```

This ensures the positions table in the bottom section gets proper formatted strings even when older fields are missing.

---

## 4. Layout tweaks (minimal but clearer)

In `dashboard/app.py`, focus only on the Overview + AUM + Router block.
Do **not** radically redesign; just put KPIs and donut into a sensible 2-row, 2-column structure.

Example layout (Streamlit-ish):

```python
# v7 Risk KPIs (keep as-is)

st.markdown("### Overview (v7)")
col_nav, col_dd, col_router = st.columns([1, 1, 1])
with col_nav:
    # NAV (USD/ZAR) + AUM (USD/ZAR)
with col_dd:
    # DD / ATR
with col_router:
    # Router Fill / Fallback + maybe NAV age

st.markdown("### AUM Breakdown (USD)")
col_donut, col_positions = st.columns([2, 2])
with col_donut:
    # donut chart + legend
with col_positions:
    # AUM KPIs & perhaps a small summary table, or leave empty for now

st.markdown("### Open Positions")
# positions table
```

Key points for Codex:

* Keep the existing section headings (`“Overview (v7)”`, `“AUM Breakdown (USD)”`, `“Open Positions”`).
* Ensure the donut chart stays within its column so it doesn’t crowd the rest of the metrics.
* Avoid adding large empty spacers that force the donut to a new “page”.

---

## 5. Validation

After applying the patch:

1. Syntax:

```bash
python -m py_compile dashboard/live_helpers.py dashboard/app.py dashboard/kpi_panel.py
```

(omit kpi_panel if it doesn’t exist).

2. Restart dashboard:

```bash
sudo supervisorctl restart hedge:dashboard
```

3. Verify in the UI:

* Top “v7 Risk KPIs” shows:

  * Drawdown State: normal / etc.
  * ATR Regime: not “unknown” once ATR state is present.
  * Fee / PnL: some number or “n/a” if truly not computable.
  * Router Quality: ok/whatever is in KPIs.

* Overview (v7):

  * NAV & AUM values correct (match nav_state).
  * DD / ATR card populated from kpis_v7.
  * Router Fill / Fallback shows both maker and fallback when present.

* AUM Breakdown:

  * Donut shows BTC / ETH / FDUSD / Futures / USDC / USDT slices cleanly.

* Positions:

  * notional_fmt & pnl_fmt show non-zero where applicable.
  * dd_state / atr_ratio / atr_regime columns pull from kpis_v7 symbols.

Return a short summary of:

* Which fields are now populated.
* Any remaining blanks so we can address them in a follow-up patch if needed.

---

# END OF PROMPT
