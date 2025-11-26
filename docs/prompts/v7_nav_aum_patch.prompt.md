# ✅ **v7_nav_aum_patch.prompt.md**

Paste this into Codex IDE / CLI.

````markdown
# v7 NAV + AUM Fix Patch
# Mode: Codex IDE / CLI
# Branch: v7-risk-tuning

Goal:
Fix stale NAV + missing AUM slices by:
1) Splitting nav ingestion into two streams:
   - nav_state.json → NAV, DD, series, age_s, peak
   - nav.json → AUM, portfolio asset breakdown, FX
2) Rebuilding AUM logic in state_v7.py to read nav.json
3) Ensuring dashboard uses these correct v7 surfaces
4) Surfacing DD%, ATR, fees, and router metrics (when present)
5) Guarantee accurate NAV ≈ 11,107.30 USD on testnet futures

Do not modify execution risk logic or router logic.  
Dashboard + state_v7 only.

---------------------------------------------------------
## 1. Modify dashboard/state_v7.py
---------------------------------------------------------

### A. Add loader for nav.json

Create:

```python
NAV_DETAIL_PATH = STATE_DIR / "nav.json"

def load_nav_detail(default=None):
    try:
        return _load_state_json(NAV_DETAIL_PATH, default or {}) or (default or {})
    except Exception:
        return default or {}
````

### B. Fix load_all_state():

Replace any existing AUM logic with:

```python
nav_state = load_nav_state()
nav_detail = load_nav_detail()

state["nav"] = {
    "nav_usd": float(nav_state.get("total_equity") or nav_state.get("nav") or 0.0),
    "peak_equity": float(nav_state.get("peak_equity") or 0.0),
    "drawdown_pct": float(nav_state.get("drawdown_pct") or 0.0),
    "drawdown_abs": float(nav_state.get("drawdown_abs") or 0.0),
    "age_s": float(nav_state.get("age_s") or 0.0),
    "updated_at": float(nav_state.get("updated_at") or 0.0),
}
```

### C. Build AUM using nav_detail

Look inside nav.json for one of:

* `nav_detail.asset_breakdown`
* `aum.assets`
* `aum.offexchange`
* or any per-asset keys found in that file

Implement:

```python
aum_slices = []

# 1. futures nav
futures_nav = float(nav_state.get("total_equity") or 0.0)
aum_slices.append({"label": "Futures", "usd": futures_nav})

# 2. off-exchange assets from nav.json
assets = None

if "nav_detail" in nav_detail:
    ad = nav_detail["nav_detail"]
    if "asset_breakdown" in ad:
        assets = ad["asset_breakdown"]

if assets:
    for sym, row in assets.items():
        usd_val = float(row.get("usd") or row.get("value") or 0.0)
        aum_slices.append({"label": sym.upper(), "usd": usd_val})
```

### D. Compute ZAR using whatever FX exists

Inside the same function:

```python
usd_zar = float(
    nav_detail.get("fx", {}).get("usd_zar")
    or nav_state.get("fx_usd_zar")
    or nav_state.get("usd_zar")
    or 18.0   # fallback but never crash
)

for s in aum_slices:
    s["zar"] = round(s["usd"] * usd_zar, 2)

state["aum"] = {
    "slices": aum_slices,
    "total_usd": sum(s["usd"] for s in aum_slices),
    "total_zar": sum(s["zar"] for s in aum_slices),
}
```

This gives BTC / USDC / XAUT slices as long as nav.json contains their values.

### E. Merge KPIs from kpis_v7.json

Fix any missing merges:

```python
kpis_raw = load_kpis_v7({})
state["kpis"] = {
    "nav": kpis_raw.get("nav", {}),
    "risk": kpis_raw.get("risk", {}),
    "router": kpis_raw.get("router", {}),
    "symbols": kpis_raw.get("symbols", {}),
}
```

If any of these are missing, leave them empty dicts (dashboard will show “n/a”).

### F. Correct stale NAV issue

Ensure `state["nav"]["nav_usd"]` **is NOT taken from nav.json**.
Use only nav_state.json (freshest file written by sync_state).
This prevents stale NAV (your reported NAV ≈ 11,107.30).

---

## 2. Modify dashboard/app.py to use new AUM and NAV

---

### A. Overview NAV card

Replace any use of `nav_detail["nav"]` with:

```python
nav_usd = state["nav"]["nav_usd"]
nav_age_s = state["nav"]["age_s"]
```

### B. AUM donut

Replace donut source with:

```python
slices = state["aum"]["slices"]
```

Ensure label/values come from these slices.

### C. Disable legacy AUM helpers

Remove any call to:

* build_aum_slices
* nav_helpers.build_aum_slices
* offexchange / futures v6 models

Everything must come from `state["aum"]`.

---

## 3. Fix KPI panels to use actual v7 fields

---

### A. Risk KPIs:

```python
risk = state["kpis"].get("risk", {})
dd_state = risk.get("dd_state") or "normal"
atr_regime = risk.get("atr_regime") or "unknown"
fee_pnl = risk.get("fee_pnl_ratio") or "n/a"
```

### B. Drawdown % and ATR ratio in overview:

```python
dd_pct = risk.get("drawdown_pct")
atr_ratio = risk.get("atr_ratio")
```

### C. Router KPIs:

```python
router = state["kpis"].get("router", {})
maker_fill = router.get("maker_fill_rate")
fallback = router.get("fallback_ratio")
slip_q50 = router.get("slip_q50_bps")
```

Display:
`maker_fill / fallback` or `"unknown"` if missing.

---

## 4. Positions table: use v7 symbol KPIs

---

In app.py:

```python
symbols = state["kpis"].get("symbols", {})

for pos in positions:
    sy = pos["symbol"]
    sk = symbols.get(sy, {})
    pos["dd_state"] = sk.get("dd_state")
    pos["dd_today_pct"] = sk.get("dd_today_pct")
    pos["atr_ratio"] = sk.get("atr_ratio")
    pos["atr_regime"] = sk.get("atr_regime")
```

Then format notional/pnl:

```python
pos["notional_fmt"] = f"{pos.get('notional',0):,.2f}"
pos["pnl_fmt"] = f"{pos.get('pnl',0):,.2f}"
```

---

## 5. Validation checklist

---

After patch:

1. Run:

```
python -m py_compile dashboard/state_v7.py dashboard/app.py
```

2. Restart dashboard:

```
sudo supervisorctl restart hedge:dashboard
```

3. Confirm in UI:

* NAV now matches ≈ 11,107.30 USD
* NAV age ~0–1s
* AUM donut shows BTC / XAUT / USDC / Futures
* KPI fields populated where telemetry exists
* Positions table shows PnL and ATR/DD if present
* Router fill/fallback values show correctly

# END OF PATCH
````