# PATCH SCOPE: Dashboard v7 Overhaul (Investor View + FX-aware AUM)

# Files:
#   - execution/state_publish.py
#   - dashboard/nav_helpers.py
#   - dashboard/kpi_panel.py
#   - dashboard/app.py
#
# GOAL:
#   1) Keep "Overview (v7)" as the primary investor page.
#   2) Add USD→ZAR FX awareness for NAV/AUM (including tooltips on the AUM donut).
#   3) Surface more v7 KPIs (DD%, ATR, router stats) in a compact way.
#   4) Add "last updated" + data age indicators and stale-data warning.
#   5) Leave legacy v6 tabs intact but clearly secondary.

# ---------------------------------------------------------------------------
# 1) STATE PUBLISH: add FX (usd→zar) into kpis_v7.json
#
# In execution/state_publish.py, where build_kpis_v7(...) constructs the KPI
# payload that is written to logs/state/kpis_v7.json:
#
#   - Extend the kpis dict to include an "fx" block, using the current
#     usd→zar rate already available in the runtime (coingecko / price utils).
#
#   Example target shape (do not hard-code values):
#
#     kpis["fx"] = {
#         "usd_zar": usd_zar_rate,   # float or None
#         "ts": now_ts_iso_or_epoch, # consistent with other timestamps
#     }
#
#   - Reuse any existing usd→zar price source in the repo (e.g. exchange_utils
#     or a cached coingecko price). If no FX value is available, set the field
#     to None but still include the "fx" block.
#
#   - Do NOT change the structure of existing kpi fields. Only add "fx".
#
# ---------------------------------------------------------------------------
# 2) NAV HELPERS: AUM donut slices with optional ZAR and snapshot age
#
# In dashboard/nav_helpers.py:
#
#   a) Extend build_aum_slices(nav_snapshot: dict) so it can optionally
#      include ZAR-valued slices when the FX rate is available.
#
#      - Either:
#          def build_aum_slices(nav_snapshot: Dict[str, Any], usd_zar: float | None = None) -> List[Dict[str, Any]]:
#        or read usd_zar from an injected kpis payload if that’s cleaner.
#
#      - For each slice (including "Futures" and each symbol in aum["offexchange"]):
#          - value → USD value (existing behaviour)
#          - qty   → quantity (existing behaviour)
#          - NEW: "zar" → value * usd_zar if usd_zar is not None, else None
#
#      - Keep the default behaviour (no FX) working by making usd_zar optional.
#
#   b) Add a generic snapshot age helper:
#
#         def snapshot_age_seconds(payload: Dict[str, Any]) -> Optional[float]:
#             """
#             Compute age of a snapshot (nav.json or kpis_v7.json) in seconds,
#             based on 'ts' / 'updated_ts' / similar fields.
#             """
#
#      - Reuse the existing _to_epoch_seconds logic and the pattern from
#        nav_state_age_seconds, but make it generic enough for nav.json / kpis_v7.json.
#
#   c) Export new helpers in __all__ if needed so app.py can import them.
#
# ---------------------------------------------------------------------------
# 3) KPI PANEL: expose FX and richer KPIs to Overview
#
# In dashboard/kpi_panel.py:
#
#   a) In load_kpis(), after loading kpis_v7.json:
#
#      - Return the full dict (unchanged), but ensure the "fx" block is preserved.
#      - No breaking changes needed; we just want kpi_panel.render_* functions to
#        be able to access kpis.get("fx", {}).
#
#   b) In render_kpis_overview(kpis: dict):
#
#      - Keep existing metrics:
#          NAV, AUM, Fee/PnL, Drawdown State, ATR Regime, Router Quality,
#          Maker Fill, Fallback Rate.
#
#      - Add a small FX display if available:
#
#          fx = kpis.get("fx") or {}
#          usd_zar = fx.get("usd_zar")
#          if usd_zar is not None:
#              st.caption(f"FX: 1 USD ≈ {usd_zar:.2f} ZAR")
#
#      - Optionally, derive and show drawdown % (if present):
#
#          dd_pct = (kpis.get("drawdown") or {}).get("dd_pct")
#          # This can remain in the existing Drawdown State metric or be
#          # included in the caption/label (e.g. "cautious (0.11%)").
#
#      - Keep the panel compact; do not add new columns beyond a small caption
#        or at most one extra metric card if it remains readable.
#
# ---------------------------------------------------------------------------
# 4) DASHBOARD APP: Overview (v7) enhancements
#
# In dashboard/app.py, in main():
#
#   a) Ensure we only load kpis_v7 once at the top of main():
#
#       kpis_v7 = load_kpis_v7()   # existing call
#
#   b) Inside the "Overview (v7)" tab:
#
#       with tabs[0]:
#           st.subheader("Overview (v7)")
#           nav_v7 = load_nav_with_aum()
#           kpis_overview = kpi_panel.load_kpis()
#
#           # Extract NAV + AUM
#           nav_total = (nav_v7.get("aum") or {}).get("futures") or nav_v7.get("nav_total")
#           aum_total = (nav_v7.get("aum") or {}).get("total")
#
#           # Extract FX
#           fx = (kpis_overview.get("fx") or {}) if isinstance(kpis_overview, dict) else {}
#           usd_zar = fx.get("usd_zar")
#
#   c) Top row metrics:
#
#       - Keep 3 main cards:
#           - NAV (Futures) – still in USD
#           - AUM – still in USD
#           - DD / ATR – unchanged label; content can stay as "dd_state / atr_regime".
#
#       - Below or next to these, add a small caption that shows FX if available:
#
#           if usd_zar:
#               st.caption(f"FX: 1 USD ≈ {usd_zar:.2f} ZAR")
#
#   d) AUM donut:
#
#       - When calling build_aum_slices(nav_v7), pass usd_zar if the function
#         signature was updated:
#
#           slices = build_aum_slices(nav_v7, usd_zar=usd_zar)
#
#       - For the Altair chart, extend the tooltip to include ZAR:
#
#           tooltip=["label", "value", "qty", "zar"]
#
#         where "zar" comes from the slices dict.
#
#       - Optionally, rename axes / legend more clearly:
#           - Title: "AUM by Asset (USD)"
#
#   e) Data age + last updated:
#
#       - Use snapshot_age_seconds(nav_v7) and/or snapshot_age_seconds(kpis_overview)
#         from nav_helpers to compute freshness:
#
#           age_nav = snapshot_age_seconds(nav_v7) or None
#           age_kpis = snapshot_age_seconds(kpis_overview) or None
#           age = age_kpis or age_nav
#
#       - Show a small freshness badge near the top:
#
#           if age is not None:
#               if age < 60:
#                   st.success(f"Data age: {age:.0f}s")
#               elif age < 300:
#                   st.warning(f"Data age: {age:.0f}s")
#               else:
#                   st.error(f"Data age: {age:.0f}s (stale)")
#
#       - Also show "Last updated" if there is a timestamp in kpis_overview["ts"]
#         or nav_v7["ts"]:
#
#           ts = kpis_overview.get("ts") or nav_v7.get("ts")
#           # Convert to human-readable UTC string and show as caption.
#
#   f) Positions snapshot:
#
#       - The current code already loads positions_state.json into pos_doc/pos_source,
#         and then in the Overview (v7) tab uses "items = pos_doc.get('items')".
#
#       - Keep that behaviour, but make sure:
#           * If pos_doc is not a dict, fallback to [].
#           * The table uses symbol, side, qty, pnl, notional in that order when present.
#
#   g) Legacy/static tabs:
#
#       - Do NOT remove any v6-specific tabs (Intel, Pipeline, Router, Positions),
#         but:
#           * Ensure the Overview (v7) tab is first and clearly labeled.
#           * Optionally add a small caption at the top of each legacy tab:
#
#               st.caption("Legacy v6 telemetry — for internal/ops use.")
#
#         This keeps them available without confusing investors.
#
# ---------------------------------------------------------------------------
# 5) TESTING / VALIDATION
#
# After Codex applies the patch:
#
#   1) Compile:
#        python -m py_compile \
#          execution/state_publish.py \
#          dashboard/nav_helpers.py \
#          dashboard/kpi_panel.py \
#          dashboard/app.py
#
#   2) Run executor + sync_state:
#        PYTHONPATH=. ENV=prod ALLOW_PROD_SYNC=1 \
#          python execution/executor_live.py   # or via supervisor in testnet/dry_run
#
#   3) Run dashboard:
#        PYTHONPATH=. streamlit run dashboard/app.py
#
#   4) In the browser:
#        - Navigate to the "Overview (v7)" tab.
#        - Confirm:
#            * NAV (Futures) and AUM display in USD.
#            * FX caption appears (1 USD ≈ X ZAR) once usd_zar is available.
#            * AUM donut shows slices for "Futures", "BTC", "XAUT", "USDC"
#              with tooltips showing USD, qty, and ZAR.
#            * KPI panel shows NAV, AUM, Fee/PnL, Drawdown, ATR, Router stats as before.
#            * Data age badge reflects freshness correctly (green <60s, yellow <300s,
#              red if older).
#            * "Last updated" timestamp is shown in UTC.
#            * Positions table shows any open positions from positions_state.json.
#        - Kill executor temporarily:
#            * Verify that, after a short delay, the data age warning turns
#              yellow/red to signal staleness.
#
#   5) Verify that legacy tabs still load and render, with the Overview (v7) tab
#      as the clear, primary investor-facing view.
