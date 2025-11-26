# PATCH SCOPE: v7 Dashboard – Investor Overview & AUM Donut
#
# Files:
#   - dashboard/app.py
#   - dashboard/nav_helpers.py
#   - dashboard/dashboard_utils.py
#   - (new) dashboard/kpi_panel.py
#   - (optional) dashboard/aum_panel.py or integrate donut into nav_helpers
#
# GOAL:
#   1) Add an "Overview (v7)" page to the dashboard focused on investor-facing state.
#   2) Implement an AUM donut chart that uses the new AUM block from nav.json.
#   3) Implement a KPI panel that reads kpis_v7.json.
#   4) Keep existing technical panels (pipeline, router health, intel) intact.
#
# CONTEXT:
#   - v7 Telemetry writes:
#       logs/state/nav.json       (with "aum" block from nav.py)
#       logs/state/kpis_v7.json   (from state_publish.build_kpis_v7)
#   - sync_state.py mirrors these into the dashboard state dir similarly to nav/positions/router.
#   - dashboard/app.py currently wires Streamlit pages with panels:
#       nav_helpers.py, pipeline_panel.py, router_health.py, intel_panel.py, live_helpers.py, dashboard_utils.py.
#
# REQUIREMENTS:
#
# 1) Overview (v7) page in dashboard/app.py
#
#   - Add a new top-level page/section, e.g. "Overview (v7)" or "Investor Overview".
#   - Layout (roughly):
#       - Row 1: NAV card + AUM card + Drawdown/ATR status card
#       - Row 2: AUM donut on the left, Risk & Router KPIs on the right
#       - Row 3: Short positions table (reuse existing helpers)
#
#   - Use existing layout utilities from dashboard_utils where possible for consistent styling.
#   - Keep the existing pages (pipeline/intel/router health) accessible via sidebar or tabs.
#
# 2) AUM helpers in dashboard/nav_helpers.py
#
#   - Add a helper to load nav state with AUM:
#
#       def load_nav_with_aum(state_dir: str = None) -> dict:
#           """
#           Load nav.json from the mirrored state_dir and return the snapshot.
#           Returns {} if missing or malformed.
#           """
#
#   - Add a helper to build AUM slices for charts:
#
#       def build_aum_slices(nav_snapshot: dict) -> list[dict]:
#           """
#           Build donut slices from the nav['aum'] block.
#
#           Returns a list of:
#             { "label": <str>, "value": <float>, "qty": <float or None> }
#
#           Slices:
#             - "Futures" from aum["futures"]
#             - For each key in aum["offexchange"], use:
#                 label = symbol
#                 value = usd_value (0 if missing)
#                 qty = qty (from offexchange[symbol]["qty"])
#           """
#
#         - Handle missing aum gracefully: return empty list.
#         - Treat negative or None values as 0 for the donut.
#
#   - Provide a simple wrapper to render the donut in Streamlit (either here or in aum_panel.py).
#     The donut should be dynamic: no hard-coded symbol set; iterate over whatever keys exist
#     in aum["offexchange"].
#
# 3) KPI panel module: dashboard/kpi_panel.py
#
#   - New module with something like:
#
#       def load_kpis(state_dir: str = None) -> dict:
#           """
#           Load kpis_v7.json from the mirrored state_dir.
#           Returns {} if missing or malformed.
#           """
#
#       def render_kpis_overview(kpis: dict):
#           """
#           Render a compact KPI block for the Overview page using Streamlit.
#
#           Suggested cards/fields:
#             - NAV / AUM:
#                 nav_total = kpis.get("nav", {}).get("nav_total")
#                 aum_total = kpis.get("aum_total")
#             - Drawdown:
#                 dd_state = kpis.get("dd_state") or kpis.get("drawdown", {}).get("dd_state")
#                 dd_pct = kpis.get("drawdown", {}).get("dd_pct")
#             - Volatility:
#                 atr_regime = kpis.get("atr_regime") or kpis.get("atr", {}).get("atr_regime")
#             - Fees vs PnL:
#                 fee_pnl_ratio = kpis.get("fee_pnl_ratio") or kpis.get("fee_pnl", {}).get("fee_pnl_ratio")
#             - Router:
#                 router_quality = kpis.get("router_quality")
#                 maker_fill_ratio = kpis.get("router_stats", {}).get("maker_fill_ratio")
#                 fallback_ratio = kpis.get("router_stats", {}).get("fallback_ratio")
#           """
#
#         - Render as Streamlit metric cards or columns; keep it compact and readable.
#         - Handle missing fields by showing "–" or similar.
#
# 4) Wiring Overview in app.py
#
#   - In dashboard/app.py, add a new page option (e.g. via sidebar selectbox or tabs):
#       pages = ["Overview (v7)", "Pipeline", "Router Health", "Intel", ...]
#
#   - For "Overview (v7)":
#       - Determine state_dir using the same logic as other panels (where nav.json is read).
#       - Call:
#           nav_snapshot = nav_helpers.load_nav_with_aum(state_dir)
#           kpis = kpi_panel.load_kpis(state_dir)
#
#       - Top row:
#           - Show NAV (nav_snapshot["nav_total"]) and AUM (nav_snapshot["aum"]["total"])
#             using st.metric or equivalent.
#           - Show dd_state and atr_regime in a small status card.
#
#       - Middle row:
#           - Left: AUM donut from nav_helpers.build_aum_slices(nav_snapshot).
#           - Right: kpi_panel.render_kpis_overview(kpis).
#
#       - Bottom row:
#           - Use existing positions helper from live_helpers or nav_helpers to show a brief
#             positions table (symbol, side, qty, upnl).
#
#   - Ensure all reads are robust to missing files; the page should still render with
#     "no data" messages rather than crashing.
#
# 5) Styling and consistency
#
#   - Use existing Streamlit styling patterns (columns, st.metric, st.table).
#   - Do not introduce new dependencies beyond what dashboard already uses.
#   - Respect the v7 state contract (logs/state/*.json) as documented; do not change
#     any state schemas in this patch.
#
# TESTING / VALIDATION:
#
# 1. Import checks:
#       python -m py_compile dashboard/app.py \
#                               dashboard/nav_helpers.py \
#                               dashboard/dashboard_utils.py \
#                               dashboard/kpi_panel.py
#
# 2. Run dashboard:
#       PYTHONPATH=. BINANCE_TESTNET=1 DRY_RUN=1 streamlit run dashboard/app.py
#
# 3. With executor running (so nav.json and kpis_v7.json are fresh), open the dashboard
#    in a browser and:
#       - Navigate to "Overview (v7)".
#       - Confirm:
#           - NAV and AUM metrics render.
#           - AUM donut shows slices for Futures + BTC/XAUT/USDC (from offexchange_holdings).
#           - Risk & Router KPIs display (dd_state, atr_regime, router_quality, fee_pnl_ratio).
#           - Positions table shows 0..N active futures positions.
#
# 4. Confirm that existing pages (pipeline, router health, intel) still work as before.

# == Patch Notes == #
• - Added v7-aware NAV helpers to dashboard/nav_helpers.py, including loading nav.json with the AUM
    block and converting it into donut slices.
  - Introduced dashboard/kpi_panel.py to load kpis_v7.json and render compact KPI cards (NAV/AUM,
    drawdown/ATR, fee ratio, router stats).
  - Updated dashboard/app.py with a new “Overview (v7)” tab showing NAV/AUM metrics, AUM donut
    (Altair), KPIs, and a positions snapshot, while keeping existing tabs intact.
  - Cleaned imports and compiled all dashboard modules successfully.

  Tests: python -m py_compile dashboard/app.py dashboard/nav_helpers.py dashboard/dashboard_utils.py
  dashboard/kpi_panel.py.

  Next steps: run the dashboard (streamlit run dashboard/app.py with fresh state files) and verify
  the Overview (v7) tab shows NAV/AUM, donut slices from off-exchange holdings, KPI cards, and
  positions.