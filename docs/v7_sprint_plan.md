# GPT-Hedge — v7 Sprint Plan (v7-risk-tuning)

Branch: `v7-risk-tuning`  
Scope: Risk tuning, telemetry KPIs, dashboard polish, investor access

---

## 1. Sprint Objectives

1. **Risk Tuning v7**
   - Add richer diagnostics around risk decisions (thresholds, observations, nav context).
   - Introduce or tighten ATR/volatility regime awareness and DD-mode gating.
   - Make fee drag and router quality visible to risk sizing (even if advisory in first pass).

2. **Telemetry & KPIs**
   - Expose a **KPI block** consumable by the dashboard:
     - Sharpe / expectancy state
     - ATR regime
     - Drawdown state
     - Router KPIs (maker fill, fallback ratio, slippage)
     - Fee/PnL ratio
   - Ensure state files are **stable schemas** and contract-documented.

3. **AUM Donut & NAV Clean-up**
   - Eliminate any treasury/reserve weirdness from NAV.
   - Build a **portfolio-level AUM view** including:
     - Futures NAV
     - Spot / off-Binance assets: BTC, XAUT, USDC (values you provided)
   - AUM donut is:
     - Backed by a structured state file
     - Dynamic with hover PnL per slice.

4. **Investor-Facing Dashboard Access**
   - Harden dashboard UX with the new KPIs and AUM donut.
   - Configure NGINX + Basic Auth for investor access.
   - Keep `/healthz` and similar ops endpoints unprotected.

5. **Low-Frequency Telegram Alerts**
   - Re-enable alerts at “a few times a week” cadence.
   - Focus on:
     - 4h close summaries
     - Major DD/risk state changes
     - Major ATR regime shifts / router degradation.

---

## 2. Workstreams

### WS1 — Risk Tuning & Diagnostics

**Goals**
- Risk decisions expose a clear “why”, not just allowed/veto.
- Easily see caps, thresholds, nav freshness, symbol/tier context.

**Tasks**
- [ ] Extend `RiskDecision` (or risk detail dict) with:
  - `thresholds` (per cap, nav freshness, DD limits)
  - `observations` (nav_age_s, nav_sources_ok, symbol_gross, tier_gross, open_positions_count)
  - `gate` / `source` tags (e.g. `risk_limits`, `risk_engine_v6`, `nav_guard`).
- [ ] Ensure veto logs (`risk_vetoes.jsonl` / veto files) carry these fields.
- [ ] Add tests (or smoke checks) that assert presence of the new keys.

**Acceptance Criteria**
- Given a veto, we can reconstruct *exactly why* it fired from structured fields.
- No “silent” caps – every veto has at least one threshold/observation attached.

---

### WS2 — Telemetry KPIs & State Contract

**Goals**
- Compute and expose KPIs in a stable state file (e.g. `logs/state/kpis_v7.json` or piggy-back onto existing state where reasonable).
- Dashboard reads from this, not from ad-hoc calculations.

**Tasks**
- [ ] Define a **KPI state schema** (doc + implementation):
  - `sharpe_state` or expectancy bucket
  - `atr_regime` (low/normal/high or bucketed)
  - `dd_state` (e.g. normal / cautious / defensive)
  - Router KPIs aggregated (maker_fill, fallback ratio, slippage percentiles)
  - Fee/PnL ratio (e.g. fees paid / gross PnL over a window).
- [ ] Wire KPI computation into the existing telemetry refresh loop (executor tick, risk snapshot, or intel refresh), *without* heavy extra I/O.
- [ ] Document schema in a new v7 telemetry doc.

**Acceptance Criteria**
- KPI state file exists, is updated regularly, and dashboard can read it without extra computation.
- Schema is documented and stable (no breaking changes without doc updates).

---

### WS3 — AUM Donut & NAV Clean-up

**Goals**
- NAV used for trading and dashboard is **transparent** and **consistent**.
- AUM donut shows total “portfolio” including BTC, XAUT, USDC, and futures NAV.

**Tasks**
- [ ] Identify current NAV computation paths (futures + spot/other).
- [ ] Remove / neutralise any “treasury/reserve” hacks that distort NAV.
- [ ] Introduce a **portfolio AUM state** file, e.g. `logs/state/aum_v7.json`:
  - Per-bucket USD values:
    - `futures_nav_usd`
    - `btc_spot_usd`
    - `xaut_spot_usd`
    - `usdc_spot_usd`
  - Per-bucket PnL (if available).
- [ ] Update dashboard AUM donut to:
  - read from AUM state file
  - show dynamic slices and hover PnL.

**Acceptance Criteria**
- A single, documented AUM state file.
- Donut matches the underlying state values and updates with time.

---

### WS4 — Dashboard & NGINX Investor Access

**Goals**
- Dashboard looks “fund ready”.
- External investors can log in over HTTPS.

**Tasks**
- [ ] Add new KPI panel + AUM donut to dashboard layout.
- [ ] Ensure all reads are from documented state files (no hidden APIs).
- [ ] Create or refine `ops/nginx_site.conf` and/or deploy config:
  - Reverse proxy to dashboard port.
  - Basic auth for `/`.
  - No auth for `/healthz`.
- [ ] Document the steps to add new investor users.

**Acceptance Criteria**
- Dashboard exposes AUM donut + KPI block.
- NGINX fronted dashboard is accessible with username/password.

---

### WS5 — Telegram Alerts (Low Frequency)

**Goals**
- Useful, low-noise alerting.

**Tasks**
- [ ] Re-enable Telegram bot integration with:
  - 4h close summary pipeline, or
  - scheduled cron/stats emitter.
- [ ] Alerts include:
  - high-level NAV/DD
  - ATR regime
  - major router quality change
- [ ] Keep frequency to “a couple times a week” by design (e.g. send only on regime changes rather than every bar).

**Acceptance Criteria**
- Telegram channel receives occasional, meaningful alerts.
- No noisy bar-by-bar spam.

---

## 3. Sprint Logistics

- Duration: 1–2 weeks (depending on your availability).
- Branches:
  - `v7-risk-tuning` (base)
  - feature branches per WS if preferred (e.g. `v7-ws1-risk`, `v7-ws2-kpi`, etc.).
- Review:
  - Small, incremental PRs.
  - Codex patches focused on one WS at a time.

---
