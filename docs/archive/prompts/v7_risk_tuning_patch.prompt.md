# v7 Risk Tuning Patch Prompt
# Mode: Codex IDE / CLI
# Branch: v7-risk-tuning

You are a code-editing agent working directly on this repo.

## Goal

Implement **v7 risk tuning** and enriched risk diagnostics while keeping behaviour safe and auditable.

Do NOT reference historical versions (v5/v6). Work only from the current code.

---

## Scope

Work only in these areas unless explicitly required:

- `execution/risk_engine_v6.py`
- `execution/risk_limits.py`
- `execution/executor_live.py`
- `execution/utils/execution_health.py`
- `execution/state_publish.py`
- `execution/sync_state.py`
- `execution/intel/*` (only if needed for ATR regime / Sharpe state / DD state)
- `dashboard/*` (read-only for now; we just ensure telemetry is present)

---

## Requirements

### 1. Enriched Risk Diagnostics

1. Extend the risk decision/diagnostic path so each veto or clamp has:
   - `gate`: which sub-system made the decision (e.g. `"nav_guard"`, `"risk_limits"`, `"risk_engine_v6"`).
   - `thresholds`: dictionary of relevant caps/limits (e.g. `daily_loss_limit_pct`, `max_trade_nav_pct`, `nav_freshness_seconds`).
   - `observations`: dictionary with current observed values (e.g. `nav_age_s`, `nav_usd`, `current_gross_notional`, `tier_gross_notional`, `open_positions_count`).
2. Ensure these fields are:
   - propagated into veto logs (`risk_vetoes.jsonl` or equivalent),
   - present in any veto JSON files the executor writes for screener or risk blocks.
3. Keep the existing behaviour unless explicitly improved; we are adding visibility, not changing rules in this patch.

### 2. ATR / Volatility Regime & DD State

1. Identify how ATR or volatility metrics are computed today.
2. Add a **regime classification helper**, e.g.:
   - low / normal / high volatility based on ATR relative to a rolling baseline.
3. Attach a **`dd_state`** or **`risk_mode`** concept (if not present already) which maps current drawdown into:
   - `"normal"`, `"cautious"`, `"defensive"`.
4. Ensure these fields are exposed in:
   - risk snapshot / execution health structures,
   - any state files used by the dashboard (e.g. `risk_snapshot.json` or a dedicated KPI state).

We don’t have to auto-tune caps yet; just make the regimes and states visible and testable.

### 3. Fee/PnL Visibility (Advisory)

1. Identify where realised PnL and fees are tracked (order metrics, PnL tracker, etc).
2. Add a small helper that computes a **Fee / PnL ratio** over a recent window (e.g. last N trades or last M hours).
3. Surface this as:
   - part of a KPI block in telemetry (e.g. `kpis.fee_pnl_ratio`),
   - and/or an intel snapshot.

Do not wire this into sizing yet; phase 1 is visibility only.

### 4. Telemetry Contract

1. Decide where to place the v7 KPI state:
   - Either an extension of an existing state file (e.g. `risk_snapshot.json`), or
   - A new file (e.g. `logs/state/kpis_v7.json`).
2. Write code to produce this KPI state, containing at least:
   - `atr_regime`
   - `dd_state` or `risk_mode`
   - `fee_pnl_ratio` (if computable)
   - a summary router quality score or basic KPIs (maker fill rate, fallback ratio, slippage indicator) if cheap.
3. Ensure all new fields are written via the standard state publish helpers and follow the established writing pattern (tmp + replace, timestamps, etc).

---

## Constraints

- Preserve backwards compatibility where possible; avoid breaking dashboards until we wire them explicitly to v7 KPIs.
- No long-running CPU work in the main executor loop; reuse existing metrics / caches if possible.
- Add or update tests where they exist; if tests are missing, add at least simple sanity tests.

---

## Work Plan

1. Read the current implementations of:
   - `risk_engine_v6.py`
   - `risk_limits.py`
   - `execution_health` / risk snapshot writers
   - telemetry writers and state publishers.
2. Design minimal but expressive data structures for:
   - thresholds, observations, gate tags.
3. Implement changes in small, focused edits with comments where necessary.
4. Run a quick import or test smoke (if available) to validate:
   - no syntax errors
   - key flows still work.
5. Summarise all changes in a short internal changelog (as comments or a small `docs/v7_risk_tuning_notes.md` if appropriate).

Return the diff(s) and a concise description of what was changed and where.
