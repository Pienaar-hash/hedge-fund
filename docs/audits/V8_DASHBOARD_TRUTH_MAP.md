# V8 Dashboard Truth Map

## Scope

This audit maps dashboard KPI surfaces to canonical data sources and formulas, and clarifies authority ordering:

1. NAV Delta PnL comes from NAV history deltas.
2. Closed-Trade Realized PnL comes from closed episodes only.
3. Veto metrics are split by risk gate vs doctrine gate.

Phase 1 is observability-only. No execution behavior is changed.

## KPI Truth Map

| Dashboard Label | Surface | Source File(s) | Calculation | Notes |
|---|---|---|---|---|
| NAV | KPI strip | logs/state/nav_state.json | nav_usd or nav or total_equity | Current futures wallet NAV surface. |
| NAV Delta PnL | KPI strip | logs/nav_log.json, logs/state/episode_ledger.json (fallback) | If span_ok(all_time): NAV(now)-NAV(first); else episode_ledger.stats.total_net_pnl | Label intentionally names NAV delta authority. |
| Exposure | KPI strip | logs/state/nav_state.json | gross_exposure / nav_usd * 100 | Percent of current gross exposure. |
| Drawdown | KPI strip | logs/state/nav_state.json | drawdown_pct | Display-only surface. |
| Win Rate | KPI strip | logs/state/episode_ledger.json | stats.win_rate | Closed-episode win rate. |
| Risk | KPI strip | logs/state/risk_snapshot.json, logs/state/kpis_v7.json | dd_state normalization | Qualitative state badge. |
| NAV Delta PnL (24h) | Strategy Performance | logs/nav_log.json, logs/state/episode_ledger.json (fallback) | If span_ok(24h): NAV(now)-NAV(24h ago); else sum closed episode pnl in 24h | Explicitly not closed-trade-only when NAV span is valid. |
| 7d PnL | Strategy Performance | logs/nav_log.json, logs/state/episode_ledger.json (fallback) | If span_ok(7d): NAV(now)-NAV(7d ago); else 7d episode-window pnl | Same authority ordering as 24h. |
| 30d PnL | Strategy Performance | logs/nav_log.json, logs/state/episode_ledger.json (fallback) | If span_ok(30d): NAV(now)-NAV(30d ago); else 30d episode-window pnl | Same authority ordering as 24h. |
| NAV Delta PnL (All-Time) | Strategy Performance | logs/nav_log.json, logs/state/episode_ledger.json (fallback) | If span_ok(all_time): NAV(now)-NAV(first); else episode_ledger.stats.total_net_pnl | Matches KPI strip all-time value priority. |
| Win Rate | Strategy Performance | logs/state/episode_ledger.json | stats.win_rate | Closed episodes only. |
| Sharpe | Strategy Performance | logs/state/kpis_v7.json | kpis.sharpe or kpis.sharpe_ratio | Display from published KPI surface. |
| Max DD | Strategy Performance | logs/state/episode_ledger.json or kpis merge | kpis.max_drawdown / max_dd | Display-only stat. |
| Trades | Strategy Performance | logs/state/episode_ledger.json | episode_count | Closed episodes count. |
| Closed-Trade Realized PnL | NAV Composition | logs/state/episode_ledger.json | stats.total_net_pnl | Realized closed-trade PnL only. |
| Gross PnL | NAV Composition | logs/state/episode_ledger.json | stats.total_gross_pnl | Before fees. |
| Fees Paid | NAV Composition | logs/state/episode_ledger.json | stats.total_fees | Closed episodes accumulated fees. |
| Episodes Closed | NAV Composition + Episode widget | logs/state/episode_ledger.json | episode_count | Closed episodes only. |
| Risk Vetoes (24h) | Runtime Health | logs/state/risk_snapshot.json or logs/execution/risk_vetoes.jsonl | sum(veto_counts.values()) or scan JSONL in last 24h | Risk gate veto stream. |
| Doctrine Vetoes (24h) | Runtime Health | logs/doctrine_events.jsonl | count events with event_type in {ENTRY_VETO, DOCTRINE_ENTRY_VETO} in last 24h | Doctrine gate veto stream. |
| Min Notional Blocks (24h) | Runtime Health | logs/state/risk_snapshot.json or logs/execution/risk_vetoes.jsonl | veto_counts[min_notional] + veto_counts[below_min_notional] | Plumbing veto subclass of risk vetoes. |
| Maker Fill Rate | Runtime Health | logs/state/router_health.json | maker_count / order_count aggregate | Router health aggregate. |
| Fallback Ratio | Runtime Health | logs/state/router_health.json | fallback_count / order_count aggregate | Router degradation indicator. |

## PnL Authority Split

### NAV Delta PnL

- Source of truth: logs/nav_log.json.
- Represents full portfolio movement including unrealized mark-to-market and funding effects reflected in NAV.
- Used when nav span authority is valid for each window.

### Closed-Trade Realized PnL

- Source of truth: logs/state/episode_ledger.json (stats.total_net_pnl).
- Represents realized PnL from closed episodes only.
- Does not include unrealized PnL.

These are intentionally distinct and are now explicitly labeled in dashboard UI.

## Stale-State Detection

The dashboard diagnostics now tracks staleness for required V8 surfaces with explicit thresholds:

| Surface | File | Stale Threshold |
|---|---|---|
| NAV state | logs/state/nav_state.json | 300s |
| Positions state | logs/state/positions_state.json | 300s |
| Episode ledger | logs/state/episode_ledger.json | 1800s |
| Risk veto log | logs/execution/risk_vetoes.jsonl | 86400s |

Status semantics:

- ok: file exists and age <= threshold
- stale: file exists and age > threshold
- missing: file not found

## Label Additions Implemented

The dashboard now includes these labels explicitly:

1. NAV Delta PnL
2. Closed-Trade Realized PnL
3. Risk Vetoes
4. Doctrine Vetoes
5. Min Notional Blocks

## Phase 1 Guardrail

This phase is dashboard truth and observability hardening only. No doctrine, routing, or execution decision path is modified in this audit phase.
