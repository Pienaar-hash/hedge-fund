# GPT-HEDGE v7.9 — Performance Summary

**Report date:** 2026-03-30
**Source:** `logs/state/` (read-only state surfaces written by executor)
**Disclaimer:** This document is a factual snapshot. No claims of profitability are made. Performance is negative.

---

## NAV Trajectory

| Date | NAV (USDT) | Change |
|------|-----------|--------|
| 2025-12-09 | 10,000.00 | Initial deposit |
| 2026-03-01 | 9,766.04 | −2.34% |
| 2026-03-03 | 9,769.60 | −2.30% (period high) |
| 2026-03-10 | 9,483.66 | −5.16% |
| 2026-03-14 | 9,124.78 | −8.75% |
| 2026-03-17 | 9,305.05 | −6.95% |
| 2026-03-20 | 9,233.52 | −7.66% |
| 2026-03-25 | 9,128.15 | −8.72% |
| 2026-03-30 | 8,999.87 | −10.00% |

NAV source: Futures wallet only (BTC + USDT + USDC)
Current NAV: **$8,999.87 USDT**
Peak NAV: **$10,000.00 USDT** (inception)

The NAV has declined from inception with no sustained recovery. The trajectory is monotonically negative across all recorded months.

---

## Current PnL State

| Metric | Value |
|--------|-------|
| Cumulative PnL (inception) | −$1,000.13 |
| Return (inception) | −10.00% |
| Realized PnL (today) | −$0.33 |
| Unrealized PnL | $0.00 |
| Gross exposure | $0.00 |
| Open positions | 0 |

**Assessment:** Performance is negative. The system has not generated positive returns over the reporting period. No profitability claim is made.

---

## Open Positions

**Count:** 0

No active trading exposure as of the report timestamp. All positions have been closed via thesis-invalidation exits.

---

## Trade History (Episode Ledger)

| Episode | Symbol | Side | Duration | Entry | Exit | Gross PnL | Fees | Net PnL | Exit Reason |
|---------|--------|------|----------|-------|------|-----------|------|---------|-------------|
| EP_0001 | BTCUSDT | LONG | 6.59 h | $185.38 | $185.21 | −$0.17 | $0.15 | −$0.32 | THESIS_INVALIDATED |
| EP_0002 | BTCUSDT | LONG | 0.01 h | $185.31 | $185.10 | −$0.21 | $0.15 | −$0.36 | THESIS_INVALIDATED |
| EP_0003 | BTCUSDT | LONG | 0.31 h | $185.22 | $185.01 | −$0.21 | $0.15 | −$0.36 | THESIS_INVALIDATED |
| EP_0004 | BTCUSDT | LONG | 0.02 h | $185.87 | $185.01 | −$0.86 | $0.15 | −$1.00 | THESIS_INVALIDATED |
| EP_0005 | BTCUSDT | LONG | 0.03 h | $185.04 | $185.06 | +$0.02 | $0.15 | −$0.13 | THESIS_INVALIDATED |

All sampled trades were closed by `THESIS_INVALIDATED` (doctrine thesis no longer valid). Average fee per trade: ~$0.15. Net cumulative over these episodes: −$2.17.

---

## Drawdown

| Metric | Value |
|--------|-------|
| Drawdown state | NORMAL |
| Drawdown from inception peak | −10.00% (−$1,000.13) |
| Drawdown from March high | −7.88% (−$769.73) |
| Daily drawdown | 0.00% |
| Circuit breaker threshold | 10% NAV |
| Circuit breaker breach | NO |

The system is in maximum drawdown from inception. The NAV has not recovered to its initial deposit level at any point during March 2026.

---

## Fee Analysis (3-Day Window)

| Metric | Value |
|--------|-------|
| Fees paid (3d) | $5.86 |
| Realized PnL (3d) | −$1.46 |
| Fee/PnL ratio | 4.02× |

Fees exceed realized PnL by approximately 4×. Execution costs consume gross returns.

---

## Execution Quality

| Metric | Value |
|--------|-------|
| Router quality | POOR |
| Maker fill ratio | 5.26% |
| Taker fallback ratio | 9.52% |
| Slippage P50 | 0.57 bps |
| Slippage P75 | 3.48 bps |
| Slippage P95 | 43.1 bps |
| Total order attempts | 67,385 |
| Total executed orders | 25,323 |
| Recorded fills | 3,283 |

---

## Risk Mode

| Metric | Value |
|--------|-------|
| Risk mode | HALTED |
| Halt reason | `nav_stale_age=97s` (exceeds 90s threshold) |
| Risk score | 1.0 (maximum severity) |
| NAV cache health | STALE |

---

## System Liveness

| Subsystem | Status | Last Activity |
|-----------|--------|---------------|
| Signals | IDLE | 901 s ago |
| Orders | IDLE | 1,801 s ago |
| Exits | IDLE | 3,601 s ago |
| Router | IDLE | 1,801 s ago |

All subsystems idle (>15 min). Trading is halted.

---

## Portfolio Composition (Futures Wallet)

| Asset | Holdings |
|-------|----------|
| BTC | 674.201 |
| USDT | 3,325.66 |
| USDC | 5,000.00 |
| ETH | 0.00 |
| FDUSD | 0.00 |

---

## Symbol-Level Health

| Symbol | Router Quality | Sharpe State | Risk Flags | ATR Regime |
|--------|----------------|--------------|------------|------------|
| BTCUSDT | ok | poor | none | quiet |
| ETHUSDT | BROKEN | neutral | slip/fallback high | normal |
| SOLUSDT | BROKEN | neutral | slip/fallback high | quiet |
| DOGEUSDT | ok | neutral | none | unknown |
| LINKUSDT | ok | neutral | none | unknown |
| LTCUSDT | ok | neutral | none | unknown |
| SUIUSDT | ok | neutral | none | unknown |
| WIFUSDT | ok | neutral | none | unknown |

ETHUSDT and SOLUSDT router quality is BROKEN (high slippage and fallback rates).

---

## Hydra Head Allocation

| Head | Budget (NAV %) | Usage (NAV %) | Positions |
|------|----------------|---------------|-----------|
| TREND | 50% | 2.57% | 3 |
| MEAN_REVERT | 25% | 0% | 0 |
| RELATIVE_VALUE | 30% | 0% | 0 |
| CATEGORY | 20% | 0% | 0 |
| VOL_HARVEST | 20% | 0% | 0 |
| EMERGENT_ALPHA | 15% | 0% | 0 |

Only the TREND head has generated intents. All other heads are unused.

---

## Pending Intents (Not Executed)

| Symbol | Head | Side | NAV % | Score |
|--------|------|------|-------|-------|
| ETHUSDT | TREND | LONG | 0.90% | 0.452 |
| BTCUSDT | TREND | LONG | 0.90% | 0.451 |
| SOLUSDT | TREND | LONG | 0.76% | 0.381 |

These intents are generated by Hydra but blocked at the Doctrine gate (regime confidence below floor, NAV stale).

---

*Source files: `nav_state.json`, `risk_snapshot.json`, `diagnostics.json`, `kpis_v7.json`, `episode_ledger.json`, `positions_state.json`, `positions_ledger.json`, `pnl_attribution.json`, `hydra_state.json`, `execution_quality.json`*
