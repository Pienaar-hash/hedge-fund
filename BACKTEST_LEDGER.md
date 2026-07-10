# BACKTEST LEDGER

**Last updated:** 2026-06-05  
**Status:** Sparse — only entries verified against actual run artifacts should be added here.

---

## Backtest Engines

| Engine | Location | Status |
|--------|----------|--------|
| v8 (primary) | `research/backtest_engine_v8.py` | Current |
| Lightweight (scripts) | `scripts/backtest.py` | Used for quick checks |
| Signal causality audit | `research/signal_causality_audit.py` | Timing analysis tool |

**Standard assumptions (scripts/backtest.py defaults):**
- Initial capital: $10,000
- Leverage: 1× (must be overridden to match live leverage)
- Maker fee: 2 bps
- Taker fee: 4 bps
- Slippage: 1 bp
- Position fraction: 1.0 (all-in by default — override for realistic sizing)
- Data source: `data/ohlcv/` directory (local OHLCV files)

---

## Validity Checklist (required before any backtest counts)

Before recording a result below, confirm all items:

- [ ] No lookahead bias: signals computed on bar close, not bar open; no future data in indicator calculations
- [ ] Survivorship bias: universe defined before the test period begins
- [ ] Fees: taker fee ≥ 4 bps, maker ≥ 2 bps; realistic for the live environment
- [ ] Slippage: ≥ 3 bps per side on liquid assets, more on illiquid
- [ ] Leverage: matches live leverage assumption (3–4× for primary strategies)
- [ ] Position sizing: matches live inverse-volatility sizing, not fixed fraction
- [ ] Funding rate: accounted for (perpetuals charge/pay funding every 8h)
- [ ] Test period: ≥ 6 months; includes at least one major drawdown regime
- [ ] Out-of-sample: final 20% of data not used in parameter selection
- [ ] Churn guard: 120s min hold enforced in simulation

---

## Recorded Results

*No validated backtest results recorded yet. Add entries below after passing the validity checklist.*

### Template for new entries

```
### [Strategy ID] — [Asset] — [Date range]
- Engine: backtest_engine_v8 / scripts/backtest.py
- Timeframe: 15m
- Date range: YYYY-MM-DD to YYYY-MM-DD
- Universe: BTCUSDT (or list)
- Leverage: Nx
- Fees: taker Xbps, maker Xbps
- Slippage: Xbps
- Funding: included / excluded
- Net return: X%
- Sharpe (annualized): X.X
- Max drawdown: X%
- Win rate: X%
- Avg hold (bars): X
- Total trades: N
- Out-of-sample period: YYYY-MM-DD to YYYY-MM-DD
- Out-of-sample return: X%
- Validity checklist: PASSED (all items above confirmed)
- Known weaknesses: [describe]
- Artifact path: logs/research/[filename]
```

---

## Known Weaknesses (system-wide)

1. **Leverage mismatch:** `scripts/backtest.py` defaults to 1× leverage; live runs at 3–4×. Backtests using the default understates both returns and drawdowns.

2. **Funding rate omission:** Perpetual funding (±0.01% per 8h) is not modeled in `scripts/backtest.py`. On directional trades held multi-day, this materially affects net PnL.

3. **Small sample size:** With only BTC, ETH, and SOL in the active universe (per `config/pairs_universe.json`) and $10K NAV, statistical significance of any backtest result is limited. Sharpe ratios are noisy below ~500 trades.

4. **Capacity constraints:** Strategy capacities ($350–500/trade) mean the backtest at $10K NAV will show different position-sizing behavior than a scaled-up portfolio.

5. **Regime dependency:** The Hydra/Doctrine system is regime-aware; simple backtests that replay signals without a regime filter will overestimate performance in adverse regimes.

6. **No order book simulation:** Limit order fills are assumed instantaneous at close price. Real maker-order fills depend on queue position and spread; POST_ONLY orders may not fill at all.

---

## Research Outputs Location

- Backtest equity curves: `logs/research/`
- Signal analysis: `logs/research/signal_metrics_*.jsonl`
- Pipeline comparison traces: `logs/pipeline_v6_compare.jsonl`
- Shadow engine traces: `logs/pipeline_v6_shadow.jsonl`

---

## Audit Trigger

A backtest validity audit is required before any strategy is promoted from testnet to live. The audit must be performed by someone who did not write the strategy code (four-eyes principle). Document the audit result as a row in the table above with `PASSED` or `FAILED` and the specific failure reason.
