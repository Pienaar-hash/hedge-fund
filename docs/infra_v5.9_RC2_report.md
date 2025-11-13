✅ v5.9 Execution Hardening — RC2 Integration Summary
1. Telemetry Wiring — Fully Live (No Stubs Remaining)
execution/fills.py & execution/trade_logs.py

Now parse:

logs/execution/orders_executed.jsonl

logs/execution/orders_attempted.jsonl

Extract:

timestamps

notional

taker/maker fees

realized PnL

slippage (mid_before vs fill_price)

Provide canonical accessors used everywhere else (get_recent_fills, get_recent_orders, get_recent_trades, etc.).

execution/utils/metrics.py

No more placeholder values. Connected to:

fills/logs → rolling notionals

pnl_tracker → Sharpe, std, gross realized, fee totals

router metrics → slippage fallback

trade logs → hourly expectancy

These now match the new KPI tiles & risk gates 1:1.

execution/utils/vol.py, execution/utils/expectancy.py, execution/utils/toggle.py

All wired to canonical trackers, replacing in-memory stubs.

Volatility, expectancy, and symbol disablement now reflect real system state (persistent or in-memory toggle store).

Outcome:
The entire execution hardening layer now runs on real historical data, not defaults.

2. Maker-First Routing — Fully Threaded Through Executor & Screener
execution/executor_live.py:1131

The executor now:

injects maker/taker gate metadata into every order attempt

normalizes price/qty for router_ctx

tries submit_limit(effective_price, size) first

only falls back to a taker execution when:

post-only reject threshold exceeded, or

slippage bound breached

Router metrics are persisted with symbol + ts for downstream telemetry.

execution/order_router.py

PlaceOrderResult now carries the raw exchange response, useful for:

real slippage analysis

maker/taker attribution

router-effectiveness analytics

Binance GTX (post-only) TIF is enforced under maker-first mode.

Runtime config feeds:

fee tier

slippage thresholds

min child size

trading windows

maker_first on/off flag

Maker-first logic sits inside route_order so executor doesn’t need to branch manually.

signal_screener & executor routing

signal_screener.py:360 and _route_intent() now feed normalized router_ctx:

intended price

intended size

side

symbol

maker(taker)-allowed flag

effective price precomputation

3. Dashboard Hook — Real KPIs
dashboard/live_helpers.py

Tiles now pull real signals from:

fill_notional_7d()

fees_7d()

realized_slippage_bps_7d()

rolling_expectancy()

Execution tab correctly shows:

Fill efficiency

Fee/PnL ratio

Slippage (bps)

Expectancy (7d rolling)

Slippage/fee ratios now move when logs move — no static values.

4. Tests
✔ pytest -k "execution_hardening or router_metrics or execution_health" -q

Passes clean.

✔ pytest tests/test_order_metrics.py -q

Also passes, confirming router metrics compatibility.

No regressions introduced.
