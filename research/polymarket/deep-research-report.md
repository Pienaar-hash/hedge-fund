# Edge-Discovery and Implementation Blueprint for Polymarket BTC Up/Down at 15m and 5m Horizons

## Executive summary

- The BTC “Up/Down” rounds are **per-interval binary contracts** that resolve **Up if the end-of-window BTC/USD price is ≥ the start-of-window price, else Down**, with the **resolution source explicitly defined as the Chainlink BTC/USD Data Stream**. citeturn4view0turn4view1  
- Because **resolution depends on Chainlink’s Data Stream** (not exchange spot), *basis risk* is a first-class microstructure concern; using any other price proxy is an avoidable implementation error. citeturn4view0turn4view1turn24view0  
- For production-grade timing and oracle-consistent ground truth, use the **RTDS “crypto_prices_chainlink” stream** (BTC/USD) as the canonical oracle-aligned reference stream; RTDS specifies subscription format and keepalive requirements. citeturn24view0turn28view0  
- The trading venue is a hybrid CLOB: **offchain order matching, onchain settlement on Polygon**, with **EIP-712 signed orders** and explicit WebSocket market-data primitives for orderbook and trades. citeturn23search11turn8view1  
- **Taker fees are enabled** for both **15-minute** and **5-minute crypto markets**, and the fee curve peaks at **1.56% effective rate at 50% probability** (per Polymarket documentation/changelog). These fees are implementation-critical and must be modelled in EV. citeturn28view0turn15search2  
- The highest-probability “durable edge” candidates are *implementation-dominant*, not “alpha-signal” dominant:  
  - (A) **two-sided bundle / locked payout capture** when *Up + Down is purchasable for < $1 net of fees and slippage* (microstructure/arb), and  
  - (B) **fee-aware mispricing + execution-quality** (maker-first, strict staleness controls) rather than raw direction prediction. citeturn27view0turn15search2turn12search10  
- The 5m arm is structurally more fragile: less time, tighter latency budget, greater sensitivity to quote staleness and fee drag; treat it as **observe-first / shadow-first**, with explicit go/no-go gates. citeturn28view0turn8view1  
- The **Market Channel WebSocket** provides `book`, `price_change`, `last_trade_price`, `best_bid_ask`, and `market_resolved` events; ingesting these is sufficient for replayable simulation if dedup + reconciliation is done correctly. citeturn8view1  
- **Tick size can change** when prices move near extremes (e.g., >0.96 or <0.04); order construction must query/track tick size, and simulation must reproduce it. citeturn8view1turn6search20  
- Because the binding Binary Lab repo documents were not retrievable in this environment, the blueprint is written to **load and enforce** `binary_lab_limits.json`, dataset admission rules, and checkpoint doctrine *from files at runtime*; final deployment is **inconclusive** until those documents are applied and backtests are run with admitted datasets.

## Market microstructure and mechanics

### Contract design and resolution mechanics

The BTC Up/Down round is mechanically a two-outcome binary market whose settlement rule is **purely a comparison of two oracle prints**: “Up” resolves if the BTC price at the end of the specified time range is **greater than or equal to** the price at the beginning; otherwise it resolves “Down.” citeturn4view0turn4view1

Crucially, the market’s own rules emphasise that the contract is about the **Chainlink BTC/USD Data Stream**, not any exchange spot feed, and that the Chainlink website display may lag while “live data” should be obtained via Chainlink APIs. citeturn4view0turn5view0  
For engineering, the most direct “oracle-consistent” and operationally simple solution is to treat **Polymarket’s RTDS Chainlink source** (topic `crypto_prices_chainlink`, symbol `btc/usd`) as your canonical oracle-aligned stream for both (i) feature construction and (ii) ground-truth reconstruction in research replay. citeturn24view0

### Venue mechanics and execution primitives

Polymarket’s trading stack is an orderbook-based system (CLOB) described as **hybrid-decentralised**: matching is offchain while settlement is onchain on **entity["company","Polygon","l2 blockchain network"]**, with **non-custodial** semantics and **EIP-712 signed orders**. citeturn23search11turn3search4

For market-data ingestion, the **Market Channel WebSocket** (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) is the primary feed, providing:  
- `book`: snapshot + refresh on book-changing trades, including `bids`, `asks`, `timestamp`, and `hash`  
- `price_change`: incremental updates on new/cancelled orders (size=0 removes a level)  
- `last_trade_price`: trade prints with `fee_rate_bps`, price, size, side  
- `best_bid_ask`, `tick_size_change`, `new_market`, and `market_resolved` when `custom_feature_enabled` is set citeturn8view1

Tick size is not static; `tick_size_change` is explicitly documented to occur when price reaches extreme regions (price > 0.96 or < 0.04). This means “naïvely hardcoded” tick logic will break live and will bias simulation. citeturn8view1turn6search20

### Fees, rebates, and why they matter for EV

Both 15m and 5m crypto markets have **taker fees enabled**. The platform’s changelog explicitly notes 15m taker fees (Jan 5, 2026) and that 5m crypto markets launched with taker fees on (Feb 12, 2026). citeturn28view0  
Polymarket’s fee documentation emphasises that taker fees are **price-dependent**, highest near 50% probability, and decreasing toward extremes; the fee table shows a maximum effective rate of **1.56% at 50% probability** for crypto. citeturn15search2turn28view0

Fee collection is asymmetric by direction of trade: taker fees are calculated in USDC and **collected in shares on buy orders** and in USDC on sell orders. For a “hold-to-resolution” strategy that primarily enters by buying one side, your *filled share count* is impacted, altering realised payout and must be represented explicitly in both backtest and live accounting. citeturn3search1turn15search2

### Liquidity cycles, latency, and stale quote risk

Empirically, these markets can exhibit meaningful per-round volume (examples: a 15m market showing six-figure volume; a 5m market also showing six-figure volume in a single interval), but volume is highly time-varying and should be treated as a regime feature, not a constant. citeturn16view0turn16view1

Stale-quote risk is amplified by:  
- short horizons (especially 5m),  
- orderbook dynamics (`price_change` events are frequent), and  
- tick size changes at extremes. citeturn8view1  

For quote lifetime controls, Polymarket’s GTD orders include a **one-minute security threshold buffer** requirement, which is especially important for fast-cycle quoting and must be handled both in “maker-first execution rules” and in simulation constraints. citeturn12search10

## Edge hypothesis matrix and falsifiable hypotheses

### Edge hypothesis matrix

**Interpretation:** “Expected effect size” is net-of-fees EV per trade in decimal terms (e.g., 0.003 = 0.3% of $1 payout), assessed out-of-sample with latency/slippage sensitivity.

| Candidate edge | 15m expected effect | 5m expected effect | Primary implementation risk | Core test method |
|---|---:|---:|---|---|
| Conviction-band transfer (very_high > high > medium EV monotonic) | Moderate (if any true signal exists after fee drag) | Low–moderate (noise can invert band ordering) | band leakage due to execution quality differences; selection bias | stratified EV/trade by band with bootstrap CI and monotonicity constraint |
| Mispricing vs realised move (implied p vs realised Up frequency) | Moderate | Low | oracle alignment errors; fee omission makes “edge” illusory | calibration gap vs implied mid, net-of-fees payoff model, walk-forward |
| Timing edge (entry timing within round) | Moderate | Moderate–high (if microstructure is dominant) | slippage/partial fills; RTDS vs CLOB timestamp skew | bucket by entry offset (e.g., 0–60s, 60–180s, …) plus latency stress test |
| Regime-conditional edge (exclude CHOPPY-like states) | Moderate | Low | regime classifier instability; overfitting | pre-registered, frozen regime rules; OOS performance vs baseline |
| Liquidity/friction edge (maker-first, strict staleness controls) | High (durable if execution dominates) | Moderate (latency budget tighter) | fill uncertainty; adverse selection | compare maker-first vs taker baseline; simulate fills with market-channel replay |
| Two-sided bundle / “locked payout capture” when Up+Down < 1 | Potentially high, but rare post-fees | Potentially moderate, but latency-sensitive | incomplete leg fill; fee curve; end-of-round liquidity gaps | detect opportunity from full depth; simulate two-leg fills + fee impact + failure handling |

The final row reflects a widely discussed pattern in public bot implementations: two-sided purchase to lock a payout when combined cost < $1, and/or “latency/signal” entry before odds adjust. citeturn27view0  
Binary Lab doctrine (as provided) suggests interpreting “works in binary but not futures” as execution friction; this matrix therefore separates “signal” vs “friction” edges as different hypotheses rather than blending them.

### Falsifiable hypothesis table

Below are null/alternative hypotheses for each horizon (15m baseline, 5m experimental). All tests are defined in terms of *net* EV after fees/slippage models, with walk-forward OOS splits.

**H1: Conviction-band transfer (directional signal quality)**  
- 15m H0: EV(very_high) ≤ EV(high) or EV(high) ≤ EV(medium), or EV per band ≤ 0 net after fees  
- 15m H1: EV(very_high) > EV(high) > EV(medium) and EV(very_high) > 0 after fees  
- 5m H0/H1: same structure, but evaluated under stricter latency scenarios

**H2: Calibration/mispricing (market probability inefficiency)**  
- 15m H0: For all bins of implied probability p, realised Up frequency f(p) ≈ p (within CI), net EV ≈ 0  
- 15m H1: There exist stable bins where f(p) − p > 0 and net EV > 0 after fees  
- 5m H0/H1: same, plus a “noise dominance” test where any deviation disappears under mild latency shifts

**H3: Timing within round (microstructure exploitability)**  
- 15m H0: Conditional on signal strength, entry timing does not change net EV (within CI)  
- 15m H1: There exists a fixed, pre-registered timing window that increases net EV net of fees  
- 5m H0/H1: same, but must hold under high-latency stress scenario

**H4: Regime gating (CHOPPY exclusion)**  
- 15m H0: EV in “trade” regime ≤ EV in “no trade” regime, or gating reduces EV/day  
- 15m H1: EV conditional on “trade” regime improves and EV/day increases net of reduced trade count  
- 5m H0/H1: same, with emphasis on avoiding overfitting due to 5m’s larger sample but noisier dynamics

**H5: Liquidity/fee-aware execution (implementation edge)**  
- 15m H0: Maker-first execution does not improve realised fill-price and net EV after fees  
- 15m H1: Maker-first reduces effective fee burden / spread paid enough to create positive net EV  
- 5m H0/H1: same, but must hold under stricter stale-quote limits

These are intentionally falsifiable: if latency/sim assumptions erase H1 effects, the “edge” is not durable.

## Data and instrumentation specification

### Binding constraints status

The following documents were named as hard constraints but were **not accessible** via the available toolchain in this environment:  
- `ops/BINARY_LAB_WINDOW_2026-02-XX.md`  
- `ops/BINARY_LAB_DAILY_CHECKPOINT.md`  
- `ops/BINARY_LAB_30_DAY_BRIEF.md`  
- `config/binary_lab_limits.json`  
- `config/dataset_admission.json` (polymarket_snapshot + prediction_polymarket_feed)  
- `research/polymarket/market_structure_notes.md`  
- `research/polymarket/api_reference.md`

Therefore, the blueprint is written so the production system **loads these files at boot, enforces them mechanically, and refuses to trade if they are missing or their hashes mismatch the frozen config**. This preserves “binding governance” without guessing file contents.

### Exact datasets required for replayable analysis

You need two primary admitted datasets (plus mirrors), each stored with immutable partitions and time-synchronisation metadata:

**Dataset: polymarket_snapshot (market microstructure ground truth)**  
Minimum schema (fields are required unless noted):  
- `ts_ms` (int, UTC) — ingestion timestamp normalised to UTC milliseconds  
- `source` (enum: `ws_market`, `rest_clob`, `ws_rtds`)  
- `market_slug` (string) — e.g., `btc-updown-15m-<start_ts>`  
- `round_start_ts_s` (int, UTC seconds) — parsed from slug  
- `round_end_ts_s` (int, UTC seconds) — derived from horizon  
- `horizon_s` (enum: 300, 900)  
- `condition_id` (string) and `clob_token_id_up` / `clob_token_id_down` (string)  
- **Best prices:** `best_bid_up`, `best_ask_up`, `best_bid_down`, `best_ask_down`  
- **Depth:** top-N levels for each side/outcome, from `book` events (`price`, `size`) citeturn8view1  
- `tick_size` (string/decimal) and `min_order_size` (string/decimal)  
- `fee_rate_bps_up` / `fee_rate_bps_down` (numeric, fetched from CLOB where available)  
- `book_hash` (string, from WS `book.hash`), `event_seq` (optional monotone sequence)

**Dataset: prediction_polymarket_feed (model outputs and conviction bands)**  
Minimum schema:  
- `ts_ms` (int, UTC) — prediction timestamp  
- `market_slug`, `horizon_s`  
- `p_up_model` (float in [0,1]) — model probability of Up at settlement  
- `conviction_band` (enum: `medium`, `high`, `very_high`)  
- `regime` (enum: `trade`, `no_trade`, `choppy`, etc — must be frozen definitions)  
- `features_version` and `model_version` (strings)  
- `latency_budget_ms` (int) — the assumed end-to-end latency for which the prediction remains valid

**Diagnostic mirror datasets (required by doctrine)**  
- `futures_or_spot_snapshot`: a high-frequency BTC price series aligned to `ts_ms` used to test the same signals “in continuous markets.”  
- `execution_telemetry`: per-order lifecycle and failure telemetry to separate signal failure from implementation failure.

### Streaming ingestion architecture

A production-appropriate ingestion strategy uses three streams:

**Market microstructure stream:** Market Channel WebSocket (public)  
- Endpoint and event types are documented; ingest `book`, `price_change`, `last_trade_price`, `best_bid_ask`, and `tick_size_change`. citeturn8view1  
- Dedup strategy: treat (`event_type`, `asset_id`, `timestamp`, `hash`) as a natural idempotency key where present; maintain per-asset “last seen timestamp” and reject out-of-order beyond a small tolerance; periodically resynchronise with a full `book` snapshot on gap detection.

**Oracle-consistent price stream:** RTDS WebSocket (public for crypto prices)  
- Subscribe to `crypto_prices_chainlink` filtered to `btc/usd`. citeturn24view0  
- Keepalive: send `PING` every 5 seconds as specified. citeturn24view0  
- Purpose: construct `oracle_start_price` and `oracle_end_price` (using nearest tick rules defined below), and expose “oracle move features” for timing/latency studies.

**Control-plane / discovery:** Gamma discovery and/or WS market-resolved notifications  
- Use Gamma for market metadata discovery (markets/events are public REST endpoints). citeturn22search13turn18search4  
- Use `market_resolved` WS events when available to record final outcomes and close the loop in replay. citeturn8view1

### Timestamp normalisation and reconciliation rules

A single “canonical clock” is required:

- Use the CLOB server time as your skew monitor (SDK method `getServerTime()` returns Unix seconds). citeturn23search0  
- Persist `clock_skew_ms = local_time_ms - server_time_ms` for every process at 1-minute cadence; stamp every ingested event with both `recv_ts_ms` and `exchange_ts_ms` (the latter is the embedded WS timestamp where present). citeturn8view1turn24view0  
- Round boundary rule: parse `round_start_ts_s` from the slug (it matches UTC epoch in observed markets), and define `round_end_ts_s = round_start_ts_s + horizon_s`. citeturn4view0turn4view1  

**Oracle sampling rule (must be deterministic and replayable):**  
- `oracle_start_price` = last RTDS Chainlink print with `payload.timestamp ≤ round_start_ts_ms` within a maximum staleness window `S_max_ms` (config).  
- `oracle_end_price` = last RTDS Chainlink print with `payload.timestamp ≤ round_end_ts_ms` within staleness window.  
- If staleness > `S_max_ms`, mark round as **invalid for directional evaluation** (still usable for microstructure stats), because you can’t robustly reconstruct settlement.

This explicit staleness gating is the cleanest way to measure whether 5m is “noise/friction overwhelmed.”

## Statistical evaluation framework and go/no-go gates

### Payout-adjusted net EV model

For a single “buy-and-hold” trade on outcome Up at fill price `p_fill` with size `q` shares:

- Gross PnL per share:  
  - if Up wins: `+ (1 - p_fill)`  
  - if Up loses: `- p_fill`  
- With taker fees enabled, fees vary by price and are not constant; the system must compute `fee_usdc` using the documented curve / API-provided `fee_rate_bps`, and apply the “fees charged in shares on buy orders” rule (i.e., you may receive fewer shares than requested, affecting payout). citeturn3search1turn15search2turn28view0  

**Simulation must therefore track two quantities:**  
- `requested_shares` and `filled_shares_net_fee`  
- `effective_cost_usdc` (including spread/slippage)

The same structure applies to Down trades.

### Required metrics (computed per horizon and per conviction band)

All metrics must be computed separately for 15m and 5m, and also under latency stress scenarios (see below):

- EV/trade (mean and median), and EV/day (using observed trade frequency and limits)  
- Win rate by conviction band, plus monotonicity constraint test  
- Band separation: ΔEV between adjacent bands, and ΔWinRate  
- Calibration: implied probability (from midpoint of best bid/ask) vs realised frequency, including calibration gap curves  
- Net expectancy after fees/slippage (primary metric)  
- Maximum drawdown and drawdown duration (in “strategy accounting”, not blended into core NAV)  
- Loss-streak autocorrelation (to detect regime dependence and clustering)

### Robustness methods

**Bootstrap confidence intervals:**  
- Use block bootstrap at the *round level* (not per fill) to preserve intra-round dependence, with blocks sized to one day (or a fixed number of rounds) for conservative CI estimation.

**Walk-forward / out-of-sample:**  
- Use a rolling split (e.g., 14 days train / 7 days validate, roll by 7) for research; once live, enforce the Binary Lab 30-day freeze by locking parameters and only evaluating forward.

**Sensitivity analysis (must be reported):**  
- Fee curve: test ±25% fee multiplier (because fee policies can change over time; maker rebate percentages are explicitly noted as changeable by the platform). citeturn15search1turn28view0  
- Latency: inject execution delay distributions (e.g., 50ms, 250ms, 1000ms, 2500ms) between signal timestamp and fill timestamp.  
- Quote staleness: require that market best bid/ask is no older than `Q_max_ms` at order submission; test at multiple `Q_max_ms`.  
- Tick size changes: ensure order rounding adheres to the tick size in effect at the simulated submission time. citeturn8view1turn6search20  

### Minimum sample sizing guidance

Because a binary payout has high per-trade variance, statistical confidence requires either (i) very large N or (ii) meaningful effect size. Given there are ~96 15m rounds/day and ~288 5m rounds/day, a practical minimum for initial go/no-go is:

- 15m: ≥ 2,000 eligible rounds (≈ 21 days if trading many rounds; fewer if heavily gated)  
- 5m: ≥ 4,000 eligible rounds (≈ 14 days if trading many rounds)

These are not “guarantees,” but operationally realistic thresholds for bootstrap stability given time-varying microstructure.

### Go/no-go rubric with exact thresholds

All thresholds below are **net of fees and slippage** and must hold in the final walk-forward segment.

**Baseline eligibility (both horizons):**  
- Data completeness: ≥ 98% of targeted rounds have valid `oracle_start_price` and `oracle_end_price` under the fixed staleness rule. citeturn24view0  
- Execution integrity: ≥ 99% of submitted orders have terminal status recorded (filled/cancelled/rejected) with reason codes; no silent drops.

**Deploy 15m only (go):**  
- Net EV/trade ≥ +0.0025 (0.25 cents per $1 payout) with 95% block-bootstrap **lower CI > 0**  
- Net EV/day ≥ +0.10% of allocated strategy capital/day under the hard concurrency cap (read from `binary_lab_limits.json`)  
- Conviction monotonicity: EV(very_high) > EV(high) > EV(medium) in OOS, and EV(very_high) − EV(medium) ≥ 0.0050  
- Max drawdown (strategy accounting) ≤ 6% over the OOS segment  
- Loss-streak autocorrelation: |ρ₁| ≤ 0.15 (indicates no strong unmodelled clustering)

**Add 5m with guardrails (go, experimental):**  
- All 15m gates passed, plus:  
- 5m net EV/trade ≥ +0.0035 with 95% lower CI > 0 under a latency stress of +250ms  
- 5m “noise dominance” test: EV under +1000ms latency remains ≥ 50% of the +0ms EV (otherwise it’s likely a pure latency arb and not durable)  
- 5m max drawdown ≤ 4% and worst loss streak length ≤ 8 (in the OOS window)

**Remain observe-only (no-go):**  
- Any of the above gates fail, or the results are positive only under unrealistically low latency assumptions, or the edge disappears when you apply the documented fee curve. citeturn15search2turn28view0  

These gates are intentionally conservative: they prioritise “durable and implementable” over “paper alpha.”

## Implementation blueprint and Binary Lab governance mapping

### System architecture for data ingestion

**Component graph (production):**

- **MarketDiscoveryService**  
  - Responsibility: find the active BTC Up/Down markets for 15m and 5m horizons and map them to CLOB token IDs  
  - Sources: Gamma API for market metadata (public) citeturn22search13turn18search4  
  - Output: `active_rounds.json` (immutable per minute) containing `{market_slug, horizon_s, round_start_ts_s, token_ids, condition_id}`

- **MarketDataIngestor (WS Market Channel)**  
  - Responsibility: subscribe to token IDs, ingest `book`, `price_change`, `last_trade_price`, `best_bid_ask`, `tick_size_change`, `market_resolved` citeturn8view1  
  - Output: append-only `polymarket_snapshot` partitions (and raw WS logs for forensic replay)

- **OraclePriceIngestor (RTDS)**  
  - Responsibility: subscribe to `crypto_prices_chainlink` with filter for `btc/usd`, enforce 5s ping keepalive, persist time-series citeturn24view0  
  - Output: `oracle_chainlink_btcusd.parquet` (or jsonl), plus integrity metrics (gap counts, staleness)

- **SignalService**  
  - Responsibility: compute frozen features, produce `prediction_polymarket_feed` with conviction bands and regime tags  
  - Governance: frozen model + frozen thresholds for 30 days once deployed (per doctrine)

- **ExecutionEngine**  
  - Responsibility: deterministic rule application, order creation, state machine, enforce limits  
  - Uses: authenticated CLOB endpoints with two-level auth model citeturn23search13turn12search7  
  - Order types:  
    - default: post-only maker-first for fee/spread reduction, fallback to FOK marketable order when necessary citeturn12search10  
    - GTD for auto-expiry, respecting 60s buffer rule citeturn12search10  
  - Batch orders: optionally use batch placement up to 15 orders per request (for paired execution or multi-round catch-up) citeturn12search10turn28view0  

- **AccountingService (separate state)**  
  - Responsibility: maintain independent ledger of positions, realised/unrealised PnL, fees, and rebates (if applicable)  
  - Output: `strategy_accounting/*` isolated from core NAV

- **Observability + CheckpointService**  
  - Responsibility: produce daily checkpoint artefacts and enforce “kill line” from `binary_lab_limits.json`  
  - Output: `ops/checkpoints/YYYY-MM-DD/…` plus metrics export

**Rate-limit discipline:**  
All REST polling must remain far below published limits; Cloudflare throttles when exceeded, which can introduce latency spikes that directly destroy 5m viability. citeturn3search11turn6search11

### Canonical event schema for `binary_lab_trades.jsonl` extension

A minimal extension that supports both 15m and 5m and preserves replayability:

- `ts_ms` — event time (UTC ms)  
- `run_id` — immutable UUID for the strategy process  
- `strategy_id` — e.g., `pm_btc_updown_15m_v1` or `pm_btc_updown_5m_v0`  
- `config_hash` — SHA-256 of the frozen config bundle  
- `horizon_s` — 900 or 300  
- `market_slug`, `condition_id`, `token_id`  
- `side` — BUY/SELL  
- `intent` — `entry`, `exit`, `hedge_leg1`, `hedge_leg2`, `cancel`, `redeem`  
- `order_type` — GTC/GTD/FOK/FAK, `post_only` boolean citeturn12search10  
- `price_limit`, `size_requested`, `size_filled_gross`, `size_filled_net_fee`  
- `fee_usdc`, `fee_collected_in` — `shares` or `usdc` citeturn3search1  
- `best_bid`, `best_ask`, `spread` — at decision time  
- `latency_budget_ms`, `observed_latency_ms`  
- `status` — filled/partial/cancelled/rejected + `reject_reason`  
- `oracle_start_price`, `oracle_end_price`, `resolved_outcome` (when known)  
- `p_up_model`, `conviction_band`, `regime`

This schema is deliberately “event-sourced”: you can rebuild positions and PnL from it without any hidden state.

### Backtest and simulation design

**Core design principle:** replay must reproduce the same decision inputs as live. Therefore, base the simulation on recorded WS `book` and RTDS oracle prints, not on down-sampled price history alone.

**Entry/exit rule families to simulate:**

1. **Directional hold-to-resolution (baseline “binary alpha”)**  
   - Entry: one order per round at a deterministic time offset from round start  
   - Exit: none; hold until `market_resolved` then redeem

2. **Timing optimisation (still directional)**  
   - Same as baseline but with a fixed entry window (e.g., enter only between +60s and +180s), pre-registered before live

3. **Two-sided bundle / locked payout capture**  
   - Condition: combined executable cost of Up + Down < 1 − margin (fees/slippage buffer)  
   - Execution: either atomic batch submission (preferred) or strict two-leg with failure handling  
   - This pattern is explicitly described in public bot write-ups as “hedging arbitrage.” citeturn27view0  

**Fill model (deterministic, conservative):**  
- For taker-style execution: consume available book levels from `asks` (for BUY) or `bids` (for SELL) at time `t_submit + latency_ms`. citeturn8view1  
- For maker-style (post-only): fill only if subsequent `last_trade_price` or `book` updates indicate that price level traded through; add adverse selection penalty in sensitivity mode.

**Fee model:**  
- Apply taker fee curve by price; use `fee_rate_bps` sources from WS trade events where available and/or the documented curve; enforce rounding to at least the smallest documented fee granularity where applicable. citeturn15search2turn8view1  
- Apply “fees collected in shares on buy orders” rule explicitly. citeturn3search1  

**Latency/slippage model:**  
- Parameterise end-to-end delay as a distribution; stress test to identify regimes where edge flips sign.

**Invalid/void handling:**  
- If oracle prices cannot be reconstructed within staleness limits, mark round invalid for directional EV.  
- If `market_resolved` is missing, fall back to Gamma/chain resolution fields (must be logged); if still missing, treat as unresolved and exclude from EV.

### Live shadow mode plan

Shadow mode must be a full end-to-end run (ingest → signal → “paper order” → simulated fill → accounting) with **zero capital**:

- Execute “paper orders” using the same order construction logic, but do not submit.  
- Record the counterfactual fill using the same fill model as backtest at real-time.  
- Compare shadow PnL vs backtest PnL for the same period; large divergence indicates implementation gaps (the “diagnostic mirror” principle).

Shadow duration minimums:  
- 15m: 10 days continuous  
- 5m: 14 days continuous (to sample more market regimes)

### Day 0–Day 30 operations plan

Because the repo checkpoint docs were not available, the following is an implementation template that should be mapped 1:1 onto your existing `ops/BINARY_LAB_DAILY_CHECKPOINT.md` commands once retrieved.

**Day 0 (launch) hard requirements:**  
- Config bundle freeze: compute and persist `config_hash`; refuse to trade if runtime hash differs  
- Confirm `binary_lab_limits.json` loaded and active; log limits snapshot to the checkpoint artefact  
- Confirm all ingest streams healthy (WS Market, RTDS Chainlink BTC/USD) with gap metrics = 0 for last 10 minutes citeturn8view1turn24view0  

**Daily cadence (Day 1–Day 30):**  
- Generate checkpoint:  
  - data completeness (oracle + market snapshots)  
  - number of rounds targeted vs traded  
  - realised EV/day and drawdown  
  - breach log (any limit triggers / near-misses)  
- Freeze integrity check: verify unchanged `config_hash`  
- Breach handling:  
  - if kill-line breached: stop trading immediately, cancel outstanding orders, persist post-mortem bundle  
  - if concurrency cap exceeded (should be impossible if enforced): treat as severity-1 incident, halt

**No adaptive tuning:**  
- During Day 0–Day 30 the system can *observe* metrics but must not change thresholds/params. Only “stop trading” is allowed when kill-lines trigger (governance constraint).

### Ranked implementation backlog

**P0 must-build (blocking for any edge evaluation)**  
- WS Market Channel ingestor with dedup/reconciliation and snapshot persistence citeturn8view1  
- RTDS Chainlink BTC/USD ingestor with PING keepalive and staleness metrics citeturn24view0  
- Deterministic round segmentation (slug parse → start/end timestamps) consistent across 15m/5m citeturn4view0turn4view1  
- Fee-aware accounting (buy-fee-in-shares handling) citeturn3search1turn15search2  
- Limits + kill-line enforcement loaded from `binary_lab_limits.json` (hard stop if file missing)  
- Immutable `binary_lab_trades.jsonl` event-sourcing extension (schema above)  
- Shadow mode runner that uses identical strategy logic and logs divergences vs realised market fills

**P1 edge enhancers (increase durability, reduce friction)**  
- Maker-first execution with strict staleness cancellation, post-only logic, and GTD expiry handling citeturn12search10turn8view1  
- Two-leg bundling engine with explicit “leg2 failure” handling (cancel/hedge rules)  
- Regime classifier (pre-registered, frozen) trained on admitted data only  
- Diagnostic mirror pipeline (same signals on continuous BTC market data)

**P2 optional research**  
- Model-based calibration mapping: transform model probabilities into “execution-aware” target entry prices  
- Market maker rebate tracking (only if acting as maker; rebate policies can change over time) citeturn15search1turn28view0  
- Advanced microstructure features: depth imbalance, orderbook toxicity proxies, tick-size-change proximity triggers citeturn8view1  

### Top failure modes and mitigations

1. **Oracle mismatch (basis risk): using spot instead of Chainlink stream**  
   - Mitigation: use RTDS Chainlink BTC/USD as canonical oracle-aligned stream; block trading if RTDS gaps exceed staleness threshold. citeturn24view0turn4view0  

2. **Fee-blind backtests that disappear live**  
   - Mitigation: strict fee modelling with documented curve and buy-fee-in-shares accounting; sensitivity tests ±25% fee level. citeturn15search2turn3search1turn28view0  

3. **Stale quotes / book drift (especially 5m)**  
   - Mitigation: maker-first with rapid cancel-on-change using `price_change` updates; enforce max quote age; fallback to FOK with tight worst-price cap. citeturn8view1turn12search10  

4. **Tick size change breaks order placement and biases sim**  
   - Mitigation: subscribe to `tick_size_change` and/or query tick size; round order prices accordingly; simulate with the same tick regime. citeturn8view1turn6search20  

5. **Rate-limit throttling induces latency spikes and kills 5m viability**  
   - Mitigation: WS-first architecture; REST only for metadata refresh; keep requests far under limits; treat throttling as an incident metric. citeturn3search11turn6search11  

### Final recommendation

**Remain observe-only (inconclusive) — with a 15m-first deployment path prepared.**

Rationale:  
- The engineering blueprint can be made production-ready with the documented Polymarket interfaces (Market Channel WS, RTDS Chainlink stream, fee-aware execution). citeturn8view1turn24view0turn15search2  
- However, the request explicitly treats specific Binary Lab repo doctrine and config files as *binding governance*. Those files were not retrievable here, so any “deploy” recommendation would be non-compliant with the stated constraint set (limits, checkpoint commands, dataset admission rules, and window governance cannot be verified or enforced from actual file contents).  
- Additionally, without running admitted-dataset backtests and shadow-mode divergence checks, evidence for “durable, implementable edge” is insufficient by the user’s own standard (“If evidence is insufficient, return inconclusive with exact additional data required.”).

**Exact additional data required to move from inconclusive → go/no-go (minimum set):**  
- The full contents of the seven reference documents and configs (especially `binary_lab_limits.json` and dataset admission rules).  
- At least 21 consecutive days of 15m-round WS+RTDS recorded data (market + oracle streams) and the corresponding `prediction_polymarket_feed` outputs, enabling walk-forward analysis under the fixed fee model. citeturn8view1turn24view0turn28view0  
- Shadow-mode logs showing live-vs-sim fill divergence metrics across those same days.

## Thirty-day experiment template

**Template applies to both horizons; all parameters frozen at Day 0.**  
Populate and commit as two configs: `pm_btc_updown_15m_v1` and `pm_btc_updown_5m_v0`.

**Common fields (both):**  
- `config_hash`: generated at Day 0  
- `horizon_s`: 900 or 300  
- `entry_schedule`: fixed offsets from round start (e.g., `[+90s]`), plus a hard cutoff time  
- `conviction_thresholds`: mapping from `p_up_model` to bands (`medium/high/very_high`)  
- `regime_rules`: deterministic, frozen classifier rules  
- `execution_mode`: maker-first (post-only) with fallback to FOK; slippage caps  
- `quote_staleness_max_ms`, `oracle_staleness_max_ms`  
- `per_trade_risk`: fixed size by band (no martingale, no escalation)  
- `no_same_direction_stacking`: true  
- `concurrency_cap`, `kill_line`: loaded from `binary_lab_limits.json` at runtime  
- `observability`: enabled (event-sourced logs + daily checkpoint artefacts)

**15m populated defaults (baseline):**  
- `entry_offset_s = 90`  
- `cutoff_before_end_s = 120` (do not enter inside last 2 minutes)  
- `min_expected_edge = 0.0035` (net) to trade a very_high band  
- `trade_frequency_cap = 48 rounds/day` (example; must be replaced by actual limits file)

**5m populated defaults (experimental, guardrails):**  
- `entry_offset_s = 20`  
- `cutoff_before_end_s = 60`  
- `min_expected_edge = 0.0060` (net) (higher bar due to fragility)  
- `allowed_regimes = {strong_trend_only}` (pre-registered)  
- `latency_budget_ms = 250` (if exceeded, do not trade)

These defaults are placeholders pending the binding repo limits and admission schema; the system must refuse to trade until they are validated against the actual doctrine/config files.