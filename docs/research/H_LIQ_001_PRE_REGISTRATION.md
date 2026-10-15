# H-LIQ-001 Pre-Registration — Previous-Day Structural Sweep/Reclaim

```text
hypothesis_id            = H-LIQ-001
repository                = Pienaar-hash/hedge-fund
repository_base_sha       = cf25dd1727804ee8e014eb5ed96bcf0884f510ab
research_branch           = research/liquidity-sweep-reclaim-001
pre_registration_timestamp = 2026-09-03T00:00:00Z
pre_registration_version = 1
```

This document is frozen before any forward-return outcome is examined. Once results
exist, the primary specification below must not change. Any substantive correction
before the first result run must be recorded explicitly in a dated addendum at the
end of this file, with the hash of the corrected document recomputed and both hashes
retained in `H_LIQ_001_PRE_REGISTRATION.sha256`.

## Research question

Does penetration and rapid reclaim of an ex-ante previous-day high or low produce a
materially different, cost-adjusted 15-minute forward-return distribution from
comparable crossings where price remains accepted beyond the level?

## Mechanism under test

```text
known price level (PDH/PDL)
  -> contemporaneous first crossing
  -> bounded excursion
  -> fixed-time reclaim classification
  -> conditional forward return
```

Explicitly NOT under test:

```text
past directional variable -> future directional return
```

No claim is made about resting stop orders, institutional intent, or manipulation.
Only observable price behaviour around an ex-ante structural level is tested.

## Venue and symbols

```text
venue   = Binance USD-M perpetual futures
symbols = BTCUSDT, ETHUSDT, SOLUSDT
```

## Primary dataset

```text
aggTrades   (Binance public historical archive, monthly files)
1m klines   (Binance public historical archive, monthly files)
```

Source: `https://data.binance.vision/data/futures/um/monthly/{aggTrades,klines}/...`

## Research period

```text
2022-01-01T00:00:00Z through 2025-12-31T23:59:59Z
```

No partial 2026 data. Every requested month must have a proven, hashed manifest
entry before it contributes events. Missing or corrupt source files are recorded as
a documented gap for that symbol-month and are excluded, never forward-filled or
reconstructed from klines.

### Documented environment constraint (recorded before any result is examined)

This tranche executes inside a disk- and time-bounded research sandbox (single
worktree, single disk volume). Full raw retention of every monthly aggTrades archive
for all 48 months x 3 symbols was verified to require substantially more storage than
the environment provides. The ingestion pipeline therefore hashes each monthly
archive on download and retains only a compact three-column (price, quantity,
transact_time) derived extract before deleting the raw archive. This preserves the
provenance hash of the original source file without requiring the raw file to remain
on disk. Whatever subset of the 48 x 3 symbol-months is actually completed within
the available session time is recorded, file by file, in the source manifest
(Section 8 of the tranche instructions). Symbol-months not completed are recorded as
`gap_status = NOT_INGESTED_ENVIRONMENT_BUDGET`, not as zero activity, and are excluded
from event construction. This constraint is disclosed here, before any outcome is
computed, precisely so it cannot be invoked after the fact to explain away a result.

## Structural level definition

```text
PDH(D) = max(high) over completed UTC day D-1
PDL(D) = min(low)  over completed UTC day D-1
```

Levels become observable at `00:00:00 UTC` on day `D`. No information from day `D`
may contribute to constructing its own PDH/PDL. Only the first outside crossing of
each level during day `D` may create a candidate event.

## Crossing definition

```text
PDH: previous_trade.price <= PDH  and  current_trade.price > PDH
PDL: previous_trade.price >= PDL  and  current_trade.price < PDL
```

`t0` = exchange timestamp of the current (crossing) aggregate trade.

## Classification timestamp

```text
tC = t0 + 5 minutes
```

No signal, entry proxy, or outcome may reference information before `tC`.

## Sweep/reclaim definition

Upper (PDH):

```text
max(trade.price for trade in [t0, tC]) <= PDH * 1.0025
VWAP30(tC) <= PDH * 0.9998
```

Lower (PDL):

```text
min(trade.price for trade in [t0, tC]) >= PDL * 0.9975
VWAP30(tC) >= PDL * 1.0002
```

`VWAP30(tC)` is the trade-notional-weighted VWAP over the 30 seconds ending at `tC`.

## Accepted-breakout control definition

```text
PDH acceptance: VWAP30(tC) >= PDH * 1.0005
PDL acceptance: VWAP30(tC) <= PDL * 0.9995
```

Events between the reclaim and acceptance thresholds enter neither primary class
(intentional deadband).

## Entry proxy

```text
entry_target = tC + 1 second
```

Use the first aggregate trade at or after `entry_target`. If no valid trade occurs
within 5 seconds, invalidate the event (`reason = no_execution_proxy`).

## Outcome definition

```text
primary horizon   = 15 minutes after tC
secondary horizons = 5 minutes, 60 minutes (non-gating)

upper: signed_return = -ln(exit_price / entry_price)
lower: signed_return = +ln(exit_price / entry_price)
```

Positive signed return always means movement in the hypothesised reversal direction.

## Cost assumptions

```text
primary round-trip cost = 15 bp
stress                  = 25 bp, 40 bp
```

No maker-fill assumption. Taker-like execution assumed.

## Pre-event matching variables (strictly prior to t0)

```text
trailing 30-minute realised volatility (from completed 1m returns)
trailing 30-minute aggregate-trade quote notional
UTC 4-hour block
calendar quarter
symbol
level side (PDH / PDL)
```

## Matched control algorithm

Eligible accepted-breakout controls: same symbol, same level side, same calendar
quarter, same UTC 4-hour block, realised volatility within factor 2, quote volume
within factor 2. Choose the unused control minimizing
`abs(ln(RV_sweep/RV_control)) + abs(ln(vol_sweep/vol_control))`, no replacement,
deterministic tie-break by earliest control timestamp. Sweeps without an eligible
control are retained in unconditional statistics only.

## Event independence (overlap embargo)

After accepting an event for a symbol, reject further candidates for that symbol
with `t0 < prior_event.tC + 60 minutes`. Cross-symbol simultaneous events are
retained and handled via UTC-week cluster resampling, not deletion.

## Minimum sample gate

```text
qualified sweep/reclaim events >= 500 total
qualified sweep/reclaim events >= 100 per symbol
matched sweep/control pairs    >= 400
```

Below this: `VERDICT = INADEQUATE SAMPLE`, not `FALSIFIED`. This gate is not lowered
under any circumstance, including the documented environment constraint above.

## Primary statistics

```text
P1 = median 15-minute signed sweep/reclaim return, net of 15 bp cost
P2 = median matched difference: sweep/reclaim signed return - accepted-breakout signed return
```

Also reported: mean, SD, quantiles, count, hit rate (secondary), MFE, MAE, symbol
and year attribution.

## Inference

10,000-resample UTC-week cluster bootstrap, 95% CI, deterministic seed (recorded).
Matched-pair permutation test as secondary confirmation.

## Pre-registered success criteria (ALL must hold)

```text
1. sample gate clears
2. pooled median gross 15-minute reversal return           >= 30 bp
3. pooled median net return after 15 bp cost                >= 15 bp
4. UTC-week cluster-bootstrap 95% CI for net median entirely > 0
5. median matched sweep-minus-acceptance difference          >= 20 bp
6. cluster-bootstrap 95% CI for matched difference entirely > 0
7. pooled median remains positive under 25 bp stress cost
8. matched-difference point estimate positive for BTC, ETH, and SOL individually
9. matched-difference point estimate positive in >= 3 of 4 calendar years actually covered
```

Criterion 9 is evaluated over whatever calendar years the actually-ingested data
spans, per the documented environment constraint; it is not waived.

## Falsification rule

If data integrity and sample gates pass but any primary criterion above fails:
`H-LIQ-001 = FALSIFIED`. No post-hoc adjustment of the 5-minute window, 25bp
penetration cap, 2bp reclaim threshold, primary horizon, symbol universe, PDH/PDL
definition, cost assumption, or matching method is permitted to rescue the result.
Such a change would constitute a new hypothesis with a new ID.

## Predeclared robustness checks (non-gating, cannot convert FAIL to PASS)

```text
penetration cap: 10bp, 50bp
classification horizon: 2min, 10min
level class: previous-week high/low
time split: first half vs second half of ingested period
symbol: BTC / ETH / SOL individually
cost: 25bp, 40bp
funding-time proximity: near vs away from scheduled funding
structural confluence: PWH/PWL within 10bp vs not
```

## Stop condition

Exactly one frozen primary run over whatever data is actually ingested and hashed
within this tranche. Then stop. Do not begin H-LIQ-002. Do not purchase L2 data. Do
not alter production.

## Forbidden during this tranche

Modifying weekly production files, `execution/signal_screener.py`,
`execution/orderbook_features.py`, any live executor, sizing, risk, scheduler,
collector daemon, paper-trading path, or production config. No orders. No capital
exposure. No runtime claims beyond what is directly inspected.

---

*Hash of this document at freeze time is recorded in
`docs/research/H_LIQ_001_PRE_REGISTRATION.sha256`.*
