# PDH/PDL rejection candle screen — result, 2026-10-30

**Decision: REJECT / DEPRIORITIZE THE SCREENED MECHANISM.**

The five-minute candle rejection does not show an economically useful 15-minute
reversal effect at 15 bp round-trip costs. Among **1,415 rejections**, median gross
return was **+4.60 bp** and median net return **−10.40 bp**. The **1,530 acceptance**
controls had median gross **+4.58 bp**, net **−10.42 bp**. Rejection's median advantage
was **0.02 bp**, with a mean advantage of **1.36 bp**. Its own pooled mean was
**−2.20 bp gross / −17.20 bp net**. These results do not justify more precise
validation of this screen or opening the held-out 2025 data.

This is **PDH_PDL_CANDLE_SCREEN_001**, an exploratory candle screen, **not** the
frozen H-LIQ-001 primary test. H-LIQ-001 remains UNTESTED; its previous results,
pre-registration and source-contract evidence were not changed. No trade-ID
attribution or ingestion-governance work was resumed.

## Fixed design and timing

The [screen definition](PDH_PDL_CANDLE_SCREEN.md) was written and hashed before
market-data acquisition and outcome inspection. Its thresholds and the code were
unchanged throughout the single run. Definition SHA-256:
`950e32f558862216ec7c8f88f0a7cc254ea2332ad64d02cad25a8229ef108dd9`.
Code SHA-256: `85afc0044a85ad318ddb15adc44a3f02e11a1d6fc3caef32e2a12b08ed478d6f`.

Venue: Binance USD-M perpetuals, BTCUSDT/ETHUSDT/SOLUSDT, 2022–2024 UTC. Previous
completed-day highs/lows require a complete 1,440-minute day. The first minute
strictly above PDH / below PDL is the only candidate for that side each day.
Five bars, including that penetration minute, form the observation window. At its
close, a strict close back inside is rejection and a strict close outside is
acceptance; equality is excluded. No intrabar ordering, trade VWAP, penetration
cap, filters, matching or parameter search. Same-bar first penetrations of both
levels are excluded as ambiguous.

Entry is the following minute's open, after the fifth bar has closed; exit is
exactly 15 minutes after entry. All 21 required minute timestamps must be
contiguous. Both classes use the same reversal sign: negative log price return
at PDH, positive log price return at PDL, multiplied by 10,000. Deduct 15 bp once.
No event can use 2025 context. Day openings already outside a level count as
penetrations. At most two events per symbol/day; no overlap embargo.

## Sample and effects

All returns below are **basis points**. Each gross/net pair subtracts the same
15 bp cost. R−A compares unpaired class means or medians, not matched event pairs;
equal costs cancel in that difference. Results are equally weighted by event.

| Group | Rejection n | Acceptance n | Rejection mean gross / net | Rejection median gross / net | Acceptance mean gross / net | Acceptance median gross / net | R−A mean / median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pooled | 1,415 | 1,530 | -2.20 / -17.20 | 4.60 / -10.40 | -3.56 / -18.56 | 4.58 / -10.42 | 1.36 / 0.02 |
| BTCUSDT | 455 | 519 | 0.65 / -14.35 | 2.83 / -12.17 | -2.45 / -17.45 | 3.81 / -11.19 | 3.10 / -0.98 |
| ETHUSDT | 488 | 504 | 1.82 / -13.18 | 6.06 / -8.94 | -4.31 / -19.31 | 4.54 / -10.46 | 6.13 / 1.52 |
| SOLUSDT | 472 | 507 | -9.12 / -24.12 | 4.54 / -10.46 | -3.96 / -18.96 | 6.64 / -8.36 | -5.16 / -2.10 |
| 2022 | 452 | 544 | -3.58 / -18.58 | 5.69 / -9.31 | 0.62 / -14.38 | 4.30 / -10.70 | -4.20 / 1.38 |
| 2023 | 517 | 425 | -0.28 / -15.28 | 4.61 / -10.39 | -4.74 / -19.74 | 9.07 / -5.93 | 4.46 / -4.46 |
| 2024 | 446 | 561 | -3.03 / -18.03 | 3.53 / -11.47 | -6.73 / -21.73 | 1.83 / -13.17 | 3.70 / 1.69 |

Every rejection **symbol** median net return is negative, and so is every
rejection **year** median net return. Mean differences favor rejection in BTC/ETH
but favor acceptance in SOL. Median differences favor rejection only in ETH among
symbols, and in 2022/2024 among years; their pooled value is essentially zero.
There is no consistent, cost-surviving effect.

### Symbol-year detail

| Group | Rejection n | Acceptance n | Rejection mean gross / net | Rejection median gross / net | Acceptance mean gross / net | Acceptance median gross / net | R−A mean / median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT-2022 | 143 | 193 | 2.23 / -12.77 | 1.30 / -13.70 | 2.80 / -12.20 | 3.81 / -11.19 | -0.56 / -2.51 |
| BTCUSDT-2023 | 168 | 140 | 0.53 / -14.47 | 2.46 / -12.54 | 3.89 / -11.11 | 11.52 / -3.48 | -3.35 / -9.06 |
| BTCUSDT-2024 | 144 | 186 | -0.80 / -15.80 | 3.78 / -11.22 | -12.67 / -27.67 | -3.60 / -18.60 | 11.87 / 7.38 |
| ETHUSDT-2022 | 163 | 176 | 3.70 / -11.30 | 6.20 / -8.80 | -0.88 / -15.88 | 2.28 / -12.72 | 4.58 / 3.92 |
| ETHUSDT-2023 | 173 | 148 | -2.12 / -17.12 | 5.77 / -9.23 | -6.76 / -21.76 | 6.28 / -8.72 | 4.64 / -0.52 |
| ETHUSDT-2024 | 152 | 180 | 4.30 / -10.70 | 6.55 / -8.45 | -5.65 / -20.65 | 5.32 / -9.68 | 9.95 / 1.23 |
| SOLUSDT-2022 | 146 | 175 | -17.42 / -32.42 | 8.16 / -6.84 | -0.27 / -15.27 | 9.76 / -5.24 | -17.15 / -1.60 |
| SOLUSDT-2023 | 176 | 137 | 0.74 / -14.26 | 6.89 / -8.11 | -11.37 / -26.37 | 9.73 / -5.27 | 12.12 / -2.84 |
| SOLUSDT-2024 | 150 | 195 | -12.61 / -27.61 | -0.09 / -15.09 | -2.07 / -17.07 | 2.17 / -12.83 | -10.54 / -2.25 |

All nine rejection symbol-year median net returns are negative. The smallest
class in any symbol-year contains **137 observations**, so this rejection is not
caused by failure to meet the screen's minimum sample rule.

## Fixed screening decision

All criteria were required to justify more precise validation. No criterion was
changed after examining the result.

| Criterion | Observed | Result |
| --- | --- | --- |
| At least 50 of each class in every symbol-year | Minimum 137 | PASS |
| Pooled rejection median net ≥ 5 bp | −10.40 bp | FAIL |
| Pooled R−A median difference ≥ 10 bp | +0.02 bp | FAIL |
| Rejection median net and median difference both positive for all symbols | 0 of 3 satisfy both | FAIL |
| Both positive in at least two years | 0 of 3 satisfy both | FAIL |

The screen fails far below its economic criteria: positive gross medians of a few
basis points do not survive the stated costs, and rejection barely distinguishes
itself from acceptance. Stop; do not tune thresholds or inspect 2025 to rescue it.
This is a practical rejection of this candle-observable formulation, not formal
falsification of the different trade-level H-LIQ-001 hypothesis.

## Data coverage and limitations

Official source: [Binance public archive documentation](https://github.com/binance/binance-public-data),
USD-M 1-minute klines. Retained **111 archives**: 108 monthly files for the screen
and three daily 2021-12-31 files used only for previous-day levels. All 111 match
their published SHA-256 checksum companions and pass ZIP CRC validation. Total
ZIP bytes: **189,920,281**. Parsed candles including dependencies: **4,731,840**.

| Symbol | Complete screen days | Missing whole days | Additional unavailable-level days |
| --- | ---: | ---: | ---: |
| BTCUSDT | 1,096 / 1,096 | 0 | 0 |
| ETHUSDT | 1,096 / 1,096 | 0 | 0 |
| SOLUSDT | 1,091 / 1,096 | 5 | 2 |

SOL's checksum-verified official files omit **7,200 minute intervals**: 2022-02-26
through 2022-02-28 (4,320 minutes), and 2022-04-01 through 2022-04-02 (2,880).
Those five days are absent. March 1 and April 3 also lack complete previous-day
levels and are excluded. No gaps are filled or reconstructed from another source.
There were no observed duplicate/out-of-order timestamps or corrupt OHLC values.

Of the two potential daily sides, no penetration occurred 1,218 times for BTC,
1,199 for ETH and 1,193 for SOL. Exact fifth-close equality excluded one ETH and
six SOL candidates. No double-penetration ambiguities or unavailable exit windows
occurred among otherwise eligible candidates. The source/gap files identify the
missing intervals and per-archive counts explicitly.

This is one venue and one fixed, exploratory definition, without matching,
confidence intervals or multiple-testing claims. Overlapping events and shared
market shocks make samples dependent. Candle wicks cannot reconstruct exact trade
crossing times, intrabar sequence or the original 30-second VWAP. Candle opens are
hypothetical proxies; 15 bp is an assumed all-in cost, not demonstrated fills.
Missing SOL periods limit coverage. No claim is made about stop orders or market
intent. These limitations do not create evidence for a precision-validation step
when the observed net effect is negative across every symbol and year.

**2025 was neither downloaded nor inspected.** A future study would require
separate authorization; this result supplies no reason to initiate it.

## Implementation and verification

One standalone [research script](../../research/liquidity/candle_screen.py), using
existing requests/NumPy and the isolated Python 3.10 environment; one download
worker; one day plus 20 following minutes in memory. No new framework or dependency.
The script does not import production or trade-level ingestion modules.

**12 focused offline tests passed**, covering previous-day levels, first crossings,
classification symmetry/equality, next-minute entry, exact 15-minute holding time,
cost applied once, ambiguous bars, missing context, midnight and 2025 exclusion.
**12 `make test-fast` checks passed.** Focused lint and whitespace checks passed.
Event-output checks confirm next-minute entry, exact holding time, costs and no
2025 exits; definition/code hashes match their before-run snapshots.

Single-run process wall time: **278.41 s**; peak RSS **53,484 KiB (52.23 MiB)**.
Existing host cgroup enforced 2 GiB memory, zero swap allowance and one CPU quota;
no memory-limit or OOM events occurred. Approximately 30.86 GiB disk remained.
These are execution details for this screen, not another capacity investigation.

## Artifacts and stop

Evidence root: `/srv/policyops/data/PDH-PDL-CANDLE-SCREEN-001`.
Run artifacts: `run-20261030/results.json`, `events.csv`, `sources.json`, `gaps.json`
and `runtime.json`; original archives/checksums remain in `run-20261030/archives`.
The root retains the before-run definition/code, logs and test results. The JSON
contains full-precision counts/effects and all fixed decision gates.

No existing tracked repository file was changed. New files are the definition,
script, focused tests and this result. No production changes, trading paths,
paper trading, capital interaction or new research-governance framework.
**Stop after this screen.**
