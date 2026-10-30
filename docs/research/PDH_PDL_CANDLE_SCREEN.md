# PDH/PDL candle rejection screen — fixed definition, 2026-10-30

This is **PDH_PDL_CANDLE_SCREEN_001**, a cheap exploratory screen, not the frozen
H-LIQ-001 primary test. Existing H-LIQ-001 files/results remain unchanged.

Source: official Binance USD-M perpetual futures 1-minute OHLC candles for
BTCUSDT, ETHUSDT and SOLUSDT. Initial screen: 2022-01-01 through 2024-12-31 UTC.
**2025 is held out: do not download or inspect it in this screen.** One daily
2021-12-31 candle archive per symbol supplies the first day's structural levels.
Monthly archives and their published SHA-256 checksums are retained locally.

For day D, PDH/PDL are the maximum high/minimum low over the complete UTC day D-1.
Require all 1,440 unique, ordered minute bars for both level day and crossing day;
exclude incomplete days, never fill gaps. Only the first bar with high > PDH and
first bar with low < PDL are candidates, at most two per symbol/day. Opening gaps
outside a level count as penetrations. If both first penetrations occur in the same
bar, exclude both as ambiguous; do not infer intrabar order or choose later crossings.

The five-minute observation window includes the penetration bar and four following
bars. At its close, upper rejection means close < PDH; upper acceptance means
close > PDH. Lower rejection means close > PDL; lower acceptance means close < PDL.
Exact equality is excluded. No penetration cap, VWAP, volume filter, matching,
parameter search or trade-level data. The state is not knowable before that close.

Entry: open of the next minute (penetration bar index + 5). Exit: open exactly
15 minutes after entry (index + 20). Require a contiguous 21-minute sequence,
including boundary context from the following screen day where necessary. Never
use a 2025 bar; events whose windows exceed the screen end are excluded explicitly.
Signed gross return in bp = sign * 10,000 * ln(exit_open / entry_open), with sign
-1 for upper and +1 for lower, for **both classes**. Net = gross minus 15 bp once.
These are hypothetical candle-open proxies, not execution evidence.

Keep both first-level candidates when at different minutes; no overlap embargo.
Report dependence/overlap as a limitation, not independent statistical significance.
Report rejection/acceptance counts, mean/median gross/net, and rejection-minus-
acceptance mean/median differences pooled, by symbol, year and symbol-year.
Equal costs cancel in between-class differences. No bootstrap or p-value claim.

Before inspecting outcomes, fix this screening decision:
justify more precise validation only if both classes have at least 50 observations
in every symbol-year cell, pooled rejection median net >= 5 bp (gross >= 20 bp),
pooled rejection-minus-acceptance median difference >= 10 bp, rejection median net
and median difference > 0 for all three symbols, and both positive in at least two
of three years. Otherwise **reject/deprioritize this candle-screen mechanism**.
These modest economic margins select a survivor for further testing; they do not
establish tradable edge or formally falsify H-LIQ-001. No thresholds may be tuned
after the run. A survivor justifies a separately authorized 2025 validation; it does
not authorize opening the held-out data now.

Implementation: one research script and focused offline tests, using existing
requirements/Python environment. One download worker; process one day of candles
plus twenty next-day bars in memory. Preserve source sizes/hashes, gaps, event rows,
code/definition hashes and results. Run under the existing host's 2 GiB/no-swap,
one-CPU cgroup. No production integration, governance cards or new framework.
Stop after this one screen and its result, regardless of direction.
