# H-LIQ-001 acquisition addendum — 2026-10-30

Status: H-LIQ-001 remains UNTESTED. This authorizes infrastructure work only.

The original pre-registration file and its tracked SHA-256 remain byte-for-byte
unchanged: `16c7ad13079c0f8c95743a362520b143decaa3007bdb0321fd79eac5b5e59e63`.
This separately hashed addendum records an acquisition/provenance change from
monthly archives to official Binance USD-M daily archives with companion
`.CHECKSUM` verification and fixed-size, uncompressed column partitions.
It changes no event definition, cost, matching method, sample gate or success
threshold. Source archives and checksums are retained for this benchmark.

Before any market-data request, preselect **BTCUSDT, 2022-01-02 UTC** (the first
full study day with its previous day also inside the frozen research period).
Acquire exactly three daily archives: that day's aggTrades and 1-minute klines
for 2022-01-01 and 2022-01-02. No date substitution or outcome-based selection.
No cross-day aggregate-trade completeness claim is possible from this scope.

One worker; at most 100,000 rows per partition; one numeric library thread;
2 GiB cgroup memory maximum, zero swap allowance, one CPU quota; 900-second
whole-job timeout; 512 MiB compressed / 2 GiB decompressed / 10 million rows
per source; 4 GiB total output/write budget; minimum 3 GiB available disk and
1 GiB host MemAvailable, with 3 GiB MemAvailable required at initial admission
and host full-memory-pressure avg10 at most 1 percent. Stop on any violated limit, checksum/integrity failure
or cgroup memory-limit event. Preserve source evidence and explicit failure
records. No retries that select a different day or broaden coverage.

Proof is a one-day ingestion benchmark PASS/FAIL, with source hashes, sizes,
row counts, gaps, deterministic partition hashes, elapsed time, peak RSS,
cgroup memory events, host health and remaining disk. PASS applies only to
these inputs and limits. It is not a hypothesis verdict or proof of full-history
capacity. The existing monthly/full-tape APIs remain unsuitable for this host.

Before any later historical experiment, reconcile the original permission for
partial coverage with the four-year consistency criterion in a separate reviewed
record, preserving original hashes and thresholds. This addendum does not resolve
or waive that issue. Downstream bounded event reconstruction is outside this task.

Official source capability: https://github.com/binance/binance-public-data
(daily/monthly public data, futures schemas, SHA-256 .CHECKSUM companions).
