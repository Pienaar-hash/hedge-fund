# H-LIQ-001 bounded daily ingestion benchmark — 2026-10-30

**Infrastructure benchmark: FAIL. H-LIQ-001: UNTESTED.**

The one authorized benchmark stopped at the integrity gate. All three selected
archives passed published SHA-256 verification and ZIP CRC checks. Full-day
partitioning stayed within the enforced memory limit. Three discontinuities
between underlying trade-ID ranges triggered the deliberately conservative
acceptance rule. No date substitution, retry, historical expansion or outcome
analysis followed. The real-archive deterministic replay was not reached.

## Repository and acquisition boundary

Host: policy-ops-01; repository: `/home/policyops/research/hedge-fund`;
remote: `https://github.com/Pienaar-hash/hedge-fund.git`.
Branch: `research/liquidity-sweep-reclaim-001`.
HEAD at implementation and measurement: `b63adec9b6b200340f08826d31ad4ee482931047`.
Frozen base `cf25dd1727804ee8e014eb5ed96bcf0884f510ab` is an ancestor,
exactly one commit behind HEAD. Initial working tree was clean.

The original pre-registration and hash file are unchanged. Original SHA-256:
`16c7ad13079c0f8c95743a362520b143decaa3007bdb0321fd79eac5b5e59e63`.
The separate dated acquisition addendum retains that identity and has SHA-256:
`80d4489ebda5d6373cdd4275de87ba336c5d2f9b8cf40f0a6efb980c0c9ec84d`.
It changes no hypothesis parameter or success threshold. Partial historical
coverage versus the four-year consistency criterion remains unresolved for any
future hypothesis run.

Preselection, recorded before requests: BTCUSDT, **2022-01-02 UTC**, with that
day's aggTrades and 1m klines plus previous-day 1m klines. This is the first day
whose previous-day dependency is also within the frozen 2022–2025 period.

## Implemented scope

`research/liquidity/daily_ingestion.py` implements daily source identities,
companion checksum verification, capped streaming download, capped CSV lines,
explicit numeric parsing and fixed-size little-endian `.npy` column partitions.
It validates timestamp units/day/order, aggregate and underlying ID sequences,
numeric values, OHLC bounds and minute coverage across partition boundaries.
ZIP directory size/member count and declared/actual decompressed size are bounded
before accepting the source. Source archives/checksums remain on disk; failures
retain manifests and diagnostics. No complete trade tape is reconstructed.

`research/liquidity/benchmark_daily.py` fixes the symbol/date/dependencies,
requires the cgroup/CPU/thread limits and frozen repository boundary, and records
source, partition, process and host evidence. Both code modules, tests and addendum
were snapshotted before acquisition and hashed in `benchmark-result.json`.
Those files remained unchanged after measurement.

The legacy monthly loader and full-tape event engine are unchanged and remain
unsuitable for full-history processing on this host. No production path invokes
the new research modules.

## Enforced limits and host acceptance

Python 3.10.22 and all repository requirements use the previously verified isolated
environment at `/srv/policyops/data/H-LIQ-001/infra-20261030/venv`.
One synchronous worker, numerical threads set to one, low scheduling priority.

| Control | Enforced setting |
| --- | --- |
| Cgroup v2 memory.max | 2,147,483,648 bytes |
| Cgroup memory.swap.max | 0 |
| CPU quota | 100000 / 100000, one CPU |
| TasksMax | 16 |
| External timeout | 900 s; TERM, then KILL after 10 s |
| Partition rows | At most 100,000 |
| Per-source limits | 512 MiB compressed, 2 GiB CSV, 10 million rows |
| CSV line | At most 4,096 bytes |
| Whole-job write budget | 4 GiB |
| Disk floor | 3 GiB available, checked during work |
| Host RAM floor | 3 GiB MemAvailable at admission; 1 GiB during work |
| Host full memory PSI | avg10 at most 1 percent |

Physical RAM is 4,005,433,344 bytes, with no swap and two CPUs. The memory cap
leaves 1,857,949,696 bytes (1.73 GiB) outside the research job. This does not
reserve resources for unrelated services. No system configuration was changed.

## Measured sources and gaps

All source identities are under
`https://data.binance.vision/data/futures/um/daily/`.
Full URLs, expected/actual SHA-256 values, coverage, partition sizes/hashes and
gap records are retained in the evidence manifest.

| Source | ZIP bytes | CSV bytes | Rows | Observed coverage/gaps |
| --- | ---: | ---: | ---: | --- |
| BTCUSDT 2022-01-01 1m | 65,314 | 162,667 | 1,440 | No missing minute intervals |
| BTCUSDT 2022-01-02 1m | 64,860 | 162,410 | 1,440 | No missing minute intervals |
| BTCUSDT 2022-01-02 aggTrades | 13,571,348 | 66,671,670 | 1,002,681 | Zero aggregate-ID gaps; three unrepresented underlying IDs |

Each ZIP digest matched its source `.CHECKSUM`; each CSV was read to EOF with
CRC verification. Aggregate timestamps were ordered and inside the selected UTC
day. The largest observed inter-trade silence was 4,479 ms; leading/trailing
silence was 99/20 ms. Inactivity is not classified as missing data. Cross-day
aggregate-ID continuity was not assessed in this one-day trade scope.

Underlying ranges have one unrepresented ID after each aggregate ID
998946303, 998946314 and 999064862. The implementation counts these in
`missing_underlying_trade_ids` and returns INVALID/GAPS_DETECTED, causing the
benchmark to fail after all trade rows were parsed. These labels describe the
guard's decision, **not proven missing aggregate records or corrupt archives**.

Binance documents that aggregate data includes market trades and excludes
insurance-fund and ADL trades. Thus underlying-ID contiguity is stronger than
aggregate-ID contiguity; the three observations cannot automatically establish
missing primary observations. The exclusions do not prove the cause of these
specific discontinuities. Source-contract review is needed before changing the
acceptance rule; the failed result has not been relabelled.
[Official Binance aggregate-trade contract](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Compressed-Aggregate-Trades-List).

## Measured capacity and validation

| Measurement | Result |
| --- | ---: |
| End-to-end process wall time, including startup | 33.88 s |
| Driver elapsed time | 32.05 s |
| aggTrades download/verification/partitioning | 29.69 s |
| Peak process RSS | 53,612 KiB / 52.36 MiB |
| Peak cgroup charged memory | 98,418,688 bytes / 93.86 MiB |
| Cgroup max / OOM / OOM-kill events | 0 / 0 / 0 |
| Source ZIP total | 13,701,522 bytes |
| Derived partitions | 48,253,616 bytes, 11 trade partitions plus 2 kline partitions |
| Minimum sampled host MemAvailable | 3,258,535,936 bytes / 3.03 GiB |
| Host memory full avg10 maximum | 0.00 percent |
| Disk after benchmark | 33,392,136,192 bytes / 31.10 GiB |

Host memory PSI cumulative counters did not increase during the run; load average
was 0.09 before and 0.52 after (two CPUs). Disk declined by about 59.6 MiB.
Resource acceptance passed; integrity acceptance failed. Process RSS and cgroup
charge are different accounting measures. Measurements include the integrity-failed
full-day extraction and do not demonstrate successful end-to-end acceptance.

Focused liquidity tests: **63 passed**, including 24 new offline daily-ingestion
checks/parameter cases. Tests cover checksum failure before parsing, CRC failure,
day/unit errors, sequence gaps/duplicates across partitions, hard allocation sizes,
download/row/byte/time/disk/RAM/pressure limits and deterministic partition replay.
`make test-fast`: **12 passed**. Focused lint and whitespace checks passed.
Tests used the same enforced memory/CPU limits. Fixture tests compute no real-data
outcomes. Determinism is demonstrated on fixtures; real-archive file hashes exist,
but the scheduled real-archive replay was skipped when the integrity gate failed.

## Evidence and bounded successor

Evidence root: `/srv/policyops/data/H-LIQ-001/daily-20261030`.
It contains the launcher, code snapshot, source selection timestamp, test logs,
GNU time output, before/after cgroup controls/events, per-source manifests and
FAILURE record, original ZIPs/checksums and partition hashes. `source-manifest.json`
was assembled from all three per-source manifests after the stop, without further
ingestion/replay. Its SHA-256 is:
`2008047763ed526cd70d28d95231279132812129b54d5d5028aeb61aadc6d5fd`.

The next proposed bounded action is a source-contract review of the underlying-ID
guard using the retained observations and official schema. Any correction should
have explicit tests and a new reviewed benchmark authorization. Do not silently
weaken the guard and rerun until PASS. No additional archives are needed for this
review. If a subsequent benchmark is authorized, retain the same day, one worker,
100,000-row partitions, cgroup, timeout and disk/RAM floors; finish local archive
replay before any broader capacity claim.

Only after ingestion acceptance could a separately authorized streaming event
engine be assessed. It would need bounded rolling context, prior-day levels,
cross-midnight handling and compact event storage; the current full-array engine
is not that implementation. No four-year runtime/storage estimate is established
by this single-day input. Full-history retention does not fit an assumed
capacity plan merely because this day fits.

No hypothesis verdict, forward-return evidence or trading edge is established.
No frozen hypothesis edits, weekly-production changes, service restarts or capital
interaction occurred. Stop upheld; no automatic expansion.
