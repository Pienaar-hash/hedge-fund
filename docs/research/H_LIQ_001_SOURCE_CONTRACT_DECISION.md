# H-LIQ-001 source-contract reconciliation — 2026-10-30

Decision: **all three specific anomalies remain UNRESOLVED_SOURCE_ANOMALY**.
The previous infrastructure benchmark remains **FAIL**. H-LIQ-001 remains
**UNTESTED**. The diagnostic classification is corrected; acceptance is still
blocked. No benchmark was rerun, day substituted, new period acquired, full-day
partition replay performed or outcome examined.

## Repository and preserved evidence

Host policy-ops-01; repository `/home/policyops/research/hedge-fund`;
branch `research/liquidity-sweep-reclaim-001`;
HEAD during investigation: `b63adec9b6b200340f08826d31ad4ee482931047`.
The six existing uncommitted files were snapshotted before this investigation.
All **88 files** inside the previous benchmark directory were SHA-256 inventoried
before and after this work and are unchanged, including the FAIL result,
INVALID/GAPS_DETECTED source manifest, original archives and partitions.
The previous benchmark report, acquisition addendum/hash and benchmark driver
were not edited.

Frozen pre-registration SHA-256 remains:
`16c7ad13079c0f8c95743a362520b143decaa3007bdb0321fd79eac5b5e59e63`.
No hypothesis, date range, event definition or success threshold changed.

Evidence root: `/srv/policyops/data/H-LIQ-001/source-contract-20261030`.
The original BTCUSDT 2022-01-02 archive was rehashed locally against its retained
manifest and source checksum identity:
`87f3d9e800bf9abec643ce0e9add98eedc927bea4c8006da652235ea779cc429`.
No market-data request was made.

## Established observations

A read-only streaming scan of the retained ZIP selected eight original CSV rows
per anomaly: aggregate IDs `a-3` through `a+4`, where `a` is the ID before the
discontinuity. All **24 original rows**, including the original header, are
preserved byte-for-byte in three CSV context files and in the regression fixture.
The scan stopped after the final context; it did not rebuild daily partitions.
The inspection script and GNU time record are retained in the evidence directory.
It ran under the 2 GiB/no-swap cgroup and one CPU quota, with bounded lines and
memory-mapped existing partitions. No full-day arrays were loaded.

The 24 rows match the original six partition columns (aggregate ID, underlying
first/last IDs, timestamp, price and quantity). The 12 relevant partition column
files in partitions 4 and 6 match their retained SHA-256 values. Context JSON also
records original CSV line numbers, uncompressed byte offsets, data-row indices,
partition locations and context hashes.

All three anomaly pairs have consecutive aggregate IDs, strictly increasing
timestamps and non-overlapping, increasing underlying ranges. Their eight-row
contexts also have consecutive aggregate IDs, nondecreasing timestamps and no
underlying overlap. No missing aggregate ID or ordering defect was observed in
these windows. The prior full-day manifest reports zero aggregate-ID gaps;
this investigation independently checked the windows, not the entire trade day.

| Consecutive aggregate IDs | Previous f–l | Next f–l | Unrepresented underlying ID | Previous → next UTC time on 2022-01-02 | Original partition / row pair (zero-based) |
| --- | --- | --- | --- | --- | --- |
| 998946303 → 998946304 | 1776452762–1776452762 | 1776452764–1776452769 | 1776452763 | 14:49:14.680 → 14:49:15.346 (666 ms) | 4 / 84215–84216 |
| 998946314 → 998946315 | 1776452800–1776452800 | 1776452802–1776452802 | 1776452801 | 14:49:17.219 → 14:49:17.282 (63 ms) | 4 / 84226–84227 |
| 999064862 → 999064863 | 1776671920–1776671920 | 1776671922–1776671922 | 1776671921 | 16:55:29.540 → 16:55:29.554 (14 ms) | 6 / 2774–2775 |

Every pair is inside an original 100,000-row partition, away from a chunk boundary.
The discontinuities are present in the source CSV itself; chunk partitioning did
not introduce them.

## Source contract and limits of attribution

Binance's [USD-M aggregate-trade documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Compressed-Aggregate-Trades-List)
identifies `a` as the aggregate-trade ID, `f`/`l` as underlying first/last trade IDs
and `T` as the timestamp. Its market-trade eligibility rule excludes insurance-fund
and ADL trades. The official [public archive documentation](https://github.com/binance/binance-public-data)
connects futures aggTrades archives with the aggregate-trade endpoint and describes
the same seven source fields. Both sources were checked on 2026-10-30.

Therefore `next.f != previous.l + 1` is not equivalent to a missing aggregate-trade
record. Treating these two observations as the same data defect was an unsupported
validation assumption. Feed eligibility provides a possible explanation for skipped
underlying IDs; it does not identify the cause of these three discontinuities.

The retained aggregate rows carry no event-type evidence for the unrepresented
underlying IDs. No contemporaneous raw-trade audit, exchange attribution or
2022-specific eligibility/version guarantee was supplied. Current documentation
alone cannot prove that any of the three IDs belonged to an excluded insurance-fund
or ADL event. Aggregate-ID continuity also cannot prove absence of upstream
omissions before aggregate IDs were assigned.

| Candidate classification | Finding for each of the three anomalies |
| --- | --- |
| Source-permitted underlying-ID exclusion | Compatible with documented eligibility; specific cause not proved |
| Missing aggregate-trade record | No missing aggregate ID observed at these boundaries; not established |
| Ordering or overlap inconsistency | None observed in original contexts or their saved partitions |
| Unresolved source anomaly | **Retained classification for all three** |

## Minimal validation correction

Only classification/diagnostics changed in the existing daily module. New manifests
identify `validation_rule_version = 2`:

| Observation | Validation result | Acceptance |
| --- | --- | --- |
| No detected sequence/minute gaps | VALID / OK_WITHIN_DAY | Existing checks still apply |
| Missing aggregate IDs or 1m intervals | INVALID / GAPS_DETECTED | Blocked |
| Underlying-ID discontinuity without an aggregate-ID gap | REVIEW_REQUIRED / UNDERLYING_ID_DISCONTINUITY_UNRESOLVED | **Blocked** |
| Duplicate/reversed IDs, overlap, timestamp regression | IntegrityError, unchanged | Blocked |

New diagnostics distinguish `missing_aggregate_ids`, `unrepresented_underlying_ids`
(count of ID values between adjacent ranges) and `underlying_id_discontinuities`
(count of affected boundaries). They do not claim that an unrepresented ID is a
missing primary record. The new rule removes the misleading
`missing_underlying_trade_ids` field; bounded samples use
`unrepresented_underlying_ids` instead of `missing_raw_ids`.
If both kinds of discontinuity occur, INVALID retains precedence while both counts
are recorded. The ingestion wrapper still raises for every status other than VALID
and preserves REVIEW_REQUIRED in its failure diagnostics. No automatic exception
or source-permitted allowlist was introduced. Old manifests are immutable evidence
of rule version 1 and have not been rewritten or promoted.

## Exact changes and validation

Relative to the start of this investigation:

- Modified `research/liquidity/daily_ingestion.py`: separate diagnostics and
  unresolved status, rule version, preserve unresolved failure status; acceptance
  remains blocked.
- Modified `tests/research/liquidity/test_daily_ingestion.py`: update renamed field
  assertion; add 14 regression cases for original contexts, independent gap types,
  overlaps/order and rejection through the ingestion wrapper.
- Added `tests/research/liquidity/fixtures/h_liq_001_underlying_id_context.json`:
  three original eight-row windows, their hashes and original archive identity.
- Added this decision record. No other existing repository file was changed.

Focused suite: **77 passed in 5.08 s** (7.66 s process wall time), including the
original 63 cases and 14 new cases. Tests use only offline fixtures, with the three
eight-row windows tested both in one chunk and across a synthetic chunk boundary.
The wrapper rejection test mocks local download bytes and proves REVIEW_REQUIRED
cannot enter the accepted path. `make test-fast`: **12 passed**. Focused lint and
whitespace checks passed. Tests ran in the existing isolated Python 3.10.22
environment under the same 2 GiB/no-swap and one CPU controls.
No historical benchmark or real-data event/outcome test was run.

## Authorized closeout — 2026-10-30

The user decided to retain all three anomalies as unresolved and close this bounded
investigation. Further attribution is **not a prerequisite for closing this review**.
No open-ended exchange investigation or event-specific attribution is requested.

The closeout records these separate conclusions:

- `REVIEW_REQUIRED` remains blocking under the current integrity contract. No
  admission exception or allowlist was added.
- The three anomalies remain unattributed. They are not proven missing market
  data, missing aggregate IDs, duplicate records, timestamp reversals, overlaps
  or chunk-boundary defects.
- The original one-day benchmark remains **FAIL**, with its result and source
  evidence unchanged. The corrected validator was not rerun against that day.
- Measured resource performance is established for that selected day independently
  of admission: 1,002,681 aggregate rows, 33.88 s process wall time, 53,612 KiB
  (52.36 MiB) peak RSS, 2 GiB/no-swap cgroup and zero memory-limit/OOM events.
  These measurements do not establish full-history capacity or source acceptance.
- No new date, historical outcome analysis, benchmark rerun or ingestion expansion
  is authorized. H-LIQ-001 remains **UNTESTED**.

The coding agent reviewed the complete pending research change: archive identity
and checksum handling, bounded partition allocations, sequence checks, unresolved
status propagation, benchmark acceptance and failure paths, fixture provenance,
tests and documentary boundaries. No blocking defect in the source-contract
correction was identified. No functional code was changed during closeout.
This is a review by the same coding agent, **not an independent code review**;
the reported tests are verification evidence, not independent certification.

The commit includes eight previously untracked files together, so the correction
does not depend on an absent ingestion module or benchmark record:

- `research/liquidity/daily_ingestion.py`
- `research/liquidity/benchmark_daily.py`
- `tests/research/liquidity/test_daily_ingestion.py`
- `tests/research/liquidity/fixtures/h_liq_001_underlying_id_context.json`
- `docs/research/H_LIQ_001_DAILY_INGESTION_ADDENDUM.md`
- `docs/research/H_LIQ_001_DAILY_INGESTION_ADDENDUM.sha256`
- `docs/research/H_LIQ_001_DAILY_BENCHMARK_RESULT.md`
- `docs/research/H_LIQ_001_SOURCE_CONTRACT_DECISION.md`

Closeout checks rerun only the 77 offline liquidity tests and 12 `make test-fast`
checks under the 2 GiB/no-swap and one CPU limits, plus focused lint, staged
whitespace/scope checks, frozen/addendum hash checks and preservation of all 88
original benchmark files. The commit containing this record identifies the
closeout revision; the investigation and measured code identities above remain
historical provenance.

Stop after committing the reviewed correction. Any future admission of data with
documented, source-permitted underlying-ID gaps requires a **separate explicit
evidence-policy decision before outcomes are examined**. This closeout grants no
such admission and no permission to resume historical work. No production changes,
service restarts or capital interaction.
