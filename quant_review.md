 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/quant_review.md b/quant_review.md
new file mode 100644
index 0000000000000000000000000000000000000000..35f37c5a2919fe04451a767ab0a6d3ed3fac2c31
--- /dev/null
+++ b/quant_review.md
@@ -0,0 +1,47 @@
+# Quant & Infra Review
+
+## Executive Summary
+- Exchange NAV computation drops to a static capital base whenever the futures wallet reads as zero or the positions API fails, leaving the confirmed NAV cache stale and masking true drawdowns; reserves valuation also fans out unthrottled spot requests that will rate-limit under load.【F:execution/nav.py†L27-L174】【F:execution/reserves.py†L12-L73】
+- Risk gating only fails closed on stale NAV if `nav_freshness_seconds` is configured, so the default behaviour is fail-open; moreover, NAV freshness relies on file mtimes that stop updating whenever position fetches fail, defeating the safeguard.【F:execution/risk_limits.py†L41-L507】【F:execution/nav.py†L27-L62】
+- Firestore sync processes force the environment namespace to `prod`, ship placeholder payloads, and silently downgrade to no-op clients, so dashboards can drift without alarms; supervisor heartbeats never propagate failure context.【F:execution/sync_state.py†L10-L156】【F:execution/hedge_sync.py†L8-L27】【F:utils/firestore_client.py†L206-L289】【F:execution/sync_daemon.py†L38-L129】
+- Logging rewrites full JSONL files on every write and lacks backpressure, while lingering ML and dashboard helpers remain bundled despite not feeding live execution, increasing maintenance surface and test debt.【F:execution/log_utils.py†L21-L129】【F:execution/ml/train.py†L1-L160】
+- Unit coverage for the reviewed areas consists of two narrow happy-path tests, leaving NAV aging, Firestore sync, and fail-closed logic unverified under failure scenarios.【F:tests/test_nav_age.py†L1-L28】【F:tests/test_symbol_routes.py†L1-L21】
+
+## NAV Pipeline (exchange + reserves)
+1. `_futures_nav_usdt` only persists a "confirmed" NAV when both balances and positions succeed; any transient positions failure leaves `logs/cache/nav_confirmed.json` frozen while the risk module keeps accepting stale MTIME data.【F:execution/nav.py†L27-L62】 Consider caching the last good NAV with an explicit stale flag and surfacing partial failures.
+2. If the futures wallet returns `0` (liquidation, withdrawal, or auth lapse) the trading NAV reverts to the static `capital_base_usdt`, instantly hiding large losses and breaking downstream caps.【F:execution/nav.py†L150-L160】 Replace with a fail-closed zero (or last confirmed) plus alerting.
+3. Treasury NAV loops over configured assets and hits Binance futures prices by default; non-futures assets (e.g., equities, bank cash) will raise and be silently dropped from totals.【F:execution/nav.py†L68-L108】 Add explicit venue routing, currency overrides, and alerting when holdings are skipped.
+4. Reserves valuation makes sequential `get_price(..., venue="spot")` calls without rate limiting or caching, so large reserve sets will trip Binance HTTP throttles and zero-out valuations silently.【F:execution/reserves.py†L28-L73】 Batch CoinGecko pulls, reuse price caches, and bubble failures up to reporting NAV.
+
+## Risk Limits & Fail-Closed Behaviour
+1. NAV freshness enforcement only triggers when `nav_freshness_seconds > 0`; default configs omit it, so `fail_closed_on_nav_stale` never fires and stale NAVs pass through.【F:execution/risk_limits.py†L470-L507】 Make freshness thresholds mandatory and default fail-closed.
+2. `get_nav_age` trusts file mtimes; because `_persist_confirmed_nav` skips writes on position API failures, the age never advances and the gate keeps trading on stale data.【F:execution/risk_limits.py†L41-L68】【F:execution/nav.py†L27-L62】 Persist a heartbeat regardless of position fetch outcome and track source health separately.
+3. `_emit_veto` logs to JSONL, but the log helper rewrites entire files and can block the execution loop during bursts; risk gating lacks non-blocking fallback or sampling.【F:execution/risk_limits.py†L117-L140】【F:execution/log_utils.py†L21-L106】 Switch to append-only writes (e.g., `os.open` with `O_APPEND`) or queue-based logging.
+
+## Dashboard / Firestore Sync
+1. `_force_env_to_prod` hard-codes Firestore writes to the prod namespace, so running the sync locally can overwrite live telemetry.【F:execution/sync_state.py†L18-L55】 Respect caller-provided `ENV` with opt-in overrides.
+2. `hedge_sync.py` still publishes empty placeholder payloads every minute, which overwrites real NAV/position docs when invoked and hides integration regressions.【F:execution/hedge_sync.py†L8-L27】 Replace or remove this script; wire it to the real readers or guard with `--dry-run`.
+3. `get_db` silently returns a no-op client whenever credentials are missing or Firestore packages are absent, with only a stderr warning; downstream syncs treat the no-op as success and dashboards fall back to stale local files without alerting.【F:utils/firestore_client.py†L206-L289】 Surface hard failures (or structured health metrics) so supervisors can trip.
+4. `sync_daemon` reads local JSON files without freshness checks, pushes the raw series to Firestore, and records only "✔ sync ok" even when Firestore writes fail inside `sync_*`; heartbeats never indicate degraded state.【F:execution/sync_daemon.py†L38-L129】 Add try/except around each Firestore call, track last-success timestamps, and publish failure counters.
+
+## Logging & Supervisor Reliability
+- `JsonlLogger` copies the entire log file into a temp file on every write, so busy services (risk vetos, heartbeats) will thrash disk and block threads; rotation also gzips synchronously on the hot path.【F:execution/log_utils.py†L21-L106】 Replace with append-only writes plus asynchronous rotation.
+- Supervisor-facing scripts (`sync_state`, `sync_daemon`) rely on `print` statements for critical errors; these are not structured or rate-limited, hindering downstream alerting.【F:execution/sync_state.py†L18-L156】【F:execution/sync_daemon.py†L38-L129】 Promote them to structured log events or Firestore telemetry.
+
+## Legacy / Bloat Inventory
+- The ML training stack (`execution/ml/*.py`) remains bundled with plotting dependencies and calibration logic yet is disconnected from live sizing; it bloats deployment packages and the test matrix.【F:execution/ml/train.py†L1-L160】 Consider extracting to a separate research repo or marking it optional.
+- Streamlit helpers maintain multiple NAV loaders (`dashboard/live_helpers.py`, `dashboard/dashboard_utils.py`) that duplicate executor logic; consolidate around one source of truth once Firestore is authoritative.【F:dashboard/dashboard_utils.py†L19-L123】
+
+## Tests & Coverage Gaps
+- `test_nav_age_selects_newest` only verifies the happy path of `_NAV_SNAPSHOT_PATHS` mtimes; it does not cover missing files, stale caches, or the risk gate interactions.【F:tests/test_nav_age.py†L1-L28】
+- `test_symbol_routes.py` exercises the CoinGecko fallback but leaves failure and timeout paths untested; no tests cover reserves valuation, Firestore sync, or fail-closed gating.【F:tests/test_symbol_routes.py†L1-L21】
+
+## Security-Sensitive Files
+- `config/firebase_creds.json` (and its `firebase.json` symlink) contains a live Google service-account private key and should be rotated or removed from the repo immediately.【F:config/firebase_creds.json†L1-L13】
+
+## Prioritized Codex Patch Queue
+1. **Harden NAV freshness & fail-closed guardrails.** Persist NAV snapshots even on partial failures, default `fail_closed_on_nav_stale` to true with mandatory freshness thresholds, and surface stale-source alerts in risk veto logs.【F:execution/nav.py†L27-L160】【F:execution/risk_limits.py†L41-L507】
+2. **Secure Firestore sync pipeline.** Respect `ENV`, remove placeholder writers, make `get_db` failures fatal, and emit structured heartbeat telemetry with failure counters.【F:execution/sync_state.py†L18-L156】【F:execution/hedge_sync.py†L8-L27】【F:utils/firestore_client.py†L206-L289】【F:execution/sync_daemon.py†L38-L129】
+3. **Modernize logging primitives.** Replace the temp-file rewrite pattern with append-only logging or buffered queues to keep risk and sync loops non-blocking under load.【F:execution/log_utils.py†L21-L106】
+4. **Quarantine legacy ML tooling.** Move `execution/ml` (and associated heavy deps) behind an optional extra or separate package to slim production deployments and clarify maintenance boundaries.【F:execution/ml/train.py†L1-L160】
+5. **Expand failure-mode test coverage.** Add pytest cases for NAV stale handling, reserves valuation errors, and Firestore sync fallbacks to prevent regressions.【F:tests/test_nav_age.py†L1-L28】【F:tests/test_symbol_routes.py†L1-L21】 
EOF
)