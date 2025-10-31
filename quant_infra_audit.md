# Quant Infra Audit — Execution / Dashboard / Scripts
_Date: 2025-10-31_

## Executive Summary
- Trading executor hides critical telemetry failures and forgets throttling state on restart, leaving Firestore dashboards and risk controls blind during outages.
- Operator-facing surfaces (Streamlit dashboard, doctor CLI) re-execute production code paths on every refresh without caching, introducing unnecessary exchange calls and inconsistent situational awareness.
- Supporting scripts rely on ad-hoc environment usage and network calls without retries, creating drift between tooling output and live system reality.

## Execution Module
- **Silent telemetry degradation** — `execution/executor_live.py:133` swaps in a no-op publisher when `execution.state_publish` import fails, causing `publish_*` calls to succeed silently while Firestore stops receiving intents/orders. Treat missing publisher modules as fatal, or surface a hard alert before continuing.
- **Risk throttles reset on restart** — `execution/risk_limits.py:403` keeps burst/error counters and cooldowns only in-memory (`RiskState`). An executor restart wipes these guards, allowing immediate re-entry after repeated vetoes. Persist lightweight state (e.g., JSON in `logs/cache/`) and reload on boot so drawdown / burst protections survive restarts.
- **Binance client lacks resiliency** — `execution/exchange_utils.py:129` issues single-shot HTTP requests without retries or jitter, so transient 5xx/429 responses bubble up as order send failures. Wrap `_S.request` with bounded exponential backoff (respecting Binance rate limits) and capture structured error flags for downstream risk vetoes.
- **Tight coupling of intent generation and execution** — `execution/executor_live.py:1867` invokes `generate_intents` synchronously inside the trading loop. A slow strategy or import error blocks order routing and heartbeats. Consider isolating the screener behind a queue (Redis, pub/sub) or at minimum guarding single-strategy failures behind timeouts so the executor keeps publishing risk updates.

## Dashboard Module
- **Dashboard recomputes production signals** — `dashboard/app.py:684` calls `generate_intents` on every Streamlit refresh, reimporting trading strategies and potentially triggering network fetches. Replace with Firestore snapshots (already produced by executor) or memoise results with `st.cache_data` gated by timestamp to keep the UI read-only.
- **Doctor snapshot is heavyweight and uncached** — `dashboard/app.py:797` pulls in `collect_doctor_snapshot`, which in turn performs live Firestore reads, log scanning, and FX lookups each render (`scripts/doctor.py:406`). Move this call behind an async cron (writing JSON to `logs/cache/doctor.json`) or add explicit throttling + timeout to avoid 10s refresh stalls.
- **UI depends on private risk internals** — `dashboard/live_helpers.py:53` calls `RiskGate._portfolio_nav()`, a private helper that could change without warning. Expose a supported API on `RiskGate` (e.g., `portfolio_nav()`) and update the dashboard to consume that, removing the TypeIgnore and improving stability under future refactors.
- **Large log tails loaded per request** — `dashboard/app.py:727` reads ~200 KB from `logs/screener_tail.log` into memory each refresh. Tighten tail windows, move heavy parsing into background jobs, or offer pagination to reduce UI latency and file descriptor churn.

## Scripts Module
- **Doctor cannot see live positions** — `scripts/doctor.py:194` passes `None` into `utils_get_live_positions`, which immediately raises and returns an empty list. As a result, doctor reports zero positions even when exposure exists. Instantiate a UMFutures client (mirroring the executor) or default to the cached synced state file so operator checks are accurate.
- **Tooling duplicates business logic without retries** — `scripts/doctor.py:220` and `scripts/doctor.py:406` fetch CoinGecko FX and Firestore data directly with bare `requests`, lacking retry/backoff symmetry with execution. Adopt the shared helpers in `execution.exchange_utils` / `utils.firestore_client` or wrap calls in shared retry utilities to keep doctor output aligned with production behaviour.
- **Screener probes assume pristine configs** — `scripts/screener_probe.py:10` loads `config/risk_limits.json` without validation and assumes missing tiers are OK, leading to misleading “would_emit” results when configs drift. Add schema validation (pydantic/voluptuous) and require explicit NAV inputs to turn the probe into a reliable dry-run.

## Cross-Cutting Opportunities
- Harden Firestore publishing by unifying telemetry writes: export a single `StatePublisher` package consumed by executor, sync daemon, and dashboard workers, with circuit breakers and health pings.
- Introduce a lightweight background orchestrator (systemd timer or cron) to precompute heavy doctor/dashboard artefacts and persist to `logs/cache/`, so interactive surfaces stay responsive.
- Add integration tests that boot a fake Binance/Firestore stack (pytest fixtures) and exercise executor/sync daemon loops, ensuring risk state persistence and telemetry publishing function under restart scenarios.
- Capture operational runbooks in `docs/` (e.g., how to rehydrate `RiskState` or rotate API keys) so automated audits can highlight divergence between desired and actual runtime behaviour.

