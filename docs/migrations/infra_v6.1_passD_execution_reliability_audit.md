## PASS D — Execution Reliability & Telemetry

Scope: exchange utils (Binance UM), router/executor error handling, execution health telemetry, operator CLIs.

### Error classes & classification
- `execution.exchange_utils.classify_binance_error` normalises Binance UM errors into categories: `network`, `rate_limit`, `auth`, `bad_request`, `exchange_server`, `http_error`, `config`, `unknown`, with `retriable` flag and status/code/msg.
- `get_um_client`, balance/position fetches, `_req` backoff, and `send_order` failures record structured errors via `record_execution_error`, including classification and context (symbol/payload).
- Client-style errors without a `requests.Response` (e.g., `status_code`/`error_code` on `ClientError`) are still bucketed into `rate_limit`/`auth`/`bad_request` via the classifier so router/executor health sees the right category.
- Router taker submission failures attach classification to `route_decision` logs and error payloads; executor `_dispatch` HTTP errors log `order_error` with classification, reason, and retry flag. Pipeline shadow/compare failures now register component errors instead of silently passing.

### Retry/backoff & fail-closed behaviour
- `_req` keeps bounded exponential backoff (`BINANCE_MAX_RETRIES`, `BINANCE_BACKOFF_*`) and now tags final failures with classification.
- Executor `_send_order` adds bounded retries for transient dispatch errors controlled by runtime (`config/runtime.yaml`):
  - `execution.max_transient_retries` (default 1)
  - `execution.transient_retry_backoff_s` (default 1.0s)
  - `execution.error_cooldown_sec` (default 60s) reused by symbol cooldowns.
- Maker-first/taker routing still fails closed on config/risk vetoes; fatal config (missing keys, bad UM init) installs stub client and records `config`/`auth` errors.

### Execution health snapshot schema
- `execution.utils.execution_health.compute_execution_health` now returns schema `execution_health_v1` with:
  - `router`, `risk`, `vol`, `sizing` sections (existing fields preserved).
  - `errors`: counts + last_error per component (router, exchange, nav/risk/pipeline as recorded).
  - `components`: mirrors sub-sections for downstream consumers.
- Error registry (`record_execution_error`) captures counts per component/symbol; `reset_error_registry` for tests/ops hygiene.
- Executor writes periodic execution health snapshots to `logs/execution/execution_health.jsonl` and `logs/state/execution_health.json` alongside nav/router/risk state.

### Operator CLIs
- `scripts/exec_debug.py`: reads latest `logs/state/nav.json`, `router_health.json`, `execution_health.json`, prints concise summary and paths; `--show-json` dumps raw payloads. Exits non-zero if state files missing.
- `scripts/route_debug.py`: dry-run router inspector (no orders sent). Args: `--symbol`, `--side`, `--notional`, optional `--price`, `--spread-bps`, `--reduce-only`, `--json`. Prints routed mode, policy quality/bias, offset, and veto reasons.

### Logging / JSON contracts
- Structured logs under `logs/execution/*` now include top-level `type` and `context` (e.g., `route_decision`, `order_error`, `execution_health`). Router error logs include classification, retry flag, and payload context.

### Tests added
- Exchange utils: transient `_req` retry path and error classification (rate-limit vs auth).
- Router failure path: send-order HTTPError records router component error (visible in execution health).
- Execution health: schema + error registry surfaced in snapshots.
