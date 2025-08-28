Title: Integrate risk checks, Firestore audits (incl. blocked), and Risk Status KPI

Summary
- Gate all orders through execution.risk_limits.check_order (or wrapper) before placement.
- Publish Firestore audits for intents, order request/response/errors, close events, and risk blocks.
- Add “Risk Status” KPI to the Streamlit overview (open gross vs NAV cap usage).
- Keep tests green; add light lint/type fixes to touched files.

Changes
- execution/risk_limits.py
  - Add check_order, will_violate_exposure, and extend RiskState with cooldown/error tracking (note_fill, last_fill_ts, note_error, errors_in).
  - Backward-compatible can_open_position wrapper; legacy RiskConfig path unchanged.
  - Deduplicate clamp_order_size and tidy imports.

- execution/executor_live.py
  - Load risk limits config, compute nav/current gross/symbol open_qty, and call check_order before placing.
  - On block, log and publish order audit with phase="blocked" (reason, details).
  - Publish intent/request/response/error/close audits.
  - Note fills and errors in RiskState.
  - Minor lint/type cleanups, robust optional imports.

- execution/state_publish.py
  - Add audit append helpers: publish_intent_audit, publish_order_audit, publish_close_audit.

- dashboard/app.py
  - Add Risk Status KPI (open gross vs cap with % used) sourced from config and Firestore positions.
  - Add Risk Blocks table: reads hedge/{ENV}/state/audit_orders_* and shows latest blocked intents.
  - Clean up small lint/style issues and remove duplicate exit plans block.

Validation
- pytest: 13 passed.
- ruff: clean for modified files.
- mypy: clean for modified files (local config added for scope).

Notes
- Risk config loaded from RISK_LIMITS_CONFIG or config/risk_limits.json.
- Dashboard and executor fail-soft if Firestore or creds are unavailable.
- Non-touched modules still have lint issues; scope limited to files changed in this PR.

Reviewers
- @codex — please review risk checks, audit coverage, and dashboard additions.

