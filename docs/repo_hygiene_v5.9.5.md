Repo Hygiene Summary — v5.9.5

Date: v5.9.5
Purpose: Prepare the repository for Codex 5.1 multi-file patching and the v5.10 Execution Intelligence sprint by removing legacy modules, stabilizing entrypoints, and eliminating historical noise that caused patch reversions or mis-routing.

1. Legacy Modules Archived

All deprecated, superseded, unused, or misleading modules were moved to:

archive/deprecated_v5.9.5/


This includes:

Execution

execution/ml/

execution/ml_old/

execution/leaderboard_sync.py

execution/hedge_sync.py

execution/pipeline_probe.py

execution/orderbook_features.py

execution/backtest/ (if present)

Strategies / Backtests

strategies/

backtests/

scripts/backtest.py

scripts/backtest_*

Dashboard Legacy

dashboard/legacy/

dashboard/v3/

dashboard/old_nav_parsers.py

dashboard/old_state.py

Old Audits / Reports / PR Artifacts

docs/audit_*

docs/telemetry_audit_*

docs/*v5.4* → docs/*v5.8*

logs/repo_hygiene_report.md

logs/import_graph.json

Legacy Scripts

scripts/old_*

scripts/debug_*

scripts/portfolio_probe.py

Misc Runtime Clutter

state/

tmp/

backup/

patches/

*.patch files

2. Stubs Added for Deprecated Modules

To prevent Codex from attempting to regenerate retired modules or mis-patching imports, 3-line stubs were added under original paths for:

execution/pipeline_probe.py

execution/leaderboard_sync.py

execution/hedge_sync.py

execution/ml/__init__.py

Any other removed top-level legacy entrypoints

Each stub states:

Deprecated in v5.9.5.
Original implementation moved to archive/deprecated_v5.9.5/.
Kept as a stub so Codex does not attempt to regenerate it.

3. Codex Entrypoints Declared

A new authoritative reference was added:

docs/codex_entrypoints_v5.9.5.md


This file explicitly enumerates:

Execution (LIVE)

execution/executor_live.py

execution/order_router.py

execution/risk_limits.py

execution/signal_screener.py

execution/utils/*

execution/firestore_utils.py

execution/telegram_utils.py

execution/router_metrics.py

execution/drawdown_tracker.py

execution/sync_state.py

Dashboard (LIVE)

dashboard/app.py

dashboard/live_helpers.py

dashboard/router_health.py

dashboard/dashboard_utils.py

Tests driving Codex behavior

tests/test_execution_hardening_*.py

tests/test_execution_health.py

tests/test_execution_alerts.py

tests/test_router_metrics_*.py

tests/test_symbol_toggle_bootstrap.py

Supervisor Entrypoints

hedge:executor → execution/executor_live.py

hedge:dashboard → dashboard/app.py

hedge:sync_state → execution/sync_state.py

This file is now the single source of truth for Codex 5.1 patch scope.

4. Codex Ignore File Added

.codexignore now suppresses noisy or irrelevant directories:

archive/
logs/
tmp/
backup/
state/
*.patch
*.ipynb
**/__pycache__/
tests/fixtures/*


This reduces Codex’s context and prevents legacy modules from influencing multi-file reasoning.

5. Repo Root Normalized

The repo is now intentionally streamlined:

Kept:

README.md
docs/
config/
dashboard/
execution/
scripts/
tests/
supervisor.conf
requirements.txt


Removed or archived:

old audit .md files

legacy dev notes

unused root-level code

6. Outcome

The v5.9.5 hygiene pass achieves:

✔ Clean Codex context (no more patch regressions)
✔ Explicit entrypoints for deterministic patch generation
✔ Old ML/strategy/backtests fully isolated
✔ Dashboard and executor modules clearly scoped
✔ Reduced merge conflicts and noise
✔ Fully prepared foundation for v5.10 Execution Intelligence Layer