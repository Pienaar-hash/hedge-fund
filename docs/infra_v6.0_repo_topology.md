# v6.0 Pre-Migration Repo Topology Audit

> Point-in-time snapshot of the v5.10/v6-pre repo. For a concise “current reality” summary, see `docs/ARCHITECTURE_CURRENT.md`. Target blueprint/intent live in `docs/v6.0_Master_Architecture_Map.md` and `docs/v6.0_architecture_brief.md`.

## 1. Repo Structure & Layer Mapping

### 1.1 Repo Layout
- `execution/` – core live stack: `executor_live.py` orchestrates the intent → risk → sizing → router → exchange flow while telemetry hooks are stubbed (`execution/executor_live.py:94-1760`); routers (`execution/order_router.py:474-683`), exchange adapter (`execution/exchange_utils.py:1-1210`), size model (`execution/size_model.py:1-360`), intel modules (`execution/intel/*.py`), gates (`execution/risk_limits.py:1-1570`), and telemetry/state publishers (`execution/state_publish.py`, `execution/sync_state.py:1-200`, `execution/firestore_utils.py:1-136`) live here along with helpers such as `nav.py`, `reserves.py`, `pnl_tracker.py`, `drawdown_tracker.py`, `capital_allocator.py`, and router metrics (`execution/router_metrics.py:1-80`).
- `core/` – shared abstractions for configurable strategies, currently just `strategy_base.py` exposing the `Strategy` class that logs results (`core/strategy_base.py:12-53`).
- `dashboard/` – Streamlit UI plus helpers: `app.py`/`main.py` bootstrap, `live_helpers.py` pulls execution intel (`dashboard/live_helpers.py:1-190`), `router_health.py` aggregates JSONL telemetry, and utility files (`dashboard/dashboard_utils.py`, `dashboard/nav_helpers.py`, `dashboard/async_cache.py`) format state.
- `scripts/` – operator tooling: screener probe (`scripts/screener_probe.py:19-75`), registry toggler (`scripts/registry_ctl.py:9-78`), doctor/diagnostic scripts (`scripts/doctor.py`, `scripts/balance_doctor.py`, `scripts/fs_doctor.py`), ML retrain utilities (`scripts/ml_*`), deploy aids (`scripts/go_live_now.sh`, `scripts/run_executor_once.sh`), and hygiene scripts.
- `config/` – runtime declaratives including strategy/risk configs (`config/strategy_config.json:1-302`, `config/risk_limits.json:2-46`), registry (`config/strategy_registry.json:1-23`), universe metadata (`config/pairs_universe.json:1-20`, `config/symbol_tiers.json:1-6`), dashboard knobs (`config/dashboard.yml:1-16`), reserves/assets (`config/reserves.json:1-4`, `config/assets.json:1-5`), and settings toggles (`config/settings.json:1-6`).
- `tests/` – extensive pytest suite spanning live execution contracts, routers, risk, telemetry/dashboards, ML shims, and config parsing (e.g., `tests/test_risk_limits.py:1-200`, `tests/test_router_smoke.py:1-175`, `tests/test_execution_health.py:1-90`).
- `research/`, `ml/`, `models/` – experimentation layer: notebooks/scripts for factor fusion and RL sizing (`research/factor_fusion/*.py`, `research/rl/*.py`, `research/rl_sizer/*.py`), ML telemetry utilities (`ml/telemetry.py:1-160`), and serialized models (`models/*.pkl`, `models/registry.json`).
- `docs/` – operations + architecture references, including the v6 blueprint (`docs/v6.0_Master_Architecture_Map.md:1-210`), brief (`docs/v6.0_architecture_brief.md:1-210`), pre-flight notes (`docs/v6.0-alpha1_pre_flight.txt`), and prior audits/reports.
- `archive/` – quarantined v5.x modules (`archive/deprecated_v5.9.5/strategies/*.py`, `archive/deprecated_v5.7/*`) plus legacy configs; these mirror code that has since been replaced.
- DevOps/ops: `deploy/` carries supervisor configs for executor/dashboard/ML (`deploy/supervisor/*.conf`), `ops/hedge.conf` is the host-level supervisor file, `cron/rotate_exec_logs.sh` defines log maintenance, and `bin/run-*.sh` wrappers start services. `.github/workflows/ci.yml:1-21` defines CI, while `infrastructure/` currently contains empty docker-compose/Dockerfile scaffolds awaiting content.
- Supporting roots: `data/` (historical CSV snapshots), `logs/` (local telemetry artifacts), `reports/` (HTML investor decks), `telegram/heartbeat.py` (alerting), `gpt_schema/` (frozen contract snapshot for GPT agents), `run_all.sh`/`Makefile` (entrypoints), and metadata such as `infra_v6.0_presprint_codex_audit.md`.

### 1.2 Architecture Layer Mapping
#### Execution Engine
- Live orchestration: `_run_executor` in `execution/executor_live.py` manages polling, intent hydration, sizing, router dispatch, and state updates while referencing `RiskAutotuner`, router policy tags, and maker/taker offsets (`execution/executor_live.py:145-2108`).
- Sizing + math: `execution/size_model.py:229-320` enforces bucket caps, risk-derived leverage, and min notionals; `execution/position_sizing.py` + `execution/capital_allocator.py` provide supporting math.
- Router/policy: `execution/order_router.py:474-683` handles maker/taker attempts; `execution/intel/router_policy.py:10-78`, `execution/intel/maker_offset.py:1-120`, and `execution/intel/symbol_score.py:1-120` compute policy/maker offsets and symbol scores.
- Exchange adapter: REST/UM Futures interface, payload cleaners, and hedging helpers in `execution/exchange_utils.py:1-1210`, with risk-critical constants such as `_CLOSE_POSITION_QUOTES` defined at `execution/exchange_utils.py:1039-1105`.
- Supporting telemetry: router/event loggers (`execution/router_metrics.py:1-80`, `execution/events.py:1-120`, `execution/trade_logs.py`), PnL tracker (`execution/fills.py`, `execution/pnl_tracker.py`), drawdown tracker (`execution/drawdown_tracker.py`), reserves/nav modules, and maker/taker metrics in `execution/utils/metrics.py:203-260`.
- Duplicates/variants: archived strategy/execution pipeline copies in `archive/deprecated_v5.9.5/*`, along with deprecated probes (`execution/pipeline_probe.py:1-40`).

#### Strategy & Screener Layer
- Intent orchestration lives in `execution/signal_generator.py:1-250`, which imports hard-coded modules (`strategies.momentum`, `strategies.relative_value`) despite those packages only existing in `archive/deprecated_v5.9.5/strategies/*.py`.
- Screener and doctor gates: `_entry_gate_result` + `would_emit` blend listing/orderbook/tier/risk vetos in `execution/signal_screener.py:250-410`, while `execution/signal_doctor.py:1-90` adds fast sanity checks.
- Registry/external tooling: `config/strategy_registry.json:1-23` stores id-level metadata; `_load_registry()` in `execution/signal_generator.py:79-139` injects concurrency & confidence, and `scripts/registry_ctl.py:9-78` toggles entries. `scripts/screener_probe.py:19-75` replays `would_emit` for operations.
- Universe/tiers: `execution/universe_resolver.py:1-160` merges tier listings, discovery feeds, and exchange filters; support configs live in `config/symbol_tiers.json:1-6`, `config/pairs_universe.json:1-20`, and `config/settings.json:1-6`.
- Alternate/legacy strategies only exist inside `archive/deprecated_v5.9.5/strategies`, so there is no active `strategies/` package in the main tree.

#### Risk Engine
- Canonical gating is handled by `execution/risk_limits.py:36-1565`, including `RiskState`, `check_order`, concurrency counters, nav freshness, drawdown clamps, and global constants (`SYMBOL_NOTIONAL_CAP`, `SYM_DD_CAP_PCT`).
- Risk tuning/intel: `execution/risk_autotune.py` and `execution/drawdown_tracker.py` generate size multipliers; caches and veto logs sit under `logs/execution/*` and `screener_state.json`.
- Config surfaces: `config/risk_limits.json:2-47`, `config/symbol_tiers.json:1-6`, and risk fragments embedded in `config/strategy_config.json:12-144` drive these gates.

#### Telemetry & State
- Local telemetry: `execution/state_publish.py` computes execution stats for dashboard ingestion, while `_mirror_router_metrics`, `_maybe_publish_execution_health`, `_maybe_emit_execution_alerts`, and `_maybe_publish_execution_intel` are currently stubs (`execution/executor_live.py:94-107`).
- Syncers: `execution/sync_state.py:1-200` mirrors NAV/positions to local caches with Firestore disabled, `execution/sync_daemon.py:1-120` pushes docs when allowed, and `execution/firestore_utils.py:1-136` intentionally returns No-ops for remote publishes. `execution/hedge_sync.py:1-6` is an explicit deprecated shim.
- Mirror builders + leaderboard/supervisor hooks: `execution/mirror_builders.py`, `execution/leaderboard_sync.py`, `execution/firestore_mirror.py:1-63`, and `telegram/heartbeat.py` handle derived state and alerts.
- Telemetry configs + docs: `config/dashboard.yml:1-16`, `docs/FIRESTORE.md`, and `scripts` such as `scripts/fetch_synced_state.py`, `scripts/diagnose_live_activity.py`, and `scripts/doctor.py` provide operator tooling.

#### Dashboard
- Streamlit entrypoints `dashboard/app.py` and `dashboard/main.py` compose pages; `dashboard/live_helpers.py:1-192`, `dashboard/router_health.py:1-120`, `dashboard/nav_helpers.py`, and `dashboard/dashboard_utils.py` fetch telemetry from JSONL files, risk config gates, and intel modules. Tests `tests/test_dashboard_equity.py:4-20`, `tests/test_dashboard_metrics.py:5-20`, and `tests/test_dashboard_intel_helpers.py:1-26` confirm helper functions.
- Front-end assets: `dashboard/theme.css`, `config/dashboard.yml`, and HTML exports in `reports/*.html` define presentation.

#### Research & Backtest
- Python research modules include `research/correlation_matrix.py`, factor fusion (`research/factor_fusion/fusion.py`), reinforcement-learning envs (`research/rl/market_env.py`, `research/rl_sizer/env.py`), and RL agents (`research/rl_sizer/agents.py`).
- ML telemetry/experimentation: `ml/telemetry.py:1-160`, `scripts/ml_fit.py`, `scripts/ml_retrain_daemon.py`, and serialized models in `models/*.pkl`.
- Datasets `data/*.csv` and notebooks (under `research/` and docs/investor decks) support analysis.

#### DevOps & Environments
- Runtime orchestration: supervisor configs (`deploy/supervisor/*.conf`, `ops/hedge.conf`), cron (`cron/rotate_exec_logs.sh`), shell wrappers (`bin/run-*.sh`), and `.env`-driven toggles referenced by `execution/sync_state.py:26-76`.
- CI/CD: `.github/workflows/ci.yml:1-21` installs multiple requirement files (two of which are missing) and executes linting/tests. `Makefile`, `run_all.sh`, `scripts/quick_checks.sh`, and `docs/OPERATIONS.md` round out operator guidance.
- Containerization scaffolds exist but are empty placeholders (`infrastructure/docker-compose.yml`, `infrastructure/Dockerfile`, `infrastructure/startup.sh`, `infrastructure/requirements.txt` all zero-length).

## 2. Alignment vs v6.0 Architecture
### 2.1 Execution Engine
- **Present & aligned**: Router-policy hand-offs exist (`execution/all`): intents carry metadata and `route_order()` tracks maker/taker decisions (`execution/order_router.py:474-585`). Size model enforces nav percentages consistent with tests (`execution/size_model.py:229-320`).
- **Present but misaligned/leaky**: Executor bypasses tier/concurrency fields by not passing `open_positions_count`, `tier_name`, or `current_tier_gross_notional` into `check_order` (`execution/executor_live.py:1717-1729` vs design spelled out in earlier audit `infra_v6.0_presprint_codex_audit.md:3-14`). Telemetry hooks are all `return None` (`execution/executor_live.py:94-107`), so router/health contracts never publish, starving intel/feedback loops. Router metrics omit `is_maker_final` and `used_fallback` even though intel consumers require them (`execution/order_router.py:619-682` vs `execution/utils/metrics.py:222-238`), preventing router-policy adaptation.
- **Missing**: v6 expects a feedback allocator, expectancy ingestion, and router auto-tuning graph (`docs/v6.0_architecture_brief.md:78-170`), but no module persists per-intent veto/sizing results, no expectancy_map inputs beyond static router metrics exist, and there is no `execution/feedback_allocator.py`. Multi-venue abstractions are absent (only Binance functions exist inside `execution/exchange_utils.py:1-400`).
- **Legacy/obsolete**: `execution/pipeline_probe.py:1-40` and `execution/hedge_sync.py:1-6` are marked deprecated; `archive/deprecated_v5.9.5/strategies/*.py` and `archive/deprecated_v5.7/*` contain old routers/exchange helpers.

### 2.2 Strategy & Screener
- **Present & aligned**: Screener path enforces listing, orderbook imbalance, tier caps, and risk by calling `check_order` with enriched context (`execution/signal_screener.py:286-400`), which matches v6 layering guidance.
- **Present but misaligned/leaky**: `execution/signal_generator.py:1-210` still imports hard-coded modules and uses `_STRATEGY_MODULES = ("momentum", "relative_value")` with no discoverable `strategies/` package, so there is no real strategy registry interface despite `docs/v6.0_Master_Architecture_Map.md:33-60` mandating one. Strategy config duplicates concurrency values and contains universe/risk knobs that belong in separate schemas (`config/strategy_config.json:12-210`). Tier/entry thresholds are hard-coded (±0.20 orderbook gate at `execution/signal_screener.py:313-323`) rather than data-driven as v6 demands.
- **Missing**: No shared `BaseStrategy` implementations integrate into execution; `core/strategy_base.py:12-53` exists but nothing inherits from it. `config/strategy_registry.json:1-23` lacks module/class fields, so there is no loader for actual strategies. No screener probes for ML/regime classification beyond `scripts/screener_probe.py` CLI exist; there is no multi-strategy registry view demanded by `docs/v6.0_architecture_brief.md:118-165`.
- **Legacy/obsolete**: `archive/deprecated_v5.9.5/strategies/*.py` contain the only concrete strategy modules (momentum, momentum_vol) and are not imported anywhere else.

### 2.3 Risk Engine
- **Present & aligned**: `RiskState`/`check_order` enforce nav freshness, exposure caps, and drawdown controls, and tests keep the contract honest (`execution/risk_limits.py:712-1570`, `tests/test_risk_limits.py:1-200`).
- **Present but misaligned**: Configurable clamps such as `SYMBOL_NOTIONAL_CAP` and `SYM_DD_CAP_PCT` remain module constants (`execution/risk_limits.py:36-38`, `execution/risk_limits.py:1549-1565`) instead of being read from configs, so risk tuning requires code changes. The executor path does not populate concurrency/tier context (`execution/executor_live.py:1717-1729`). Risk reason strings default to `trade_gt_10pct_equity` even when configs specify different thresholds (`execution/risk_limits.py:1079-1104`, `execution/signal_screener.py:373-378`).
- **Missing**: No canonical `RiskEngine` object is exported; functions are module-level. There is no shared risk state persistence beyond ad-hoc JSON files; `cache/risk_state.json` referenced in architecture docs is not part of this tree. Per-strategy risk overlays described in `docs/v6.0_architecture_brief.md:92-152` are nonexistent.
- **Legacy/obsolete**: Old risk modules in `archive/deprecated_v5.7/*` and manual toggles in `execution/utils/toggle.py` remain but are mostly unused.

### 2.4 Telemetry & State
- **Present & aligned**: Local JSONL loggers exist for orders, signal metrics, and vetoes (`execution/executor_live.py:80-88`). Router health readers/dashboards consume these logs (`execution/router_metrics.py:1-80`, `dashboard/router_health.py:1-110`).
- **Present but misaligned**: Firestore publishers are intentionally disabled (`execution/firestore_utils.py:48-84`, `execution/sync_state.py:37-110`) in contrast to v6’s requirement for cloud telemetry. `_mirror_router_metrics` and `_maybe_publish_execution_health/intel/alerts` are dead stubs (`execution/executor_live.py:94-107`). `execution/firestore_mirror.py:28-63` is unused, so there is no actual sync_daemon for remote dashboards. Telemetry schema drift exists across `positions.json`, `nav_log.json`, etc., with no canonical `universe.json` or risk snapshot as mandated by `docs/v6.0_Master_Architecture_Map.md:150-196`.
- **Missing**: v6 expects sync daemons to push to Firestore/S3, a statewide telemetry aggregator, and nav/reserve publishers; these are either stubs or local-only. There is no `execution/state_publish.py` test coverage beyond stats, and no trace of a `feedback_allocator` feed tying together veto + fill data.
- **Legacy/obsolete**: `execution/hedge_sync.py:1-6` is deprecated, yet still shipped. `execution/firestore_mirror.py` references Google Cloud but is unreachable, implying stale interfaces from pre-hardening releases.

### 2.5 Dashboard
- **Present & aligned**: Streamlit pages, helper functions, and router health overlays exist (`dashboard/live_helpers.py:1-192`, `dashboard/router_health.py:1-110`), and tests validate helper APIs (`tests/test_dashboard_equity.py:4-20`, `tests/test_dashboard_metrics.py:5-20`).
- **Present but misaligned**: Dashboards rely on local files rather than canonical telemetry sources, and they re-read `config/strategy_config.json`/`risk_limits.json` to compute caps, so any schema changes ripple immediately. There is no `dashboard/pages/` multi-strategy layout as described in `docs/v6.0_Master_Architecture_Map.md:200-230`.
- **Missing**: No `dashboard/pages/*.py` folder exists even though architecture docs call for multi-page experiences. There is no aggregated fund view (NAV curves/reserves) beyond helper stubs.
- **Legacy/obsolete**: HTML investor decks in `reports/*.html` reference old telemetry structures and may mislead GPT agents.

### 2.6 Research & Backtest
- **Present & aligned**: Factor fusion and RL sizing modules exist along with RL agents and telemetry loggers (`research/factor_fusion/*.py`, `research/rl_sizer/*.py`). ML telemetry library (`ml/telemetry.py:1-160`) provides a structure for logging model output.
- **Present but misaligned**: Research outputs are disconnected from live configs; there is no script to materialize `pairs_universe.json` from backtests or to sync RL results into risk caps. RL envs do not feed into `execution/signal_generator` or `strategy_registry`.
- **Missing**: Vectorbt integration, backtest orchestrators, and research notebooks referenced in `docs/v6.0_Master_Architecture_Map.md:180-210` are not in repo. There are no tests for the research modules, so regression safety is low.
- **Legacy/obsolete**: Archived momentum strategies and `archive/deprecated_v5.9.5/ml/*` remain, but nothing imports them.

### 2.7 DevOps & Environments
- **Present & aligned**: Supervisor + cron scripts exist, and CI runs lint/tests (`deploy/supervisor/*.conf`, `ops/hedge.conf`, `.github/workflows/ci.yml:1-21`).
- **Present but misaligned**: CI references `requirements-execution.txt` and `requirements-dashboard.txt` that aren’t in the repo root, so pipelines will fail to install dependencies until those files are created. There is no `.env` template committed, and docker-compose/Dockerfile scaffolds are empty files, so containerized deployments are undefined.
- **Missing**: v6 requests environment-specific supervisors (hedge-sync, dashboard, RL/ML) plus Cron/ops docs; only executor/dashboard/ML retrain configs exist. No GitHub Actions for schema validation or config linting exist.
- **Legacy/obsolete**: `deploy/supervisor-user/supervisord.conf` still references old service names; `ops/nginx_site.conf`/`deploy/nginx/hedge-dashboard.conf` mention 127.0.0.1-only addresses rather than the multi-env routing described in v6 docs.

## 3. Config & Schema Drift
### 3.1 Schema Mapping
- `config/strategy_config.json:1-210` combines global toggles (`use_futures`, `poll_seconds`), universe listings, nav settings, sizing policy, risk controls, signal parameters, ML configuration, per-strategy dictionaries (each with duplicate `max_concurrent_positions` entries), and a stray global `max_trade_nav_pct`.
- `config/risk_limits.json:2-47` defines `global`, `per_symbol`, nav freshness, quote symbol lists, and a `non_prod` block. Global keys overlap with strategy sizing (exposure caps, leverage, concurrency) and include drawdown configs absent from the strategy file.
- `config/pairs_universe.json:1-20` stores symbol lists plus overrides for leverage/min notionals that duplicate data already held in `strategy_config` and `risk_limits`.
- `config/strategy_registry.json:1-23` only contains metadata (enabled, sandbox, max concurrent, confidence, capacity) without any Python entrypoints.
- `config/settings.json:1-6` toggles `automerge_discovery` and a `features.market_maker` flag; these settings are read by `execution/universe_resolver.py:40-80`.
- `config/dashboard.yml:1-16` is purely UI layout (tail rows, KPI thresholds) and does not reference any telemetry schema.
- `config/assets.json:1-5` and `config/reserves.json:1-4` track treasury balances separately, resulting in parallel truth sets.

### 3.2 Drift Against v6 Design
- Risk schema duplication: `strategy_config` contains `max_open_positions`, `max_gross_exposure_pct`, `max_symbol_exposure_pct`, and `max_trade_nav_pct` under sizing, even though v6 dictates that only `risk_limits.json` should own those (docs/v6.0_architecture_brief.md:60-110). Meanwhile, `strategy_config` also sets `capital_per_trade` and `leverage` per strategy, conflating sizing with universe definition.
- Min-notional mismatch: `risk_limits.global.min_notional_usdt = 100` (`config/risk_limits.json:12-14`) conflicts with `sizing.min_gross_usd_per_order = 10` and `per_symbol_min_gross_usd = {"BTCUSDT": 15}` (`config/strategy_config.json:30-55`), triggering repeated risk vetoes as highlighted in `infra_v6.0_presprint_codex_audit.md:8-14`.
- Strategy duplication: Per-strategy config objects redundantly contain `bucket`, `risk`, `entry` definitions plus duplicate `max_concurrent_positions` keys (lines `config/strategy_config.json:107-196`). JSON keeps the last value (usually `1`), but operators cannot tell which is authoritative, and `_strategy_concurrency_budget()` in `execution/signal_screener.py:266-283` inherits silent overrides, causing concurrency drift.
- Universe metadata: `pairs_universe.json` and `config/universe` disagree on leverage overrides (e.g., `target_leverage` vs `per_symbol_leverage`), yet no validation ensures they converge. v6 expects a generated `universe_resolver` output bridging risk/strategy/dash, but this repo maintains three separate, manually edited files.
- Registry gaps: Without module/class references, `strategy_registry.json` can’t satisfy v6’s strategy registry interface (docs/v6.0_architecture_brief.md:118-165). Confidence/capacity metadata exists but isn’t connected to execution or telemetry.
- Dashboard/reserve divergence: `config/assets.json` stores both qty and price for each asset while `config/reserves.json` is only balances; they can easily drift. There is no script to reconcile them with `execution/reserves.py` or nav calculations.

### 3.3 Config-Test Contracts
- `tests/test_config_parsing.py:1-66` asserts current schema shapes and value presence, so any schema normalization would break those tests until updated. The test enforces overlapping keys (e.g., verifying `strategy_config` sizing has `max_gross_exposure_pct` and `risk_limits.global` contains the same keys), thereby reinforcing the drift.
- `tests/test_screener_tier_caps.py:5-150` expects risk config JSON structures with `global.tiers`, `per_symbol`, and concurrency fields, and ensures screener gating reasons contain legacy strings. v6 schema changes need to update these fixtures.
- Dashboard helpers read `config/risk_limits.json` via `RiskGate` (`dashboard/live_helpers.py:84-149`), so tests rely on that shape.
- Telemetry tests such as `tests/test_router_metrics_reader.py:7-33`, `tests/test_state_publish_stats.py:7-47`, and `tests/test_router_health_events.py:9-87` assume JSONL field names that may evolve once telemetry is reworked.

## 4. Tests vs Topology
- `tests/test_risk_limits.py:1-200` binds `execution/risk_limits.py` functions (`RiskConfig`, `RiskState`, `check_order`) by asserting kill switches, per-trade caps, nav freshness, and detail payloads. These tests form the primary regression guard for the risk engine.
- `tests/test_screener_tier_caps.py:5-150` binds `execution/signal_screener.would_emit`, ensuring portfolio/tier caps, concurrency, orderbook vetoes, and risk reason strings remain stable.
- `tests/test_exchange_dry_run.py:8-136` exercises `execution/exchange_utils.send_order` for dry-run gating, reduce-only logic, and stop closePosition toggles, preventing regressions in dual-hedge reduce-only paths.
- `tests/test_router_smoke.py:10-175`, `tests/test_order_router_ack.py:6-23`, and `tests/test_order_metrics.py:16-77` cover `execution/order_router.route_order/route_intent`, verifying maker/taker flows, dry-run behavior, result payloads, and telemetry shaping.
- `tests/test_execution_health.py:1-92`, `tests/test_execution_alerts.py:1-50`, and `tests/test_execution_intel_sizing.py:1-35` validate `execution/utils.execution_health` + intel sizing functions. They assume router metrics provide maker/fallback/slippage stats, so missing fields lead to meaningless alerts.
- Intel tests: `tests/test_expectancy_map.py:1-41`, `tests/test_symbol_score.py:1-54`, `tests/test_maker_offset.py:1-39`, and `tests/test_router_policy.py:1-34` bind expectancy, symbol scoring, maker offsets, and router policy classification to existing metrics.
- Dashboard tests: `tests/test_dashboard_equity.py:4-20`, `tests/test_dashboard_metrics.py:5-20`, and `tests/test_dashboard_intel_helpers.py:1-26` ensure helper functions deliver formatted strings and delegate to intel modules.
- Firestore/telemetry tests: `tests/test_firestore_client.py:12-51` covers retry/backoff for Firestore doc writes even though production paths disable them, `tests/test_firestore_execution_intel.py:1-15` asserts publishing is a no-op, and `tests/test_firestore_publish.py:3-7` is xfailed due to stub mismatches. `tests/test_state_publish_stats.py:7-47` anchors execution stat generation, while `tests/test_router_health_events.py:9-87` and `tests/test_router_health_v2.py:11-110` expect order metrics JSONL formats.
- Coverage assessment: Risk, screener, router metrics, and select dashboard helpers are well covered. Telemetry publish paths (`execution/firestore_utils.*`, `_maybe_publish_*` stubs), sync daemon (`execution/sync_state.py`), and DevOps scripts lack tests. Strategy registry loading, universe resolver discovery merge, and multi-strategy orchestrators remain largely untested, so v6 changes to those areas carry higher risk.

## 5. Legacy / Obsolete Surface Inventory
- **Legacy but used**: `execution/utils/toggle.py` (symbol disable cache) and `execution/risk_autotune.py` are actively imported but still reference v5.9 assumptions; they should be modernized but cannot be removed yet. `dashboard/router_health.py` consumes legacy JSONL metrics that will need adapters.
- **Legacy and unused**:
  - `execution/hedge_sync.py:1-6` and `execution/pipeline_probe.py:1-40` explicitly mark themselves deprecated.
  - Archived strategies, probes, and ML pipelines under `archive/deprecated_v5.9.5/*` and `archive/deprecated_v5.7/*` are not imported anywhere else.
  - `execution/firestore_mirror.py:28-63` is unused because `execution/firestore_utils.py` short-circuits all writes.
  - Empty container scaffolds (`infrastructure/*.yml`, `infrastructure/Dockerfile`, `infrastructure/startup.sh`) exist only as placeholders.
  - Scripts such as `scripts/polymarket_insiders.py`, `scripts/replay_logs.py`, and `scripts/smoke_exec_logging.py` are not referenced in configs/supervisor or docs, indicating historical experiments ready for quarantine.

## 6. v6.0 Gaps & To-Create Stubs
- **Canonical strategy registry module**: Implement `strategies/` package with per-strategy classes inheriting `core/strategy_base.py:12-53`, plus loader utilities to honor `config/strategy_registry.json`. Reuse `execution/signal_generator._load_registry()` but extend it to instantiate strategies by module path.
- **Universe resolver output**: Add `execution/universe_resolver.py` companion script that materializes a `universe.json` snapshot consumed by screener, risk, and dashboard; reuse tier loaders plus `config/pairs_universe.json` metadata to enforce consistent leverage/min notionals.
- **RiskEngine class wrapper**: Encapsulate the module-level functions from `execution/risk_limits.py` into a `RiskEngine` class with explicit `check_order` and `check_nav` methods, and ensure executor passes concurrency/tier context – aligning with `docs/v6.0_architecture_brief.md:92-152`.
- **Telemetry publishers**: Fill `_mirror_router_metrics`, `_maybe_publish_execution_health`, `_maybe_emit_execution_alerts`, and `_maybe_publish_execution_intel` with real Firestore/S3 writers referencing `execution/firestore_utils.py`. Extend router metrics to include maker/taker/fallback flags and realized PnL so expectancy/feedback allocators have the data they expect (`infra_v6.0_presprint_codex_audit.md:23-27`).
- **Feedback allocator + router autotune stubs**: Create modules (e.g., `execution/feedback_allocator.py`, `execution/router_autotune.py`) that ingest risk/sizer/intel outputs and provide signals for NAV caps, router bias, and symbol scoring. These can wrap existing intel modules (`execution/intel/expectancy_map.py`, `execution/intel/symbol_score.py`) until richer data lands.
- **DevOps/environment templates**: Populate `infrastructure/Dockerfile`/`docker-compose.yml` and add `.env.example` plus multi-service supervisor configs (hedge-sync, telemetry publisher) so v6’s deployment requirements are codified. Update `.github/workflows/ci.yml` to reference actual requirement files.

## 7. Remediation Plan (No patches yet)
1. **Batch 0 — Doc/topology alignment**
   - Update docs to reflect the actual repo layout and current telemetry limitations (`docs/v6.0_architecture_brief.md`, `docs/v6.0_Master_Architecture_Map.md`, new `docs/ARCHITECTURE_CURRENT.md`).
   - Files to touch: architecture docs, new topology snapshot, `infra_v6.0_presprint_codex_audit.md` cross-references.
   - Tests to keep green: doc-only change (none), but run lint to ensure no doc tooling fails.

2. **Batch 1 — Config schema normalization**
   - Split `config/strategy_config.json` into pure per-strategy settings (id, symbol, timeframe, params) and move nav/exposure/risk knobs into `config/risk_limits.json` + `config/pairs_universe.json`. Generate `pairs_universe` from risk/tiers and add schema validation tests.
   - Files: `config/strategy_config.json`, `config/risk_limits.json`, `config/pairs_universe.json`, `config/strategy_registry.json`, helper scripts/tests (`execution/signal_generator.py`, `execution/signal_screener.py`, `execution/universe_resolver.py`, `tests/test_config_parsing.py`, `tests/test_screener_tier_caps.py`).
   - Tests: `tests/test_config_parsing.py`, `tests/test_screener_tier_caps.py`, `tests/test_risk_limits.py`, plus any new schema validators.

3. **Batch 2 — Execution & risk contract clarity**
   - Ensure executor passes concurrency/tier context to `check_order`, rename risk reasons to match config, and expose per-symbol caps via `RiskEngine` wrapper. Implement telemetry fields (maker final/fallback, realized PnL) in router metrics.
   - Files: `execution/executor_live.py`, `execution/risk_limits.py`, `execution/order_router.py`, `execution/utils/metrics.py`, `execution/intel/router_policy.py`, `execution/intel/expectancy_map.py`.
   - Tests: `tests/test_risk_limits.py`, `tests/test_screener_tier_caps.py`, `tests/test_router_smoke.py`, `tests/test_router_policy.py`, `tests/test_expectancy_map.py`, `tests/test_router_metrics_effectiveness.py`.

4. **Batch 3 — Telemetry & dashboard truthfulness**
   - Implement `_mirror_router_metrics`/`_maybe_publish_execution_health/_intel/_alerts`, extend `execution/firestore_utils.py` to optionally publish locally and to Firestore/S3, and add a canonical `universe.json`/`risk_snapshot.json`. Adjust dashboard helpers to read the new canonical telemetry instead of configs.
   - Files: `execution/executor_live.py`, `execution/firestore_utils.py`, `execution/state_publish.py`, `execution/sync_state.py`, `dashboard/live_helpers.py`, `dashboard/router_health.py`, `scripts/doctor.py`.
   - Tests: `tests/test_execution_health.py`, `tests/test_execution_alerts.py`, `tests/test_router_health_events.py`, `tests/test_router_health_v2.py`, `tests/test_dashboard_*`, `tests/test_state_publish_stats.py`.

5. **Batch 4 — DevOps & CI readiness**
   - Provide real container specs and `.env.example`, fix CI requirements, and ensure supervisor/cron definitions cover sync + telemetry. Add tests or linting for infrastructure manifests.
   - Files: `.github/workflows/ci.yml`, `infrastructure/Dockerfile`, `infrastructure/docker-compose.yml`, `infrastructure/startup.sh`, `deploy/supervisor/*.conf`, `ops/hedge.conf`, new `.env.example`.
   - Tests: run CI (pytest + lint), plus any new integration tests for container builds and supervisor validation scripts.
