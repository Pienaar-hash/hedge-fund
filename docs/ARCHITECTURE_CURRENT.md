# ARCHITECTURE_CURRENT.md — v5.10 / v6-pre Reality Map

This document is the **current-state** companion to the v6 blueprint in `docs/v6.0_Master_Architecture_Map.md` and the rollout intent in `docs/v6.0_architecture_brief.md`. It distills the point-in-time repo topology captured in `docs/infra_v6.0_repo_topology.md` so Batch 0 has an explicit source of truth.

## 1. Repo Layout

- **execution/** – Live trading stack. `executor_live.py` runs the intent → risk → sizing → router → exchange loop with telemetry hooks mostly stubbed; supporting modules handle risk gates (`risk_limits.py`), routers (`order_router.py`), exchange adapters (`exchange_utils.py`), size math (`size_model.py`, `capital_allocator.py`), and telemetry emitters (`state_publish.py`, `sync_state.py`).
- **core/** – Shared abstractions, currently only `strategy_base.py` for basic Strategy objects used by signal generators.
- **dashboard/** – Streamlit operator UI (`app.py`, `main.py`) with helpers (`live_helpers.py`, `router_health.py`, `dashboard_utils.py`, `nav_helpers.py`, `async_cache.py`) that read telemetry JSONL plus risk config.
- **scripts/** – Operator tooling such as `screener_probe.py`, `registry_ctl.py`, doctor/diagnostic utilities (`doctor.py`, `balance_doctor.py`, `fs_doctor.py`), ML retrain scripts, and release helpers (`go_live_now.sh`, `run_executor_once.sh`).
- **config/** – Runtime JSON/YAML describing strategies (`strategy_config.json`), risk knobs (`risk_limits.json`), registry (`strategy_registry.json`), universe/tier metadata (`pairs_universe.json`, `symbol_tiers.json`), dashboard settings, reserves/assets, and toggles in `settings.json`.
- **tests/** – Pytest suite covering risk gates, screener tier caps, routers, telemetry formatters, dashboard helpers, exchange dry-runs, and Firestore stubs (e.g., `tests/test_risk_limits.py`, `tests/test_router_smoke.py`, `tests/test_state_publish_stats.py`).
- **research/**, **ml/**, **models/** – Notebooks and experimental code for factor fusion, RL sizing, telemetry tooling, and serialized model payloads referenced by scripts.
- **docs/** – Architecture, operations, and audit notes including the v6 blueprint/brief, pre-flight checklists, and historical infra reports.
- **DevOps & supporting roots** – `deploy/` supervisor configs, `ops/hedge.conf`, `cron/rotate_exec_logs.sh`, service wrappers in `bin/`, `.github/workflows/ci.yml`, placeholder `infrastructure/` docker assets, and operational data/telemetry roots (`data/`, `logs/`, `reports/`, `telegram/`).

## 2. v5.10 Pipeline Summary

`execution/executor_live.py` (see `_run_executor`) drives the production path:

1. **Signal ingestion** – `execution/signal_generator.py` pulls configured strategy modules, applies screeners, and produces intents annotated with volatility/tier metadata.
2. **Risk gate** – Intents call into module-level helpers in `execution/risk_limits.py` (no `RiskEngine` class yet) via `check_order`, but executor fails to forward tier/concurrency context, so caps are partially enforced.
3. **Sizing** – `execution/size_model.py` along with `capital_allocator.py` convert approved intents into USD notionals, bucket caps, leverage, and quantity payloads.
4. **Routing** – `execution/order_router.py` uses `execution/intel/router_policy.py`, `execution/intel/maker_offset.py`, and `execution/intel/symbol_score.py` to pick maker/taker paths; router metrics omit several v6-required fields.
5. **Exchange adapter** – `execution/exchange_utils.py` normalizes payloads and calls Binance REST/UM-F endpoints; reduce-only and close-position helpers are implemented but tuned for a single venue.
6. **State/telemetry** – `execution/state_publish.py` and `_maybe_publish_*` hooks exist but default to no-ops, so telemetry relies on JSONL files without remote sinks. `execution/sync_state.py` mirrors files locally with Firestore disabled.

## 3. Known Gaps vs v6.0 Blueprint

- **RiskEngine abstraction** – Current risk module exposes free functions, yet the v6 blueprint expects a `RiskEngine` object with cached limits and explicit APIs (`docs/v6.0_Master_Architecture_Map.md#4.-risk-engine`, `infra_v6.0_repo_topology.md §6`).
- **StrategyRegistry & Strategy package** – Config files exist (`config/strategy_registry.json`) but there is no `strategies/` implementation directory; `execution/signal_generator.py` hard-codes modules, so the registry cannot load real strategy classes.
- **Canonical universe snapshot** – Config drift across `strategy_config.json`, `risk_limits.json`, and `pairs_universe.json` persists, and no `universe.json` output exists despite being required for dashboard + telemetry parity.
- **Telemetry publishers** – `_mirror_router_metrics`, `_maybe_publish_execution_health`, `_maybe_emit_execution_alerts`, and `_maybe_publish_execution_intel` short-circuit with `return None`, so telemetry consumers get stale or missing data.
- **Feedback allocator & router autotune** – No modules consume expectancy metrics or router outcomes to tune offsets; v6’s feedback loop is entirely absent.
- **Multi-venue/DevOps scaffolding** – `infrastructure/*` files are empty placeholders; `.github/workflows/ci.yml` references requirement files that do not exist, and there is no `.env.example` enumerating runtime controls.
- **Legacy Firestore surfaces** – `execution/firestore_utils.py` intentionally no-ops remote publishing and `execution/firestore_mirror.py` remains unused, so v6 telemetry contracts lack real storage destinations.

## 4. Test Coverage Overview

- **Strong anchors** – Risk gating (`tests/test_risk_limits.py`), screener tier enforcement (`tests/test_screener_tier_caps.py`), router flows (`tests/test_router_smoke.py`, `tests/test_order_router_ack.py`), and exchange dry-runs (`tests/test_exchange_dry_run.py`) bind the current contracts tightly. Intel math (`tests/test_expectancy_map.py`, `tests/test_symbol_score.py`, `tests/test_maker_offset.py`) and dashboard helpers (`tests/test_dashboard_equity.py`, `tests/test_dashboard_metrics.py`) also have coverage.
- **Moderate coverage** – Telemetry metric shaping and state publishing via `tests/test_router_health_events.py`, `tests/test_router_health_v2.py`, and `tests/test_state_publish_stats.py` exercise data readers but still rely on stubbed publishers.
- **Weak spots** – Firestore utilities (`tests/test_firestore_publish.py` is xfailed), executor telemetry hooks, sync daemons, DevOps scripts, strategy registry loading, and universe resolver flows lack meaningful tests. These gaps mean v6 migrations in telemetry, config schemas, and registry layers will need fresh regression suites.

## 5. Legacy / Deprecated Surfaces

- `execution/hedge_sync.py`, `execution/pipeline_probe.py`, and scripts like `scripts/polymarket_insiders.py` / `scripts/replay_logs.py` are flagged as deprecated in the audit.
- Archived strategy and ML assets live under `archive/deprecated_v5.9.5/*` and `archive/deprecated_v5.7/*`; they are not imported anywhere else but remain in-tree.
- `execution/firestore_mirror.py` is effectively dead because `execution/firestore_utils.py` turns every publish into a no-op.
- DevOps scaffolds (`infrastructure/docker-compose.yml`, `infrastructure/Dockerfile`, `infrastructure/startup.sh`) are zero-byte placeholders waiting for Batch 4.
- Dashboard router health code consumes legacy JSONL formats that omit the new maker/taker/fallback fields, meaning telemetry consumers will require adapters during the v6 rollout.
