# infra_v6.1 Repo Topology

## Active surfaces
- `execution/` — v6 runtime brain (executor_live, risk_engine_v6, risk_limits, router, intel, nav, pipeline shadow/compare, events, utils).
- `dashboard/` — Streamlit UI reading state/telemetry snapshots from execution.
- `scripts/` — Ops/probes/CLIs (pipeline shadow heartbeat/compare, router autotune probes, expectancy builders, doctor).
- `config/` — Risk/universe definitions, router/runtime configs, strategy registry, dashboard settings.
- `tests/` — Pytest suite covering executor, router, risk, intel, pipeline, dashboard.
- `docs/` — Architecture, audits, contracts, release notes, prompts (legacy moved under docs/archive/).
- `ops/` — Supervisor manifests and operational configs.
- `bin/` — Launch wrappers for dashboard/runtime; kept thin and aligned with deploy scripts.

## Archived / legacy
- `archive/` — v5 and experimental artifacts (old prompts, diff files, duplicate module trees like `gpt_schema/`, experimental dirs `core/`, `cron/`, `data/`, `ml/`, `models/`, `pr/`, `py/`, `repo/`, `research/`, `telegram/`, `infrastructure/`, `reports/`, `status/`, `test/`, cached JSON like `tmp_router_health.json`, `screener_state.json`, binary blobs like `execution.tar.gz`). Not on `PYTHONPATH`.

## Notes
- `PYTHONPATH` must point to repo root; avoid shadow trees (e.g., `gpt_schema/`) in active path.
- Tests and scripts should import only from `execution/*` and `dashboard/*`.
- Runtime state (`logs/`, caches) must remain unversioned; keep `.gitignore` intact.
