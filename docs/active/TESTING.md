# Testing Topology (v7.9)

We split tests into clear lanes:

- `tests/unit/` — fast, deterministic, no external I/O.
- `tests/integration/` — multi-module/stateful tests (use tmp_path when possible).
- `tests/dashboard/` — dashboard-only observer tests.
- `tests/scripts/` — CLI/script behavior tests.
- `tests/research/` — offline research/backtest suites run in a separate CI lane.
- `tests/legacy/` — v5/v6-era tests kept for reference; not part of the green bar.

Markers (see `pytest.ini`):
- `@pytest.mark.unit`
- `@pytest.mark.integration`
- `@pytest.mark.runtime` (longer/stateful; excluded from fast lane)
- `@pytest.mark.legacy`

Canonical commands:

```bash
# Day-to-day fast lane
make test-fast

# Core CI-equivalent lane
make test

# Research lane
make test-research

# Runtime slice
make test-runtime

# Core coverage baseline
make cov-core

# Full sweep (includes legacy)
make test-all
```

Agent guidance:
- Keep `test-fast` green for typical patches.
- If you change runtime/state surfaces, also run `test-runtime`.
- Use `make cov-core` when changing execution, dashboard, prediction, or treasury logic.
- `tests/integration/test_pub_tick_heartbeat.py` covers the `_pub_tick()` boundary heartbeat (`logs/execution/pub_tick_heartbeat.jsonl`); always monkeypatch `executor_live._PUB_TICK_HEARTBEAT_LOG` to an in-memory/tmp_path logger — this repo's `logs/` directory is the live executor's real log directory, not a test fixture.
