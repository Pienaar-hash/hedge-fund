# Testing Topology (v7.5)

We split tests into three lanes:

- `tests/unit/` — fast, side-effect-free tests (pure functions/classes).
- `tests/integration/` — runtime/state/dashboard integration tests.
- `tests/legacy/` — v5/v6-era tests kept for reference; not part of v7.5 green runs.

Canonical commands:

```bash
# Daily green bar
PYTHONPATH=. pytest tests/unit tests/integration -q

# Run legacy set explicitly
PYTHONPATH=. pytest tests/legacy -m legacy -q
```

Notes:
- `pytest.ini` points default discovery to unit + integration only.
- Legacy tests may be unstable or skipped; they are informational only.
