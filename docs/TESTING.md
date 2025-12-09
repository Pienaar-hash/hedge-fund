# Testing Topology (v7.6)

We split tests into clear lanes:

- `tests/unit/` — fast, deterministic, no external I/O.
- `tests/integration/` — multi-module/stateful tests (use tmp_path when possible).
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

# Runtime slice
make test-runtime

# Full sweep (includes legacy)
make test-all
```

Agent guidance:
- Keep `test-fast` green for typical patches.
- If you change runtime/state surfaces, also run `test-runtime`.
