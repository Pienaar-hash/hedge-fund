# PROJECT_MAP.md — GPT Hedge v5.6

## Directory Structure
~~~
hedge-fund/
│
├── execution/                 # Live trading and telemetry
│   ├── executor_live.py       # Main loop (ACK/FILL split + PnL)
│   ├── order_router.py        # Order routing — preserves raw status
│   ├── risk_limits.py         # Exposure and drawdown caps
│   ├── state_publish.py       # Publishes fills / PnL to Firestore
│   ├── signal_doctor.py       # Router health and signal validation
│   └── ml/                    # ML screeners and predictors
│
├── scripts/
│   ├── backfill_fills_pnl.py  # Rebuilds fills + realized PnL
│   ├── doctor.py              # Router health / NAV diagnostics
│   └── screener_probe.py      # Manual screener tests
│
├── dashboard/
│   ├── app.py                 # Main Streamlit app
│   ├── router_health.py       # Router latency + PnL tiles
│   └── main.py                # Legacy entrypoint
│
├── config/
│   ├── strategy_config.json   # Strategy parameters
│   └── firebase_creds.json    # Secrets (not in git)
│
├── utils/                     # Shared helpers
├── tests/                     # Unit + integration tests
└── docs/                      # Documentation suite
~~~

## Documentation Suite
| File | Purpose |
|------|---------|
| **AGENTS.md** | Architecture & agent behavior |
| **OPERATIONS.md** | Runtime commands & runbook |
| **PROJECT_MAP.md** | Repository layout overview |
| **CONTRIBUTING.md** | PR, lint, and test policy |

## Build & Quality Checks
Run core validation before every commit:
~~~bash
pytest -q          # Unit & integration tests
ruff check .       # Lint (auto-fix if needed)
mypy .             # Static type checking
~~~

## Development Workflow
1. **Branching**  
   - Create a new branch per feature or hotfix (`v5.6-router-fix`, etc.).  
   - Rebase or merge only after all tests + lint + mypy pass.

2. **Testing**  
   - Unit tests live in `/tests`.  
   - Integration tests verify order routing, fill tracking, and PnL updates.  
   - Use `pytest -q --disable-warnings` for clean output.

3. **Continuous Integration**  
   - Local pre-commit: run `ruff check . && mypy . && pytest -q`.  
   - CI pipeline mirrors this to block regressions.

4. **Documentation Updates**  
   - `AGENTS.md` → architecture or behavioral changes.  
   - `OPERATIONS.md` → command/runtime changes.  
   - `PROJECT_MAP.md` → new modules or layout shifts.

## Version Reference
- **System version:** v5.6 — Execution Integrity Hotfix  
- **Audit date:** November 2025  
- **Auditor:** GPT-5 Codex Quant  
- **Artifacts:**  
  - `/mnt/data/repo_audit_summary.md`  
  - `/mnt/data/repo_audit_findings.csv`  
  - `/mnt/data/repo_patch_suggestions.json`

## Notes
- All fill, PnL, and telemetry logic now trace back to **executor_live → router → Firestore → dashboard**.  
- Backfill utilities live under `scripts/` and integrate with the `order_fill` / `order_close` schema.  
- Documentation and tests are versioned together starting v5.6; always commit updated docs alongside code changes.
