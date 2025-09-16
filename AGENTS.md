# AGENTS.md — GPT Hedge

## How to run
- Unit tests: `pytest -q`
- Lint: `ruff check .`
- Type-check: `mypy .`

## Project map
- execution/: live trading, sync, risk, telegram
 - dashboard/: Streamlit app (`app.py` and `main.py`)
- config/: strategy_config.json, firebase_creds.json (not in git)
- utils/: shared helpers

## Commands
- One-shot signal+sync: `ENV=prod PYTHONPATH=. ONE_SHOT=1 python -m execution.executor_live`
- Long-run executor: `ENV=prod PYTHONPATH=. python -m execution.executor_live`
- Dashboard: `streamlit run dashboard/app.py --server.port=8501`

## Guardrails
- Never commit secrets; read `config/firebase_creds.json` if present.
- Use `execution/risk_limits.py` to enforce caps.
- Prefer small PRs with tests. Always run tests+ruff+mypy before commit.

## Acceptance for PRs
- ✅ `pytest -q` green
- ✅ `ruff check .` clean (or auto-fix)
- ✅ `mypy .` clean (or suppressions documented)
