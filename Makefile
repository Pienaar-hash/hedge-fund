PYTHON ?= python

.PHONY: smoke
smoke:
	@PYTHONPATH=. $(PYTHON) scripts/smoke_test.py

.PHONY: test
test:
	@PYTHONPATH=. pytest tests/unit tests/integration -q

.PHONY: test-fast
test-fast:
	@PYTHONPATH=. pytest tests/unit tests/integration -m "not runtime and not legacy" -q

.PHONY: test-runtime
test-runtime:
	@PYTHONPATH=. pytest tests/integration -m "runtime" -q

.PHONY: test-all
test-all:
	@PYTHONPATH=. pytest -q

.PHONY: lint
lint:
	@PYTHONPATH=. ruff check .

.PHONY: format
format:
	@PYTHONPATH=. black execution dashboard tests

.PHONY: deadcode
deadcode:
	@PYTHONPATH=. vulture execution dashboard > deadcode.txt

.PHONY: lint-docs
lint-docs:
	@npx markdownlint \"docs/**/*.md\"

.PHONY: pretag-v7.6
pretag-v7.6:
	@PYTHONPATH=. $(PYTHON) scripts/preflight_v7_6.py

.PHONY: aw-status
aw-status:
	@PYTHONPATH=. $(PYTHON) scripts/aw_status.py

.PHONY: ecs-status
ecs-status:
	@PYTHONPATH=. $(PYTHON) scripts/ecs_status.py

.PHONY: ecs-calibration
ecs-calibration:
	@PYTHONPATH=. $(PYTHON) scripts/ecs_calibration_audit.py

.PHONY: ecs-phase-map
ecs-phase-map:
	@PYTHONPATH=. $(PYTHON) scripts/ecs_phase_map.py

.PHONY: ecs-regime-pnl
ecs-regime-pnl:
	@PYTHONPATH=. $(PYTHON) scripts/ecs_regime_pnl.py

.PHONY: ecs-regret
ecs-regret:
	@PYTHONPATH=. $(PYTHON) scripts/ecs_regret_curve.py

.PHONY: ecs-v2-eval
ecs-v2-eval:
	@PYTHONPATH=. $(PYTHON) scripts/ecs_shadow_v2_eval.py

.PHONY: aw-verify
aw-verify:
	@PYTHONPATH=. $(PYTHON) scripts/activation_verify.py

.PHONY: aw-preflight
aw-preflight:
	@PYTHONPATH=. $(PYTHON) scripts/activation_verify.py --preflight

.PHONY: heartbeat
heartbeat:
	@PYTHONPATH=. $(PYTHON) scripts/telegram_daily_heartbeat.py --dry-run

.PHONY: heartbeat-send
heartbeat-send:
	@PYTHONPATH=. TELEGRAM_ENABLED=1 $(PYTHON) scripts/telegram_daily_heartbeat.py

.PHONY: heartbeat-test
heartbeat-test:
	@PYTHONPATH=. TELEGRAM_ENABLED=1 $(PYTHON) scripts/telegram_daily_heartbeat.py --test
.PHONY: fund-ops
fund-ops:
	@PYTHONPATH=. $(PYTHON) ops/fund_ops_monthly.py