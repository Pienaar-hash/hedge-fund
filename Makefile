PYTHON ?= $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; else echo python; fi)

.PHONY: smoke
smoke:
	@PYTHONPATH=. $(PYTHON) scripts/smoke_test.py

.PHONY: test
test:
	@PYTHONPATH=. $(PYTHON) -m pytest tests/unit tests/integration tests/dashboard tests/scripts -q

.PHONY: test-fast
test-fast:
	@PYTHONPATH=. $(PYTHON) -m pytest tests/unit tests/integration tests/dashboard tests/scripts -m "not runtime and not legacy" -q

.PHONY: test-runtime
test-runtime:
	@PYTHONPATH=. $(PYTHON) -m pytest tests/integration -m "runtime" -q

.PHONY: test-research
test-research:
	@PYTHONPATH=. $(PYTHON) -m pytest tests/research -q

.PHONY: test-all
test-all:
	@PYTHONPATH=. $(PYTHON) -m pytest tests/unit tests/integration tests/dashboard tests/scripts tests/research tests/legacy -q

.PHONY: cov-core
cov-core:
	@PYTHONPATH=. $(PYTHON) -m pytest tests/unit tests/integration tests/dashboard tests/scripts \
		--cov=execution --cov=dashboard --cov=prediction --cov=treasury \
		--cov-report=term --cov-report=xml:coverage-core.xml -q

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
