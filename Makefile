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
