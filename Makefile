PYTHON ?= python

.PHONY: smoke
smoke:
	@PYTHONPATH=. $(PYTHON) scripts/smoke_test.py

.PHONY: test
test:
	@PYTHONPATH=. pytest tests/unit tests/integration -q
