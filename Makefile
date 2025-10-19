PYTHON ?= python

.PHONY: smoke
smoke:
	@PYTHONPATH=. $(PYTHON) scripts/smoke_test.py
