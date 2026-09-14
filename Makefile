PYTHON ?= python3

.PHONY: help test lint check build reproduce posters inventory

help:
	@echo "test       Run scientific and reproducibility checks"
	@echo "lint       Check maintained Python code"
	@echo "check      Run lint and tests"
	@echo "build      Build the Python source distribution and wheel"
	@echo "reproduce  Recompute registered results from saved inputs"
	@echo "posters    Rebuild and verify both posters from saved inputs"
	@echo "inventory  Inventory research artifacts without moving them"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src/wnt_pinn tests

check: lint test

build:
	$(PYTHON) -m build

reproduce:
	$(PYTHON) -m wnt_pinn.cli reproduce --all

posters:
	$(PYTHON) -m wnt_pinn.cli publications build --id poster_first --id poster_second

inventory:
	$(PYTHON) -m wnt_pinn.cli artifacts inventory
