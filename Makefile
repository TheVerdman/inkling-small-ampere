PYTHON ?= python3
UV ?= $(if $(wildcard .tools/uv-bootstrap/bin/uv),.tools/uv-bootstrap/bin/uv,uv)
STORAGE_PATH ?= .
DOCTOR_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help bootstrap doctor doctor-strict inspect-checkpoint model-memory test lint format format-check typecheck check

help:
	@echo "Inkling-Small Ampere development commands"
	@echo ""
	@echo "  make doctor         Collect an immutable local environment report"
	@echo "  make doctor-strict  Require the four-A100 target contract"
	@echo "  make inspect-checkpoint  Read pinned checkpoint headers only"
	@echo "  make model-memory   Project profiles and all four-rank placements"
	@echo "  make bootstrap      Install pinned uv locally and sync the dev environment"
	@echo "  make check          Run formatting, lint, types, and tests"
	@echo "  make format         Apply Ruff formatting and safe lint fixes"

bootstrap:
	./scripts/bootstrap.sh

doctor:
	PYTHONPATH=src $(PYTHON) -m inkling_ampere.environment \
		--contract configs/hardware/a2-ultragpu-4g.json \
		--software-lock configs/hardware/software-lock.json \
		--storage-path "$(STORAGE_PATH)" \
		$(DOCTOR_ARGS)

doctor-strict:
	PYTHONPATH=src $(PYTHON) -m inkling_ampere.environment \
		--contract configs/hardware/a2-ultragpu-4g.json \
		--software-lock configs/hardware/software-lock.json \
		--storage-path "$(STORAGE_PATH)" \
		--strict \
		$(DOCTOR_ARGS)

inspect-checkpoint:
	PYTHONPATH=src $(UV) run --frozen python scripts/inspect_checkpoint.py \
		--repository thinkingmachines/Inkling-Small \
		--revision b2d4f225a02032c5d154bff748ab5a00c5ca26e4

model-memory:
	PYTHONPATH=src $(UV) run --frozen python scripts/model_memory.py
	PYTHONPATH=src $(UV) run --frozen python scripts/simulate_sharding.py

test:
	$(UV) run --frozen pytest

lint:
	$(UV) run --frozen ruff check .

format:
	$(UV) run --frozen ruff format .
	$(UV) run --frozen ruff check --fix .

format-check:
	$(UV) run --frozen ruff format --check .

typecheck:
	$(UV) run --frozen mypy src tests

check: format-check lint typecheck test
