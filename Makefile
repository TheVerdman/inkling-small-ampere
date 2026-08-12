PYTHON ?= python3
UV ?= $(if $(wildcard .tools/uv-bootstrap/bin/uv),.tools/uv-bootstrap/bin/uv,uv)
GATE_E_PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(UV) run --frozen python)
PYTEST ?= $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,$(UV) run --frozen pytest)
RUFF ?= $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,$(UV) run --frozen ruff)
MYPY ?= $(if $(wildcard .venv/bin/mypy),.venv/bin/mypy,$(UV) run --frozen mypy)
STORAGE_PATH ?= .
DOCTOR_ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help bootstrap doctor doctor-strict inspect-checkpoint model-memory multimodal-local-check vertex-gate-e-dry-run test lint format format-check typecheck check

help:
	@echo "Inkling-Small Ampere development commands"
	@echo ""
	@echo "  make doctor         Collect an immutable local environment report"
	@echo "  make doctor-strict  Require the four-A100 target contract"
	@echo "  make inspect-checkpoint  Read pinned checkpoint headers only"
	@echo "  make model-memory   Project profiles and all four-rank placements"
	@echo "  make multimodal-local-check  Verify media patches, fixtures, admission, and native dry run"
	@echo "  make vertex-gate-e-dry-run  Render fail-closed Vertex, bootstrap, probe, and edge gates"
	@echo "  make bootstrap      Install pinned uv locally and sync the dev environment"
	@echo "  make check          Run hermetic formatting, lint, types, and tests"
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

multimodal-local-check:
	PYTHONPATH=src:. $(GATE_E_PYTHON) scripts/apply_runtime_patchset.py \
		--project-root . \
		--multimodal \
		--verify-only
	PYTHONPATH=src:. $(GATE_E_PYTHON) scripts/gpu/validate_multimodal_native_engine.py \
		--profile configs/serving/responses-2k-multimodal-bringup-v1.json \
		--research-manifest manifests/multimodal-research-control-v1.json \
		--dry-run \
		--output /tmp/inkling-multimodal-native-dry-run.json

vertex-gate-e-dry-run:
	PYTHONPATH=src $(GATE_E_PYTHON) scripts/render_vertex_gate_e.py
	PYTHONPATH=src $(GATE_E_PYTHON) -m inkling_ampere.serving.bootstrap \
		--profile configs/serving/responses-2k-bringup-v1.json \
		--plan configs/serving/vertex-gate-e-plan-v1.json \
		--model-path /tmp/inkling-small-ampere \
		--dry-run
	PYTHONPATH=src $(GATE_E_PYTHON) -m inkling_ampere.serving.storage_probe \
		--plan configs/serving/vertex-gate-e-plan-v1.json \
		--dry-run
	PYTHONPATH=src $(GATE_E_PYTHON) -m inkling_ampere.serving.edge \
		--profile configs/serving/responses-2k-bringup-v1.json \
		--plan configs/serving/vertex-gate-e-plan-v1.json \
		--dry-run

test:
	$(PYTEST)

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

format-check:
	$(RUFF) format --check .

typecheck:
	$(MYPY) src tests

check: format-check lint typecheck test
