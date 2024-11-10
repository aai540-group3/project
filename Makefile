# Makefile

# Load environment variables
-include .env

# Define variables
PROJECT := diabetic-readmission-risk-prediction
VERSION := $(shell python setup.py --version 2>/dev/null || echo "0.0.0")
PYTHON := python3
VENV_DIR := pipeline/.venv
UV := uv
PYTHON_FILES := $(shell find . -type f -name "*.py" -not -path "./.venv*/*")
TEST_FILES := $(shell find pipeline/tests -type f -name "test_*.py")
CONF_FILES := $(shell find pipeline/conf -type f -name "*.yaml")
INFRA_FILES := $(shell find pipeline/infrastruct -type f)
PIPELINE_MAIN := pipeline/pipeline/__main__.py
SCRIPTS_DIR := scripts
DOCS_DIR := docs

# Requirements files
REQUIREMENTS_BASE := pipeline/requirements/base.txt
REQUIREMENTS_DVC := pipeline/requirements/dvc.txt
REQUIREMENTS_AUTOGLUON := pipeline/requirements/autogluon.txt
REQUIREMENTS_EXPLORE := pipeline/requirements/explore.txt
REQUIREMENTS_FEATURIZE := pipeline/requirements/featurize.txt
REQUIREMENTS_INFRASTRUCT := pipeline/requirements/infrastruct.txt
REQUIREMENTS_LOGISTIC := pipeline/requirements/logistic.txt
REQUIREMENTS_NEURAL := pipeline/requirements/neural.txt
REQUIREMENTS_PREPROCESS := pipeline/requirements/preprocess.txt
REQUIREMENTS_INGEST := pipeline/requirements/ingest.txt
REQUIREMENTS_DEPLOY := pipeline/requirements/deploy.txt
REQUIREMENTS_OPTIMIZE := pipeline/requirements/optimize.txt
REQUIREMENTS_EVALUATE := pipeline/requirements/evaluate.txt
REQUIREMENTS_REGISTER := pipeline/requirements/register.txt
REQUIREMENTS_MONITORING := pipeline/requirements/monitoring.txt
REQUIREMENTS_SERVE := pipeline/requirements/serve.txt

# DVC Remote
DVC_REMOTE := s3://$(BUCKET_NAME)/dvcstore

# Ports for local services
MLFLOW_PORT := 5000
FEAST_PORT := 6566
WANDB_PORT := 8080

# Timestamp for experiment names
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

.PHONY: all help setup clean venv test lint format install install-dev docs run combine verify-setup verify-paths check-dependencies build dist upload deploy serve monitor dvc-init dvc-remote-add dvc-repro dvc-push dvc-pull dvc-status dvc-doctor aws-configure mlflow-server wandb-local feast-apply feast-ui ingest preprocess explore featurize train evaluate register deploy-model serve-model monitor-model optimize

all: run

help:
	@echo "Makefile for $(PROJECT)"
	@echo "-----------------------"
	@echo "make setup: Set up the project (venv, install dependencies, DVC)"
	@echo "make clean: Remove temporary files and build artifacts"
	@echo "make venv: Create a virtual environment"
	@echo "make install: Install base requirements"
	@echo "make install-dev: Install development requirements"
	@echo "make install-dvc: Install DVC requirements"
	@echo "make dvc-init: Initialize DVC"
	@echo "make dvc-remote-add: Add the DVC remote repository (if needed - configure DVC_REMOTE first)"
	@echo "make dvc-repro: Reproduce the entire DVC pipeline"
	@echo "make dvc-push: Push data to DVC remote storage"
	@echo "make dvc-pull: Pull data from DVC remote storage"
	@echo "make dvc-status: Check DVC status"
	@echo "make dvc-doctor: Diagnose the DVC repository"
	@echo "make aws-configure: Configure AWS credentials (if using an S3 remote)"
	@echo "make mlflow-server: Start a local MLflow server"
	@echo "make wandb-local: Start a local WandB server"
	@echo "make feast-apply: Apply Feast feature repo configuration"
	@echo "make feast-ui: Start Feast UI locally"
	@echo "make <stage>: Run a specific DVC stage (e.g., make ingest)"
	@echo "make test: Run tests"
	@echo "make lint: Run linters"
	@echo "make format: Format code"
	@echo "make docs: Generate documentation"
	@echo "make run: Run the entire pipeline using Python"
	@echo "make combine: Combine source code files"
	@echo "make verify-setup: Verify project setup"
	@echo "make verify-paths: Verify project paths"
	@echo "make check-dependencies: Check dependency conflicts"
	@echo "make build: Build the project"
	@echo "make dist: Create distributable packages"
	@echo "make upload: Upload packages to PyPI"
	@echo "make deploy: Deploy the model"
	@echo "make serve: Serve the model locally"
	@echo "make monitor: Run model monitoring"
	@echo "make repro: Run 'dvc repro --force' (shortcut)"

# --- Setup targets ---
setup: venv install install-dvc

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(UV) venv $(VENV_DIR); \
	fi

install: venv
	pip install -r $(REQUIREMENTS_BASE)
	$(UV) pip install -e . --python $(VENV_DIR)/bin/python

install-dvc: venv
	pip install dvc[all]


# --- AWS targets ---
aws-configure:
	@if ! aws sts get-caller-identity &> /dev/null; then \
		aws configure set aws_access_key_id $(AWS_ACCESS_KEY_ID); \
		aws configure set aws_secret_access_key $(AWS_SECRET_ACCESS_KEY); \
		aws configure set default.region $(AWS_REGION); \
	fi


# --- DVC targets ---
dvc-init:
	@if [ ! -f ".dvc/config" ]; then \
		dvc init --subdir; \
	fi

dvc-remote-add: dvc-init
	@if ! grep -q "$(DVC_REMOTE)" .dvc/config; then \
		dvc remote add --default dvcstore $(DVC_REMOTE); \
	fi

dvc-repro: install-dvc aws-configure dvc-doctor
	cd pipeline && dvc repro --force

dvc-push: install-dvc aws-configure
	dvc push -r dvcstore

dvc-pull: install-dvc aws-configure
	dvc pull -r dvcstore

dvc-status: install-dvc
	dvc status -r dvcstore

dvc-doctor: install-dvc
	dvc doctor

# --- Individual DVC stage targets ---
ingest: install-dvc aws-configure
	cd pipeline && dvc exp run -n ingest-$(TIMESTAMP) --queue && dvc queue start

preprocess: install-dvc aws-configure
	cd pipeline && dvc exp run -n preprocess-$(TIMESTAMP) --queue && dvc queue start

explore: install-dvc aws-configure
	cd pipeline && dvc exp run -n explore-$(TIMESTAMP) --queue && dvc queue start

featurize: install-dvc aws-configure
	cd pipeline && dvc exp run -n featurize-$(TIMESTAMP) --queue && dvc queue start

train: install-dvc aws-configure
	cd pipeline && dvc exp run -n train-$(TIMESTAMP) --queue && dvc queue start

evaluate: install-dvc aws-configure
	cd pipeline && dvc exp run -n evaluate-$(TIMESTAMP) --queue && dvc queue start

register: install-dvc aws-configure
	cd pipeline && dvc exp run -n register-$(TIMESTAMP) --queue && dvc queue start

deploy-model: install-dvc aws-configure
	cd pipeline && dvc exp run -n deploy-$(TIMESTAMP) --queue && dvc queue start

serve-model: install-dvc aws-configure
	cd pipeline && dvc exp run -n serve-$(TIMESTAMP) --queue && dvc queue start

monitor-model: install-dvc aws-configure
	cd pipeline && dvc exp run -n monitor-$(TIMESTAMP) --queue && dvc queue start

optimize: install-dvc aws-configure
	cd pipeline && dvc exp run -n optimize-$(TIMESTAMP) --queue && dvc queue start


# --- MLflow target ---
mlflow-server: install-dev
	$(VENV_DIR)/bin/mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) &

# --- WandB target ---
wandb-local: install-dev
	wandb local --port $(WANDB_PORT) &

# --- Feast targets ---
feast-apply: install-dev
	$(VENV_DIR)/bin/python pipeline/features/store.py apply

feast-ui: install-dev
	feast ui --port $(FEAST_PORT) &


# --- Other targets ---
clean:
	rm -rf $(VENV_DIR) .uv_cache .ruff_cache outputs pipeline.egg-info build dist .venv-*
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.egg-info" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.log" -delete
	rm -rf debug dvclive .dvc/tmp mlruns .venv

test: install-dev
	$(VENV_DIR)/bin/pytest tests/

lint: install-dev
	$(VENV_DIR)/bin/black --check $(PYTHON_FILES)
	$(VENV_DIR)/bin/flake8 $(PYTHON_FILES)
	$(VENV_DIR)/bin/isort --check-only $(PYTHON_FILES)
	$(VENV_DIR)/bin/mypy pipeline

format: install-dev
	$(VENV_DIR)/bin/black $(PYTHON_FILES)
	$(VENV_DIR)/bin/isort $(PYTHON_FILES)

docs: install-dev
	$(MAKE) -C $(DOCS_DIR) html

run:
	cd pipeline && dvc repro --force --verbose

combine: install-dev
	$(PYTHON) $(SCRIPTS_DIR)/combine.py

verify-setup: install-dev
	$(PYTHON) $(SCRIPTS_DIR)/verify_setup.py

verify-paths: install-dev
	$(PYTHON) $(SCRIPTS_DIR)/verify_paths.py

check-dependencies: install-dev
	$(PYTHON) $(SCRIPTS_DIR)/check_dependencies.py

build:
	$(PYTHON) setup.py sdist bdist_wheel

dist: build

upload: dist
	$(VENV_DIR)/bin/twine upload dist/*

deploy: install-dev
	$(VENV_DIR)/bin/python pipeline/stages/deploy.py

serve: install
	$(UV) uvicorn pipeline.serve.app:app --host 0.0.0.0 --port 8000 --reload --python $(VENV_DIR)/bin/python

monitor: install
	$(VENV_DIR)/bin/python pipeline/monitoring/app.py

repro:
	cd pipeline && dvc repro --force