# Makefile
.PHONY: all setup clean venv test lint

# Load environment variables
include .env
export

# Variables
PYTHON := python3
VENV_DIR := .venv
UV := uv

all: setup

setup: venv
	$(UV) pip install -e .

venv:
	$(UV) venv $(VENV_DIR)
	$(UV) pip install --upgrade --requirement requirements/base.txt --python $(VENV_DIR)/bin/python

setup-autogluon:
	$(UV) venv .venv-autogluon
	$(UV) pip install --requirement requirements/autogluon.txt --python .venv-autogluon/bin/python

setup-explore:
	$(UV) venv .venv-explore
	$(UV) pip install --requirement requirements/explore.txt --python .venv-explore/bin/python

setup-featurize:
	$(UV) venv .venv-featurize
	$(UV) pip install --requirement requirements/featurize.txt --python .venv-featurize/bin/python

setup-infrastruct:
	$(UV) venv .venv-infrastruct
	$(UV) pip install --requirement requirements/infrastruct.txt --python .venv-infrastruct/bin/python

setup-logistic:
	$(UV) venv .venv-logistic
	$(UV) pip install --requirement requirements/logistic.txt --python .venv-logistic/bin/python

setup-neural:
	$(UV) venv .venv-neural
	$(UV) pip install --requirement requirements/neural.txt --python .venv-neural/bin/python

setup-preprocess:
	$(UV) venv .venv-preprocess
	$(UV) pip install --requirement requirements/preprocess.txt --python .venv-preprocess/bin/python

setup-ingest:
	$(UV) venv .venv-ingest
	$(UV) pip install --requirement requirements/ingest.txt --python .venv-ingest/bin/python

setup-deploy:
	$(UV) venv .venv-deploy
	$(UV) pip install --requirement requirements/deploy.txt --python .venv-deploy/bin/python

setup-optimize:
	$(UV) venv .venv-optimize
	$(UV) pip install --requirement requirements/optimize.txt --python .venv-optimize/bin/python

setup-evaluate:
	$(UV) venv .venv-evaluate
	$(UV) pip install --requirement requirements/evaluate.txt --python .venv-evaluate/bin/python

setup-register:
	$(UV) venv .venv-register
	$(UV) pip install --requirement requirements/register.txt --python .venv-register/bin/python

setup-monitoring:
	$(UV) venv .venv-monitoring
	$(UV) pip install --requirement requirements/monitoring.txt --python .venv-monitoring/bin/python

setup-serve:
	$(UV) venv .venv-serve
	$(UV) pip install --requirement requirements/serve.txt --python .venv-serve/bin/python

setup-all: setup-autogluon setup-explore setup-featurize setup-infrastruct \
          setup-logistic setup-neural setup-preprocess \
          setup-ingest setup-deploy setup-optimize setup-evaluate \
          setup-register setup-monitoring setup-serve

clean:
	rm -rf $(VENV_DIR)
	rm -rf .venv-*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

test:
	$(VENV_DIR)/bin/pytest tests/

lint:
	$(VENV_DIR)/bin/black .
	$(VENV_DIR)/bin/flake8 .
	$(VENV_DIR)/bin/isort .
	$(VENV_DIR)/bin/mypy src/

dev-setup: venv
	$(UV) pip install -e ".[dev]" --python $(VENV_DIR)/bin/python
