################################################################################
# GLOBALS                                                                      #
################################################################################

PROJECT      := pipeline
VENV_DIR     := .venv
VENV_PYTHON  := $(VENV_DIR)/bin/python

################################################################################
# PHONY TARGETS                                                                #
################################################################################

.PHONY: all run help

################################################################################
# DEFAULT TARGET                                                               #
################################################################################

.DEFAULT_GOAL := run

################################################################################
# COMMANDS                                                                     #
################################################################################

all: run

## run: Setup virtual environment and run the pipeline
run:
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install dvc
	@echo "Running pipeline..."
	@cd $(PROJECT) && $(VENV_PYTHON) -m dvc repro -f
	@echo "Pipeline complete."

################################################################################
# Help                                                                         #
################################################################################

## help: Display available Make commands
help:
	@echo "Available targets:"
	@echo "  run         - Setup virtual environment and run the DVC pipeline (default)."
	@echo "  help        - Displays this help message."
