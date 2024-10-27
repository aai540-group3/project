################################################################################
# GLOBALS                                                                      #
################################################################################

PROJECT      := pipeline
VENV_DIR     := .venv
CLEAN_TARGETS := build debug **/*pycache__* *.pyc *.pyo *.log outputs reports/*/*

# Virtual environment paths and commands
ifeq ($(OS),Windows_NT)
    VENV_PYTHON  := $(VENV_DIR)\Scripts\python.exe
    RM_VENV      := rmdir /s /q $(VENV_DIR)
else
    VENV_PYTHON  := $(VENV_DIR)/bin/python
    RM_VENV      := rm -rf $(VENV_DIR)
endif

################################################################################
# PHONY TARGETS                                                                #
################################################################################

.PHONY: all clean run help

################################################################################
# DEFAULT TARGET                                                               #
################################################################################

.DEFAULT_GOAL := help  # Default target is 'help'

################################################################################
# COMMANDS                                                                     #
################################################################################

## all: Default target - display help
all: help

## clean: Remove the virtual environment, build artifacts, and temporary files
clean:
	@echo "Cleaning up..."
	$(RM_VENV)
	@echo "Cleaning build artifacts, temporary files, and reports..."
	@find . -type d \( -name "build" -o -name "debug" -o -name "__pycache__" -o -name "outputs" -o -name "reports" \) -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.log" \) -delete
	@echo "Cleanup complete."

## run: Run the entire DVC pipeline
run:
	@echo "Running the DVC pipeline..."
	@cd $(PROJECT) && $(VENV_PYTHON) -m dvc repro -f
	@echo "DVC pipeline complete."

################################################################################
# Help                                                                         #
################################################################################

## help: Display available Make commands
help:
	@echo "Available targets:"
	@echo "  clean       - Remove virtual environment, build artifacts, and temporary files."
	@echo "  run         - Run the entire DVC pipeline."
	@echo "  help        - Displays this help message."