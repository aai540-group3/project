################################################################################
# GLOBALS                                                                      #
################################################################################

PROJECT      := project
PYTHON       := python3.10
PIP          := $(PYTHON) -m pip
AWS_BUCKET   := s3://mlops-artifacts-aai540-group3/data/
LOCAL_DATA   := data/
VENV_DIR     := .venv
CLEAN_TARGETS := build debug **pycache** *.pyc *.pyo *.log outputs

# Determine the platform to set the path to the virtual environment's Python and Pip
ifeq ($(OS),Windows_NT)
    VENV_PYTHON  := $(VENV_DIR)\Scripts\python.exe
    VENV_PIP     := $(VENV_DIR)\Scripts\pip.exe
    ACTIVATE_CMD := $(VENV_DIR)\Scripts\activate.bat
    RM_VENV      := rmdir /s /q $(VENV_DIR)
else
    VENV_PYTHON  := $(VENV_DIR)/bin/python
    VENV_PIP     := $(VENV_DIR)/bin/pip
    ACTIVATE_CMD := source $(VENV_DIR)/bin/activate
    RM_VENV      := rm -rf $(VENV_DIR)
endif

################################################################################
# PHONY TARGETS                                                                #
################################################################################

.PHONY: all install pipx uv aws clean lint fmt sync_down sync_up env help activate data

################################################################################
# DEFAULT TARGET                                                               #
################################################################################

.DEFAULT_GOAL := help

################################################################################
# COMMANDS                                                                     #
################################################################################

## all: Default target to display help
all: help

## install: Install all necessary components and set up the environment
install: pipx uv aws env
	@echo "----------------------------------------"
	@echo "Installation complete."
	@echo "To activate the virtual environment, run:"
	@echo "    make activate"
	@echo "----------------------------------------"

## pipx: Install pipx if not present
pipx:
	@command -v pipx >/dev/null || { \
		echo "pipx not found. Installing pipx..."; \
		$(PIP) install --user pipx; \
		echo "pipx installed. Ensure ~/.local/bin is in your PATH."; \
	}

## uv: Install uv using pipx
uv: pipx
	@command -v uv >/dev/null || { \
		echo "uv not found. Installing uv via pipx..."; \
		pipx install uv; \
		echo "uv installed."; \
	}

## aws: Install AWS CLI if not present
aws:
	@command -v aws >/dev/null || { \
		echo "AWS CLI not found. Installing AWS CLI..."; \
		$(PIP) install --user awscli; \
		echo "AWS CLI installed. Ensure ~/.local/bin is in your PATH."; \
	}

## clean: Remove the virtual environment, compiled Python files, and build artifacts
clean:
	@echo "Cleaning up..."
	$(RM_VENV)
	@echo "Cleaning build artifacts and compiled Python files recursively..."
	@find . -type d \( -name "build" -o -name "debug" -o -name "__pycache__" -o -name "outputs" \) -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.log" \) -delete
	@echo "Cleanup complete."

## lint: Lint the project using Ruff
lint:
	@echo "Linting the project..."
	ruff check $(PROJECT)
	@echo "Linting complete."

## fmt: Format the project using Ruff
fmt:
	@echo "Formatting the project..."
	ruff fix $(PROJECT)
	@echo "Formatting complete."

## sync_down: Download data from S3
sync_down: aws
	@echo "Syncing data down from S3..."
	aws s3 sync $(AWS_BUCKET) $(LOCAL_DATA)
	@echo "Data synced down."

## sync_up: Upload data to S3
sync_up: aws
	@echo "Syncing data up to S3..."
	aws s3 sync $(LOCAL_DATA) $(AWS_BUCKET)
	@echo "Data synced up."

## env: Set up the Python virtual environment using `uv` and install dependencies
env: uv
	@echo "----------------------------------------"
	@echo "Creating virtual environment using uv..."
	@echo "----------------------------------------"
ifeq ($(OS),Windows_NT)
	@uv venv $(VENV_DIR) --python=$(PYTHON) --prompt "$(PROJECT)" --allow-existing
else
	@uv venv $(VENV_DIR) --python=$(PYTHON) --prompt "$(PROJECT)" --allow-existing
endif
	@echo "Virtual environment '$(VENV_DIR)' created."

	@echo "----------------------------------------"
	@echo "Activating virtual environment and installing dependencies..."
	@echo "----------------------------------------"
ifeq ($(OS),Windows_NT)
	@call $(ACTIVATE_CMD) && uv pip install -r requirements.txt
else
	@bash -c "source $(VENV_DIR)/bin/activate && uv pip install -r requirements.txt"
endif
	@echo "Dependencies installed."

	@echo "----------------------------------------"
	@echo "Virtual environment '$(VENV_DIR)' is ready."
	@echo "----------------------------------------"

	# Output the activation command
ifeq ($(OS),Windows_NT)
	@echo "call $(ACTIVATE_CMD)"
else
	@echo "source $(VENV_DIR)/bin/activate"
endif

## activate: Display activation instructions
activate:
	@echo "To activate the virtual environment, run:"
ifeq ($(OS),Windows_NT)
	@echo "    call $(ACTIVATE_CMD)"
else
	@echo "    $(ACTIVATE_CMD)"
endif

## data: Generate dataset
data: env
	@echo "Generating dataset..."
	uv run python $(PROJECT)/dataset.py
	@echo "Dataset generation complete."

################################################################################
# Help                                                                         #
################################################################################

## help: Display available make commands
help:
	@echo "Available targets:"
	@echo "  all           - Default target. Displays this help message."
	@echo "  install       - Installs dependencies and virtual environment."
	@echo "  env           - Creates the virtual environment and installs dependencies."
	@echo "  pipx          - Installs pipx if not present."
	@echo "  uv            - Installs uv using pipx."
	@echo "  aws           - Installs AWS CLI if not present."
	@echo "  clean         - Removes the virtual environment and cleans Python caches."
	@echo "  lint          - Lints the project using Ruff."
	@echo "  fmt           - Formats the project using Ruff."
	@echo "  sync_down     - Downloads data from AWS S3."
	@echo "  sync_up       - Uploads data to AWS S3."
	@echo "  activate      - Displays activation instructions for the virtual environment."
	@echo "  data          - Generates the dataset."
	@echo "  help          - Displays this help message."