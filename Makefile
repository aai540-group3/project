# Define variables for directories and file patterns
CLEAN_TARGETS := build debug **pycache** *.pyc *.pyo *.log outputs

# Default target (runs when you just type 'make')
all:

# Clean target (removes build artifacts and compiled Python files recursively)
clean:
	@echo "Cleaning build artifacts and compiled Python files recursively..."
	@find . -type d \( -name "build" -o -name "debug" -o -name "__pycache__" -o -name "outputs" \) -exec rm -rf {} +
	@find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.log" \) -delete

# Phony targets (not real files, used for dependencies)
.PHONY: all clean