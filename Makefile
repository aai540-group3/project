# Define variables for directories and file patterns
CLEAN_TARGETS := build debug __pycache__ *.pyc *.pyo *.log combined.txt

# Default target (runs when you just type 'make')
all:

# Clean target (removes build artifacts, and compiled Python files)
clean:
	@echo "Cleaning build artifacts, and compiled Python files..."
	rm -rf $(CLEAN_TARGETS)
	find . -name "__pycache__" -type d -exec rm -rf {} +  # Remove __pycache__ directories recursively

# Phony targets (not real files, used for dependencies)
.PHONY: all clean