# Diabetes Readmission Prediction System

## Overview

Machine learning system for predicting hospital readmissions for diabetic patients.

## Quick Start

### Prerequisites

- Python 3.10+
- AWS Account
- Docker

### Installation

```bash
# Clone repository
git clone https://github.com/aai540-group3/diabetes-readmission.git
cd diabetes-readmission

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.template .env
# Edit .env with your credentials
```

### Running the Pipeline

```bash
# Run complete pipeline
dvc repro

# Run specific stages
dvc repro prepare
dvc repro train
```

### Project Structure

```bash
pipeline/
├── conf/          # Configuration files
├── data/          # Data files
├── models/        # Model implementations
├── src/           # Source code
├── tests/         # Test files
└── docs/          # Documentation
```

## Development Guide

See `CONTRIBUTING.md` for development guidelines.

```markdown
# docs/CONTRIBUTING.md
# Contributing Guide

## Development Setup

1. Fork the repository

1. Clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/diabetes-readmission.git
   ```

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Use Black for formatting
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings in Google format

## Testing

- Write tests for new features
- Maintain test coverage above 80%
- Run tests before committing:

  ```bash
  pytest tests/
  ```

## Pull Request Process

- Update documentation
- Add tests
- Run linters:

  ```bash
  black .
  flake8 .
  mypy src/
  ```

- Create PR against develop branch

## Release Process

- Update version in pyproject.toml
- Update CHANGELOG.md
- Create release PR
- Tag release after merge
