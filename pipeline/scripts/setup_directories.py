#!/usr/bin/env python3
from pathlib import Path

# Define directory structure
DIRECTORIES = [
    "conf/data",
    "conf/deploy",
    "conf/experiment",
    "conf/explore",
    "conf/features",
    "conf/infrastructure",
    "conf/pipeline",
    "conf/tracking",
    "data/external",
    "data/features",
    "data/interim",
    "data/processed",
    "data/raw",
    "infrastructure",
    "metrics/data",
    "metrics/evaluation",
    "metrics/explore",
    "metrics/features",
    "metrics/infrastructure",
    "metrics/optimization/autogluon",
    "metrics/optimization/logistic",
    "metrics/optimization/neural",
    "metrics/training/autogluon",
    "metrics/training/logistic",
    "metrics/training/neural",
    "models/autogluon/artifacts/model",
    "models/autogluon/artifacts/metrics",
    "models/autogluon/artifacts/plots",
    "models/logistic/artifacts/model",
    "models/logistic/artifacts/metrics",
    "models/logistic/artifacts/plots",
    "models/neural/artifacts/model",
    "models/neural/artifacts/metrics",
    "models/neural/artifacts/plots",
    "plots/evaluation",
    "plots/explore",
    "plots/features",
    "registry",
    "deploy/huggingface",
    "serve/logs",
]


def setup_directories():
    """Create project directory structure."""
    root = Path(__file__).parent.parent

    for directory in DIRECTORIES:
        dir_path = root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / ".gitkeep").touch()

    print("Directory structure created successfully!")


if __name__ == "__main__":
    setup_directories()
