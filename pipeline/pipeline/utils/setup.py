import logging
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """Setup development environment."""
    # Install dependencies
    install_dependencies()

    # Setup pre-commit hooks
    setup_pre_commit()

    # Initialize DVC
    initialize_dvc()

    # Setup MLflow
    setup_mlflow()

    logger.info("Environment setup completed")


def install_dependencies() -> None:
    """Install project dependencies."""
    requirements_files = [
        "requirements.txt",
        "requirements-dev.txt",
    ]

    for req_file in requirements_files:
        if Path(req_file).exists():
            subprocess.run(["pip", "install", "-r", req_file], check=True)
            logger.info(f"Installed dependencies from {req_file}")


def setup_pre_commit() -> None:
    """Setup pre-commit hooks."""
    subprocess.run(["pre-commit", "install"], check=True)
    logger.info("Installed pre-commit hooks")


def initialize_dvc() -> None:
    """Initialize DVC and remote storage."""
    # Initialize DVC
    subprocess.run(["dvc", "init"], check=True)

    # Add remote storage
    subprocess.run(
        ["dvc", "remote", "add", "-d", "storage", "s3://your-bucket/dvc"], check=True
    )

    logger.info("Initialized DVC")


def setup_mlflow() -> None:
    """Setup MLflow tracking."""
    mlflow_dir = Path("mlruns")
    mlflow_dir.mkdir(exist_ok=True)
    logger.info("Setup MLflow tracking")


if __name__ == "__main__":
    setup_environment()
