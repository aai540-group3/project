# pipeline/src/utils/clean.py
import logging
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def clean_artifacts() -> None:
    """Clean generated artifacts."""
    paths_to_clean = [
        "outputs",
        "mlruns",
        "models",
        "metrics",
        "plots",
        ".pytest_cache",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".Python",
        "*.so",
    ]

    for pattern in paths_to_clean:
        for path in Path(".").rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            logger.info(f"Removed: {path}")

    logger.info("Cleaned all artifacts")

def clean_data() -> None:
    """Clean data files."""
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)
        logger.info("Cleaned data directory")

def reset_dvc() -> None:
    """Reset DVC cache."""
    subprocess.run(["dvc", "gc", "-f"], check=True)
    logger.info("Reset DVC cache")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Clean everything")
    parser.add_argument("--artifacts", action="store_true", help="Clean artifacts")
    parser.add_argument("--data", action="store_true", help="Clean data")
    parser.add_argument("--dvc", action="store_true", help="Reset DVC cache")

    args = parser.parse_args()

    if args.all or args.artifacts:
        clean_artifacts()
    if args.all or args.data:
        clean_data()
    if args.all or args.dvc:
        reset_dvc()
