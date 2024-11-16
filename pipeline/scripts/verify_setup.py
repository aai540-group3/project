#!/usr/bin/env python3
import subprocess
from pathlib import Path


def verify_project_structure() -> None:
    """Verify project directory structure."""
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

    required_dirs = [
        "conf",
        "data",
        "deploy",
        "docs",
        "infrastructure",
        "metrics",
        "models",
        "plots",
        "registry",
        "requirements",
        "scripts",
        "serve",
        "src",
        "tests",
    ]

    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)


def verify_virtual_envs() -> None:
    """Verify virtual environments."""
    script_path = Path(__file__).parent / "manage_venvs.py"
    subprocess.run(["python", str(script_path), "check", "all"], check=True)


def main():
    print("Verifying project setup...")
    verify_project_structure()
    verify_virtual_envs()
    print("Project setup verification completed")


if __name__ == "__main__":
    main()
