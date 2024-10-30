import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

VENV_CONFIGS = {
    "autogluon": PROJECT_ROOT / "requirements" / "autogluon.txt",
    "explore": PROJECT_ROOT / "requirements" / "explore.txt",
    "featurize": PROJECT_ROOT / "requirements" / "featurize.txt",
    "infrastruct": PROJECT_ROOT / "requirements" / "infrastruct.txt",
    "logistic": PROJECT_ROOT / "requirements" / "logistic.txt",
    "neural": PROJECT_ROOT / "requirements" / "neural.txt",
    "preprocess": PROJECT_ROOT / "requirements" / "preprocess.txt",
    "ingest": PROJECT_ROOT / "requirements" / "ingest.txt",
    "deploy": PROJECT_ROOT / "requirements" / "deploy.txt",
    "optimize": PROJECT_ROOT / "requirements" / "optimize.txt",
    "evaluate": PROJECT_ROOT / "requirements" / "evaluate.txt",
    "register": PROJECT_ROOT / "requirements" / "register.txt",
    "monitoring": PROJECT_ROOT / "requirements" / "monitoring.txt",
    "serve": PROJECT_ROOT / "requirements" / "serve.txt",
}


def create_venv(name: str, requirements_file: Path) -> None:
    """Create virtual environment and install requirements using uv.

    Args:
        name: Name of the virtual environment
        requirements_file: Path to requirements file
    """
    venv_path = PROJECT_ROOT / f".venv-{name}"

    print(f"Creating virtual environment at: {venv_path}")
    print(f"Using requirements from: {requirements_file}")

    # Create venv using uv
    subprocess.run(["uv", "venv", str(venv_path)], check=True)

    # Install requirements using uv
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--requirement",
            str(requirements_file),
            "--python",
            str(venv_path / "bin" / "python"),
        ],
        check=True,
    )

    # Verify the installation
    python_path = venv_path / "bin" / "python"
    if not python_path.exists():
        raise RuntimeError(f"Python interpreter not found at {python_path}")

    print(f"Created virtual environment for {name}")
    print(f"Python interpreter: {python_path}")


def delete_venv(name: str) -> None:
    """Delete virtual environment.

    Args:
        name: Name of the virtual environment
    """
    venv_path = PROJECT_ROOT / f".venv-{name}"
    if venv_path.exists():
        subprocess.run(["rm", "-rf", str(venv_path)], check=True)
        print(f"Deleted virtual environment: {venv_path}")


def list_venvs() -> List[str]:
    """List existing virtual environments.

    Returns:
        List of virtual environment names
    """
    return [
        path.name[5:]  # Remove '.venv-' prefix
        for path in PROJECT_ROOT.glob(".venv-*")
        if path.is_dir()
    ]


def check_venv(name: str) -> bool:
    """Check if virtual environment exists and has all requirements installed.

    Args:
        name: Name of the virtual environment

    Returns:
        True if virtual environment is properly set up
    """
    venv_path = PROJECT_ROOT / f".venv-{name}"
    if not venv_path.exists():
        return False

    python_path = venv_path / "bin" / "python"
    if not python_path.exists():
        return False

    # Check if requirements are installed using uv
    try:
        result = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(python_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        installed = set(result.stdout.splitlines())

        # Read requirements
        with open(VENV_CONFIGS[name]) as f:
            required = set(
                line.strip() for line in f if line.strip() and not line.startswith("#")
            )

        return all(
            any(req.startswith(r.split("==")[0]) for r in installed) for req in required
        )
    except subprocess.CalledProcessError:
        return False


def verify_environment() -> None:
    """Verify the execution environment."""
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")

    # Check if requirements directory exists
    requirements_dir = PROJECT_ROOT / "requirements"
    if not requirements_dir.exists():
        raise RuntimeError(f"Requirements directory not found: {requirements_dir}")

    # Verify all requirement files exist
    for name, req_file in VENV_CONFIGS.items():
        if not req_file.exists():
            raise RuntimeError(f"Requirements file not found: {req_file}")

    print("Environment verification completed successfully")


def main():
    # Verify environment
    verify_environment()

    # Ensure uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

    parser = argparse.ArgumentParser(description="Manage virtual environments")
    parser.add_argument(
        "action",
        choices=["create", "delete", "list", "check"],
        help="Action to perform",
    )
    parser.add_argument(
        "name",
        nargs="?",
        choices=list(VENV_CONFIGS.keys()) + ["all"],
        help="Virtual environment name",
    )

    args = parser.parse_args()

    if args.action == "list":
        existing = list_venvs()
        if existing:
            print("\nExisting virtual environments:")
            for venv in existing:
                venv_path = PROJECT_ROOT / f".venv-{venv}"
                python_path = venv_path / "bin" / "python"
                status = "✓" if python_path.exists() else "✗"
                print(f"  {status} {venv} ({venv_path})")
        else:
            print("No virtual environments found")
        return

    if args.name is None and args.action != "list":
        parser.error("name is required for this action")

    if args.name == "all":
        names = list(VENV_CONFIGS.keys())
    else:
        names = [args.name]

    for name in names:
        try:
            if args.action == "create":
                create_venv(name, VENV_CONFIGS[name])
            elif args.action == "delete":
                delete_venv(name)
            elif args.action == "check":
                status = "OK" if check_venv(name) else "Missing requirements"
                venv_path = PROJECT_ROOT / f".venv-{name}"
                print(f"{name}: {status} ({venv_path})")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {name}: {e}")
            if args.name != "all":
                raise


if __name__ == "__main__":
    main()
