from pathlib import Path
from omegaconf import OmegaConf
import os

def verify_paths():
    # Load config
    config = OmegaConf.load("conf/config.yaml")

    # Resolve runtime directory
    runtime_cwd = Path.cwd()
    config.paths.root = str(runtime_cwd)

    # Resolve all paths
    resolved_config = OmegaConf.to_container(config, resolve=True)

    # Print and verify all paths
    print("\nVerifying paths:")
    for key, value in resolved_config["paths"].items():
        path = Path(value)
        exists = path.exists()
        print(f"{key}: {path} {'✓' if exists else '✗'}")

        # Create directory if it doesn't exist
        if not exists and key not in ["venv"]:  # Skip venv as it's created by uv
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")

if __name__ == "__main__":
    verify_paths()
