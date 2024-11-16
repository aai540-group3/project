import subprocess
from pathlib import Path
from omegaconf import OmegaConf


def run_pipeline():
    # Load config to get paths
    config = OmegaConf.load("conf/config.yaml")

    # Resolve runtime directory
    runtime_cwd = Path.cwd()
    config.paths.root = str(runtime_cwd)

    # Run DVC pipeline
    subprocess.run(["dvc", "repro"], check=True)


if __name__ == "__main__":
    run_pipeline()
