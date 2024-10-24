#!/usr/bin/env python3
import logging
from pathlib import Path
import site

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_FILE = "ray/_private/accelerators/nvidia_gpu.py"
PATCH_FILE = "src/nvidia_gpu.py"


def patch_nvidia_gpu():
    """Patches the nvidia_gpu.py file in the Ray installation."""
    logger.info("Searching for: %s in Python site-packages", TARGET_FILE)

    for site_package_path in site.getsitepackages():
        nvidia_gpu_path = Path(site_package_path) / TARGET_FILE
        if nvidia_gpu_path.exists():
            logger.info("Found: %s", nvidia_gpu_path.parent)

            try:
                with open(PATCH_FILE, "r") as patch_file:
                    patch_content = patch_file.read()

                with open(nvidia_gpu_path, "w") as target_file:
                    target_file.write(patch_content)

                logger.info("Patched: %s", nvidia_gpu_path)
                return

            except (FileNotFoundError, OSError) as e:
                logger.error("Error patching %s: %s", nvidia_gpu_path, e)
                return

    logger.warning("Target file not found: %s", TARGET_FILE)


def main():
    patch_nvidia_gpu()


if __name__ == "__main__":
    main()
