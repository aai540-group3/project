#!/usr/bin/env python3
"""
src.utils

This utility script patches the `nvidia_gpu.py` file in the Ray installation to ensure compatibility
with specific NVIDIA GPU configurations. It searches for the target file within Python's site-packages
and replaces it with a patched version if found.

Attributes:
    TARGET_FILE (str): Relative path to the target file in the Ray installation.
    PATCH_FILE (str): Path to the local patch file to be used.

Example:
    To execute the patch, run:

        $ python utils.py
"""

import logging
import site
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_FILE = "ray/_private/accelerators/nvidia_gpu.py"
PATCH_FILE = "src/nvidia_gpu.py"


def patch_nvidia_gpu():
    """Patches the `nvidia_gpu.py` file in Ray installation with a custom patch.

    This function searches Python's site-packages for the specified `TARGET_FILE`.
    If the target file is found, it replaces the file's contents with the patch from `PATCH_FILE`.

    Logs the success or failure of each step, including:
        - Paths searched
        - Success of the patch process
        - Any errors encountered

    Raises:
        FileNotFoundError: If the patch file `PATCH_FILE` is not found.
        OSError: If there is an issue opening or writing to the target file.
    """
    logger.info("Searching for target file: %s in Python site-packages", TARGET_FILE)

    for site_package_path in site.getsitepackages():
        nvidia_gpu_path = Path(site_package_path) / TARGET_FILE
        if nvidia_gpu_path.exists():
            logger.info("Target file found at: %s", nvidia_gpu_path.parent)

            try:
                with open(PATCH_FILE, "r") as patch_file:
                    patch_content = patch_file.read()

                with open(nvidia_gpu_path, "w") as target_file:
                    target_file.write(patch_content)

                logger.info("Successfully patched: %s", nvidia_gpu_path)
                return

            except (FileNotFoundError, OSError) as e:
                logger.error("Error patching %s: %s", nvidia_gpu_path, e)
                return

    logger.warning("Target file not found: %s", TARGET_FILE)


def main():
    """Main function to execute the patching process."""
    patch_nvidia_gpu()


if __name__ == "__main__":
    main()
