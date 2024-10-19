#!/usr/bin/env python3
"""
.. module:: scripts.fix-nvidia-gpu
   :synopsis: Update NVIDIA GPU accelerator type detection in Ray for UTF-16BE encoded device names.

This script fixes the NVIDIA GPU accelerator type detection in the Ray library to handle UTF-16BE encoded device names.
It locates the nvidia_gpu.py file within the project's environment and modifies the
get_current_node_accelerator_type method to include a fallback decoding mechanism.

Usage:
    Run the script with Hydra: `python fix-nvidia-gpu.py`

Requirements:
    - Python 3.x
    - Hydra

Note:
    Ensure that you have the necessary permissions to modify files in the environment.
"""

import logging
import os
import re
import site
import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base=None,
)
def main(cfg: DictConfig = None):
    """Main function to update the nvidia_gpu.py file."""

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    nvidia_gpu_py = find_nvidia_gpu_py()
    if nvidia_gpu_py:
        update_nvidia_gpu_py(nvidia_gpu_py)
    else:
        print("Failed to locate nvidia_gpu.py. Check your Python environment.")
        sys.exit(1)


def find_nvidia_gpu_py() -> Optional[Path]:
    """Locate the nvidia_gpu.py file within the active Python environment."""

    for path in site.getsitepackages():
        nvidia_gpu_py = (
            Path(path) / "ray" / "_private" / "accelerators" / "nvidia_gpu.py"
        )
        if nvidia_gpu_py.exists():
            print(f"Found nvidia_gpu.py in: {nvidia_gpu_py.parent}")
            return nvidia_gpu_py

    print("Could not find nvidia_gpu.py in the active Python environment.")
    return None


def update_nvidia_gpu_py(file_path):
    """Update the get_current_node_accelerator_type method in the nvidia_gpu.py
    file.

    This function reads the content of the nvidia_gpu.py file,
    identifies the get_current_node_accelerator_type method, and
    replaces it with an updated version that includes a fallback
    mechanism for decoding device names.

    :param file_path: Path to the nvidia_gpu.py file
    :type file_path: Path
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Define the patterns to match and replace
    old_pattern = re.compile(
        r"@staticmethod\s*\n"
        r"\s*def get_current_node_accelerator_type.*?"
        r"return cuda_device_type\s*\n",
        re.DOTALL,
    )

    new_content = """
    @staticmethod
    def get_current_node_accelerator_type() -> Optional[str]:
        import ray._private.thirdparty.pynvml as pynvml

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return None  # pynvml init failed
        device_count = pynvml.nvmlDeviceGetCount()
        cuda_device_type = None
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            device_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(device_name, bytes):
                try:
                    device_name = device_name.decode("utf-16be")
                except UnicodeDecodeError as e:
                    device_name = device_name.decode("utf-8")
            cuda_device_type = (
                NvidiaGPUAcceleratorManager._gpu_name_to_accelerator_type(device_name)
            )
        pynvml.nvmlShutdown()
        return cuda_device_type

"""

    # Replace the old content with the new content
    new_content = old_pattern.sub(new_content, content)

    # Write the updated content back to the file
    with open(file_path, "w") as file:
        file.write(new_content)

    print(f"Successfully updated {file_path}")


if __name__ == "__main__":
    main()
