#!/usr/bin/env python3
"""
.. module:: scripts.update-nvidia-gpu
   :synopsis: Update NVIDIA GPU accelerator type detection in Ray for UTF-16BE encoded device names.

This script updates the NVIDIA GPU accelerator type detection in the Ray library to handle UTF-16BE encoded device names.
It locates the nvidia_gpu.py file within the project's virtual environment and modifies the
get_current_node_accelerator_type method to include a fallback decoding mechanism.

Usage:
    Run the script to automatically locate and update the nvidia_gpu.py file in the project's virtual environment.

Requirements:
    - Python 3.x
    - A virtual environment named '.venv' in the current or parent directories

Note:
    Ensure that you have the necessary permissions to modify files in the virtual environment.
    This script should be run from within the project directory or a subdirectory thereof.
"""

import re
import sys
from pathlib import Path


def find_nvidia_gpu_py():
    """Locate the nvidia_gpu.py file within the project's virtual environment.

    This function searches for the .venv directory starting from the
    current working directory and moving up through parent directories.
    Once found, it constructs the path to the nvidia_gpu.py file within
    the site-packages directory.

    :return: Path to the nvidia_gpu.py file if found, None otherwise
    :rtype: Path or None
    """
    # Start from the current working directory
    current_dir = Path.cwd()

    # Search for .venv directory
    venv_dir = None
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / ".venv").exists():
            venv_dir = parent / ".venv"
            break

    if not venv_dir:
        print("Could not find .venv directory.")
        return None

    # Find the site-packages directory
    site_packages = None
    for path in venv_dir.rglob("site-packages"):
        if path.is_dir():
            site_packages = path
            break

    if not site_packages:
        print("Could not find site-packages directory.")
        return None

    # Find the nvidia_gpu.py file
    nvidia_gpu_py = (
        site_packages / "ray" / "_private" / "accelerators" / "nvidia_gpu.py"
    )

    if not nvidia_gpu_py.exists():
        print(f"Could not find nvidia_gpu.py in {nvidia_gpu_py}")
        return None

    return nvidia_gpu_py


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

    new_content = """    @staticmethod
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
    """Main execution block of the script.

    This block locates the nvidia_gpu.py file and updates its content if
    found. If the file cannot be located, it exits with an error
    message.
    """
    nvidia_gpu_py = find_nvidia_gpu_py()
    if nvidia_gpu_py:
        update_nvidia_gpu_py(nvidia_gpu_py)
    else:
        print(
            "Failed to locate nvidia_gpu.py. Please check your virtual environment setup."
        )
        sys.exit(1)
