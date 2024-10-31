from setuptools import find_packages, setup

setup(
    name="pipeline",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "dvc[all]",
        "uv",
        "virtualenv",
    ],
)
