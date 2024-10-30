from setuptools import find_packages, setup

setup(
    name="pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "omegaconf",
        "boto3",
        "mlflow",
    ],
)
