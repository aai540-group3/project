"""
Test Configuration
===============

.. module:: tests.conftest
   :synopsis: Test fixtures and configuration

.. moduleauthor:: aai540-group3
"""

import os
from pathlib import Path
from typing import Dict, Generator

import pytest
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

@pytest.fixture(scope="session")
def test_config() -> Dict:
    """Load test configuration.

    :return: Test configuration
    :rtype: Dict
    """
    config_path = Path("conf/config.yaml")
    cfg = OmegaConf.load(config_path)

    # Override settings for testing
    cfg.experiment.name = "test"
    cfg.monitoring.enabled = False
    cfg.tracking.enabled = False

    return cfg

@pytest.fixture(scope="session")
def sample_data() -> pd.DataFrame:
    """Generate sample data for testing.

    :return: Sample DataFrame
    :rtype: pd.DataFrame
    """
    np.random.seed(42)
    n_samples = 1000

    data = {
        "age": np.random.normal(65, 10, n_samples),
        "time_in_hospital": np.random.poisson(5, n_samples),
        "num_procedures": np.random.poisson(3, n_samples),
        "num_medications": np.random.poisson(8, n_samples),
        "num_lab_procedures": np.random.poisson(10, n_samples),
        "number_diagnoses": np.random.poisson(7, n_samples),
        "gender": np.random.choice([0, 1], n_samples),
        "readmitted": np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }

    return pd.DataFrame(data)

@pytest.fixture(scope="function")
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary directory for tests.

    :param tmp_path: Temporary path
    :type tmp_path: Path
    :yield: Temporary directory path
    :rtype: Generator[Path, None, None]
    """
    yield tmp_path

@pytest.fixture(scope="session")
def mock_aws_credentials() -> None:
    """Mock AWS credentials for testing."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

@pytest.fixture(scope="session")
def mock_mlflow(tmp_path_factory) -> str:
    """Setup mock MLflow tracking.

    :param tmp_path_factory: Temporary path factory
    :return: MLflow tracking URI
    :rtype: str
    """
    tracking_uri = str(tmp_path_factory.mktemp("mlruns"))
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    return tracking_uri
