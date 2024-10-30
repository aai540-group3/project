import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Generator

@pytest.fixture(scope="session")
def test_config() -> Dict:
    """Load test configuration."""
    from omegaconf import OmegaConf
    return OmegaConf.load("conf/config.yaml")

@pytest.fixture(scope="session")
def test_data() -> pd.DataFrame:
    """Generate synthetic test data."""
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
    """Create temporary directory."""
    yield tmp_path
