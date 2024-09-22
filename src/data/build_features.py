import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def identify_features(df: pd.DataFrame):
    """Identifies categorical and numerical features in a DataFrame."""
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return categorical_cols, numerical_cols


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing data."""

    logger.info("Starting feature engineering process...")

    # Create total procedures feature
    df["total_procedures"] = df["num_lab_procedures"] + df["num_procedures"]

    # Create total number of visits feature
    df["total_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # Create average procedures per visit feature
    df["avg_procedures_per_visit"] = df["total_procedures"] / df["total_visits"]
    df["avg_procedures_per_visit"] = df["avg_procedures_per_visit"].fillna(0)

    # Create feature for ratio of lab procedures to total procedures
    df["lab_procedure_ratio"] = df["num_lab_procedures"] / df["total_procedures"]
    df["lab_procedure_ratio"] = df["lab_procedure_ratio"].fillna(0)

    # Create feature for medication intensity
    df["medication_intensity"] = df["num_medications"] / df["time_in_hospital"]

    # Log the new features created
    new_features = [
        "total_procedures",
        "total_visits",
        "avg_procedures_per_visit",
        "lab_procedure_ratio",
        "medication_intensity",
    ]
    logger.info(f"New features created: {', '.join(new_features)}")

    return df


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        interim_data_path = Path(to_absolute_path(cfg.dataset.path.interim)) / "data.csv"
        processed_data_path = Path(to_absolute_path(cfg.dataset.path.processed)) / "data_featured.csv"

        logger.info("Loading interim data...")
        df = pd.read_csv(interim_data_path)

        logger.info("Performing feature engineering...")
        df_featured = create_features(df)

        logger.info("Saving processed data with new features...")
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df_featured.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")

        # Log some data quality metrics
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Processed data shape: {df_featured.shape}")
        logger.info(
            f"New columns added: {', '.join(set(df_featured.columns) - set(df.columns))}"
        )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
