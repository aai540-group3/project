"""
.. module:: src.data.ingestion
   :synopsis: Ingest the raw dataset.

This script loads the specified dataset using the Hugging Face Datasets library, converts it to a
Pandas DataFrame, and saves it as a CSV file in the raw data directory.
"""

import logging
from pathlib import Path

import datasets
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Ingest the raw dataset.

    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises ValueError: If the dataset configuration is missing or incomplete.
    :raises Exception: If any other error occurs during data ingestion.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Validate configuration
        if not cfg.dataset or not cfg.dataset.name:
            raise ValueError("Dataset configuration is missing or incomplete")

        # Get dataset name and output path
        dataset_name = cfg.dataset.name
        data_output_path = Path(to_absolute_path(cfg.dataset.path.raw))

        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = datasets.load_dataset(dataset_name)
        df = dataset["train"].to_pandas()

        # Save raw data
        data_output_path.mkdir(parents=True, exist_ok=True)
        output_file = data_output_path / "data.csv"

        logger.info(f"Saving raw data to {output_file}")
        df.to_csv(output_file, index=False)
        logger.info("Data ingestion completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during data ingestion: {str(e)}")
        raise


if __name__ == "__main__":
    main()
