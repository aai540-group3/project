# data/ingestion.py

"""
.. module:: data.ingestion
   :synopsis: Ingest the raw dataset.

This script loads the specified dataset using the Hugging Face Datasets library, converts it to a
Pandas DataFrame, and saves it as a CSV file in the raw data directory.
"""

import logging
import os
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(os.getenv("CONFIG_PATH"))

@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    """
    Ingest the raw dataset.

    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises ValueError: If the dataset configuration is missing or incomplete.
    :raises Exception: If any other error occurs during data ingestion.
    """
    try:
        logger.info("Configuration loaded:")
        logger.info(OmegaConf.to_yaml(cfg))

        logger.info("Ingesting raw dataset...")

        dataset_name = cfg.dataset.name
        output_file = Path(cfg.paths.data.raw.file)

        # Load dataset using Hugging Face Datasets library
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = datasets.load_dataset(dataset_name)
        df = dataset["train"].to_pandas()

        # Ensure the output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the raw data as a CSV file
        logger.info(f"Saving raw data to {output_file}")
        df.to_csv(output_file, index=False)

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during data ingestion: {str(e)}")
        raise


if __name__ == "__main__":
    main()
