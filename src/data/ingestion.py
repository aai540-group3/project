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
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        if not cfg.dataset or not cfg.dataset.name:
            raise ValueError("Dataset configuration is missing or incomplete")

        dataset_name = cfg.dataset.name
        data_output_path = Path(to_absolute_path(cfg.dataset.path.raw))

        logger.info(f"Loading dataset: {dataset_name}")
        dataset = datasets.load_dataset(dataset_name)
        df = dataset["train"].to_pandas()

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
