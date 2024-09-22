import logging
from pathlib import Path

import datasets
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.data.path
        dataset_name = cfg.data.dataset_name
        data_output_path = Path(to_absolute_path(data_paths.raw))

        logger.info("Loading dataset...")
        dataset = datasets.load_dataset(dataset_name, token=None)
        df = dataset["train"].to_pandas()

        data_output_path.mkdir(parents=True, exist_ok=True)
        output_file = data_output_path / "data.csv"

        logger.info("Saving raw data...")
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()