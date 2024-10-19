import logging
import os
from pathlib import Path
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Split the dataset into training and testing sets.
    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises ValueError: If the dataset or training split configuration is missing or incomplete.
    :raises FileNotFoundError: If the interim data file is not found.
    :raises Exception: If any other error occurs during data splitting.
    """
    try:
        logger.info("Configuration loaded:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Validate configuration
        if "dataset" not in cfg:
            raise ValueError("Dataset configuration is missing")
        if "training" not in cfg or "split" not in cfg.training:
            raise ValueError("Training split configuration is missing or incomplete")

        split_params = cfg.training.split
        interim_data_path = Path(cfg.paths.data.interim.file)
        train_data_path = Path(cfg.paths.data.processed.train_file)
        test_data_path = Path(cfg.paths.data.processed.test_file)

        logger.info(f"Interim data path: {interim_data_path}")
        logger.info(f"Train data path: {train_data_path}")
        logger.info(f"Test data path: {test_data_path}")

        # Check if interim data file exists
        if not interim_data_path.exists():
            raise FileNotFoundError(f"Interim data file not found: {interim_data_path}")

        # Load data
        logger.info(f"Loading data from {interim_data_path}")
        df = pd.read_csv(interim_data_path)

        # Split data
        logger.info("Splitting data...")
        X = df.drop(columns=["readmitted"])
        y = df["readmitted"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_params.test_size,
            random_state=split_params.random_state,
        )

        # Ensure processed data directory exists
        train_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the train and test CSV files directly to their paths
        logger.info("Saving train and test sets...")
        pd.concat([X_train, y_train], axis=1).to_csv(train_data_path, index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(test_data_path, index=False)

        logger.info(f"Train data saved to {train_data_path}")
        logger.info(f"Test data saved to {test_data_path}")

    except Exception as e:
        logger.error(f"An error occurred during data splitting: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


if __name__ == "__main__":
    main()
