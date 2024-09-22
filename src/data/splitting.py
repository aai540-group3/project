import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.data.path
        split_params = cfg.training.split

        interim_data_path = Path(to_absolute_path(f"{data_paths.interim}/data.csv"))
        train_data_path = Path(to_absolute_path(f"{data_paths.processed}/train.csv"))
        test_data_path = Path(to_absolute_path(f"{data_paths.processed}/test.csv"))

        logger.info("Loading data...")
        df = pd.read_csv(interim_data_path)

        logger.info("Splitting data...")
        X = df.drop(columns=["readmitted"])
        y = df["readmitted"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_params.test_size, random_state=split_params.random_state
        )

        logger.info("Saving train and test sets...")
        train_data_path.parent.mkdir(parents=True, exist_ok=True)

        pd.concat([X_train, y_train], axis=1).to_csv(train_data_path, index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(test_data_path, index=False)

        logger.info(f"Train data saved to {train_data_path}")
        logger.info(f"Test data saved to {test_data_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()