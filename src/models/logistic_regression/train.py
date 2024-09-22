import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        data_paths = cfg.dataset.path
        input_dir = Path(to_absolute_path(f"{data_paths.processed}/{cfg.model.name}"))
        model_output_dir = Path(to_absolute_path(f"models/{cfg.model.name}"))
        model_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training model {cfg.model.name}...")

        # Load preprocessed data
        train_data = pd.read_csv(input_dir / "train_preprocessed.csv")
        X_train = train_data.drop(columns=["readmitted"])
        y_train = train_data["readmitted"]

        model = LogisticRegression(**cfg.model.logistic_regression.params)
        model.fit(X_train, y_train)

        model_output_path = model_output_dir / "model.pkl"
        joblib.dump(model, model_output_path)

        logger.info(f"Model saved to {model_output_path}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise

if __name__ == "__main__":
    main()