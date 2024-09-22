import logging
from pathlib import Path

import hydra
import pandas as pd
from autogluon.tabular import TabularPredictor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        data_paths = cfg.dataset.path
        input_dir = Path(to_absolute_path(f"{data_paths.processed}/autogluon"))
        model_output_dir = Path(to_absolute_path("models/autogluon"))
        model_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training model autogluon...")

        train_data = pd.read_csv(input_dir / "train_preprocessed.csv")

        predictor = TabularPredictor(
            label="readmitted",
            path=str(model_output_dir),
            problem_type="binary"
        )

        predictor.fit(
            train_data=train_data,
            time_limit=cfg.model.autogluon.params.time_limit,
            presets=cfg.model.autogluon.params.presets,
        )

        # Save model - AutoGluon saves it automatically, but we'll add a marker file
        (model_output_dir / "model.pkl").touch()

        logger.info(f"Model saved to {model_output_dir}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise

if __name__ == "__main__":
    main()