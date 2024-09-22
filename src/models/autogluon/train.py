import hashlib
import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from autogluon.tabular import TabularPredictor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dvclive import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        data_paths = cfg.dataset.path
        train_data_path = Path(to_absolute_path(f"{data_paths.processed}/train.csv"))
        model_output_dir = Path(to_absolute_path("models/autogluon"))

        # Calculate hash of input data and configuration
        input_hash = calculate_input_hash(train_data_path, cfg)
        hash_file = model_output_dir / "input_hash.txt"

        if model_output_dir.exists() and hash_file.exists():
            with open(hash_file, "r") as f:
                stored_hash = f.read().strip()

            if stored_hash == input_hash:
                logger.info("Model already exists with the same input hash. Skipping training.")
                return

        logger.info("Training new AutoGluon TabularPredictor model...")

        # Load training data
        train_data = pd.read_csv(train_data_path)

        # Initialize DVCLive
        with Live() as live:
            # Log training parameters
            params_to_log = {
                "model": "autogluon",
                "label": "readmitted",
                "problem_type": "binary",
                "dataset_version": cfg.dataset.version
            }
            # Add flattened AutoGluon parameters
            for key, value in cfg.model.params.items():
                params_to_log[f"autogluon_{key}"] = value

            live.log_params(params_to_log)

            # Initialize and train the AutoGluon TabularPredictor
            predictor = TabularPredictor(
                label="readmitted",
                path=str(model_output_dir),
                problem_type="binary"
            )

            # Prepare hyperparameters
            hyperparameters = {
                'GBM': {'num_boost_round': cfg.model.params.gbm_num_boost_round}
            }

            predictor.fit(
                train_data=train_data,
                time_limit=cfg.model.params.time_limit,
                presets=cfg.model.params.presets,
                hyperparameters=hyperparameters,
                verbosity=cfg.model.params.verbosity
            )
            logger.info("AutoGluon model training completed.")

            # Save a reference to the predictor
            model_pkl_path = model_output_dir / "model.pkl"
            joblib.dump(predictor, model_pkl_path)
            logger.info(f"AutoGluon predictor reference saved to {model_pkl_path}")

            # Log the model.pkl as an artifact
            live.log_artifact(str(model_pkl_path), type="model", name="autogluon_model")

        # Save the input hash
        with open(hash_file, "w") as f:
            f.write(input_hash)
        logger.info(f"Input hash saved to {hash_file}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise

def calculate_input_hash(data_path: Path, cfg: DictConfig) -> str:
    """Calculate a combined hash of the input data and configuration."""
    with open(data_path, "rb") as f:
        data_hash = hashlib.md5(f.read()).hexdigest()

    config_str = OmegaConf.to_yaml(cfg)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    combined_hash = hashlib.md5((data_hash + config_hash).encode()).hexdigest()
    return combined_hash

if __name__ == "__main__":
    main()