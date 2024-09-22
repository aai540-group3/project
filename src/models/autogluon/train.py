"""
.. module:: src.models.autogluon.train
   :synopsis: Train an AutoGluon TabularPredictor model.

This script trains an AutoGluon TabularPredictor model for binary classification, using the
parameters specified in the Hydra configuration. It includes features to prevent redundant training
by calculating a hash of the input data and configuration and comparing it to a stored hash. The
trained model and its input hash are saved for future use.

The script also uses DVCLive to log training parameters, metrics, and artifacts.
"""

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
    """
    Train an AutoGluon TabularPredictor model.

    :param cfg: Hydra configuration object containing data paths, model parameters, and other settings.
    :type cfg: DictConfig
    :raises Exception: If an error occurs during training.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        data_paths = cfg.dataset.path
        train_data_path = Path(
            to_absolute_path(f"{data_paths.processed}/train.csv")
        )
        model_output_dir = Path(to_absolute_path("models/autogluon"))
        model_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        # Calculate hash of input data and configuration
        input_hash = calculate_input_hash(train_data_path, cfg)
        hash_file = model_output_dir / "input_hash.txt"
        model_pkl_path = model_output_dir / "model.pkl"

        # Check if model file exists before checking hashes
        if model_pkl_path.exists():
            # Check if model already exists with the same input hash
            if hash_file.exists():
                with open(hash_file, "r") as f:
                    stored_hash = f.read().strip()

                if stored_hash == input_hash:
                    logger.info(
                        "Model already exists with the same input hash. Skipping training."
                    )
                    # Load the existing model
                    predictor = joblib.load(model_pkl_path)
                    logger.info(f"Loaded existing model from {model_pkl_path}")
                    return predictor  # Return the loaded predictor
            else:
                logger.info("Hash file doesn't exist. Proceeding to training.")
        else:
            logger.info("Model file doesn't exist. Proceeding to training.")


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
                "dataset_version": cfg.dataset.version,
            }
            # Add flattened AutoGluon parameters
            for key, value in cfg.model.params.items():
                params_to_log[f"autogluon_{key}"] = value

            live.log_params(params_to_log)

            # Initialize and train the AutoGluon TabularPredictor
            predictor = TabularPredictor(
                label="readmitted",
                path=str(model_output_dir),
                problem_type="binary",
            )

            # Prepare hyperparameters
            hyperparameters = {
                "GBM": {"num_boost_round": cfg.model.params.gbm_num_boost_round}
            }

            predictor.fit(
                train_data=train_data,
                time_limit=cfg.model.params.time_limit,
                presets=cfg.model.params.presets,
                hyperparameters=hyperparameters,
                verbosity=cfg.model.params.verbosity,
            )
            logger.info("AutoGluon model training completed.")

            try:
                # Save a reference to the predictor
                joblib.dump(predictor, model_pkl_path)
                logger.info(
                    f"AutoGluon predictor reference saved to {model_pkl_path}"
                )

                # Log the model.pkl as an artifact
                live.log_artifact(
                    str(model_pkl_path), type="model", name="autogluon_model"
                )

                # Only save the hash if model saving was successful
                with open(hash_file, "w") as f:
                    f.write(input_hash)
                logger.info(f"Input hash saved to {hash_file}")

            except Exception as e:
                logger.error(f"Error saving model or hash: {e}")
                # Optionally, delete the hash file if model saving failed
                if hash_file.exists():
                    hash_file.unlink()
                raise

        return predictor  # Return the trained predictor

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


def calculate_input_hash(data_path: Path, cfg: DictConfig) -> str:
    """
    Calculate a combined hash of the input data and configuration.

    :param data_path: Path to the training data file.
    :type data_path: Path
    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :return: MD5 hash of the combined input data and configuration.
    :rtype: str
    """
    with open(data_path, "rb") as f:
        data_hash = hashlib.md5(f.read()).hexdigest()

    config_str = OmegaConf.to_yaml(cfg)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    combined_hash = hashlib.md5(
        (data_hash + config_hash).encode()
    ).hexdigest()
    return combined_hash


if __name__ == "__main__":
    main()