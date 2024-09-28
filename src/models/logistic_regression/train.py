"""
.. module:: src.models.logistic_regression.train
   :synopsis: Train a Logistic Regression model.

This script trains a Logistic Regression model for binary classification. It utilizes Hydra for configuration
management and DVCLive for logging parameters, metrics, and artifacts. To prevent redundant training, the script
calculates a hash of the input data and configuration and compares it with a stored hash. If the hashes match,
training is skipped. The trained model is saved as a ``.pkl`` file, along with the input hash for future reference.
"""

import hashlib
import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

from dvclive import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Train a Logistic Regression model.

    :param cfg: Hydra configuration object containing data paths, model parameters, and other settings.
    :type cfg: DictConfig
    :raises Exception: If an error occurs during training.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        data_paths = cfg.dataset.path
        input_dir = Path(
            to_absolute_path(f"{data_paths.processed}/{cfg.model.name}")
        )
        model_output_dir = Path(
            to_absolute_path(f"models/{cfg.model.name}")
        )
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate hash of input data and configuration
        input_hash = calculate_input_hash(
            input_dir / "train_preprocessed.csv", cfg
        )
        hash_file = model_output_dir / "input_hash.txt"
        model_output_path = model_output_dir / "model.pkl"

        # Check if model file exists before checking hashes
        if model_output_path.exists():
            # Check if model already exists with the same input hash
            if hash_file.exists():
                with open(hash_file, "r") as f:
                    stored_hash = f.read().strip()

                if stored_hash == input_hash:
                    logger.info(
                        "Model already exists with the same input hash. Skipping training."
                    )
                    # Load the existing model
                    model = joblib.load(model_output_path)
                    logger.info(f"Loaded existing model from {model_output_path}")
                    return model  # Return the loaded model
            else:
                logger.info("Hash file doesn't exist. Proceeding to training.")
        else:
            logger.info("Model file doesn't exist. Proceeding to training.")

        logger.info(
            f"Training Logistic Regression model {cfg.model.name}..."
        )

        # Load preprocessed training data
        train_data = pd.read_csv(input_dir / "train_preprocessed.csv")
        X_train = train_data.drop(columns=["readmitted"])
        y_train = train_data["readmitted"]

        # Initialize DVCLive
        with Live() as live:
            try:
                # Log training parameters
                params_to_log = {
                    "model": cfg.model.name,
                    "label": "readmitted",
                    "problem_type": "binary",
                    "dataset_version": cfg.dataset.version,
                }
                # Flatten logistic regression parameters
                for key, value in cfg.model.logistic_regression.params.items():
                    params_to_log[f"logistic_regression_{key}"] = value

                live.log_params(params_to_log)

                # Initialize and train the Logistic Regression model
                model = LogisticRegression(**cfg.model.logistic_regression.params)
                model.fit(X_train, y_train)
                logger.info("Model training completed.")

                # Log the trained model as an artifact
                model_output_path = model_output_dir / "model.pkl"
                joblib.dump(model, model_output_path)
                live.log_artifact(
                    str(model_output_path),
                    type="model",
                    name=f"{cfg.model.name}_model",
                )
                logger.info(
                    f"Model saved and logged as artifact at {model_output_path}"
                )

            finally:
                # Always end the DVC Live run
                live.end()

        # Save the input hash to prevent redundant trainings
        with open(hash_file, "w") as f:
            f.write(input_hash)
        logger.info(f"Input hash saved to {hash_file}")

        return model  # Return the trained model

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


def calculate_input_hash(data_path: Path, cfg: DictConfig) -> str:
    """
    Calculate a combined hash of the input data and configuration.

    :param data_path: Path to the preprocessed training data file.
    :type data_path: Path
    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :return: MD5 hash of the combined input data and configuration.
    :rtype: str
    """
    with open(data_path, "rb") as f:
        data_hash = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()

    config_str = OmegaConf.to_yaml(cfg)
    config_hash = hashlib.md5(config_str.encode(), usedforsecurity=False).hexdigest()

    combined_hash = hashlib.md5(
        (data_hash + config_hash).encode(),
        usedforsecurity=False
    ).hexdigest()
    return combined_hash


if __name__ == "__main__":
    main()
