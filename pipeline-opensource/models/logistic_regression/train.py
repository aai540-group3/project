"""
.. module:: models.logistic_regression.train
   :synopsis: Train a Logistic Regression model.

This script trains a Logistic Regression model for binary classification. It utilizes Hydra for configuration
management and DVCLive for logging parameters, metrics, and artifacts. To prevent redundant training, the script
calculates a hash of the input data and configuration and compares it with a stored hash. If the hashes match,
training is skipped. The trained model is saved as a `.pkl` file, along with the input hash for future reference.

All outputs are saved in the model's artifacts directory.
"""

import hashlib
import logging
import os
from pathlib import Path

import hydra
import joblib
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression

from dvclive import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base=None,
)
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

        # Use Hydra-managed paths
        artifacts_dir = Path(
            to_absolute_path(cfg.paths.models.logistic_regression.artifacts)
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        train_data_path = Path(
            to_absolute_path(cfg.paths.models.logistic_regression.preprocessed_data)
        )
        model_output_path = Path(
            to_absolute_path(cfg.paths.models.logistic_regression.model_file)
        )
        hash_file = Path(
            to_absolute_path(cfg.paths.models.logistic_regression.hash_file)
        )
        params_file = Path(
            to_absolute_path(cfg.paths.models.logistic_regression.params_file)
        )

        # Calculate hash of input data and configuration
        input_hash = calculate_input_hash(train_data_path, cfg)

        # Initialize DVCLive
        live = Live(dir=str(artifacts_dir), dvcyaml=False)

        try:
            # Log training parameters
            params_to_log = {
                "model": cfg.model.name,
                "label": "readmitted",
                "problem_type": "binary",
                "dataset_version": cfg.dataset.version,
            }
            for key, value in cfg.model.params.items():
                params_to_log[f"logistic_regression_{key}"] = value

            live.log_params(params_to_log)

            # Save parameters to YAML file
            with open(params_file, "w") as f:
                yaml.dump(params_to_log, f)

            # Check if model file exists before checking hashes
            if model_output_path.exists() and hash_file.exists():
                with open(hash_file, "r") as f:
                    stored_hash = f.read().strip()

                if stored_hash == input_hash:
                    logger.info(
                        "Model already exists with the same input hash. Skipping training."
                    )
                    model = joblib.load(model_output_path)
                    logger.info(f"Loaded existing model from {model_output_path}")

                    # Even if training is skipped, ensure artifacts are logged
                    live.log_artifact(
                        str(model_output_path),
                        type="model",
                        name="logistic_regression_model",
                    )
                    live.log_artifact(
                        str(params_file),
                        type="params",
                        name="logistic_regression_params",
                    )
                    live.end()
                    return  # Exit the function after logging artifacts

            # Proceed to training if model doesn't exist or hashes don't match
            logger.info(f"Training Logistic Regression model {cfg.model.name}...")

            # Load preprocessed training data
            train_data = pd.read_csv(train_data_path)
            X_train = train_data.drop(columns=["readmitted"])
            y_train = train_data["readmitted"]

            # Initialize and train the Logistic Regression model
            model = LogisticRegression(**cfg.model.params)
            model.fit(X_train, y_train)
            logger.info("Model training completed.")

            # Save the trained model
            joblib.dump(model, model_output_path)
            logger.info(f"Model saved to {model_output_path}")

            # Log the trained model as an artifact
            live.log_artifact(
                str(model_output_path), type="model", name="logistic_regression_model"
            )
            live.log_artifact(
                str(params_file), type="params", name="logistic_regression_params"
            )

            # Save the input hash
            with open(hash_file, "w") as f:
                f.write(input_hash)
            logger.info(f"Input hash saved to {hash_file}")

        finally:
            live.end()

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
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
        (data_hash + config_hash).encode(), usedforsecurity=False
    ).hexdigest()
    return combined_hash


if __name__ == "__main__":
    main()
