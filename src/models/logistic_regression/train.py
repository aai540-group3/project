import hashlib
import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from dvclive import Live
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

        # Calculate hash of input data and configuration to prevent redundant trainings
        input_hash = calculate_input_hash(input_dir / "train_preprocessed.csv", cfg)
        hash_file = model_output_dir / "input_hash.txt"

        if model_output_dir.exists() and hash_file.exists():
            with open(hash_file, "r") as f:
                stored_hash = f.read().strip()

            if stored_hash == input_hash:
                logger.info("Model already exists with the same input hash. Skipping training.")
                return

        logger.info(f"Training Logistic Regression model {cfg.model.name}...")

        # Load preprocessed training data
        train_data = pd.read_csv(input_dir / "train_preprocessed.csv")
        X_train = train_data.drop(columns=["readmitted"])
        y_train = train_data["readmitted"]

        # Initialize DVCLive
        with Live() as live:
            # Log training parameters
            params_to_log = {
                "model": cfg.model.name,
                "label": "readmitted",
                "problem_type": "binary",
                "dataset_version": cfg.dataset.version
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
            live.log_artifact(str(model_output_path), type="model", name=f"{cfg.model.name}_model")
            logger.info(f"Model saved and logged as artifact at {model_output_path}")

        # Save the input hash to prevent redundant trainings
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