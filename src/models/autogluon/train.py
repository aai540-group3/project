import hashlib
import logging
import shutil
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
        train_data_path = Path(to_absolute_path(f"{data_paths.processed}/train.csv"))
        model_output_dir = Path(to_absolute_path("models/autogluon"))
        final_model_dir = Path(to_absolute_path("models/autogluon_final"))

        # Calculate hash of input data and configuration
        input_hash = calculate_input_hash(train_data_path, cfg)
        hash_file = final_model_dir / "input_hash.txt"

        if final_model_dir.exists() and hash_file.exists():
            with open(hash_file, "r") as f:
                stored_hash = f.read().strip()

            if stored_hash == input_hash:
                logger.info(f"Model already exists with the same input hash. Skipping training.")
                return

        logger.info(f"Training new model autogluon...")

        train_data = pd.read_csv(train_data_path)

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

        # Copy the best model (WeightedEnsemble_L3) to the final model directory
        final_model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = model_output_dir / "models" / "WeightedEnsemble_L3"
        if best_model_path.exists():
            shutil.copytree(best_model_path, final_model_dir / "WeightedEnsemble_L3", dirs_exist_ok=True)

        # Copy necessary files for loading the model
        shutil.copy(model_output_dir / "predictor.pkl", final_model_dir / "predictor.pkl")
        shutil.copy(model_output_dir / "learner.pkl", final_model_dir / "learner.pkl")

        # Save the input hash
        with open(hash_file, "w") as f:
            f.write(input_hash)

        logger.info(f"Best model saved to {final_model_dir}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise

def calculate_input_hash(data_path, cfg):
    # Calculate hash of input data
    with open(data_path, "rb") as f:
        data_hash = hashlib.md5(f.read()).hexdigest()

    # Calculate hash of configuration
    config_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode()).hexdigest()

    # Combine hashes
    return hashlib.md5((data_hash + config_hash).encode()).hexdigest()

if __name__ == "__main__":
    main()