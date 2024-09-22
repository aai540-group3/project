import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from autogluon.tabular import TabularPredictor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        data_paths = cfg.dataset.path
        test_data_path = Path(to_absolute_path(f"{data_paths.processed}/test.csv"))
        model_dir = Path(to_absolute_path(f"models/{cfg.model.name}"))

        logger.info(f"Evaluating model {cfg.model.name}...")

        test_df = pd.read_csv(test_data_path)
        y_true = test_df["readmitted"]
        X_test = test_df.drop(columns=["readmitted"])

        predictor = TabularPredictor.load(str(model_dir))
        y_pred = predictor.predict(X_test)
        y_pred_proba = predictor.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'),
        }

        metrics_output_path = Path(to_absolute_path(f"reports/metrics/{cfg.model.name}_metrics.json"))
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {metrics_output_path}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise

if __name__ == "__main__":
    main()