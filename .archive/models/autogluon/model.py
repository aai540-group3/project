import os
import yaml
import logging
import pandas as pd
from autogluon.tabular import TabularPredictor
from dvclive import Live
from src.utils import (
    calculate_metrics,
    setup_artifacts,
    load_data,
    split_data,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(mode: str) -> dict:
    """Load configuration for the specified mode (quick/full)."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[mode]


def train_autogluon(mode: str) -> None:
    """Train AutoGluon models based on the provided configuration."""
    config = load_config(mode)
    artifacts_path = config["paths"]["artifacts"]
    setup_artifacts(artifacts_path, config["paths"]["subdirs"])

    live = Live(dir=os.path.join(artifacts_path, "metrics"), dvcyaml=False)

    try:
        df = load_data(config["paths"]["data"])
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            df, config["model"]["label"], config["training"]["splits"]
        )

        # Combine features and target for training, validation, and testing
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Train the AutoGluon model
        predictor = TabularPredictor(
            label=config["model"]["label"], path=artifacts_path
        ).fit(train_data, tuning_data=val_data)

        # Evaluate the model
        test_pred = predictor.predict(test_data)
        test_pred_proba = predictor.predict_proba(test_data)
        metrics = calculate_metrics(y_test, test_pred, test_pred_proba)

        # Save metrics and generate plots
        save_metrics(metrics, artifacts_path)
        plot_confusion_matrix(y_test, test_pred, artifacts_path, config["plots"])
        plot_roc_curve(y_test, test_pred_proba, metrics, artifacts_path, config["plots"])

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if live:
            live.end()


if __name__ == "__main__":
    mode = os.getenv("MODE", "quick").lower()
    train_autogluon(mode)
