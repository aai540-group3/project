import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from dvclive import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.data.path
        model_params = cfg.model.params

        train_data_path = Path(to_absolute_path(f"{data_paths.processed}/train_preprocessed.csv"))
        model_output_path = Path(to_absolute_path(cfg.model.model_output_path))

        logger.info("Loading preprocessed training data...")
        train_df = pd.read_csv(train_data_path)

        X_train = train_df.drop(columns=["readmitted"])
        y_train = train_df["readmitted"]

        logger.info("Creating Logistic Regression model...")
        model = LogisticRegression(**model_params)

        logger.info("Creating final pipeline...")
        clf = Pipeline(steps=[("classifier", model)])

        logger.info("Training the model...")
        with Live() as live:
            clf.fit(X_train, y_train)

            logger.info("Logging training completion metric...")
            live.log_metric("training_completed", 1)

            logger.info("Performing cross-validation...")
            cross_val_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
            cv_accuracy = cross_val_scores.mean()
            logger.info(f"Cross-Validation Accuracy: {cv_accuracy * 100:.2f}%")

            logger.info("Logging cross-validation metrics...")
            live.log_metric("cv_accuracy", cv_accuracy)
            for idx, score in enumerate(cross_val_scores):
                live.log_metric(f"cv_fold_{idx+1}_accuracy", score)

        logger.info("Saving the model...")
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_output_path)
        logger.info(f"Model saved to {model_output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()