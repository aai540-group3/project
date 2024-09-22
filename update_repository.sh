#!/usr/bin/env bash

# Correcting dvc.yaml
cat << 'EOF' > dvc.yaml
stages:
  ingest:
    cmd: >
      python src/data/ingestion.py
    deps:
      - src/data/ingestion.py
    params:
      - dataset
    outs:
      - data/raw/data.csv

  clean:
    cmd: >
      python src/data/cleaning.py
    deps:
      - src/data/cleaning.py
      - data/raw/data.csv
    outs:
      - data/interim/data.csv

  feature_engineering:
    cmd: >
      python src/data/build_features.py
    deps:
      - src/data/build_features.py
      - data/interim/data.csv
    params:
      - feature_engineering
    outs:
      - data/processed/data_featured.csv

  split:
    cmd: >
      python src/data/splitting.py
    deps:
      - src/data/splitting.py
      - data/processed/data_featured.csv
    params:
      - training.split
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  preprocess:
    foreach: ${models}
    do:
      cmd: >
        python src/models/${item}/preprocessing.py model=${item}
      deps:
        - src/models/${item}/preprocessing.py
        - data/processed/train.csv
      params:
        - models
        - model=${item}
      outs:
        # Outputs are different for each model
        - data/processed/${item}/train_preprocessed.${item == 'autogluon' ? 'csv' : 'pkl'}
        - data/processed/${item}/preprocessor.joblib
        - data/processed/${item}/y_train.pkl

  train:
    foreach: ${models}
    do:
      cmd: >
        python src/models/${item}/train.py model=${item}
      deps:
        - src/models/${item}/train.py
        - data/processed/${item}/
        - configs/model/${item}.yaml
      params:
        - models
        - model=${item}
      outs:
        - models/${item}/

  evaluate:
    foreach: ${models}
    do:
      cmd: >
        python src/models/${item}/evaluate.py model=${item}
      deps:
        - src/models/${item}/evaluate.py
        - models/${item}/
        - data/processed/test.csv
        - data/processed/${item}/
      params:
        - models
        - model=${item}
      outs:
        - reports/metrics/${item}_metrics.json

  visualize:
    cmd: >
      python src/visualization/visualize.py
    deps:
      - src/visualization/visualize.py
      - reports/metrics/
    outs:
      - reports/figures/
EOF

# Correcting src/models/autogluon/evaluate.py
cat << 'EOF' > src/models/autogluon/evaluate.py
import logging
from pathlib import Path
import json

import hydra
import pandas as pd
from autogluon.tabular import TabularPredictor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.dataset.path
        test_data_path = Path(to_absolute_path(f"{data_paths.processed}/test.csv"))
        model_output_dir = Path(to_absolute_path(cfg.model.model_output_path))

        logger.info(f"Evaluating model {cfg.model.name}...")

        test_df = pd.read_csv(test_data_path)
        y_true = test_df["readmitted"]

        predictor = TabularPredictor.load(str(model_output_dir))
        y_pred = predictor.predict(test_df)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, predictor.predict_proba(test_df)[1]),
        }

        metrics_output_path = Path(to_absolute_path(f"reports/metrics/{cfg.model.name}_metrics.json"))
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {metrics_output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
EOF

# Correcting src/models/logistic_regression/evaluate.py
cat << 'EOF' > src/models/logistic_regression/evaluate.py
import logging
from pathlib import Path
import json

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.dataset.path
        test_data_path = Path(to_absolute_path(f"{data_paths.processed}/test.csv"))
        input_dir = Path(to_absolute_path(f"{data_paths.processed}/{cfg.model.name}"))
        model_output_path = Path(to_absolute_path(cfg.model.model_output_path))
        preprocessor_path = input_dir / "preprocessor.joblib"

        logger.info(f"Evaluating model {cfg.model.name}...")

        X_test_df = pd.read_csv(test_data_path)
        y_true = X_test_df["readmitted"]
        X_test = X_test_df.drop(columns=["readmitted"])

        preprocessor = joblib.load(preprocessor_path)
        X_test_preprocessed = preprocessor.transform(X_test)

        model = joblib.load(model_output_path)
        y_pred = model.predict(X_test_preprocessed)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, model.predict_proba(X_test_preprocessed)[:, 1]),
        }

        metrics_output_path = Path(to_absolute_path(f"reports/metrics/{cfg.model.name}_metrics.json"))
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {metrics_output_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
EOF

# Correcting src/visualization/visualize.py
cat << 'EOF' > src/visualization/visualize.py
import logging
import os
import json

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = to_absolute_path("reports/figures")
    os.makedirs(output_dir, exist_ok=True)
    models = cfg.models
    metrics_list = []

    for model_name in models:
        metrics_path = to_absolute_path(f"reports/metrics/{model_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metrics["model"] = model_name
            metrics_list.append(metrics)
        else:
            logger.warning(f"Metrics file not found for {model_name}")

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.set_index("model", inplace=True)
        metrics_df.plot(kind="bar")
        plt.title("Model Comparison")
        plt.ylabel("Metric Score")
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"))
        plt.close()
        logger.info("Visualization completed.")
    else:
        logger.warning("No metrics to visualize.")


if __name__ == "__main__":
    main()
EOF

# Correcting src/models/autogluon/train.py
cat << 'EOF' > src/models/autogluon/train.py
import logging
from pathlib import Path

import hydra
import pandas as pd
from autogluon.tabular import TabularPredictor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.dataset.path
        input_dir = Path(to_absolute_path(f"{data_paths.processed}/{cfg.model.name}"))
        preprocessed_train_data_path = input_dir / "train_preprocessed.csv"
        model_output_dir = Path(to_absolute_path(cfg.model.model_output_path))
        model_output_dir.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training model {cfg.model.name}...")

        train_df = pd.read_csv(preprocessed_train_data_path)

        predictor = TabularPredictor(
            label="readmitted",
            path=str(model_output_dir)
        ).fit(
            train_data=train_df,
            **cfg.model.params
        )

        logger.info(f"Model saved to {model_output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
EOF

# Correcting configs/config.yaml
cat << 'EOF' > configs/config.yaml
defaults:
  - data: diabetes
  - training: base
  - feature_engineering: base
  - model: null
  - _self_

dataset:
  # Inherit dataset.name and dataset.path from the defaults
model: ${model}
models: ${models}
EOF