import logging
import os

import numpy as np
import yaml
from dvclive import Live
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.utils import (apply_smote, calculate_metrics, load_data,
                       log_class_distribution, plot_confusion_matrix,
                       plot_roc_curve, preprocess_data, save_metrics,
                       scale_features, setup_artifacts, split_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(mode: str) -> dict:
    """Load configuration for the specified mode (quick/full)."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[mode]


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> dict:
    """Optimize hyperparameters using Optuna."""
    import optuna

    logger.info("Starting hyperparameter optimization...")

    def objective(trial) -> float:
        C = trial.suggest_float(
            "C",
            config["optimization"]["param_space"]["C"]["low"],
            config["optimization"]["param_space"]["C"]["high"],
            log=config["optimization"]["param_space"]["C"]["log"],
        )
        penalty = trial.suggest_categorical(
            "penalty", config["optimization"]["param_space"]["penalty"]
        )
        solver = trial.suggest_categorical(
            "solver", config["optimization"]["param_space"]["solver"]
        )
        l1_ratio = (
            trial.suggest_float(
                "l1_ratio",
                config["optimization"]["param_space"]["l1_ratio"]["low"],
                config["optimization"]["param_space"]["l1_ratio"]["high"],
            )
            if penalty == "elasticnet"
            else None
        )

        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=config["model"]["random_state"],
        )
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, val_pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["model"]["random_state"]),
    )
    study.optimize(
        objective,
        n_trials=config["model"]["optimization_trials"],
        show_progress_bar=True,
    )

    best_params = study.best_trial.params
    for param_name, param_value in best_params.items():
        logger.info(f"Best {param_name}: {param_value}")
    return best_params


def train_logistic_regression(mode: str) -> None:
    """Train a logistic regression model based on the provided configuration."""
    config = load_config(mode)
    artifacts_path = config["paths"]["artifacts"]
    setup_artifacts(artifacts_path, config["paths"]["subdirs"])

    live = Live(dir=os.path.join(artifacts_path, "metrics"), dvcyaml=False)

    try:
        df = load_data(config["paths"]["data"])
        X, y = preprocess_data(df, config["model"]["target"])
        log_class_distribution(live, y)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X, y, config["splits"]["test_size"], config["splits"]["random_state"]
        )

        X_train_balanced, y_train_balanced = apply_smote(
            X_train, y_train, config["model"]["random_state"]
        )
        log_class_distribution(live, y_train_balanced, prefix="after")

        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train_balanced, X_val, X_test
        )

        # Optimize hyperparameters
        best_params = optimize_hyperparameters(
            X_train_scaled, y_train_balanced, X_val_scaled, y_val, config
        )

        # Train final model
        final_model = LogisticRegression(
            C=best_params["C"],
            penalty=best_params.get("penalty"),
            solver=best_params["solver"],
            l1_ratio=best_params.get("l1_ratio"),
            max_iter=1000,
            random_state=config["model"]["random_state"],
        )
        final_model.fit(X_train_scaled, y_train_balanced)

        # Evaluate model
        val_pred = final_model.predict_proba(X_val_scaled)[:, 1]
        test_pred = final_model.predict_proba(X_test_scaled)[:, 1]
        metrics = calculate_metrics(y_val, val_pred, y_test, test_pred)

        save_metrics(metrics, artifacts_path)

        # Generate plots
        plot_confusion_matrix(
            y_test, (test_pred > 0.5).astype(int), artifacts_path, config["plots"]
        )
        plot_roc_curve(y_test, test_pred, metrics, artifacts_path, config["plots"])

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if live:
            live.end()


if __name__ == "__main__":
    mode = os.getenv("MODE", "quick").lower()
    train_logistic_regression(mode)
