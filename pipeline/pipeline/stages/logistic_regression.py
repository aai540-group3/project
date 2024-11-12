import hashlib
import os
import warnings
from functools import wraps
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from dvclive import Live
from imblearn.over_sampling import SMOTE
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .stage import Stage


def track(func):
    """Decorator for managing DVC Live tracking around a training process."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.live:
            self.live = Live(dir=str(Path(self.cfg.paths.metrics) / "logistic"), dvcyaml=False)
        try:
            result = func(self, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}") from e
        finally:
            if self.live:
                self.live.end()
            logger.info("Training completed")
        return result

    return wrapper


class LogisticRegession(Stage):
    """Pipeline stage for Logistic Regression model training with DVC Live tracking."""

    def __init__(self):
        """Initialize the Logistic Regression pipeline stage."""
        super().__init__()
        self.live = None
        self.model = None
        self.preprocessor = None
        self.X_test = None
        self.y_test = None

    def run(self):
        """Entry point for running model training."""
        self.train()

    @track
    def train(self):
        """Perform data preparation, model training, and evaluation."""
        warnings.filterwarnings("ignore")
        np.random.seed(self.cfg.seed)

        # Determine the mode ('quick' or 'full') from environment variable
        mode = os.getenv("MODE", "quick").lower()
        if mode not in ["quick", "full"]:
            logger.warning(f"Invalid MODE '{mode}' specified. Defaulting to 'quick' mode.")
            mode = "quick"
        logger.info(f"Selected mode: {mode}")

        # Load the corresponding configuration
        config = self.cfg.logistic.get(mode, {})
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        if not config:
            logger.error(f"No configuration found for mode '{mode}'. Exiting training.")
            return

        # Extract model settings
        model_config = config.get("model", {})
        label_column = model_config.get("label", "readmitted")
        metric = model_config.get("metric", "roc_auc")
        problem_type = model_config.get("problem_type", "binary")

        # Load data and preprocess
        data = self.load_data("features.parquet", subdir=self.cfg.paths.processed)
        X = data.drop(columns=[label_column])
        y = data[label_column]
        data_hash = hashlib.md5(pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=True).values).hexdigest()

        # Split data and log parameters
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        self.X_test, self.y_test = X_test, y_test
        self.log_params(config, data_hash, X_train, X_val, X_test)

        # Log class distribution before resampling
        logger.info(f"Class distribution in training set before resampling:\n{y_train.value_counts()}")

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=self.cfg.seed)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"Class distribution in training set after SMOTE:\n{pd.Series(y_train_res).value_counts()}")

        # Define preprocessing steps
        numeric_features = X_train_res.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X_train_res.select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Store preprocessor for feature importance calculation
        self.preprocessor = preprocessor

        # Extract training hyperparameters
        hyperparameters = config.get("hyperparameters", {})
        max_iter = config.get("max_iter", 100)
        solver = config.get("solver", "lbfgs")

        # Define the initial pipeline
        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=max_iter, solver=solver, **hyperparameters)),
            ]
        )

        logger.debug(f"Initial Pipeline Steps: {self.model.named_steps}")

        # Extract optimization settings
        optimize_config = config.get("optimize", {})
        optimize_enabled = optimize_config.get("enabled", False)
        param_space = optimize_config.get("param_space", {})
        n_trials = optimize_config.get("n_trials", 10)
        timeout = optimize_config.get("timeout", 600)
        direction = optimize_config.get("direction", "maximize")
        optimize_metric = optimize_config.get("metric", metric)

        # Run hyperparameter optimization if enabled
        if optimize_enabled:
            logger.info("Hyperparameter optimization is enabled.")
            self.model = self.optimize_hyperparameters(
                X_train_res,
                y_train_res,
                X_val,
                y_val,
                param_space,
                n_trials,
                timeout,
                direction,
                optimize_metric,
                preprocessor,
            )
            logger.debug(f"Optimized Pipeline Steps: {self.model.named_steps}")
        else:
            logger.info("Hyperparameter optimization is disabled.")

        # Train the model
        logger.debug("Training the model...")
        self.model.fit(X_train_res, y_train_res)
        logger.debug(f"Model trained with parameters: {self.model.named_steps['classifier'].get_params()}")

        # Generate predictions and calculate metrics
        y_pred, y_pred_proba = self.generate_predictions(X_test)
        if y_pred is None or y_pred_proba is None:
            logger.error("Prediction generation failed. Exiting training.")
            return
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Calculate feature importance
        feature_importance_df = self.calculate_feature_importance(
            preprocessor=self.preprocessor, classifier=self.model.named_steps["classifier"]
        )

        # Log metrics and create evaluation plots
        self.log_metrics(metrics)
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba, metrics)
        self.plot_precision_recall_curve(y_test, y_pred_proba, metrics)
        self.plot_feature_importance(feature_importance_df)

        # Save feature importance
        self.save_output(
            data=feature_importance_df,
            filename="feature_importance.csv",
            subdir=Path(self.cfg.paths.metrics) / self.name,
        )

        # Save the trained model
        self.save_model()

    def optimize_hyperparameters(
        self, X_train, y_train, X_val, y_val, param_space, n_trials, timeout, direction, metric, preprocessor
    ):
        """Optimize hyperparameters using Optuna within a pipeline."""

        def objective(trial):
            # Suggest hyperparameters based on the param_space
            params = {}
            for param_name, param_cfg in param_space.items():
                if param_cfg["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_cfg["low"], param_cfg["high"], log=param_cfg.get("log", False)
                    )
                elif param_cfg["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_cfg["choices"])

            # Create a temporary pipeline with suggested hyperparameters
            temp_model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=self.model.named_steps["classifier"].max_iter,
                            solver=self.model.named_steps["classifier"].solver,
                            **params,
                        ),
                    ),
                ]
            )

            # Train the temporary model
            temp_model.fit(X_train, y_train)

            # Predict on validation set
            y_pred_proba = temp_model.predict_proba(X_val)[:, 1]

            # Calculate the metric (e.g., ROC AUC)
            score = roc_auc_score(y_val, y_pred_proba)
            logger.debug(f"Trial {trial.number}: {metric} = {score}")

            return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best {metric}: {study.best_value}")

        # Update the model's hyperparameters with the best found
        best_params = study.best_params
        self.model.named_steps["classifier"].set_params(**best_params)

        return self.model

    def generate_predictions(self, X_test):
        """Generate predictions and prediction probabilities using the trained model."""
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            return y_pred, y_pred_proba
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            return None, None

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics."""
        try:
            accuracy = np.mean(y_pred == y_true)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
            logger.info(f"Evaluation Metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {}

    def calculate_feature_importance(
        self, preprocessor: ColumnTransformer, classifier: LogisticRegression
    ) -> pd.DataFrame:
        """Calculate feature importance based on model coefficients."""
        try:
            # Extract feature names after preprocessing
            numeric_features = preprocessor.transformers_[0][2]
            categorical_features = preprocessor.transformers_[1][2]
            categorical_transformer = preprocessor.transformers_[1][1]
            if isinstance(categorical_transformer.named_steps["onehot"], OneHotEncoder):
                ohe = categorical_transformer.named_steps["onehot"]
                ohe_features = ohe.get_feature_names_out(categorical_features)
            else:
                ohe_features = categorical_features  # Fallback if not OneHotEncoder

            all_features = list(numeric_features) + list(ohe_features)

            # Get coefficients
            coefficients = classifier.coef_[0]

            # Create DataFrame
            feature_importance = pd.DataFrame({"feature": all_features, "importance": coefficients})

            # Sort by absolute importance
            feature_importance["abs_importance"] = feature_importance["importance"].abs()
            feature_importance = feature_importance.sort_values(by="abs_importance", ascending=False).drop(
                "abs_importance", axis=1
            )

            # Debugging: Log the feature_importance DataFrame
            logger.debug(f"Feature Importance DataFrame:\n{feature_importance.head()}")

            return feature_importance.head(20)  # Top 20 features
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return pd.DataFrame()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"],
            )
            self.save_plot(
                "confusion_matrix",
                lambda data: sns.heatmap(data, annot=True, fmt="d", cmap="Blues").set(title="Confusion Matrix"),
                data=cm_df,
                save_path="plots/logistic/confusion_matrix.png",
            )
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")

    def plot_roc_curve(self, y_true, y_pred_proba, metrics):
        """Plot and save ROC curve."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

            self.save_plot(
                "roc_curve",
                lambda data: sns.lineplot(
                    data=data, x="FPR", y="TPR", label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})'
                ),
                data=roc_df,
                save_path="plots/logistic/roc_curve.png",
            )

            # Add random chance line and labels
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/logistic/roc_curve.png")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {e}")

    def plot_precision_recall_curve(self, y_true, y_pred_proba, metrics, title="Precision-Recall Curve"):
        """Plot and save Precision-Recall curve."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})

            self.save_plot(
                "precision_recall_curve",
                lambda data: sns.lineplot(
                    data=data, x="Recall", y="Precision", label=f'PR curve (AUC = {metrics["pr_auc"]:.3f})'
                ),
                data=pr_df,
                save_path="plots/logistic/precision_recall_curve.png",
            )

            # Add baseline
            plt.plot([0, 1], [0.5, 0.5], linestyle="--", color="gray", label="Baseline")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/logistic/precision_recall_curve.png")
            plt.close()
        except Exception as e:
            logger.error(f"Failed to plot Precision-Recall curve: {e}")

    def plot_feature_importance(
        self, feature_importance_df: pd.DataFrame, title: str = "Top 20 Features by Importance"
    ):
        """Plot and save feature importance based on model coefficients."""
        try:
            # Debugging: Log the DataFrame columns and head
            logger.debug(
                f"Plotting feature importance with DataFrame columns: {feature_importance_df.columns.tolist()}"
            )
            logger.debug(f"Feature Importance DataFrame:\n{feature_importance_df.head()}")

            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance_df, x="importance", y="feature", palette="viridis")
            plt.title(title)
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")

            # Save plot
            save_path = Path("plots/logistic/feature_importance.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logger.info(f"Plot 'feature_importance' saved successfully at {save_path}.")
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")

    def save_model(self):
        """Save the trained Logistic Regression model to disk."""
        try:
            model_dir = Path(self.cfg.paths.models) / "logistic"
            model_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            model_path = model_dir / "model.joblib"
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved successfully at {model_path}.")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def save_plot(self, name: str, plot_func, data: pd.DataFrame, save_path: str = None):
        """Saves a plot using the provided plotting function and data."""
        try:
            logger.debug(f"Preparing to save plot '{name}' with data:\n{data.head()}")
            plot_func(data)  # Pass 'data' to plot_func
            save_path = save_path or f"plots/{name}.png"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Plot '{name}' saved successfully at {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save plot '{name}': {e}")

    def save_output(self, data: pd.DataFrame, filename: str, subdir: Path = None):
        """Save a DataFrame as a CSV file to a specified subdirectory."""
        try:
            if subdir:
                save_dir = subdir
            else:
                # Default to 'metrics/logistic'
                save_dir = Path(self.cfg.paths.metrics) / self.name

            # Create the directory if it doesn't exist
            save_dir.mkdir(parents=True, exist_ok=True)

            # Define the full path
            save_path = save_dir / filename

            # Save the DataFrame as CSV
            data.to_csv(save_path, index=False)

            # Debugging: Log the DataFrame columns
            logger.debug(f"Feature Importance DataFrame columns: {data.columns.tolist()}")

            logger.info(f"Saved output '{filename}' successfully at {save_path}.")
        except Exception as e:
            logger.error(f"Failed to save output '{filename}': {e}")
