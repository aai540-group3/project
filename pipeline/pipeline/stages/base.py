import json
import pathlib
import sys
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self):
        """Initialize the pipeline stage."""
        self.name = self.__class__.__name__.lower()
        self.cfg = OmegaConf.load("params.yaml")

        # Set visualization defaults from config
        plt.style.use(self.cfg.visualization.style)
        sns.set_theme(
            style=self.cfg.visualization.theme.style,
            context=self.cfg.visualization.theme.context,
            font_scale=self.cfg.visualization.theme.font_scale,
            rc=self.cfg.visualization.theme.rc,
        )

        # Set color palette from config
        self.colors = self.cfg.colors

    def load_data(self, filename: str, subdir: str = "data") -> pd.DataFrame:
        """Load data from file based on extension."""
        file_path = pathlib.Path(subdir) / filename
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def split_data(self, X, y):
        """Split data into training, validation, and test sets."""
        train_size = self.cfg.splits.train
        val_size = self.cfg.splits.val
        test_size = self.cfg.splits.test

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_size), random_state=self.cfg.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_size / (val_size + test_size)), random_state=self.cfg.seed, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def log_metrics(self, metrics):
        """Log metrics to DVC Live and console."""
        for name, value in metrics.items():
            self.live.log_metric(name, value)
            logger.info(f"{name}: {value:.4f}")

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate common evaluation metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba),
        }

    def log_params(self, config, data_hash, X_train, X_val, X_test):
        """Log parameters to DVC Live."""
        params = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "features": len(X_train.columns),
            "data_hash": data_hash,
        }

        hyperparameters = config.get("hyperparameters", {})
        for param_name, param_value in hyperparameters.items():
            params[f"{self.name}_{param_name}"] = param_value

        self.live.log_params(params)

    def save_output(self, data: Union[pd.DataFrame, dict], filename: str, subdir: str = "data") -> Path:
        """Save output data in appropriate format."""
        output_path = pathlib.Path(subdir) / filename
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(data, pd.DataFrame):
                if filename.endswith(".parquet"):
                    data.to_parquet(output_path)
                elif filename.endswith(".csv"):
                    data.to_csv(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported format for DataFrame: {filename}")
            elif isinstance(data, (dict, list)):
                if filename.endswith(".json"):
                    output_path.write_text(json.dumps(data, indent=2))
                else:
                    raise ValueError(f"Unsupported format for dict/list: {filename}")
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            logger.debug(f"Saved {filename} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save {filename}: {str(e)}")
            raise

    def save_metrics(self, name: str, data: Dict[str, Any], timestamp: bool = True) -> Path:
        """Save metric data as JSON."""
        if timestamp:
            data = {"timestamp": datetime.now(timezone.utc).isoformat(), "stage": self.name, **data}
        return self.save_output(data, f"{name}.json", f"metrics/{self.name}")

    def save_plot(self, name: str, plot_func: Callable[[], Any], **kwargs) -> pathlib.Path:
        """Create and save a plot or handle special plot types based on configuration."""
        plot_config = self.cfg.plots.get(name, {})
        plot_path = pathlib.Path(self.cfg.paths.plots) / self.name / f"{name}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        try:
            plot_type = plot_config.get("type")

            if plot_type == "csv":
                data = plot_func()
                filename = plot_config.get("title", f"{name}.csv").replace(" ", "_").lower() + ".csv"
                subdir = plot_config.get("subdir", self.name)
                self.save_output(data, filename, subdir=subdir)
                logger.debug(f"Saved CSV data to {filename} in {subdir}")
                return plot_path

            else:
                plot = plot_func()
                if isinstance(plot, sns.FacetGrid) or isinstance(plot, sns.axisgrid.Grid):
                    plot.savefig(plot_path, bbox_inches="tight")
                elif isinstance(plot, plt.Figure):
                    plot.savefig(plot_path, bbox_inches="tight")
                elif isinstance(plot, plt.Axes):
                    plot.figure.savefig(plot_path, bbox_inches="tight")
                else:
                    logger.warning(f"Plot function returned an unexpected type: {type(plot)}")

                logger.debug(f"Saved plot to {plot_path}")
                return plot_path

        except Exception as e:
            logger.error(f"Failed to save plot {name}: {str(e)}")
            raise

        finally:
            plt.close("all")

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"],
        )
        self.save_plot(
            "confusion_matrix",
            lambda data: sns.heatmap(data, annot=True, fmt="d", cmap="Blues").set(title=title),
            data=cm_df,
        )

    def plot_roc_curve(self, y_true, y_pred_proba, metrics, title="ROC Curve"):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

        self.save_plot(
            "roc_curve",
            lambda data: (
                sns.lineplot(
                    data=data,
                    x="FPR",
                    y="TPR",
                    label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})',
                    color=self.colors.primary,
                ),
                sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color=self.colors.neutral, label="Random Chance"),
                plt.xlabel("False Positive Rate"),
                plt.ylabel("True Positive Rate"),
                plt.title(title),
                plt.legend(),
            ),
            data=roc_df,
        )

    def plot_feature_importance(self, feature_importance: pd.DataFrame, title="Top 20 Features by Importance"):
        """Plot and save feature importance."""
        self.save_plot(
            "feature_importance",
            lambda data: sns.barplot(data=data.head(20), y="feature", x="importance").set(title=title),
            data=feature_importance,
        )
        self.save_output(
            feature_importance.reset_index(),
            "feature_importance.csv",
            subdir=str(Path(self.cfg.paths.metrics) / self.name),
        )

    def execute(self):
        """Execute the pipeline stage in a separate thread."""
        logger.info(f"EXECUTING STAGE: {self.name}")
        start_time = datetime.now()

        def thread_run():
            try:
                self.run()
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"COMPLETED STAGE: {self.name} in {duration:.2f}s")
            except Exception as e:
                logger.error(f"FAILED STAGE: {self.name}")
                logger.error(f"Error: {str(e)}")
                logger.error("Full error:", exc_info=True)
                sys.exit(1)
            finally:
                logger.complete()

        thread = threading.Thread(target=thread_run)
        thread.start()
        thread.join()

    @abstractmethod
    def run(self):
        """Execute stage-specific logic."""
        pass
