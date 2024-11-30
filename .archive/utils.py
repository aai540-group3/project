"""Machine learning pipeline utilities for diabetic readmission prediction.

This module provides core utilities for the ML pipeline workflow focused on
processing diabetic patient data, engineering features, training models, and
evaluating readmission predictions. It includes functions for loading and
processing data, applying SMOTE sampling, scaling features, calculating metrics,
and generating visualizations.

Functions
---------
patch_files_in_site_packages : function
    Patches specified files in Python site-packages directory for GPU compatibility
setup_artifacts : function
    Creates artifact directories (model, metrics, plots) for pipeline outputs
load_data : function
    Loads data from parquet files with hash verification for reproducibility
load_config : function
    Loads configuration for either 'quick' or 'full' pipeline modes
set_global_plot_config : function
    Sets matplotlib/seaborn plot styling from config parameters
preprocess_data : function
    Separates features and target variable, drops timestamp columns
split_data : function
    Creates stratified train/validation/test splits for readmission prediction
apply_smote : function
    Balances class distributions using SMOTE to handle readmission imbalance
scale_features : function
    Standardizes numerical features using sklearn's StandardScaler
log_class_distribution : function
    Records readmission class distributions and imbalance metrics using DVCLive
save_metrics : function
    Saves model evaluation metrics (accuracy, AUC, etc.) to JSON
calculate_metrics : function
    Computes classification metrics for both validation and test sets
plot_confusion_matrix : function
    Generates labeled confusion matrix visualization for model predictions
plot_roc_curve : function
    Creates ROC curve plot comparing validation and test performance
set_plot_style : function
    Applies consistent plot styling based on global configuration

Notes
-----
This module integrates with DVC for experiment tracking and metric logging:

    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)

Examples
--------
Load data and prepare for model training:

>>> from pipeline.src.utils import load_data, preprocess_data, split_data
>>> df, data_hash = load_data('data/interim/data_featured.parquet')
>>> X, y = preprocess_data(df, 'readmitted')
>>> X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.2, 42)

Scale features and balance classes:

>>> X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, 42)
>>> X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
...     X_train_balanced, X_val, X_test)

Save model metrics and generate plots:

>>> metrics = calculate_metrics(y_val, val_pred, y_test, test_pred)
>>> save_metrics(metrics, artifacts_path)
>>> plot_confusion_matrix(y_test, test_pred_classes, artifacts_path)
>>> plot_roc_curve(y_test, test_pred, metrics, artifacts_path)
"""

import hashlib
import json
import logging
import shutil
import site
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PLOT_CONFIG = {}


def load_config(mode: str) -> DictConfig:
    """Load configuration for the specified mode (quick/full) using OmegaConf.

    :param mode: The mode to load the configuration for (e.g., 'quick' or 'full').
    :return: A dictionary containing the configuration for the specified mode.
    """
    config = OmegaConf.load("params.yaml")
    return OmegaConf.to_container(config[mode], resolve=True)


def set_global_plot_config(config: DictConfig) -> None:
    """Set the global plotting configuration from the loaded config.

    :param config: The configuration dictionary containing plotting settings.
    """
    global PLOT_CONFIG
    PLOT_CONFIG = config.plots


logger = logging.getLogger(__name__)


def patch_files_in_site_packages(patch_config: List[Dict[str, str]]) -> None:
    """Patches specified files in the Python site-packages directory.
    This function iterates over a list of patch configurations, searching for
    each specified target file in the Python site-packages directory and
    replacing its content with that of the provided patch file.
    :param patch_config: A list of dictionaries containing target and patch file paths.
    """
    for patch in patch_config:
        target_file = Path(patch["target_file"])
        patch_file = Path(patch["patch_file"])

        logger.info(f"Searching for: {target_file} in Python site-packages")

        if not patch_file.exists():
            logger.error(f"Patch file not found: {patch_file}")
            continue

        # Read patch content once, outside the loop
        try:
            patch_content = patch_file.read_text()
        except OSError as e:
            logger.error(f"Error reading patch file {patch_file}: {e}")
            continue

        # Track if we found and patched the target
        target_found = False

        for site_package_path in map(Path, site.getsitepackages()):
            target_path = site_package_path / target_file

            if not target_path.exists():
                continue

            logger.info(f"Found target at: {target_path.parent}")
            target_found = True

            try:
                target_path.write_text(patch_content)
                logger.info(f"Successfully patched: {target_path}")
            except OSError as e:
                logger.error(f"Error patching {target_path}: {e}")
                continue

        if not target_found:
            logger.warning(f"Target file not found: {target_file}")


def setup_artifacts(artifacts_path: Path, subdirs: list) -> None:
    """Create artifact directories.

    :param artifacts_path: The path where artifacts will be stored.
    :param subdirs: A list of subdirectory names to create.
    """
    if artifacts_path.exists():
        shutil.rmtree(artifacts_path)
    for subdir in subdirs:
        (artifacts_path / subdir).mkdir(parents=True, exist_ok=True)


def load_data(data_path: str) -> tuple:
    """Load data from a parquet file and calculate its hash.

    :param data_path: The path to the parquet file.
    :return: A tuple containing the DataFrame and its hash.
    """
    logger.info("Loading data...")
    with open(data_path, "rb") as f:
        data_hash = hashlib.md5(f.read()).hexdigest()
    df = pd.read_parquet(data_path)
    return df, data_hash


def preprocess_data(df: pd.DataFrame, target_column: str) -> tuple:
    """Preprocess the data by dropping unnecessary columns and separating features and target.

    :param df: The DataFrame containing the data.
    :param target_column: The name of the target column.
    :return: A tuple containing the features (X) and target (y).
    """
    columns_to_drop = ["event_timestamp", "created_timestamp"]
    X = df.drop(columns=[target_column] + columns_to_drop)
    y = df[target_column]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> tuple:
    """Split data into training, validation, and test sets.

    :param X: The features DataFrame.
    :param y: The target Series.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The random state for reproducibility.
    :return: A tuple containing the training, validation, and test sets.
    """
    X_train_val, _, y_train_val, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return train_test_split(
        X_train_val,
        y_train_val,
        test_size=PLOT_CONFIG["val_size"],
        random_state=random_state,
        stratify=y_train_val,
    )


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> tuple:
    """Apply SMOTE to the training data.

    :param X_train: The training features.
    :param y_train: The training target.
    :param random_state: The random state for reproducibility.
    :return: A tuple containing the balanced features and target.
    """
    logger.info("Applying SMOTE to training data...")
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> tuple:
    """Scale features using StandardScaler.

    :param X_train: The training features.
    :param X_val: The validation features.
    :param X_test: The test features.
    :return: A tuple containing the scaled training, validation, and test features, along with the scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def log_class_distribution(live, y: pd.Series, prefix: str = "before") -> None:
    """Log class distribution and imbalance ratio.

    :param live: The live logging object.
    :param y: The target variable.
    :param prefix: A prefix for the log entries.
    """
    class_distribution = y.value_counts().to_dict()
    imbalance_ratio = max(class_distribution.values()) / min(
        class_distribution.values()
    )
    live.log_params(
        {
            f"class_distribution_{prefix}": {
                str(k): int(v) for k, v in class_distribution.items()
            },
            f"imbalance_ratio_{prefix}": float(imbalance_ratio),
        }
    )
    logger.info(f"Class imbalance ratio {prefix} SMOTE: {imbalance_ratio:.2f}")


def save_metrics(metrics: dict, artifacts_path: Path) -> None:
    """Save metrics to a JSON file.

    :param metrics: The metrics dictionary to save.
    :param artifacts_path: The path where metrics will be saved.
    """
    with open(artifacts_path / "metrics" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


def calculate_metrics(
    y_val: np.ndarray, val_pred: np.ndarray, y_test: np.ndarray, test_pred: np.ndarray
) -> dict:
    """Calculate evaluation metrics for validation and test sets.

    :param y_val: The validation target variable.
    :param val_pred: The predicted values for the validation set.
    :param y_test: The test target variable.
    :param test_pred: The predicted values for the test set.
    :return: A dictionary containing the calculated metrics.
    """
    return {
        "val_accuracy": accuracy_score(y_val, (val_pred > 0.5).astype(int)),
        "val_precision": precision_score(y_val, (val_pred > 0.5).astype(int)),
        "val_recall": recall_score(y_val, (val_pred > 0.5).astype(int)),
        "val_f1": f1_score(y_val, (val_pred > 0.5).astype(int)),
        "val_auc": roc_auc_score(y_val, val_pred),
        "val_avg_precision": average_precision_score(y_val, val_pred),
        "test_accuracy": accuracy_score(y_test, (test_pred > 0.5).astype(int)),
        "test_precision": precision_score(y_test, (test_pred > 0.5).astype(int)),
        "test_recall": recall_score(y_test, (test_pred > 0.5).astype(int)),
        "test_f1": f1_score(y_test, (test_pred > 0.5).astype(int)),
        "test_auc": roc_auc_score(y_test, test_pred),
        "test_avg_precision": average_precision_score(y_test, test_pred),
    }


def plot_confusion_matrix(
    y_test: np.ndarray, test_pred_classes: np.ndarray, artifacts_path: Path
) -> None:
    """Generate and save confusion matrix plot.

    :param y_test: The true labels for the test set.
    :param test_pred_classes: The predicted classes for the test set.
    :param artifacts_path: The path where the plot will be saved.
    """
    set_plot_style()

    plt.figure(figsize=PLOT_CONFIG["figure"]["figsize"])
    cm = confusion_matrix(y_test, test_pred_classes)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=PLOT_CONFIG["confusion_matrix"]["xticklabels"],
        yticklabels=PLOT_CONFIG["confusion_matrix"]["yticklabels"],
    )
    plt.title("Confusion Matrix", pad=20)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        artifacts_path / "plots" / "confusion_matrix.png",
        dpi=PLOT_CONFIG["figure"]["dpi"],
        bbox_inches="tight",
    )
    plt.close()


def plot_roc_curve(
    y_val: np.ndarray,
    val_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
    metrics: dict,
    artifacts_path: Path,
) -> None:
    """Generate and save ROC curve plot.

    :param y_val: The validation target variable.
    :param val_pred: The predicted values for the validation set.
    :param y_test: The test target variable.
    :param test_pred: The predicted values for the test set.
    :param metrics: The metrics dictionary containing AUC values.
    :param artifacts_path: The path where the plot will be saved.
    """
    set_plot_style()

    plt.figure(figsize=PLOT_CONFIG["figure"]["figsize"])
    fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)
    plt.plot(
        fpr_val,
        tpr_val,
        color=PLOT_CONFIG["colors"]["primary"],
        label=f"Validation (AUC = {metrics['val_auc']:.3f})",
    )

    fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
    plt.plot(
        fpr_test,
        tpr_test,
        color=PLOT_CONFIG["colors"]["secondary"],
        label=f"Test (AUC = {metrics['test_auc']:.3f})",
    )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(PLOT_CONFIG["roc_curve"]["xlabel"])
    plt.ylabel(PLOT_CONFIG["roc_curve"]["ylabel"])
    plt.title(PLOT_CONFIG["roc_curve"]["title"], pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        artifacts_path / "plots" / "roc_curve.png",
        dpi=PLOT_CONFIG["figure"]["dpi"],
        bbox_inches="tight",
    )
    plt.close()


def set_plot_style() -> None:
    """Set the plotting style based on the global configuration."""
    sns.set_style(PLOT_CONFIG["style"])
    sns.set_context(PLOT_CONFIG["context"], font_scale=PLOT_CONFIG["font_scale"])
