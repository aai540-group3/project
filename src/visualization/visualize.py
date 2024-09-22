import os
import re

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

# Define metrics and plots to visualize
METRICS_TO_VISUALIZE = [
    ("dvclive/plots/metrics/cv_accuracy.tsv", "cv_accuracy", "CV Accuracy"),
    ("dvclive_evaluate/plots/metrics/accuracy.tsv", "accuracy", "Evaluation Accuracy"),
    ("dvclive_evaluate/plots/metrics/precision.tsv", "precision", "Precision"),
    ("dvclive_evaluate/plots/metrics/recall.tsv", "recall", "Recall"),
    ("dvclive_evaluate/plots/metrics/roc_auc.tsv", "roc_auc", "ROC AUC"),
]


def sanitize_filename(filename):
    """Replaces invalid characters in filenames with underscores."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load configurations
    data_paths = cfg.data.path

    # Create output directory for figures
    output_dir = to_absolute_path("reports/figures")
    os.makedirs(output_dir, exist_ok=True)

    # Load and visualize metrics
    for metric_file, metric_name, metric_label in METRICS_TO_VISUALIZE:
        try:
            df = pd.read_csv(to_absolute_path(metric_file), sep="\t")
            plt.plot(df["step"], df[metric_name], label=metric_label)
        except FileNotFoundError:
            print(f"Warning: Metric file not found: {metric_file}")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.title("Model Performance Metrics")
    plt.savefig(os.path.join(output_dir, "model_performance.png"))
    plt.close()  # Close the plot to avoid overlapping plots

    # Load and visualize training data distribution
    train_data_path = to_absolute_path(f"{data_paths.processed}/train.csv")
    train_df = pd.read_csv(train_data_path)

    # Visualize target variable distribution (readmitted)
    train_df["readmitted"].value_counts().plot(kind="bar")
    plt.xlabel("Readmitted")
    plt.ylabel("Count")
    plt.title("Distribution of Readmission in Training Data")
    plt.savefig(os.path.join(output_dir, "readmission_distribution.png"))
    plt.close()

    # Visualize distribution of numerical features
    numerical_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    for col in numerical_cols:
        plt.hist(train_df[col], bins=20)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col} in Training Data")
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.close()

    # Visualize distribution of categorical features
    categorical_cols = train_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in categorical_cols:
        train_df[col].value_counts().plot(kind="bar")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col} in Training Data")

        # Sanitize the filename before saving
        sanitized_filename = sanitize_filename(col)
        plt.savefig(os.path.join(output_dir, f"{sanitized_filename}_distribution.png"))
        plt.close()


if __name__ == "__main__":
    main()
