import json
import logging
import math
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS_TO_VISUALIZE = [
    ("dvclive/plots/metrics/cv_accuracy.tsv", "cv_accuracy", "CV Accuracy"),
    ("dvclive_evaluate/plots/metrics/accuracy.tsv", "accuracy", "Evaluation Accuracy"),
    ("dvclive_evaluate/plots/metrics/precision.tsv", "precision", "Precision"),
    ("dvclive_evaluate/plots/metrics/recall.tsv", "recall", "Recall"),
    ("dvclive_evaluate/plots/metrics/roc_auc.tsv", "roc_auc", "ROC AUC"),
]

MAX_COLS_TO_PLOT = 20  # Maximum number of columns to plot in a single figure


def sanitize_filename(filename: str) -> str:
    """Replaces invalid characters in filenames with underscores."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in filename)


def plot_metric(ax, df, metric_name, metric_label):
    # Ensure 'step' and metric columns are numeric
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")

    ax.plot(df["step"], df[metric_name], label=metric_label)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} Over Time")
    ax.legend()


def convert_column_type(series):
    """Attempt to convert a series to numeric or datetime type."""
    if series.dtype == "object":
        # Try to convert to numeric
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() / len(series) > 0.9:
            return numeric_series

        # Try to convert to datetime
        datetime_series = pd.to_datetime(series, errors="coerce")
        if datetime_series.notna().sum() / len(series) > 0.9:
            return datetime_series

        # Return the original series (likely categorical)
        return series
    return series


def plot_distribution(ax, data, column):
    series = convert_column_type(data[column])
    if pd.api.types.is_numeric_dtype(series):
        sns.histplot(data=series.dropna(), kde=True, ax=ax)
    elif pd.api.types.is_datetime64_any_dtype(series):
        series.dropna().hist(ax=ax, bins=20)
    else:
        value_counts = series.value_counts()
        value_counts.plot(kind="bar", ax=ax)

    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {column}")
    ax.tick_params(axis="x", rotation=45)


def plot_feature_distributions(df, cols, output_dir):
    n_cols = len(cols)
    if n_cols == 0:
        logger.info("No columns to plot.")
        return

    n_cols_per_fig = min(n_cols, MAX_COLS_TO_PLOT)
    n_rows = math.ceil(n_cols_per_fig / 5)  # 5 columns per row
    n_figs = math.ceil(n_cols / MAX_COLS_TO_PLOT)

    for fig_num in range(n_figs):
        fig, axes = plt.subplots(n_rows, 5, figsize=(20, 5 * n_rows))
        axes = axes.flatten()

        start_idx = fig_num * MAX_COLS_TO_PLOT
        end_idx = min((fig_num + 1) * MAX_COLS_TO_PLOT, n_cols)

        for idx, col in enumerate(cols[start_idx:end_idx]):
            plot_distribution(axes[idx], df, col)
            logger.info(f"Plotted distribution for column: {col}")

        # Hide unused subplots
        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_distributions_{fig_num+1}.png"))
        plt.close()


def process_metric_files(metrics_to_visualize):
    processed_metrics = {}
    for metric_file, metric_name, _ in metrics_to_visualize:
        try:
            df = pd.read_csv(to_absolute_path(metric_file), sep="\t")
            # Ensure 'step' and metric columns are numeric before saving
            df["step"] = pd.to_numeric(df["step"], errors="coerce")
            df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
            processed_metrics[metric_file] = {
                "data": df.to_dict(orient="records"),
                "x": "step",
                "y": metric_name,
            }
        except FileNotFoundError:
            logger.warning(f"Metric file not found: {metric_file}")
    return processed_metrics


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.data.path
        output_dir = to_absolute_path("reports/figures")
        os.makedirs(output_dir, exist_ok=True)

        # Process and save metrics for DVC Studio
        processed_metrics = process_metric_files(METRICS_TO_VISUALIZE)
        with open(os.path.join(output_dir, "dvc_metrics.json"), "w") as f:
            json.dump(processed_metrics, f)

        # Plot metrics
        fig, axes = plt.subplots(
            len(METRICS_TO_VISUALIZE), 1, figsize=(10, 5 * len(METRICS_TO_VISUALIZE))
        )
        if len(METRICS_TO_VISUALIZE) == 1:
            axes = [axes]
        for idx, (metric_file, metric_name, metric_label) in enumerate(
            METRICS_TO_VISUALIZE
        ):
            if metric_file in processed_metrics:
                df = pd.DataFrame(processed_metrics[metric_file]["data"])
                # Ensure correct data types after reconstructing the DataFrame
                df["step"] = pd.to_numeric(df["step"], errors="coerce")
                df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
                plot_metric(axes[idx], df, metric_name, metric_label)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_performance.png"))
        plt.close()

        # Load and visualize training data distribution
        train_data_path = to_absolute_path(f"{data_paths.processed}/train.csv")
        train_df = pd.read_csv(train_data_path)

        # Ensure 'readmitted' column is of the correct type
        train_df["readmitted"] = train_df["readmitted"].astype(str)

        # Visualize target variable distribution
        plt.figure(figsize=(10, 5))
        sns.countplot(x="readmitted", data=train_df)
        plt.title("Distribution of Readmission in Training Data")
        plt.savefig(os.path.join(output_dir, "readmission_distribution.png"))
        plt.close()

        # Visualize feature distributions
        feature_cols = [col for col in train_df.columns if col != "readmitted"]
        plot_feature_distributions(train_df, feature_cols, output_dir)

        logger.info(f"All visualizations saved to {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()