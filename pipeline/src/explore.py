#!/usr/bin/env python3
"""
pipeline.src.explore

Data exploration and label issues detection using Cleanlab.

This module performs exploratory data analysis and detects label issues using Cleanlab.
It generates visualizations and summary reports based on detected issues.

Attributes:
    logger (Logger): Module-level logger configured to display info messages.

Example:
    To run the module, execute the following:

        $ python explore.py

"""

import os
import sys
import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_plot(plt, filename: str, bbox_inches: str = "tight", pad_inches: float = 0.1) -> None:
    """Saves a matplotlib plot to a file with specified settings.

    Args:
        plt: The matplotlib.pyplot instance.
        filename (str): The path where the plot will be saved.
        bbox_inches (str, optional): Bounding box for the saved figure. Defaults to "tight".
        pad_inches (float, optional): Padding for the saved figure. Defaults to 0.1.

    """
    plt.tight_layout()
    plt.savefig(filename, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=300, format="png")
    plt.close()
    logger.info(f"Saved plot: {filename}")


def analyze_label_issues(label_issues_df: pd.DataFrame) -> Dict:
    """Analyzes label issues and generates confidence and misclassification statistics.

    Args:
        label_issues_df (pd.DataFrame): DataFrame containing detected label issues with
            'cleanlab_confidence', 'actual_label', and 'predicted_label' columns.

    Returns:
        dict: Analysis results including total issues, confidence statistics, misclassification patterns,
              and high-confidence issue counts.

    """
    total_issues = len(label_issues_df)
    logger.info(f"Total label issues detected: {total_issues}")

    confidence_stats = {
        "mean_confidence": label_issues_df["cleanlab_confidence"].mean(),
        "median_confidence": label_issues_df["cleanlab_confidence"].median(),
        "min_confidence": label_issues_df["cleanlab_confidence"].min(),
        "max_confidence": label_issues_df["cleanlab_confidence"].max(),
    }

    misclassification_patterns = pd.crosstab(
        label_issues_df["actual_label"], label_issues_df["predicted_label"], margins=True
    )

    dataset_size = len(label_issues_df)
    issue_percentage = (total_issues / dataset_size) * 100

    high_confidence_issues = label_issues_df[
        label_issues_df["cleanlab_confidence"] >= 0.9
    ]

    return {
        "total_issues": total_issues,
        "confidence_stats": confidence_stats,
        "misclassification_patterns": misclassification_patterns,
        "issue_percentage": issue_percentage,
        "high_confidence_issues": len(high_confidence_issues),
    }


def create_cleanlab_visualizations(label_issues_df: pd.DataFrame, save_dir: str) -> None:
    """Generates and saves Cleanlab analysis visualizations.

    Args:
        label_issues_df (pd.DataFrame): DataFrame of label issues.
        save_dir (str): Directory where visualizations are saved.

    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=label_issues_df, x="cleanlab_confidence", bins=30, kde=True)
    plt.title("Distribution of Confidence Scores in Label Issues")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    save_plot(plt, f"{save_dir}/cleanlab_confidence_distribution.png")

    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(
        label_issues_df["actual_label"], label_issues_df["predicted_label"]
    )
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix of Label Issues")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    save_plot(plt, f"{save_dir}/cleanlab_confusion_matrix.png")

    if isinstance(label_issues_df.get("feature_importance", None), pd.Series):
        plt.figure(figsize=(12, 6))
        label_issues_df["feature_importance"].sort_values(ascending=True).plot(kind="barh")
        plt.title("Feature Importance in Label Issues")
        plt.xlabel("Importance Score")
        save_plot(plt, f"{save_dir}/cleanlab_feature_importance.png")


def generate_cleanlab_report(analysis_results: Dict, save_path: str) -> None:
    """Generates a report of Cleanlab analysis and saves it to a text file.

    Args:
        analysis_results (Dict): The analysis results dictionary from `analyze_label_issues`.
        save_path (str): Path to save the report.

    """
    with open(save_path, "w") as f:
        f.write("Cleanlab Analysis Report\n")
        f.write("=======================\n\n")

        f.write("1. Overall Statistics\n")
        f.write(f"Total label issues detected: {analysis_results['total_issues']}\n")
        f.write(f"Percentage of dataset: {analysis_results['issue_percentage']:.2f}%\n\n")

        f.write("2. Confidence Statistics\n")
        for metric, value in analysis_results["confidence_stats"].items():
            f.write(f"{metric}: {value:.3f}\n")
        f.write("\n")

        f.write("3. Misclassification Patterns\n")
        f.write(str(analysis_results["misclassification_patterns"]))
        f.write("\n\n")

        f.write("4. High Confidence Issues\n")
        f.write(f"Number of high confidence issues: {analysis_results['high_confidence_issues']}\n")

        f.write("\n5. Recommendations\n")
        f.write("- Review high confidence issues first\n")
        f.write("- Focus on patterns in misclassification matrix\n")
        f.write("- Consider data quality in areas with highest issue rates\n")
    logger.info(f"Report generated at: {save_path}")


def main():
    """Executes the data exploration and Cleanlab analysis pipeline.

    Loads and preprocesses data, performs Cleanlab analysis, generates visualizations,
    and saves results to the specified directories.

    Raises:
        FileNotFoundError: If required data files are missing.
        Exception: For general data loading or processing errors.

    """
    for dir_path in ["reports", "reports/plots", "reports/cleanlab"]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured existence of directory: {dir_path}")

    try:
        raw_df = pd.read_parquet("data/raw/data.parquet")
        processed_df = pd.read_parquet("data/interim/data_cleaned.parquet")
        logger.info("Data loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Error: Required data files not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred while reading the files: {e}")
        sys.exit(1)

    df = processed_df if len(processed_df["readmitted"].unique()) >= 2 else raw_df
    if df is raw_df:
        logger.warning("Using raw data as processed data lacks sufficient class distribution.")

    try:
        df["readmitted"] = df["readmitted"].map({1: "Readmitted", 0: "Not Readmitted"})

        plt.figure(figsize=(10, 6))
        sns.countplot(x="readmitted", data=df)
        plt.title("Distribution of Readmission")
        save_plot(plt, "reports/plots/readmitted_distribution.png")

        logger.info("Performing Cleanlab analysis...")
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop("readmitted", axis=1), df["readmitted"], test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        cl = CleanLearning(clf=clf)
        cl.fit(X_train, y_train)

        pred_probs = cl.predict_proba(X_test)
        label_issues = find_label_issues(
            labels=y_test.values,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence"
        )

        label_issues_df = pd.DataFrame(
            {
                "actual_label": y_test.iloc[label_issues],
                "predicted_label": cl.predict(X_test)[label_issues],
                "cleanlab_confidence": pred_probs[label_issues].max(axis=1),
            }
        )

        analysis_results = analyze_label_issues(label_issues_df)
        create_cleanlab_visualizations(label_issues_df, "reports/cleanlab")
        generate_cleanlab_report(analysis_results, "reports/cleanlab/detailed_analysis.txt")
        logger.info("Cleanlab analysis completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
