import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import tensorflow as tf
from autogluon.tabular import TabularPredictor
from seaborn import sns

# Set the style and font settings
plt.style.use("seaborn-v0_8")
sns.set_theme()

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics
    models = ["autogluon", "logistic_regression", "neural_network"]
    metrics_list = []
    for model_name in models:
        metrics_path = f"models/{model_name}/artifacts/metrics/metrics.json"
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

        # Plot model comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Model Performance Comparison", fontsize=16)

        metrics = [
            ("test_accuracy", "Accuracy"),
            ("test_precision", "Precision"),
            ("test_recall", "Recall"),
            ("test_f1", "F1-Score"),
            ("test_auc", "AUC-ROC"),
            ("test_pr_auc", "PR-AUC"),
        ]

        for i, (metric, label) in enumerate(metrics):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            metrics_df[metric].plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title(label)
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
        plt.savefig(os.path.join(output_dir, "model_comparison.png"))
        plt.close()
    else:
        logger.warning("No metrics to visualize.")

    # Load test data for SHAP
    X_test = pd.read_parquet("data/processed/test.parquet")

    # --- AutoGluon SHAP ---
    model_path = "models/autogluon/artifacts/model"
    predictor = TabularPredictor.load(model_path)
    explainer = shap.TreeExplainer(predictor)
    shap_values = explainer.shap_values(X_test)
    output_dir = "models/autogluon/artifacts/plots"
    os.makedirs(output_dir, exist_ok=True)

    # SHAP Summary Plot (AutoGluon)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    # SHAP Force Plot (AutoGluon) - Example for first instance in X_test
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X_test.iloc[0, :],
        show=False,
        matplotlib=True,
    )
    plt.savefig(os.path.join(output_dir, "shap_force_plot.png"))
    plt.close()

    # SHAP Importance Plot (AutoGluon)
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(output_dir, "shap_importance.png"))
    plt.close()

    # --- Logistic Regression SHAP ---
    model_path = "models/logistic_regression/artifacts/model"
    model = joblib.load(os.path.join(model_path, "model.joblib"))
    explainer = shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)
    output_dir = "models/logistic_regression/artifacts/plots"
    os.makedirs(output_dir, exist_ok=True)

    # SHAP Summary Plot (Logistic Regression)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    # SHAP Force Plot (Logistic Regression) - Example for first instance in X_test
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X_test.iloc[0, :],
        show=False,
        matplotlib=True,
    )
    plt.savefig(os.path.join(output_dir, "shap_force_plot.png"))
    plt.close()

    # SHAP Importance Plot (Logistic Regression) - Using bar plot for consistency
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(output_dir, "shap_importance.png"))
    plt.close()

    # --- Neural Network SHAP ---
    model_path = "models/neural_network/artifacts/model"
    model = tf.keras.models.load_model(os.path.join(model_path, "model.h5"))
    explainer = shap.DeepExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)
    output_dir = "models/neural_network/artifacts/plots"
    os.makedirs(output_dir, exist_ok=True)

    # SHAP Summary Plot (Neural Network)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

    # SHAP Force Plot (Neural Network) - Example for first instance in X_test
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X_test.iloc[0, :],
        show=False,
        matplotlib=True,
    )
    plt.savefig(os.path.join(output_dir, "shap_force_plot.png"))
    plt.close()

    # SHAP Importance Plot (Neural Network)
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(output_dir, "shap_importance.png"))
    plt.close()

    logger.info("Visualization and SHAP explanations completed.")


if __name__ == "__main__":
    main()