"""
.. module:: src.visualization.visualize
   :synopsis: Generate visualizations for model comparison.

This script generates a bar plot comparing the performance of different models based on their
evaluation metrics. It reads the evaluation metrics from JSON files for each model specified in
the configuration. If metrics files are found, the script creates a bar plot comparing the models
across various metrics and saves the plot as an image.
"""

import json
import logging
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Generate visualizations for model comparison.

    :param cfg: Hydra configuration object containing the list of models to compare.
    :type cfg: DictConfig
    :raises Exception: If an error occurs during visualization.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        output_dir = to_absolute_path("reports/figures")
        os.makedirs(output_dir, exist_ok=True)
        models = cfg.models
        metrics_list = []

        for model_name in models:
            metrics_path = to_absolute_path(
                f"reports/metrics/{model_name}_metrics.json"
            )
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
            plt.ylim(0, 1)  # Set y-axis limits for better visualization
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "model_comparison.png"))
            plt.close()
            logger.info("Visualization completed.")
        else:
            logger.warning("No metrics to visualize.")

    except Exception as e:
        logger.error(f"An error occurred during visualization: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


if __name__ == "__main__":
    main()
