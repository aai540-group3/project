"""
Visualization Utilities
====================

.. module:: pipeline.utils.visualization.manager
   :synopsis: Data visualization and plotting utilities

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from omegaconf import DictConfig


class VisualizationManager:
    """Manager for data visualization."""

    def __init__(self, cfg: DictConfig):
        """Initialize visualization manager.

        :param cfg: Visualization configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self._setup_style()
        self.artifacts = []

    def _setup_style(self) -> None:
        """Configure visualization style."""
        if self.cfg.style:
            plt.style.use(self.cfg.style)

        # Set default figure size and DPI
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = self.cfg.dpi

    def generate_visualizations(self, df: pd.DataFrame) -> None:
        """Generate standard visualizations.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        """
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate standard visualizations
            if self.cfg.plots.get("correlation", True):
                self.plot_correlation_matrix(df, output_dir / "correlation_matrix.png")

            if self.cfg.plots.get("distributions", True):
                self.plot_feature_distributions(df, output_dir / "feature_distributions.png")

            if self.cfg.plots.get("missing_values", True):
                self.plot_missing_values(df, output_dir / "missing_values.png")

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")

    def plot_correlation_matrix(self, df: pd.DataFrame, output_path: Union[str, Path], **kwargs) -> None:
        """Plot correlation matrix.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        :param output_path: Output path
        :type output_path: Union[str, Path]
        :param kwargs: Additional plotting arguments
        """
        try:
            plt.figure(figsize=self.cfg.plots.correlation.get("figsize", (12, 10)))

            # Calculate correlation matrix
            corr = df.select_dtypes(include=[np.number]).corr()

            # Plot heatmap
            sns.heatmap(
                corr,
                annot=self.cfg.plots.correlation.get("annot", True),
                cmap=self.cfg.plots.correlation.get("cmap", "coolwarm"),
                **kwargs,
            )

            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            self.artifacts.append(str(output_path))

        except Exception as e:
            logger.error(f"Failed to plot correlation matrix: {e}")

    def plot_feature_distributions(self, df: pd.DataFrame, output_path: Union[str, Path], **kwargs) -> None:
        """Plot feature distributions.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        :param output_path: Output path
        :type output_path: Union[str, Path]
        :param kwargs: Additional plotting arguments
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) - 1) // n_cols + 1

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

            if n_rows == 1:
                axes = [axes]

            for idx, col in enumerate(numeric_cols):
                row = idx // n_cols
                col_idx = idx % n_cols

                sns.histplot(data=df, x=col, ax=axes[row][col_idx], **kwargs)
                axes[row][col_idx].set_title(f"Distribution of {col}")

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            self.artifacts.append(str(output_path))

        except Exception as e:
            logger.error(f"Failed to plot feature distributions: {e}")

    def plot_missing_values(self, df: pd.DataFrame, output_path: Union[str, Path], **kwargs) -> None:
        """Plot missing values heatmap.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        :param output_path: Output path
        :type output_path: Union[str, Path]
        :param kwargs: Additional plotting arguments
        """
        try:
            plt.figure(figsize=(10, 6))

            # Calculate missing value percentages
            missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)

            # Plot missing values
            sns.barplot(x=missing.values, y=missing.index)
            plt.title("Missing Values (%)")
            plt.xlabel("Percentage Missing")

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            self.artifacts.append(str(output_path))

        except Exception as e:
            logger.error(f"Failed to plot missing values: {e}")

    def plot_feature_importance(self, importance_df: pd.DataFrame, output_path: Union[str, Path], **kwargs) -> None:
        """Plot feature importance.

        :param importance_df: Feature importance DataFrame
        :type importance_df: pd.DataFrame
        :param output_path: Output path
        :type output_path: Union[str, Path]
        :param kwargs: Additional plotting arguments
        """
        try:
            plt.figure(figsize=(12, 6))

            sns.barplot(data=importance_df, x="importance", y="feature", **kwargs)

            plt.title("Feature Importance")
            plt.xlabel("Importance Score")
            plt.ylabel("Feature")

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            self.artifacts.append(str(output_path))

        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")

    def plot_learning_curves(self, history: Dict[str, List[float]], output_path: Union[str, Path], **kwargs) -> None:
        """Plot learning curves.

        :param history: Training history
        :type history: Dict[str, List[float]]
        :param output_path: Output path
        :type output_path: Union[str, Path]
        :param kwargs: Additional plotting arguments
        """
        try:
            plt.figure(figsize=(12, 6))

            for metric, values in history.items():
                plt.plot(values, label=metric, **kwargs)

            plt.title("Learning Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.legend()

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            self.artifacts.append(str(output_path))

        except Exception as e:
            logger.error(f"Failed to plot learning curves: {e}")

    def get_artifacts(self) -> List[str]:
        """Get list of generated artifacts.

        :return: List of artifact paths
        :rtype: List[str]
        """
        return self.artifacts
