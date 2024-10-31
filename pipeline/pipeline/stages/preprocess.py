"""
Data Preprocessing Stage
========================

.. module:: pipeline.stages.preprocess
   :synopsis: Data preprocessing with detailed logging and tracking

.. moduleauthor:: aai540-group3
"""

import json
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

from pipeline.stages.base import PipelineStage
from pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class PreprocessStage(PipelineStage):
    """Data preprocessing stage implementation with detailed logging and tracking."""

    def __init__(self, cfg: DictConfig):
        """Initialize data preprocessing stage."""
        super().__init__(cfg)
        self.start_time = time.time()
        self.preprocessing_errors: List[Dict[str, str]] = []

        # Set stage_config to cfg.data.preprocessing
        self.stage_config = cfg.data.get("preprocessing", {})
        self.data_config = cfg.data

        # Initialize metadata structure
        self.metadata: Dict[str, Any] = {
            "preprocessing": {
                "steps_completed": [],
                "errors": [],
                "execution_time": None,
            },
            "config": OmegaConf.to_container(self.stage_config, resolve=True),
        }

    def run(self) -> None:
        """Execute preprocessing pipeline."""
        logger.info("Starting data preprocessing")
        success = False

        try:
            # Load raw data
            data_path = self.get_path("raw") / "data.csv"
            df = pd.read_csv(data_path, low_memory=False)
            logger.info(f"Initial data loaded with shape {df.shape}")
            self._log_data_quality(df, "initial")

            # Clean data
            df = self._clean_data(df)
            self.metadata["preprocessing"]["steps_completed"].append("clean_data")

            # Handle missing values
            df = self._handle_missing_values(df)
            self.metadata["preprocessing"]["steps_completed"].append(
                "handle_missing_values"
            )

            # Remove invalid entries
            df = self._remove_invalid_entries(df)
            self.metadata["preprocessing"]["steps_completed"].append(
                "remove_invalid_entries"
            )

            # Handle categorical variables
            df = self._handle_categorical(df)
            self.metadata["preprocessing"]["steps_completed"].append(
                "handle_categorical"
            )

            # Create diagnosis level features
            df = self._create_diagnosis_features(df)
            self.metadata["preprocessing"]["steps_completed"].append(
                "create_diagnosis_features"
            )

            # Handle duplicates
            df = self._handle_duplicates(df)
            self.metadata["preprocessing"]["steps_completed"].append(
                "handle_duplicates"
            )

            # Handle outliers
            df = self._handle_outliers(df)
            self.metadata["preprocessing"]["steps_completed"].append("handle_outliers")

            # Process target variable
            df = self._process_target(df)
            self.metadata["preprocessing"]["steps_completed"].append("process_target")

            # Final data quality check
            self._log_data_quality(df, "final")

            # Generate Metrics and Plots
            self._generate_metrics(df)
            self._generate_plots(df)

            # Save preprocessed data
            self._save_data(df)

            # Record execution time
            self.metadata["preprocessing"]["execution_time"] = (
                time.time() - self.start_time
            )

            # Save metadata
            self._save_metadata()

            success = True
            logger.info("Data preprocessing completed successfully")

        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            self.preprocessing_errors.append({"error": str(e)})
            raise RuntimeError(f"Data preprocessing failed: {str(e)}")

        finally:
            self._log_final_status(success)
            self._cleanup_tracking()

    def _generate_metrics(self, df: pd.DataFrame) -> None:
        """Generate and save metrics."""
        try:
            logger.info("Generating metrics for data preprocessing")

            # Data Stats
            data_stats = {
                "shape": df.shape,
                "missing_values": int(df.isnull().sum().sum()),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            }

            # Feature Stats
            feature_stats = df.describe(include="all").to_dict()

            # Data Quality
            data_quality = {
                "initial": {
                    "shape": self.metadata["preprocessing"]["steps_completed"],
                    # Add more initial data quality metrics if needed
                },
                "final": {
                    "shape": df.shape,
                    "missing_values": int(df.isnull().sum().sum()),
                    # Add more final data quality metrics if needed
                },
            }

            # Save Metrics
            metrics_dir = self.get_path("metrics") / "data"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            with open(metrics_dir / "preprocess_data_stats.json", "w") as f:
                json.dump(data_stats, f, indent=4)

            with open(metrics_dir / "preprocess_feature_stats.json", "w") as f:
                json.dump(feature_stats, f, indent=4)

            with open(metrics_dir / "preprocess_data_quality.json", "w") as f:
                json.dump(data_quality, f, indent=4)

            logger.info("Metrics generated and saved successfully")

            # Log metrics to tracking system
            if self.tracking_initialized:
                self.log_metrics(data_stats)
                self.log_metrics(feature_stats)
                self.log_metrics(data_quality)

        except Exception as e:
            logger.warning(f"Failed to generate metrics: {e}")

    def _generate_plots(self, df: pd.DataFrame) -> None:
        """Generate and save plots."""
        try:
            logger.info("Generating plots for data preprocessing")

            plots_dir = self.get_path("metrics") / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Data Distribution Plot
            plt.figure(figsize=(10, 6))
            df.hist(bins=50, figsize=(20, 15))
            plt.tight_layout()
            data_distribution_path = plots_dir / "data_distribution.png"
            plt.savefig(data_distribution_path)
            plt.close()
            logger.debug(f"Saved data distribution plot to {data_distribution_path}")

            # Feature Distributions Plot
            plt.figure(figsize=(10, 6))
            sns.pairplot(df.select_dtypes(include=[np.number]))
            feature_distributions_path = plots_dir / "feature_distributions.png"
            plt.savefig(feature_distributions_path)
            plt.close()
            logger.debug(
                f"Saved feature distributions plot to {feature_distributions_path}"
            )

            # Correlation Matrix Plot
            plt.figure(figsize=(12, 10))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            correlation_matrix_path = plots_dir / "correlation_matrix.png"
            plt.title("Correlation Matrix")
            plt.savefig(correlation_matrix_path)
            plt.close()
            logger.debug(f"Saved correlation matrix plot to {correlation_matrix_path}")

            # Target Distribution Plot
            target_col = self.stage_config.get(
                "target_column", self.data_config.get("target_column")
            )
            if target_col in df.columns:
                plt.figure(figsize=(8, 6))
                sns.countplot(x=target_col, data=df)
                plt.title(f"Target Variable Distribution: {target_col}")
                target_distribution_path = plots_dir / "target_distribution.png"
                plt.savefig(target_distribution_path)
                plt.close()
                logger.debug(
                    f"Saved target distribution plot to {target_distribution_path}"
                )

            logger.info("Plots generated and saved successfully")

            # Log plots to tracking system
            if self.tracking_initialized:
                self.log_artifact(str(data_distribution_path))
                self.log_artifact(str(feature_distributions_path))
                self.log_artifact(str(correlation_matrix_path))
                if target_col in df.columns:
                    self.log_artifact(str(target_distribution_path))

        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

    def _save_data(self, df: pd.DataFrame) -> None:
        """Save preprocessed data."""
        try:
            # Ensure that all columns have appropriate data types
            # Convert columns with object dtype but integer values to int type
            object_cols = df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(int)
                    logger.debug(f"Converted column '{col}' to integer type")

            output_dir = self.get_path("interim")
            output_dir.mkdir(parents=True, exist_ok=True)
            data_path = output_dir / "data_cleaned.parquet"
            feature_names_path = output_dir / "feature_names.csv"

            df.to_parquet(data_path, index=False)
            logger.info(f"Saved preprocessed data to {data_path}")

            # Save feature names
            feature_names = df.columns.tolist()
            pd.DataFrame(feature_names, columns=["features"]).to_csv(
                feature_names_path, index=False
            )
            logger.info(f"Saved feature names to {feature_names_path}")

            # Log artifacts if tracking is enabled
            self.log_artifact(str(data_path))
            self.log_artifact(str(feature_names_path))

        except Exception as e:
            logger.error(f"Error in _save_data: {e}")
            self.preprocessing_errors.append({"step": "save_data", "error": str(e)})
            raise

    # Existing methods (_clean_data, _handle_missing_values, etc.) remain unchanged

    def _save_metadata(self) -> None:
        """Save preprocessing metadata."""
        try:
            output_path = self.get_path("interim") / "preprocessing_metadata.json"

            # Add summary statistics
            self.metadata["preprocessing"]["errors"] = self.preprocessing_errors
            self.metadata["preprocessing"]["timestamp"] = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(output_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Saved preprocessing metadata to {output_path}")

            # Log metadata as artifact
            self.log_artifact(str(output_path))

        except Exception as e:
            logger.error(f"Error in _save_metadata: {e}")
            raise

    def _log_final_status(self, success: bool) -> None:
        """Log final preprocessing status."""
        if not self.tracking_initialized:
            return

        # Log overall metrics
        overall_metrics = {
            "data_preprocessing_success": int(success),
            "preprocessing_steps_completed": len(
                self.metadata["preprocessing"]["steps_completed"]
            ),
            "preprocessing_duration_seconds": time.time() - self.start_time,
        }

        self.log_metrics(overall_metrics)

        if self.preprocessing_errors:
            error_count = len(self.preprocessing_errors)
            failure_rate = (
                error_count / len(self.metadata["preprocessing"]["steps_completed"])
                if len(self.metadata["preprocessing"]["steps_completed"]) > 0
                else 0
            )
            self.log_metrics(
                {
                    "preprocessing_errors": error_count,
                    "failure_rate": failure_rate,
                }
            )


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        stage = PreprocessStage(cfg)
        stage.run()

    main()
