import pathlib
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ucimlrepo
from loguru import logger

from pipeline.stages.base import PipelineStage


class Ingest(PipelineStage):
    """Pipeline stage for data ingestion."""

    @logger.catch()
    def run(self):
        """Ingest data from UCI ML Repository and perform initial data analysis."""
        # Suppress specific pandas warnings
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

        # Fetch data from UCI ML Repository
        logger.info(f"Fetching dataset with ID: {self.cfg.dataset.id}")
        data = ucimlrepo.fetch_ucirepo(id=self.cfg.dataset.id)

        # Convert fetched data to a single DataFrame
        logger.debug("Converting fetched data to DataFrame")
        df = pd.concat([data.data.features, data.data.targets], axis=1)

        # Define paths for saving plots
        plots_dir = pathlib.Path(self.cfg.paths.plots) / self.name
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Plots will be saved to: {plots_dir}")

        logger.info("Generating Missing Values Heatmap")
        missing_data = df.isnull().mean().to_frame("missing_rate")

        plt.figure(figsize=(12, 8))
        sns.heatmap(missing_data, cmap="YlOrRd", cbar=True)
        plt.title("Missing Values Distribution")
        plt.xlabel("Features")
        plt.ylabel("Missing Rate")
        plt.tight_layout()

        missing_plot_path = plots_dir / "missing_values.png"
        plt.savefig(missing_plot_path, bbox_inches="tight")
        logger.debug(f"Saved Missing Values Heatmap to: {missing_plot_path}")
        plt.close()

        logger.info("Generating Data Types Distribution Bar Plot")
        dtype_counts = df.dtypes.value_counts().to_frame("count").reset_index()
        dtype_counts.rename(columns={"index": "dtype"}, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=dtype_counts, x="dtype", y="count")
        plt.title("Data Types Distribution")
        plt.xlabel("Data Type")
        plt.ylabel("Count")
        plt.tight_layout()

        dtype_plot_path = plots_dir / "data_types.png"
        plt.savefig(dtype_plot_path, bbox_inches="tight")
        logger.debug(f"Saved Data Types Distribution plot to: {dtype_plot_path}")
        plt.close()

        logger.info("Saving dataset outputs")
        outputs = {
            "data.parquet": df,
            "data.csv": df,
            "metadata.json": data.metadata,
            "variables.json": data.variables.to_dict(orient="records"),
        }

        for filename, content in outputs.items():
            self.save_output(content, filename, subdir="data/raw")
            logger.debug(f"Saved {filename} to data/raw")

        logger.info("Calculating data quality metrics")
        quality_metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_per_column": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "unique_values_per_column": df.nunique().to_dict(),
        }

        logger.info("Saving metrics")
        self.save_metrics(
            "metrics",
            {
                "dataset": {
                    "id": self.cfg.dataset.id,
                    "name": self.cfg.dataset.name,
                    "source": self.cfg.dataset.source,
                },
                "quality_metrics": quality_metrics,
                "files_created": list(outputs.keys()) + ["missing_values.png", "data_types.png"],
            },
        )
        logger.debug("Metrics saved successfully")
