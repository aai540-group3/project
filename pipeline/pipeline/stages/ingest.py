import warnings

import pandas as pd
import seaborn as sns
import ucimlrepo

from pipeline.stages.base import PipelineStage


class Ingest(PipelineStage):
    """Pipeline stage for data ingestion."""

    def run(self):
        """Ingest data from UCI ML Repository."""
        # Fetch data
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        data = ucimlrepo.fetch_ucirepo(id=self.cfg.dataset.id)

        # Convert to DataFrame
        df = pd.concat([data.data.features, data.data.targets], axis=1)

        # Missing values heatmap
        missing_data = df.isnull().mean().to_frame("missing_rate")
        self.save_plot(
            "missing_values",
            lambda data, **kwargs: sns.heatmap(data, **{k: v for k, v in kwargs.items() if k != "title"}),
            data=missing_data,
            cmap="YlOrRd",
            title="Missing Values Distribution",
        )

        # Data types distribution
        dtype_counts = df.dtypes.value_counts().to_frame("count")
        self.save_plot(
            "data_types",
            lambda data, x, y: sns.barplot(data=data, x=x, y=y),
            data=dtype_counts,
            x=dtype_counts.index.astype(str),
            y="count",
            title="Data Types Distribution",
        )

        # Save outputs
        outputs = {
            "data.parquet": df,
            "data.csv": df,
            "metadata.json": data.metadata,
            "variables.json": data.variables.to_dict(orient="records"),
        }

        for filename, content in outputs.items():
            self.save_output(content, filename, "data/raw")

        # Calculate data quality metrics
        quality_metrics = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_per_column": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "unique_values_per_column": df.nunique().to_dict(),
        }

        # Save metrics
        self.save_metrics(
            "metrics",
            {
                "dataset": {
                    "id": self.cfg.dataset.id,
                    "name": self.cfg.dataset.name,
                    "source": self.cfg.dataset.source,
                },
                "quality_metrics": quality_metrics,
                "files_created": list(outputs.keys()),
            },
        )
