# src/stages/ingest.py
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from omegaconf import DictConfig
from ucimlrepo import fetch_ucirepo

from .base import PipelineStage

logger = logging.getLogger(__name__)

class DataIngestionStage(PipelineStage):
    """Data ingestion stage."""

    def run(self) -> None:
        """Execute data ingestion."""
        self.tracker.start_run(run_name="ingest")

        try:
            # Fetch UCI data
            logger.info("Fetching UCI diabetes dataset...")
            diabetes_data = fetch_ucirepo(id=296)

            # Extract components
            X = diabetes_data.data.features
            y = diabetes_data.data.targets
            metadata = diabetes_data.metadata
            variables = diabetes_data.variables

            # Combine features and target
            df = pd.concat([X, y], axis=1)

            # Log data statistics
            self._log_data_stats(df)

            # Save data
            self._save_data(df, metadata, variables)

            logger.info("Data ingestion completed successfully")

        finally:
            self.tracker.end_run()

    def _log_data_stats(self, df: pd.DataFrame) -> None:
        """Log data statistics."""
        stats = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        self.tracker.log_metrics({"data_stats": stats})

    def _save_data(
        self,
        df: pd.DataFrame,
        metadata: Dict,
        variables: pd.DataFrame
    ) -> None:
        """Save data and metadata."""
        # Create output directories
        raw_dir = Path(self.cfg.paths.raw)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Save data in multiple formats
        df.to_parquet(raw_dir / "data.parquet")
        df.to_csv(raw_dir / "data.csv", index=False)

        # Save metadata
        with open(raw_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save variable definitions
        variables.to_csv(raw_dir / "variables.csv", index=False)
