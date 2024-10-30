import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from feast import FeatureStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from ..utils.experiment import ExperimentTracker

logger = logging.getLogger(__name__)


class FeatureEngineeringStage:
    """Feature engineering stage with Feast integration."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tracker = ExperimentTracker(cfg, cfg.experiment.name)
        self.store = FeatureStore(repo_path=cfg.feast.repo_path)

    def run(self) -> None:
        """Execute feature engineering pipeline."""
        self.tracker.start_run(run_name="feature_engineering")

        try:
            # Load preprocessed data
            df = pd.read_parquet(self.cfg.paths.interim / "data_cleaned.parquet")

            # Create basic features
            df = self._create_basic_features(df)

            # Create interaction features
            df = self._create_interaction_features(df)

            # Create ratio features
            df = self._create_ratio_features(df)

            # Apply transformations
            df = self._apply_transformations(df)

            # Scale features
            df = self._scale_features(df)

            # Add timestamps for Feast
            df = self._add_timestamps(df)

            # Create feature views and register with Feast
            self._create_feast_features(df)

            # Save feature metadata
            self._save_feature_metadata(df)

            # Save featured data
            self._save_featured_data(df)

            logger.info("Feature engineering completed successfully")

        finally:
            self.tracker.end_run()

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features."""
        # Total medications
        medication_cols = self.cfg.features.medication_columns
        df["total_medications"] = df[medication_cols].sum(axis=1)

        # Medication density
        df["medication_density"] = df["total_medications"] / df["time_in_hospital"]

        # Service utilization
        df["total_encounters"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )

        df["encounter_per_time"] = df["total_encounters"] / df["time_in_hospital"]

        # Procedure-related features
        df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"]
        df["lab_procedures_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"]

        # Diagnostic density
        df["diagnoses_per_encounter"] = df["number_diagnoses"] / (
            df["total_encounters"] + 1
        )

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        for feat1, feat2 in self.cfg.features.interactions:
            if feat1 in df.columns and feat2 in df.columns:
                feature_name = f"{feat1}_x_{feat2}"
                df[feature_name] = df[feat1] * df[feat2]

        return df

    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features."""
        ratio_features = {
            "procedure_medication_ratio": ("num_procedures", "num_medications"),
            "lab_procedure_ratio": ("num_lab_procedures", "num_procedures"),
            "diagnosis_procedure_ratio": ("number_diagnoses", "num_procedures"),
        }

        for name, (num, denom) in ratio_features.items():
            if num in df.columns and denom in df.columns:
                df[name] = df[num] / (df[denom] + 1)  # Add 1 to avoid division by zero

        return df

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations."""
        # Log transformation for skewed features
        for col in self.cfg.features.log_transform:
            if col in df.columns:
                df[f"{col}_log1p"] = np.log1p(df[col])

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        if not self.cfg.features.scaling.enabled:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_col = self.cfg.preprocessing.target_column

        # Exclude target and specific columns from scaling
        cols_to_scale = [
            col
            for col in numeric_cols
            if col != target_col and col not in self.cfg.features.scaling.exclude
        ]

        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Save scaler
        joblib.dump(scaler, self.cfg.paths.features / "scaler.joblib")

        return df

    def _add_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add timestamps for Feast."""
        df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
        df["created_timestamp"] = pd.Timestamp.now(tz="UTC")
        return df

    def _create_feast_features(self, df: pd.DataFrame) -> None:
        """Create and register feature views with Feast."""
        for group in self.cfg.features.feature_groups:
            # Create feature view
            source = FileSource(
                path=str(self.cfg.paths.features / f"{group.name}.parquet"),
                event_timestamp_column="event_timestamp",
                created_timestamp_column="created_timestamp",
            )

            features = [
                Feature(name=feat_name, dtype=ValueType.FLOAT)
                for feat_name in group.features
            ]

            view = FeatureView(
                name=f"{group.name}_view",
                entities=["patient_id"],
                ttl=timedelta(days=365),
                features=features,
                batch_source=source,
                online=True,
            )

            # Save group data
            group_df = df[
                group.features + ["patient_id", "event_timestamp", "created_timestamp"]
            ]
            group_df.to_parquet(self.cfg.paths.features / f"{group.name}.parquet")

            # Apply feature view
            self.store.apply([view])

    def _save_feature_metadata(self, df: pd.DataFrame) -> None:
        """Save feature metadata."""
        metadata = {
            "features": {
                col: {
                    "dtype": str(df[col].dtype),
                    "nunique": df[col].nunique(),
                    "missing": df[col].isnull().sum(),
                    "stats": df[col].describe().to_dict()
                    if pd.api.types.is_numeric_dtype(df[col])
                    else None,
                }
                for col in df.columns
            },
            "feature_groups": self.cfg.features.feature_groups,
            "transformations": {
                "log_transform": self.cfg.features.log_transform,
                "scaling": self.cfg.features.scaling.enabled,
                "interactions": self.cfg.features.interactions,
            },
        }

        with open(self.cfg.paths.features / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_featured_data(self, df: pd.DataFrame) -> None:
        """Save featured data."""
        df.to_parquet(self.cfg.paths.features / "data_featured.parquet")
