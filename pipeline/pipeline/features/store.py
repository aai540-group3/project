import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from feast import (
    Entity,
    Feature,
    FeatureService,
    FeatureStore,
    FeatureView,
    FileSource,
    ValueType,
)
from feast.repo_config import RepoConfig

logger = logging.getLogger(__name__)


class DiabetesFeatureStore:
    """Feature store for diabetes readmission data."""

    def __init__(self, cfg: Dict):
        """Initialize feature store.

        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.store: Optional[FeatureStore] = None
        self.feature_views: Dict[str, FeatureView] = {}
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize Feast feature store."""
        repo_config = RepoConfig(
            registry=self.cfg.feast.registry_path,
            project=self.cfg.feast.project,
            provider="aws",
            offline_store={"type": "file"},
            online_store={
                "type": "dynamodb",
                "region": self.cfg.aws.region,
                "table_name": self.cfg.feast.online_store.table_name,
            },
            entity_key_serialization_version=2,
        )

        self.store = FeatureStore(repo_config)

    def create_feature_views(self, data_path: Path) -> None:
        """Create feature views.

        Args:
            data_path: Path to feature data
        """
        # Create patient entity
        patient = Entity(
            name="patient_id",
            value_type=ValueType.INT64,
            description="Patient identifier",
        )

        # Create feature views for each group
        for group in self.cfg.feast.feature_views:
            source = FileSource(
                path=str(data_path / f"{group.name}.parquet"),
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
                ttl=timedelta(days=group.ttl),
                features=features,
                batch_source=source,
                online=True,
            )

            self.feature_views[group.name] = view

    def apply_feature_views(self) -> None:
        """Apply feature views to store."""
        if not self.store:
            raise RuntimeError("Feature store not initialized")

        self.store.apply([*self.feature_views.values()])

    def get_historical_features(
        self, entity_df: pd.DataFrame, feature_refs: List[str]
    ) -> pd.DataFrame:
        """Get historical features.

        Args:
            entity_df: Entity DataFrame
            feature_refs: List of feature references

        Returns:
            DataFrame with historical features
        """
        if not self.store:
            raise RuntimeError("Feature store not initialized")

        return self.store.get_historical_features(
            entity_df=entity_df, features=feature_refs
        ).to_df()

    def materialize_incremental(self, end_date: datetime) -> None:
        """Materialize features incrementally.

        Args:
            end_date: End date for materialization
        """
        if not self.store:
            raise RuntimeError("Feature store not initialized")

        self.store.materialize_incremental(end_date=end_date)

    def get_online_features(self, entity_rows: List[Dict]) -> Dict[str, List]:
        """Get online features.

        Args:
            entity_rows: List of entity dictionaries

        Returns:
            Dictionary of feature values
        """
        if not self.store:
            raise RuntimeError("Feature store not initialized")

        return self.store.get_online_features(
            entity_rows=entity_rows, features=self._get_all_feature_refs()
        ).to_dict()

    def _get_all_feature_refs(self) -> List[str]:
        """Get all feature references.

        Returns:
            List of feature references
        """
        feature_refs = []
        for view_name, view in self.feature_views.items():
            feature_refs.extend([f"{view_name}:{feat.name}" for feat in view.features])
        return feature_refs
