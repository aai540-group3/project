"""
Feature Store Integration
=========================

.. module:: pipeline.utils.feature_store
   :synopsis: Feature store management and integration

.. moduleauthor:: aai540-group3
"""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from feast import Entity, Feature, FeatureStore, FeatureView, ValueType
from feast.data_source import FileSource
from omegaconf import DictConfig

from ..utils.logging import get_logger

logger = get_logger(__name__)


class FeatureStoreManager:
    """Feature store management."""

    def __init__(self, cfg: DictConfig):
        """Initialize feature store manager.

        :param cfg: Feature store configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.store = FeatureStore(repo_path=cfg.feast.repo_path)
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize feature store."""
        # Create entities
        patient = Entity(
            name="patient_id",
            value_type=ValueType.INT64,
            description="Patient identifier",
        )

        # Create feature views for each group
        for group in self.cfg.feast.feature_views:
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
                ttl=timedelta(days=group.ttl),
                features=features,
                batch_source=source,
                online=True,
            )

            # Apply to store
            self.store.apply([patient, view])

    def get_historical_features(
        self, entity_df: pd.DataFrame, feature_refs: List[str]
    ) -> pd.DataFrame:
        """Get historical features.

        :param entity_df: Entity DataFrame
        :type entity_df: pd.DataFrame
        :param feature_refs: Feature references
        :type feature_refs: List[str]
        :return: Historical features
        :rtype: pd.DataFrame
        """
        return self.store.get_historical_features(
            entity_df=entity_df, features=feature_refs
        ).to_df()

    def get_online_features(self, entity_rows: List[Dict]) -> Dict[str, List]:
        """Get online features.

        :param entity_rows: Entity rows
        :type entity_rows: List[Dict]
        :return: Online features
        :rtype: Dict[str, List]
        """
        return self.store.get_online_features(
            entity_rows=entity_rows, features=self._get_all_feature_refs()
        ).to_dict()

    def _get_all_feature_refs(self) -> List[str]:
        """Get all feature references.

        :return: Feature references
        :rtype: List[str]
        """
        feature_refs = []
        for view in self.store.list_feature_views():
            feature_refs.extend([f"{view.name}:{feat.name}" for feat in view.features])
        return feature_refs

    def materialize_incremental(self, end_date: datetime) -> None:
        """Materialize features incrementally.

        :param end_date: End date
        :type end_date: datetime
        """
        self.store.materialize_incremental(end_date=end_date)
